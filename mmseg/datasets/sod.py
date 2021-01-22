import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmseg.core import mean_iou, F_measure, P_R, MAE
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class SODDataset(Dataset):

    CLASSES = ('background', 'salient')

    PALETTE = [[0, 0, 0], [255, 255, 255]]


    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_file = osp.join(img_dir, img_name + img_suffix)
                    img_info = dict(filename=img_file)
                    if ann_dir is not None:
                        seg_map = osp.join(ann_dir, img_name + seg_map_suffix)
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_file = osp.join(img_dir, img)
                img_info = dict(filename=img_file)
                if ann_dir is not None:
                    seg_map = osp.join(ann_dir,
                                       img.replace(img_suffix, seg_map_suffix))
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix)

        return result_files, tmp_d

    def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        # i = 0
        for img_info in self.img_infos:
            gt_seg_map = mmcv.imread(
                img_info['ann']['seg_map'], flag='unchanged', backend='pillow')
            maxm = (gt_seg_map).max()
            if maxm > 1:
                gt_seg_map = gt_seg_map / maxm

            gt_seg_maps.append(gt_seg_map.astype(np.float))

        return gt_seg_maps

    def evaluate(self, results, metric='F_measure', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        allowed_metrics = ['MAE', 'F_measure', 'P_R']

        if isinstance(metric, str):
            metric = [metric]

        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps()
        for eval_name in metric:
            if eval_name not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(eval_name))
            if eval_name == 'F_measure':
                Fmeasure, P, R = F_measure(results, gt_seg_maps, thersholds=0.5, belt=0.3, global_com=False)
                summary_str = ''
                summary_str += 'thersholds=0.5, Fmeasure results:'
                line_format = '{:>10} {:>10} {:>10}\n'
                summary_str += line_format.format('Fmeasure', 'P', 'R')
                Fmeasure_str = '{:.4f}'.format(Fmeasure)
                P_str = '{:.4f}'.format(P)
                R_str = '{:.4f}'.format(R)
                summary_str += line_format.format('global', Fmeasure_str, P_str, R_str)
                eval_results['Fmeasure'] = Fmeasure
                eval_results['P'] = P
                eval_results['R'] = R
            elif eval_name == 'P_R':
                Ps, Rs, AP, maxFmeasure = P_R(results, gt_seg_maps, steps=0.1, thersholds=None, belt=0.3, global_com=False)
                summary_str = ''
                summary_str += 'P_R results:'
                line_format = '{:>10} {:>10} \n'
                summary_str += line_format.format('AP', 'maxFmeasure')
                AP_str = '{:.4f}'.format(AP)
                maxFmeasure_str = '{:.4f}'.format(maxFmeasure)
                summary_str += line_format.format('global', AP_str, maxFmeasure_str)
                eval_results['maxFmeasure'] = maxFmeasure
                eval_results['AP'] = AP
                eval_results['Ps'] = Ps
                eval_results['Rs'] = Rs
            elif eval_name == 'MAE':
                mae = MAE(results, gt_seg_maps, global_com=False)
                eval_results['MAE'] = mae
                summary_str = ''
                summary_str += 'MAE results: '
                line_format = '{:>10}'
                summary_str += line_format.format('MAE')
                MAE_str = '{:.4f}'.format(mae)
                summary_str += line_format.format('global', MAE_str)
                eval_results['MAE'] = mae
            else:
                pass

        print_log(summary_str, logger)

        return eval_results
