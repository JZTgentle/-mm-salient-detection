from PIL import Image
import numpy as np
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmsegmentation format')
    parser.add_argument('gt_path', help='gt_path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    gt_path = args.gt_path
    out_path = args.out_dir
    if args.out_dir is None:
        out_dir = osp.join(gt_path)
    else:
        out_dir = args.out_dir
    files = os.listdir(gt_path)
    num_imgs = len(files)
    n = 0
    for filename in files:
        img = np.array(Image.open(gt_path+filename))
        if filename[-3:] != 'png':
            print("is not png: ", filename)
            filename = filename[:-3]
            filename += 'png'
        if img.shape[-1] == 3:
            print("channel > 1: ", filename)
            n += 1
            img = img.sum(-1)
        img = Image.fromarray(img.astype('uint8'))
        img.save(out_path+filename)

    print('Done! processing imgs: ', n/num_imgs)


if __name__ == '__main__':
    main()