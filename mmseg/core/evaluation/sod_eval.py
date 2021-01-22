import numpy as np


def F_measure(results, gt_seg_maps, thersholds=0.5, belt=0.3, global_com=False):
    """Calculate Intersection and Union (IoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        belt (float): belt of F-measure
        global_com (bool): True :compute P and R in per img; False : compute together in all imgs

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    if global_com:
        TP = ((gt_seg_maps == 1) * (results == 1)).sum()
        FP = ((gt_seg_maps == 0) * (results == 1)).sum()
        FN = ((gt_seg_maps == 1) * (results == 0)).sum()
        R = TP/(TP+FN+1)
        P = TP/(TP+FP+1)
    else:
        prec = np.zeros((num_imgs, ), dtype=np.float)
        reca = np.zeros((num_imgs, ), dtype=np.float)
        for i in range(num_imgs):
            if results[i].shape[1] == 2:
                results[i] = np.argmax(results[i], axis=1).reshape(gt_seg_maps[i].shape)
            else:
                results[i] = (results[i]>=thersholds).reshape(gt_seg_maps[i].shape)
            TP = ((gt_seg_maps[i] == 1) * (results[i] == 1)).sum()
            FP = ((gt_seg_maps[i] == 0) * (results[i] == 1)).sum()
            FN = ((gt_seg_maps[i] == 1) * (results[i] == 0)).sum()
            reca[i] = TP/(TP+FN+1)
            prec[i] = TP/(TP+FP+1)
        P = prec.mean()
        R = reca.mean()
    Fmeasure=((1+belt)*P*R)/(belt*P+R)

    return Fmeasure, P, R


def P_R(results, gt_seg_maps, steps=0.1, thersholds=None, belt=0.3, global_com=False):
    """Calculate Intersection and Union (IoU)

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        steps (float): steps of thersholds, if thersholds is None, use it.
        thersholds (list[float]): thersholds list
        global_com (bool): True :compute P and R in per img; False : compute together in all imgs

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    if thersholds is None :
        thersholds = np.arange(0, 1+steps, steps)
    Ps, Rs = [], []
    if global_com:
        for t in thersholds:
            res = results >= t
            TP = ((gt_seg_maps == 1) * (res == 1)).sum()
            FP = ((gt_seg_maps == 0) * (res == 1)).sum()
            FN = ((gt_seg_maps == 1) * (res == 0)).sum()
            Rs.append(TP/(TP+FN+1))
            Ps.append(TP/(TP+FP+1))
    else:
        for t in thersholds:
            prec = np.zeros((num_imgs, ), dtype=np.float)
            reca = np.zeros((num_imgs, ), dtype=np.float)
            for i in range(num_imgs):
                if results[i].shape[1] == 2:
                    results[i] = np.argmax(results[i], axis=1).reshape(gt_seg_maps[i].shape)
                else:
                    res = (results[i] >= t).reshape(gt_seg_maps[i].shape)
                TP = ((gt_seg_maps[i] == 1) * (res == 1)).sum()
                FP = ((gt_seg_maps[i] == 0) * (res == 1)).sum()
                FN = ((gt_seg_maps[i] == 1) * (res == 0)).sum()
                reca[i] = TP/(TP+FN+1)
                prec[i] = TP/(TP+FP+1)
            Ps.append(prec.mean())
            Rs.append(reca.mean())
    AP = Ps[0] * Rs[0]
    Fmeasure = 0
    for i in range(1, len(Ps)):
        AP += (Ps[i]-Ps[i-1]) * Rs[i]
        Fmeasure= max(((1+belt)*Ps[i]*Rs[i])/(belt*Ps[i]+Rs[i]), Fmeasure)
    return Ps, Rs, AP, maxFmeasure

def MAE(results, gt_seg_maps, global_com=False):
    """Calculate MAE

    Args:
        results (list[ndarray]): List of prediction segmentation maps
        gt_seg_maps (list[ndarray]): list of ground truth segmentation maps
        global_com (bool): True :compute P and R in per img; False : compute together in all imgs

     Returns:
         float: Overall accuracy on all images.
         ndarray: Per category accuracy, shape (num_classes, )
         ndarray: Per category IoU, shape (num_classes, )
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    if global_com:
        mae = np.abs(results, gt_seg_maps).mean()
    else:
        mae = 0
        for i in range(num_imgs):
            if results[i].shape[1] == 2:
                results[i] = np.argmax(results[i], axis=1)
            # print(i, results[i].shape, gt_seg_maps[i].shape)
            mae += np.abs(results[i].reshape(gt_seg_maps[i].shape)-gt_seg_maps[i]).mean()
        mae /= num_imgs

    return mae