import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_loss
from .utils import weight_reduce_loss



@LOSSES.register_module()
class BCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 with_logits=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(BCELoss, self).__init__()
        if with_logits:
            self.becloss = F.binary_cross_entropy_with_logits
            self.use_sigmoid = False
        else:
            self.becloss = F.binary_cross_entropy
            self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                other_lbl=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * self.becloss(
                cls_score.sigmoid().reshape(cls_score.shape),
                label.float(),
                weight= None,
                reduction=reduction)
            return loss_cls

        loss_cls = self.loss_weight * self.becloss(
            cls_score,
            label.float().reshape(cls_score.shape),
            weight= None,
            reduction=reduction)
        return loss_cls
