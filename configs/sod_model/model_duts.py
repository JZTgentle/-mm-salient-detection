_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/duts.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_sodeval.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='SODEncoderDecoder',
    decode_head=dict(
        num_classes=1,
        loss_decode=dict(
            type='BCELoss', use_sigmoid=True, with_logits=True, loss_weight=1.0)),
    auxiliary_head=dict(
        num_classes=1,
        loss_decode=dict(
            type='BCELoss', use_sigmoid=True, with_logits=True, loss_weight=1.0)))
