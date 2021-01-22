# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                 warmup='linear',
                 warmup_iters=500, 
                 warmup_ratio=0.05,
                 warmup_by_epoch=False)
# runtime settings
total_iters = 80000
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=200, metric='MAE')
