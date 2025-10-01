# Clean sweep config without custom_imports to avoid YAPF errors
_base_ = [
    '../_base_/models/fcn_unet_s5-d16-1channel.py',
    '../_base_/datasets/cat_kidney_grayscale.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# NO custom_imports - skip visualization hook for sweeps

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=500)

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    pad_val=0,
    seg_pad_val=0
)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole'),
    decode_head=dict(
        num_classes=3,
        dropout_ratio=0.2,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[1.0, 3.0, 3.0]),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.6)
        ]
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[1.0, 3.0, 3.0]),
        ]
    )
)

# Disable WandB visualizer for sweep
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# Disable visualization hook
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=False)
)
