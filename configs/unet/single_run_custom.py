_base_ = [
    '../_base_/models/fcn_unet_s5-d16-1channel.py',
    '../_base_/datasets/cat_kidney_grayscale.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# Import custom visualization hook
custom_imports = dict(imports=['custom_viz_hook'], allow_failed_imports=False)

optimizer = dict(type='SGD', lr=0.0065, momentum=0.9, weight_decay=0.0005)
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
        dropout_ratio=0.15,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3, class_weight=[1.0, 2.5, 2.5]),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.7)
        ]
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.3, class_weight=[1.0, 2.5, 2.5])
    )
)

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 project='cat-kidney-segmentation',
                 name='single-run-test-lr0065-dropout015-dice07-clahe15',
                 config=dict(
                     loss_type='CE+Dice',
                     ce_weight=0.3,
                     dice_weight=0.7,
                     model_name='UNet-S5-D16',
                     dataset='cat_kidney',
                     lr=0.0065,
                     max_iters=20000,
                     clahe_clip_limit=1.5,
                     dropout=0.15,
                     class_weights=[1.0, 2.5, 2.5]
                 )
             ))
    ],
    name='visualizer',
)

# Override train and val pipelines to use CLAHE 1.5
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHEGrayscale', clip_limit=1.5, tile_grid_size=(8, 8)),
    dict(type='GrayscalePhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5)),
    dict(type='RandomRotate', degree=10, prob=0.5, pad_val=0, seg_pad_val=0),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHEGrayscale', clip_limit=1.5, tile_grid_size=(8, 8)),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='ResetOriShape'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(batch_size=4)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mDice',
        rule='greater',
        max_keep_ckpts=3
    ),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='mDice',
        rule='greater',
        patience=6,
        min_delta=0.001
    ),
    visualization=dict(
        type='ResizedVisualizationHook',
        draw=True,
        interval=500,
        show=False
    )
)
