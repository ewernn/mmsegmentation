_base_ = [
    '../_base_/models/fcn_unet_s5-d16-1channel.py',
    '../_base_/datasets/cat_kidney_grayscale.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=12000, val_interval=500)

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255
)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
    decode_head=dict(
        num_classes=3,
        dropout_ratio=0.2,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[0.5, 2.0, 2.0]),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.6)
        ]
    )
)

visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend',
             init_kwargs=dict(
                 project='cat-kidney-segmentation',
                 name='ce-dice-0.4-0.6-clahe-lr0.01',
                 config=dict(
                     loss_type='CE+Dice',
                     ce_weight=0.4,
                     dice_weight=0.6,
                     model_name='UNet-S5-D16',
                     dataset='cat_kidney',
                     lr=0.01,
                     max_iters=12000,
                     clahe=True,
                     dropout=0.2,
                     class_weights=[0.5, 2.0, 2.0]
                 )
             ))
    ],
    name='visualizer',
)

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
        patience=3,
        min_delta=0.001
    ),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=1,
        show=False,
    )
)