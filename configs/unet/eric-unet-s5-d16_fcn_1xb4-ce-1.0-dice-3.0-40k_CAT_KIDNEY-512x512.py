_base_ = [
    '../_base_/models/fcn_unet_s5-d16-1channel.py',
    '../_base_/datasets/cat_kidney_grayscale.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# Import custom visualization hook
custom_imports = dict(imports=['custom_viz_hook'], allow_failed_imports=False)

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)  # Reduced LR for stability
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=500)

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    pad_val=0,
    seg_pad_val=0  # Changed from 255 to 0 to match background class
)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole'),
    decode_head=dict(
        num_classes=3,
        dropout_ratio=0.2,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[1.0, 2.5, 2.5]),  # Balanced weights
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=0.6)
        ]
    ),
    auxiliary_head=dict(
        num_classes=3,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4, class_weight=[1.0, 2.5, 2.5])  # Balanced weights
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
                     max_iters=20000,
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
        type='ResizedVisualizationHook',  # Custom hook that resizes images
        draw=True,  # Now enabled!
        interval=500,  # Visualize every 500 iterations
        show=False
    )
)