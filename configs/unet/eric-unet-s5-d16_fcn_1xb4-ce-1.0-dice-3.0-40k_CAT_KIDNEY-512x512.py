_base_ = [
    '../_base_/models/fcn_unet_s5-d16-1channel.py', 
    '../_base_/datasets/cat_kidney_grayscale.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole'),
    decode_head=dict(
        num_classes=3,
        loss_decode=[  # List of losses
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
            dict(type='DiceLoss', use_sigmoid=True, loss_weight=0.5)
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
                 name='combined-loss-experiment',
                 config=dict(
                     loss_type='CE+Dice',
                     ce_weight=0.5,
                     dice_weight=0.5,
                     model_name='UNet-S5-D16',
                     dataset='cat_kidney'
                 )
             ))
    ],
    name='visualizer',
)

default_hooks = dict(
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=1,
        show=False,
    )
)