_base_ = [
    '../_base_/models/deeplabv3_unet_s5-d16.py', '../_base_/datasets/solar.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    #test_cfg=dict(crop_size=(256, 256), stride=(170, 170)),
    test_cfg=dict(mode='whole'),

    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                #class_weight=[1.0, 2.0, 10.0]  # Example weights for three classes
                class_weight=[1.0, 2.0, 10.0],  # Example weights for three classes
                avg_non_ignore=True
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ])
)

# schedule modifications
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True)
)
#test_cfg = None