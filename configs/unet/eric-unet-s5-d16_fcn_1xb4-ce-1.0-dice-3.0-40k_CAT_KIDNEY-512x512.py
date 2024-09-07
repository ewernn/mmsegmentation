_base_ = [
    '../_base_/models/fcn_unet_s5-d16-1channel.py', '../_base_/datasets/cat_kidney_grayscale.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512,512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    #test_cfg=dict(crop_size=(256, 256), stride=(170, 170)),
    test_cfg=dict(mode='whole'),
#92,62,41,4 show 81 for obstruction, 
    decode_head=dict(
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
)

# schedule modifications
# default_hooks = dict(
#     visualization=dict(type='SegVisualizationHook', draw=True, interval=1, name='visualizer')
# )
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    #save_dir='/path/to/save/directory'  # Specify the save directory here
)

default_hooks = dict(
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        interval=1,
        show=False,  # Set this to False to ensure saving instead of showing
    )
)
