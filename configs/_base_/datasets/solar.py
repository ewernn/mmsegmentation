# dataset settings
dataset_type = 'SolarHoofDataset'  # Update the dataset type if needed
#data_root = 'data/voc_data'  # Update the data root path to where your dataset is stored
data_root = 'data/data_no_scale_out'  # Update the data root path to where your dataset is stored
custom_imports = dict(imports=['mmseg.datasets.hoof_solar'], allow_failed_imports=False)


# Assuming the images are of a uniform size, specify the actual size if known
img_scale = (512, 512)  # Update this based on your image sizes

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=(256, 256), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomRotate',degree=20,prob=0.5,pad_val=0),#seg_pad_val=255),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PhotoMetricDistortion', brightness_delta=32,contrast_range=(0.5, 1.5),saturation_range=(0.5, 1.5),hue_delta=18),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    #dict(type='Pad', size=(512, 512), pad_val=0),# seg_pad_val=255),
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# Test Time Augmentation (TTA) setup
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # Adjust these ratios based on the typical image scales
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=val_pipeline)
)
test_dataloader = val_dataloader


# Evaluation metrics
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator
