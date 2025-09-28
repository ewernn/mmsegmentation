# dataset settings
dataset_type = 'CatKidneyDataset'  # Update the dataset type if needed
################### TO CHANGE data_root #########################
#data_root = 'data/hoof_lat_block_voc'  # Update the data root path to where your dataset is stored
data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'  
custom_imports = dict(imports=['mmseg.datasets.cat_kidney'], allow_failed_imports=False)


# Assuming the images are of a uniform size, specify the actual size if known
#img_scale = (1000, 1000)  # Update this based on your image sizes
img_scale = (512, 512)  # Update this based on your image sizes
#color_type
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHEGrayscale', clip_limit=4.0, tile_grid_size=(8, 8)),
    dict(type='GrayscalePhotoMetricDistortion', brightness_delta=32, contrast_range=(0.5, 1.5)),
    dict(type='RandomRotate', degree=10, prob=0.5, pad_val=0, seg_pad_val=0),  # seg_pad_val=0 critical!
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    # dict(type='RemapLabels', label_map={0: 0, 38: 1, 75: 2}),  # DISABLED - masks already have [0,1,2]
    dict(type='PackSegInputs')
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='CLAHEGrayscale', clip_limit=4.0, tile_grid_size=(8, 8)),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='ResetOriShape'),  # Keep predictions at 512x512 for validation
    # dict(type='RemapLabels', label_map={0: 0, 38: 1, 75: 2}),  # DISABLED - masks already have [0,1,2]
    dict(type='PackSegInputs')
]

test_pipeline = val_pipeline

# Test Time Augmentation (TTA) setup
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # Adjust these ratios based on the typical image scales
tta_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            # [
            #     dict(type='Resize', scale_factor=r, keep_ratio=False)
            #     for r in img_ratios
            # ],
            [
                # dict(type='RandomFlip', prob=0., direction='horizontal'),
                # dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
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
    num_workers=2,
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
