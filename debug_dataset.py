import sys
import os
sys.path.insert(0, '/content/mmsegmentation')
os.chdir('/content/mmsegmentation')

from mmseg.datasets import CatKidneyDataset
import numpy as np

# Test the get_label_map method
label_map = CatKidneyDataset.get_label_map(None)
print(f"get_label_map() returns: {label_map}")

# Create a minimal dataset instance
dataset = CatKidneyDataset(
    data_root='/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
    ann_file='ImageSets/Segmentation/train.txt',
    pipeline=[
        dict(type='LoadImageFromFile', color_type='grayscale'),
        dict(type='LoadAnnotations'),
    ]
)

print(f"Dataset label_map attribute: {dataset.label_map}")
print(f"Dataset length: {len(dataset)}")

# Get first sample
sample = dataset[0]
if 'gt_seg_map' in sample:
    unique_vals = np.unique(sample['gt_seg_map'])
    print(f"First sample gt_seg_map unique values: {unique_vals}")
    if set(unique_vals) == {0, 1, 2}:
        print("✓ Remapping WORKED!")
    else:
        print("✗ Remapping FAILED!")