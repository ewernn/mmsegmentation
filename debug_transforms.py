#!/usr/bin/env python3
"""Debug script to test transform pipeline and see where labels get lost"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ewern/code/metronmind/mmsegmentation')

from mmengine.config import Config
from mmseg.datasets.transforms import (
    LoadImageFromFile, LoadAnnotations, CLAHEGrayscale,
    GrayscalePhotoMetricDistortion, RandomRotate, Resize,
    RemapLabels, PackSegInputs
)
from mmcv.transforms import Compose

# Load the config to get the exact pipeline
cfg = Config.fromfile('configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py')

# Get a sample data path
data_root = '/content/drive/MyDrive/MM/Seg/CAT_KIDNEY'
img_path = f'{data_root}/train/images/image00003.jpg'
seg_path = f'{data_root}/train/labels/image00003.png'

# Test the pipeline step by step
def test_pipeline():
    # Initial data dict
    results = {
        'img_path': img_path,
        'seg_map_path': seg_path,
        'reduce_zero_label': False,
        'seg_fields': ['gt_seg_map']
    }

    # Step 1: Load image
    loader = LoadImageFromFile(color_type='grayscale')
    results = loader(results)
    print(f"After LoadImageFromFile: img shape = {results['img'].shape}")

    # Step 2: Load annotations
    ann_loader = LoadAnnotations()
    results = ann_loader(results)
    unique_before = np.unique(results['gt_seg_map'])
    print(f"After LoadAnnotations: mask shape = {results['gt_seg_map'].shape}")
    print(f"  Unique values in mask: {unique_before}")
    print(f"  Kidney pixel counts: 38={np.sum(results['gt_seg_map']==38)}, 75={np.sum(results['gt_seg_map']==75)}")

    # Step 3: CLAHE
    clahe = CLAHEGrayscale(clip_limit=4.0, tile_grid_size=(8, 8))
    results = clahe(results)
    print(f"After CLAHE: mask unique values = {np.unique(results['gt_seg_map'])}")

    # Step 4: Photo distortion (skip for testing)

    # Step 5: Resize
    resize = Resize(scale=(512, 512), keep_ratio=False)
    results = resize(results)
    unique_after_resize = np.unique(results['gt_seg_map'])
    print(f"After Resize (1000x1000 -> 512x512):")
    print(f"  Unique values in mask: {unique_after_resize}")
    print(f"  Kidney pixel counts: 38={np.sum(results['gt_seg_map']==38)}, 75={np.sum(results['gt_seg_map']==75)}")

    # Check if we have unexpected values
    unexpected = [v for v in unique_after_resize if v not in [0, 38, 75]]
    if unexpected:
        print(f"  WARNING: Unexpected values after resize: {unexpected}")

    # Step 6: RemapLabels
    remap = RemapLabels(label_map={0: 0, 38: 1, 75: 2})
    results = remap(results)
    unique_after_remap = np.unique(results['gt_seg_map'])
    print(f"After RemapLabels:")
    print(f"  Unique values in mask: {unique_after_remap}")
    print(f"  Class counts: 0={np.sum(results['gt_seg_map']==0)}, 1={np.sum(results['gt_seg_map']==1)}, 2={np.sum(results['gt_seg_map']==2)}")

    # Check if kidneys disappeared
    if 1 not in unique_after_remap and 38 in unique_before:
        print("  ERROR: Left kidney (class 1) disappeared!")
    if 2 not in unique_after_remap and 75 in unique_before:
        print("  ERROR: Right kidney (class 2) disappeared!")

if __name__ == "__main__":
    test_pipeline()