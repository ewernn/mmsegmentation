#!/usr/bin/env python3
"""Check if left and right kidneys have different pixel counts in dataset"""

import numpy as np
from PIL import Image
import glob

data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'
masks = glob.glob(f'{data_root}/SegmentationClass/*.png')[:50]  # Sample 50 masks

left_pixels = 0
right_pixels = 0
total_pixels = 0

for mask_file in masks:
    mask = np.array(Image.open(mask_file))
    left_pixels += np.sum(mask == 1)
    right_pixels += np.sum(mask == 2)
    total_pixels += mask.size

print(f"Sampled {len(masks)} masks")
print(f"Left kidney (class 1): {left_pixels:,} pixels ({100*left_pixels/total_pixels:.2f}%)")
print(f"Right kidney (class 2): {right_pixels:,} pixels ({100*right_pixels/total_pixels:.2f}%)")
print(f"Ratio (right/left): {right_pixels/left_pixels:.2f}")

if abs(right_pixels - left_pixels) / max(right_pixels, left_pixels) > 0.2:
    print("\n⚠️ SIGNIFICANT IMBALANCE DETECTED!")
    print("This could explain why one kidney performs better than the other.")