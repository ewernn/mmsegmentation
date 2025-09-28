#!/usr/bin/env python3
"""Quick check to see if the dataset path is correct and has kidney pixels"""

import numpy as np
from PIL import Image
import glob

# Try both possible paths
paths = [
    '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC',
    '/content/drive/MyDrive/MM/Seg/CAT_KIDNEY'
]

for data_root in paths:
    print(f"\nChecking: {data_root}")

    # Check for SegmentationClass folder
    seg_pattern = f'{data_root}/SegmentationClass/*.png'
    seg_files = glob.glob(seg_pattern)

    if seg_files:
        print(f"Found {len(seg_files)} segmentation masks")

        # Check first 3 masks
        for mask_file in seg_files[:3]:
            mask = np.array(Image.open(mask_file))
            unique = np.unique(mask)
            print(f"  {mask_file.split('/')[-1]}: values={unique}, "
                  f"kidneys={'YES' if (38 in unique or 75 in unique) else 'NO'}")

    # Check for labels folder (alternative)
    label_pattern = f'{data_root}/labels/*.png'
    label_files = glob.glob(label_pattern)

    if label_files:
        print(f"Found {len(label_files)} labels")

        # Check first 3 labels
        for label_file in label_files[:3]:
            mask = np.array(Image.open(label_file))
            unique = np.unique(mask)
            print(f"  {label_file.split('/')[-1]}: values={unique}, "
                  f"kidneys={'YES' if (38 in unique or 75 in unique) else 'NO'}")