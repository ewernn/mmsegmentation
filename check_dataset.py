#!/usr/bin/env python3
"""Check dataset structure and find the RIGHT configuration"""

import os
import numpy as np
from PIL import Image

def check_voc_format():
    """Check VOC format dataset"""
    print("=" * 60)
    print("Checking VOC format dataset")
    print("=" * 60)

    data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'

    # Check structure
    folders = {
        'JPEGImages': os.path.exists(f'{data_root}/JPEGImages'),
        'SegmentationClass': os.path.exists(f'{data_root}/SegmentationClass'),
        'ImageSets/Segmentation': os.path.exists(f'{data_root}/ImageSets/Segmentation'),
    }

    print(f"Dataset root: {data_root}")
    for folder, exists in folders.items():
        print(f"  {folder}: {'✓' if exists else '✗'}")

    # Check train.txt
    train_file = f'{data_root}/ImageSets/Segmentation/train.txt'
    if os.path.exists(train_file):
        with open(train_file) as f:
            lines = f.readlines()
            print(f"\n  train.txt: {len(lines)} images")
            print(f"  First 3: {[l.strip() for l in lines[:3]]}")

    # Check actual masks
    if folders['SegmentationClass']:
        import glob
        masks = glob.glob(f'{data_root}/SegmentationClass/*.png')
        print(f"\n  Found {len(masks)} masks in SegmentationClass")

        if masks:
            # Check first mask
            mask = np.array(Image.open(masks[0]))
            unique = np.unique(mask)
            print(f"  First mask values: {unique}")
            print(f"  Has kidneys: {'YES' if (38 in unique or 75 in unique) else 'NO'}")

def check_simple_format():
    """Check simple train/val format"""
    print("\n" + "=" * 60)
    print("Checking simple train/val format")
    print("=" * 60)

    data_root = '/content/drive/MyDrive/MM/Seg/CAT_KIDNEY'

    # Check structure
    folders = {
        'train/images': os.path.exists(f'{data_root}/train/images'),
        'train/labels': os.path.exists(f'{data_root}/train/labels'),
        'val/images': os.path.exists(f'{data_root}/val/images'),
        'val/labels': os.path.exists(f'{data_root}/val/labels'),
    }

    print(f"Dataset root: {data_root}")
    for folder, exists in folders.items():
        print(f"  {folder}: {'✓' if exists else '✗'}")

    # Check train labels
    if folders['train/labels']:
        import glob
        masks = glob.glob(f'{data_root}/train/labels/*.png')
        print(f"\n  Found {len(masks)} masks in train/labels")

        if masks:
            # Check first mask
            mask = np.array(Image.open(masks[0]))
            unique = np.unique(mask)
            print(f"  First mask values: {unique}")
            print(f"  Has kidneys: {'YES' if (38 in unique or 75 in unique) else 'NO'}")

def suggest_fix():
    """Suggest the correct configuration"""
    print("\n" + "=" * 60)
    print("SUGGESTED FIX")
    print("=" * 60)

    voc_exists = os.path.exists('/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC/SegmentationClass')
    simple_exists = os.path.exists('/content/drive/MyDrive/MM/Seg/CAT_KIDNEY/train/labels')

    if voc_exists:
        print("✓ VOC format dataset exists")
        print("\nMake sure your config uses:")
        print("  data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'")
        print("  data_prefix = dict(img_path='JPEGImages', seg_map_path='SegmentationClass')")
        print("  ann_file = 'ImageSets/Segmentation/train.txt'")

    elif simple_exists:
        print("✓ Simple format dataset exists")
        print("\nYou need to update your config to use:")
        print("  data_root = '/content/drive/MyDrive/MM/Seg/CAT_KIDNEY'")
        print("  data_prefix = dict(img_path='train/images', seg_map_path='train/labels')")
        print("  # Remove or fix ann_file")

    else:
        print("✗ No valid dataset found!")
        print("Check your Google Drive mount and dataset location")

if __name__ == "__main__":
    check_voc_format()
    check_simple_format()
    suggest_fix()