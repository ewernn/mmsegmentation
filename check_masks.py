import cv2
import numpy as np
from glob import glob

data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'
mask_paths = glob(f'{data_root}/SegmentationClass/*.png')

if len(mask_paths) == 0:
    mask_paths = glob(f'{data_root}/SegmentationClass/*.jpg')

print(f"Found {len(mask_paths)} masks")

if len(mask_paths) == 0:
    print(f"No masks found in {data_root}/SegmentationClass/")
    print("Please update data_root in the script")
else:
    all_unique_values = set()

    for i, mask_path in enumerate(mask_paths[:10]):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        unique_values = np.unique(mask)
        all_unique_values.update(unique_values)

        if i < 3:
            print(f"\n{mask_path}")
            print(f"  Shape: {mask.shape}")
            print(f"  Unique values: {unique_values}")
            print(f"  Value counts: {[(val, np.sum(mask == val)) for val in unique_values]}")

    print(f"\n{'='*60}")
    print(f"All unique values across {min(10, len(mask_paths))} masks: {sorted(all_unique_values)}")
    print(f"{'='*60}")

    if all_unique_values == {0, 1, 2}:
        print("✓ Masks look correct! Classes are [0, 1, 2]")
    elif 255 in all_unique_values:
        print("✗ WARNING: Found value 255 in masks (ignore label)")
        print("  This might be causing the CUDA error")
    else:
        print(f"✗ WARNING: Unexpected values in masks: {sorted(all_unique_values)}")
        print(f"  Expected [0, 1, 2] for background, left_kidney, right_kidney")