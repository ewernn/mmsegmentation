#!/usr/bin/env python3
"""Test different CLAHE settings to see effect on kidney visibility"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def apply_clahe(img, clip_limit):
    """Apply CLAHE to grayscale image"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img)

# Test on a sample image
img_path = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC/JPEGImages/Im1876.jpg'
mask_path = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC/SegmentationClass/Im1876.png'

img = np.array(Image.open(img_path).convert('L'))
mask = np.array(Image.open(mask_path))

# Try different CLAHE settings
clip_limits = [0, 2.0, 3.0, 4.0]
results = []

for clip in clip_limits:
    if clip == 0:
        enhanced = img  # No CLAHE
    else:
        enhanced = apply_clahe(img, clip)

    # Calculate contrast in kidney regions
    left_kidney_region = enhanced[mask == 1]
    right_kidney_region = enhanced[mask == 2]
    background_region = enhanced[mask == 0]

    if len(left_kidney_region) > 0 and len(right_kidney_region) > 0:
        left_contrast = left_kidney_region.std()
        right_contrast = right_kidney_region.std()

        # Contrast between kidney and background
        left_vs_bg = abs(left_kidney_region.mean() - background_region.mean())
        right_vs_bg = abs(right_kidney_region.mean() - background_region.mean())

        print(f"\nCLAHE clip_limit={clip}:")
        print(f"  Left kidney contrast: {left_contrast:.2f}, vs background: {left_vs_bg:.2f}")
        print(f"  Right kidney contrast: {right_contrast:.2f}, vs background: {right_vs_bg:.2f}")
        print(f"  Ratio (right/left contrast): {right_contrast/left_contrast:.2f}")

        if abs(right_contrast - left_contrast) / max(right_contrast, left_contrast) > 0.15:
            print(f"  ⚠️ Kidneys have different internal contrast!")