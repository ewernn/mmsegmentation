#!/usr/bin/env python3
"""Emergency debug script to find where kidney labels are disappearing"""

import numpy as np
import os
import sys

# Add path for imports
sys.path.insert(0, '/content/mmsegmentation')

def test_raw_data():
    """Test 1: Check raw mask files directly"""
    print("=" * 50)
    print("TEST 1: Checking raw mask files")
    print("=" * 50)

    data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'

    # Check a few training masks
    for i in range(5):
        mask_file = f'{data_root}/SegmentationClass/image{i:05d}.png'
        if os.path.exists(mask_file):
            from PIL import Image
            mask = np.array(Image.open(mask_file))
            unique = np.unique(mask)
            print(f"\nMask {i}: shape={mask.shape}, unique values={unique}")
            print(f"  Pixel counts: 0={np.sum(mask==0)}, 38={np.sum(mask==38)}, 75={np.sum(mask==75)}")

            if 38 in unique or 75 in unique:
                print(f"  ✓ KIDNEY PIXELS FOUND IN RAW FILE!")
            else:
                print(f"  ✗ NO KIDNEY PIXELS IN RAW FILE!")

def test_dataloader():
    """Test 2: Check what the dataloader actually loads"""
    print("\n" + "=" * 50)
    print("TEST 2: Testing dataloader output")
    print("=" * 50)

    from mmengine.config import Config
    from mmseg.registry import DATASETS

    # Load config
    cfg = Config.fromfile('configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py')

    # Build dataset
    train_dataset = DATASETS.build(cfg.train_dataloader['dataset'])

    # Check first few samples
    for i in range(min(5, len(train_dataset))):
        data = train_dataset[i]

        # Check data_samples
        if 'data_samples' in data:
            gt_seg = data['data_samples'].gt_sem_seg.data.numpy()
            unique = np.unique(gt_seg)
            print(f"\nSample {i}: shape={gt_seg.shape}, unique values={unique}")
            print(f"  Class counts: 0={np.sum(gt_seg==0)}, 1={np.sum(gt_seg==1)}, 2={np.sum(gt_seg==2)}")

            if 1 in unique or 2 in unique:
                print(f"  ✓ KIDNEY CLASSES FOUND AFTER PIPELINE!")
            else:
                print(f"  ✗ NO KIDNEY CLASSES AFTER PIPELINE!")

def test_transform_pipeline():
    """Test 3: Step through transform pipeline manually"""
    print("\n" + "=" * 50)
    print("TEST 3: Step-by-step transform pipeline")
    print("=" * 50)

    from mmcv.transforms import Compose
    from mmseg.datasets.transforms import (
        LoadImageFromFile, LoadAnnotations, CLAHEGrayscale,
        RandomRotate, Resize, RemapLabels, PackSegInputs
    )

    # Create pipeline
    pipeline = [
        LoadImageFromFile(color_type='grayscale'),
        LoadAnnotations(),
        CLAHEGrayscale(clip_limit=4.0, tile_grid_size=(8, 8)),
        RandomRotate(degree=10, prob=0.5, pad_val=0, seg_pad_val=0),
        Resize(scale=(512, 512), keep_ratio=False),
        RemapLabels(label_map={0: 0, 38: 1, 75: 2}),
        PackSegInputs()
    ]

    data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'

    # Test on first image
    results = {
        'img_path': f'{data_root}/JPEGImages/image00000.jpg',
        'seg_map_path': f'{data_root}/SegmentationClass/image00000.png',
        'reduce_zero_label': False,
        'seg_fields': []
    }

    for i, transform in enumerate(pipeline):
        print(f"\nStep {i}: {transform.__class__.__name__}")

        if i == 0:  # LoadImageFromFile
            results = transform(results)
            print(f"  Image shape: {results['img'].shape}")

        elif i == 1:  # LoadAnnotations
            results = transform(results)
            if 'gt_seg_map' in results:
                unique = np.unique(results['gt_seg_map'])
                print(f"  Mask shape: {results['gt_seg_map'].shape}")
                print(f"  Unique values: {unique}")
                print(f"  Counts: 0={np.sum(results['gt_seg_map']==0)}, "
                      f"38={np.sum(results['gt_seg_map']==38)}, "
                      f"75={np.sum(results['gt_seg_map']==75)}")

        elif i == 4:  # After Resize
            results = transform(results)
            if 'gt_seg_map' in results:
                unique = np.unique(results['gt_seg_map'])
                print(f"  After resize unique values: {unique}")
                print(f"  Counts: 0={np.sum(results['gt_seg_map']==0)}, "
                      f"38={np.sum(results['gt_seg_map']==38)}, "
                      f"75={np.sum(results['gt_seg_map']==75)}")

        elif i == 5:  # After RemapLabels
            results = transform(results)
            if 'gt_seg_map' in results:
                unique = np.unique(results['gt_seg_map'])
                print(f"  After remap unique values: {unique}")
                print(f"  Counts: 0={np.sum(results['gt_seg_map']==0)}, "
                      f"1={np.sum(results['gt_seg_map']==1)}, "
                      f"2={np.sum(results['gt_seg_map']==2)}")

                if 1 in unique or 2 in unique:
                    print(f"  ✓ KIDNEY CLASSES PRESENT!")
                else:
                    print(f"  ✗ KIDNEY CLASSES MISSING!")
        else:
            results = transform(results)

def test_model_output():
    """Test 4: Check what the model is actually outputting"""
    print("\n" + "=" * 50)
    print("TEST 4: Model output check")
    print("=" * 50)

    import torch
    from mmengine.config import Config
    from mmseg.registry import MODELS

    # Load config
    cfg = Config.fromfile('configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py')

    # Build model
    model = MODELS.build(cfg.model)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 1, 512, 512)  # batch=1, channels=1, H=512, W=512

    with torch.no_grad():
        # Get model output
        output = model.decode_head.forward([dummy_input])
        print(f"Model output shape: {output.shape}")  # Should be [1, 3, 512, 512]
        print(f"Output channels (num_classes): {output.shape[1]}")

        # Check predictions
        preds = output.argmax(dim=1)
        unique_preds = torch.unique(preds)
        print(f"Unique predictions: {unique_preds.tolist()}")

        # Check if model can output all classes
        print(f"\nChecking if model architecture supports 3 classes:")
        print(f"  Decode head num_classes: {model.decode_head.num_classes}")
        print(f"  Auxiliary head num_classes: {model.auxiliary_head.num_classes}")

if __name__ == "__main__":
    print("EMERGENCY DEBUGGING - Finding where kidney labels disappear\n")

    try:
        test_raw_data()
    except Exception as e:
        print(f"Error in test_raw_data: {e}")

    try:
        test_dataloader()
    except Exception as e:
        print(f"Error in test_dataloader: {e}")

    try:
        test_transform_pipeline()
    except Exception as e:
        print(f"Error in test_transform_pipeline: {e}")

    try:
        test_model_output()
    except Exception as e:
        print(f"Error in test_model_output: {e}")

    print("\n" + "=" * 50)
    print("DEBUGGING COMPLETE - Check results above")
    print("=" * 50)