# MMSegmentation - Custom Medical Image Segmentation

Fork of MMSegmentation configured for cat kidney and horse hoof segmentation with grayscale images, CLAHE preprocessing, and custom metrics.

## Repository Structure

```
mmsegmentation/
├── configs/                    # Training configurations
│   ├── _base_/                # Base configs (datasets, models, schedules)
│   └── unet/                  # Custom UNet configs for medical segmentation
├── mmseg/                     # Core library
│   ├── datasets/              # Dataset loaders (cat_kidney.py, hoof_*.py)
│   ├── models/                # Model architectures and losses
│   │   ├── decode_heads/      # Segmentation heads (FCNHead, etc.)
│   │   └── losses/            # Loss functions (Dice, CrossEntropy)
│   └── evaluation/            # Metrics (IoU, Dice)
├── tools/                     # Training and testing scripts
└── eric_python_files/         # Utility scripts (test_clahe.py, etc.)
```

## Quick Start

1. Install dependencies
   ```bash
   pip install -U openmim
   mim install mmengine
   mim install mmcv>=2.0.0
   pip install -e .
   ```

2. Train model
   ```bash
   python tools/train.py configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py
   ```

3. Test model
   ```bash
   python tools/test.py configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py /path/to/checkpoint.pth
   ```

## Core Components

### Custom Datasets
**Purpose**: Load and preprocess medical images (grayscale, 1000x1000 → 512x512)
**Entry point**: `mmseg/datasets/cat_kidney.py`
**Key files**:
- `cat_kidney.py` - Cat kidney dataset with label remapping [0,38,75]→[0,1,2]
- `hoof_solar.py` - Horse hoof solar view dataset
- `hoof_lat_block.py` - Horse hoof lateral block dataset

### Data Transforms
**Purpose**: Augmentation and preprocessing pipeline
**Entry point**: `mmseg/datasets/transforms/transforms.py`
**Key files**:
- `CLAHEGrayscale` - Contrast enhancement for grayscale medical images
- `RemapLabels` - Remap segmentation mask values before training
- `GrayscalePhotoMetricDistortion` - Brightness/contrast augmentation

### Training Configuration
**Purpose**: Define model, data, training parameters
**Entry point**: `configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py`
**Base configs used**:
- `_base_/models/fcn_unet_s5-d16-1channel.py` - 1-channel grayscale UNet
- `_base_/datasets/cat_kidney_grayscale.py` - Dataset with pipelines
- `_base_/default_runtime.py` - Logging and checkpointing defaults
- `_base_/schedules/schedule_20k.py` - PolyLR scheduler with power=0.9

**Key overrides**:
- Optimizer: SGD lr=0.01, momentum=0.9, weight_decay=0.0005
- Training: max_iters=20000, val_interval=500
- Loss: CE (0.4 weight, class_weight=[1.0,2.5,2.5]) + Dice (0.6 weight)
- Test mode: slide inference (crop_size=512, stride=341)
- Dropout: 0.2 in decode head
- Hooks: EarlyStoppingHook (patience=6), save_best='mDice', SegVisualizationHook
- Logging: WandB with custom config metadata

### Model Architecture
**Purpose**: UNet encoder-decoder with auxiliary head
**Entry point**: `mmseg/models/segmentors/encoder_decoder.py`
**Key components**:
- Backbone: UNet with 1 input channel, 5 stages
- Decode head: FCNHead with dropout 0.2
- Auxiliary head: Additional supervision at intermediate layers
- Per-class Dice tracking during training (modified decode_head.py:339-350)

## Architecture

Pipeline: Load grayscale image → LoadAnnotations → CLAHE enhancement → Augmentation → Resize to 512x512 → RemapLabels [0,38,75]→[0,1,2] → PackInputs → Model (UNet) → Loss (CE+Dice) → Per-class metrics.

Data flows through transform pipeline, model outputs predictions, losses computed per-class and averaged, WandB logs training metrics including per-class Dice.

## Navigation Guide

To work on:
- Training configs → `configs/unet/eric-*.py`
- Dataset definitions → `mmseg/datasets/cat_kidney.py`, `configs/_base_/datasets/cat_kidney_grayscale.py`
- Data augmentation → `mmseg/datasets/transforms/transforms.py` (CLAHEGrayscale, RemapLabels)
- Model architecture → `configs/_base_/models/fcn_unet_s5-d16-1channel.py`
- Loss functions → `mmseg/models/losses/` (dice_loss.py, cross_entropy_loss.py)
- Training loop → `tools/train.py`
- Metrics → `mmseg/models/decode_heads/decode_head.py` (per-class Dice added at line 339)

## Recent Critical Fixes (Sep 2025)

**Fixed Issues**:
- **CRITICAL**: RemapLabels was trying to remap [0,38,75]→[0,1,2] but masks already had [0,1,2] - DISABLED RemapLabels
- RandomRotate was using default seg_pad_val=255 causing unmapped values - fixed to seg_pad_val=0
- Data preprocessor seg_pad_val was 255 causing label corruption - fixed to 0
- Auxiliary head was missing CrossEntropyLoss with class weights - added matching main head config
- Class weights optimized to [1.0, 2.5, 2.5] - aggressive weights [0.5, 5.0, 5.0] caused instability
- Learning rate set to 0.01 with PolyLR scheduler (decays to 0.0001 over 20k iterations)

## Current Status

**Working**:
- Cat kidney segmentation (3 classes: background, left_kidney, right_kidney)
- Grayscale image support with CLAHE preprocessing (clip_limit=2.0)
- Direct support for masks with values [0,1,2] (background, left_kidney, right_kidney)
- Per-class Dice metrics during training (decode_head.py:339-350)
- Early stopping (patience=3, monitors mDice)
- Best checkpoint saving (save_best='mDice', keep last 3 regular checkpoints)
- WandB logging with per-class metrics
- Custom visualization hook for grayscale images (custom_viz_hook.py)
- Validation every 500 iterations with per-class metrics
- ResetOriShape transform to handle 1000x1000→512x512 resize

**Known Issues**:
- Visualization requires custom hook due to grayscale images and size mismatch
- ResetOriShape required in val_pipeline to keep predictions at 512x512
- Label remapping must happen AFTER all transforms (before PackSegInputs) to avoid interpolation issues

## Training Pipeline Details

**Transform order** (critical):
1. LoadImageFromFile (grayscale)
2. LoadAnnotations (loads mask with original values [0,38,75])
3. CLAHEGrayscale (enhance contrast)
4. GrayscalePhotoMetricDistortion (augment brightness/contrast)
5. RandomRotate (rotate with nearest-neighbor interpolation for masks)
6. Resize (to 512x512)
7. ~~RemapLabels~~ (DISABLED - masks already have [0,1,2])
8. PackSegInputs

**Note**: RemapLabels was disabled after discovering the dataset already has values [0,1,2] instead of [0,38,75].

## Custom Modifications

**Files modified from base MMSegmentation**:
- `mmseg/datasets/transforms/transforms.py` - Added CLAHEGrayscale (line 208-270), RemapLabels (line 274-313), ResetOriShape (line 317-330)
- `mmseg/datasets/transforms/__init__.py` - Exported CLAHEGrayscale, RemapLabels, ResetOriShape
- `mmseg/datasets/cat_kidney.py` - Custom dataset with label_map override in get_label_map()
- `mmseg/models/decode_heads/decode_head.py` - Added per-class Dice computation (line 339-352)
- `custom_viz_hook.py` - Custom visualization hook for grayscale images with size correction

## Additional Documentation

See `docs/` folder for MMSegmentation documentation.
Custom scripts in `eric_python_files/`:
- `test_clahe.py` - Test different CLAHE clip_limit values on images

---

## Documentation Update Guidelines

### Core Principle
Delete first, add second. Always remove outdated content before adding new documentation.

### Update Rules
- **Present tense only** - Document what exists now
- **No history** - Delete references to previous versions
- **No promises** - Remove all "TODO" and "coming soon"
- **Test everything** - Verify every command and path
- **Keep it minimal** - If it's obvious from the code, don't document it

### When to Delete
- References to removed features
- Anything containing "previously", "will be", or "planned"
- Documentation you can't verify
- Explanations of obvious things

### When to Update
- Feature added → Update immediately
- Feature removed → Delete docs first
- Found outdated section → Delete or fix now

Remember: Documentation is complete when there's nothing left to remove.