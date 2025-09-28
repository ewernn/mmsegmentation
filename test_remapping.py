import sys
sys.path.insert(0, '/content/mmsegmentation')

from mmseg.datasets import CatKidneyDataset
from mmengine.config import Config

cfg = Config.fromfile('/content/mmsegmentation/configs/unet/eric-unet-s5-d16_fcn_1xb4-ce-1.0-dice-3.0-40k_CAT_KIDNEY-512x512.py')

dataset = CatKidneyDataset(
    data_root='/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC',
    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
    ann_file='ImageSets/Segmentation/train.txt',
    pipeline=cfg.train_pipeline
)

print(f"Dataset size: {len(dataset)}")

sample = dataset[0]
print(f"\nFirst sample keys: {sample.keys()}")

if 'data_samples' in sample:
    gt_sem_seg = sample['data_samples'].gt_sem_seg.data
    print(f"GT seg shape: {gt_sem_seg.shape}")
    print(f"GT seg unique values: {gt_sem_seg.unique()}")
    print(f"GT seg dtype: {gt_sem_seg.dtype}")
elif 'gt_seg_map' in sample:
    import numpy as np
    print(f"GT seg shape: {sample['gt_seg_map'].shape}")
    print(f"GT seg unique values: {np.unique(sample['gt_seg_map'])}")

print("\n✓ If unique values are [0, 1, 2], remapping worked!")
print("✗ If unique values are [0, 38, 75], remapping failed!")