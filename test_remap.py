#!/usr/bin/env python3
"""Test RemapLabels transform in isolation"""

import numpy as np

# Simulate the RemapLabels transform
class RemapLabels:
    def __init__(self, label_map):
        self.label_map = label_map

    def transform(self, gt_seg_map):
        remapped = np.zeros_like(gt_seg_map)
        for old_val, new_val in self.label_map.items():
            remapped[gt_seg_map == old_val] = new_val
        return remapped

# Test with synthetic data
print("Testing RemapLabels transform:")
print("-" * 40)

# Create a fake mask with kidney pixels
test_mask = np.array([
    [0, 0, 38, 38],
    [0, 75, 75, 0],
    [38, 38, 0, 0],
    [75, 0, 0, 38]
])

print(f"Original mask:\n{test_mask}")
print(f"Unique values: {np.unique(test_mask)}")

# Apply RemapLabels
remap = RemapLabels(label_map={0: 0, 38: 1, 75: 2})
remapped_mask = remap.transform(test_mask)

print(f"\nRemapped mask:\n{remapped_mask}")
print(f"Unique values: {np.unique(remapped_mask)}")

# Check if it worked
expected = np.array([
    [0, 0, 1, 1],
    [0, 2, 2, 0],
    [1, 1, 0, 0],
    [2, 0, 0, 1]
])

if np.array_equal(remapped_mask, expected):
    print("\n✓ RemapLabels works correctly!")
else:
    print("\n✗ RemapLabels is broken!")

# Test with values that shouldn't exist after resize
print("\n" + "-" * 40)
print("Testing with corrupted values (e.g., from interpolation):")

corrupted_mask = np.array([
    [0, 0, 37, 38],  # 37 is a corrupted value
    [0, 74, 75, 0],   # 74 is a corrupted value
    [38, 39, 0, 0],   # 39 is a corrupted value
    [75, 0, 0, 255]   # 255 from seg_pad_val
])

print(f"Corrupted mask:\n{corrupted_mask}")
print(f"Unique values: {np.unique(corrupted_mask)}")

remapped_corrupted = remap.transform(corrupted_mask)
print(f"\nRemapped corrupted mask:\n{remapped_corrupted}")
print(f"Unique values: {np.unique(remapped_corrupted)}")

# Count how many pixels were lost
original_kidney_count = np.sum((corrupted_mask == 38) | (corrupted_mask == 75))
remapped_kidney_count = np.sum((remapped_corrupted == 1) | (remapped_corrupted == 2))
print(f"\nOriginal kidney pixels: {original_kidney_count}")
print(f"Remapped kidney pixels: {remapped_kidney_count}")
print(f"Lost pixels: {original_kidney_count - remapped_kidney_count}")

if original_kidney_count != remapped_kidney_count:
    print("\n✗ PROBLEM: Corrupted values cause kidney pixels to be lost!")