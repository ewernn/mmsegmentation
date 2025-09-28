import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def test_clahe_settings(image_path, clip_limits=[2.0, 4.0, 20.0, 40.0]):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    for idx, clip_limit in enumerate(clip_limits):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(img)

        row = (idx + 1) // 3
        col = (idx + 1) % 3
        axes[row, col].imshow(enhanced, cmap='gray')
        axes[row, col].set_title(f'CLAHE clip_limit={clip_limit}')
        axes[row, col].axis('off')

    plt.tight_layout()

    image_name = image_path.split('/')[-1].replace('.jpg', '').replace('.png', '')
    plt.savefig(f'clahe_comparison_{image_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: clahe_comparison_{image_name}.png")

data_root = '/content/drive/MyDrive/MM/Seg/datasets/cat_kidney_dataset_csv_filtered_VOC'
test_images = glob(f'{data_root}/JPEGImages/*.jpg')[:3]

if len(test_images) == 0:
    test_images = glob(f'{data_root}/JPEGImages/*.png')[:3]

if len(test_images) == 0:
    print(f"No images found in {data_root}/JPEGImages/")
    print("Please update the data_root path in the script")
else:
    print(f"Testing CLAHE on {len(test_images)} images...")
    for img_path in test_images:
        print(f"\nProcessing: {img_path}")
        test_clahe_settings(img_path)