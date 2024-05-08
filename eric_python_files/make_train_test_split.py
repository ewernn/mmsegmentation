import os
import random

random.seed(1)  # For reproducibility
image_files = os.listdir('/Users/ewern/Desktop/code/img_segmentation/voc_data/JPEGImages')
image_filenames = [os.path.splitext(f)[0] for f in image_files if f.endswith('.jpg')]

random.shuffle(image_filenames)
split_point = int(0.8 * len(image_filenames))
train_filenames = image_filenames[:split_point]
val_filenames = image_filenames[split_point:]

with open('/Users/ewern/Desktop/code/img_segmentation/voc_data/ImageSets/Segmentation/train.txt', 'w') as f:
    f.write('\n'.join(train_filenames))

with open('/Users/ewern/Desktop/code/img_segmentation/voc_data/ImageSets/Segmentation/val.txt', 'w') as f:
    f.write('\n'.join(val_filenames))
