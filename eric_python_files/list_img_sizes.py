import os
from PIL import Image

def list_image_sizes(directory):
    # Verify if the specified directory exists
    if not os.path.isdir(directory):
        print("The specified directory does not exist.")
        return

    # Retrieve all files in the directory
    files = os.listdir(directory)

    # Filter the list to include only jpg images
    jpg_files = [file for file in files if file.lower().endswith('.jpg')]

    # Proceed only if there are jpg files found
    if not jpg_files:
        print("No jpg images found in the directory.")
        return

    # Display the size and aspect ratio of each jpg image
    for file in jpg_files:
        img_path = os.path.join(directory, file)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                print(f"File: {file} - Size: {img.size} (width x height), Aspect Ratio: {aspect_ratio:.2f}")
        except IOError:
            print(f"Could not read image: {file}")

# Example usage
directory_path = '/Users/ewern/Desktop/code/img_segmentation/voc_data/JPEGImages'
list_image_sizes(directory_path)
