import cv2
import os
import numpy as np

def canny_with_sliders(image_path):
    cv2.namedWindow('Canny Edge Detector')

    def nothing(x):
        pass

    # Initialize sliders for threshold and kernel size adjustments
    cv2.createTrackbar('Min Threshold', 'Canny Edge Detector', 50, 255, nothing)
    cv2.createTrackbar('Max Threshold', 'Canny Edge Detector', 150, 255, nothing)
    cv2.createTrackbar('Kernel Size', 'Canny Edge Detector', 1, 10, nothing)  # Max size 10, minimum size 1

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error loading image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    while True:
        # Get current positions of the sliders
        min_thresh = cv2.getTrackbarPos('Min Threshold', 'Canny Edge Detector')
        max_thresh = cv2.getTrackbarPos('Max Threshold', 'Canny Edge Detector')
        kernel_size = cv2.getTrackbarPos('Kernel Size', 'Canny Edge Detector')
        kernel_size = max(1, kernel_size)  # Ensure kernel size is at least 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        edges = cv2.Canny(gray, min_thresh, max_thresh)

        # Apply morphological closing and opening
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges_opened = cv2.morphologyEx(edges_closed, cv2.MORPH_OPEN, kernel)

        # Find and draw contours
        contours, _ = cv2.findContours(edges_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = np.zeros_like(edges)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

        # Display the edges image
        cv2.imshow('Canny Edge Detector', np.hstack((gray, contour_image)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def process_folder(folder_path):
    images = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
    if not images:
        print("No PNG images found in the directory.")
        return

    for filename in images:
        image_path = os.path.join(folder_path, filename)
        print(f"Processing {image_path}")
        canny_with_sliders(image_path)
        input("Press Enter to continue to the next image...")

folder_path = 'data/solar_w_modern_clip'
process_folder(folder_path)
