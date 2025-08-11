"""
Samara Holmes
Summer 2025

Program to process the image using the Faster R-CNN model

Reference: https://www.kaggle.com/code/benmanor/crater-object-detection-using-faster-rcnn/notebook
"""

from load_image import save_img
from skimage import io, filters, measure, morphology
from process_image import load_images_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2
from load_image import show_img
from generate_obstacles import create_grid_map, display_grid_map
from detect_crater import display_craters_on_image

import numpy as np

def extract_boxes_from_image(img):
    """
    Ensure image is a NumPy array and extract bounding boxes.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define blue range
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))

    return boxes

def create_grid_map_from_image(img, grid_size=25):
    """
    Create grid map from image with drawn bounding boxes.
    """
    boxes = extract_boxes_from_image(img)
    print("Image Shape:", img.shape)
    grid_map = create_grid_map(img.shape, boxes, grid_size)
    # fig = display_grid_map(grid_map, img, boxes, grid_size)
    plt.show()
    return grid_map


def detect_faster_rcnn_craters(img, imgID):
    """
    Detect craters using blue bounding boxes from Faster R-CNN output.
    Returns a labeled image and list of bounding box regions.
    """

    # Convert image to HSV to isolate blue color
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append((x, y, x + w, y + h))  # (xmin, ymin, xmax, ymax)

    # Draw bounding boxes on the image
    labelled_img = img.copy()
    for (xmin, ymin, xmax, ymax) in regions:
        cv2.rectangle(labelled_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue box

    print(f"Image {imgID}: Detected {len(regions)} craters.")

    return labelled_img, regions
    

if __name__ == "__main__":
    # Specify the directory
    image_directory = "results/"
    

    # Load images
    images = load_images_from_directory(image_directory)



    # Print the number of images loaded
    print(f"Loaded {len(images)} Faster R-CNN images.")

    # Detect craters
    for i in range(len(images)): 

        print(f"Type of images[{i}]:", type(images[i]))
        print(f"Content of images[{i}]:", images[i])

        # Show the original image
        show_img(images[i])

        img = images[i]
        if isinstance(img, tuple):
            img = img[0]

        img = np.array(img)

        grid_map = create_grid_map_from_image(img, grid_size=25)

        display_grid_map(grid_map, img, grid_size=25)

        save_img(img, f"rcnn_images/grid_map_{i+1}.jpg")

