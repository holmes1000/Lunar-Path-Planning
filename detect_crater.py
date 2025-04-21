"""
Samara Holmes
Spring 2025

Program to detect craters in the image
"""
from load_image import save_img
from skimage import io, filters, measure, morphology
from process_image import load_images_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2
from load_image import show_img
from generate_obstacles import create_grid_map, display_grid_map

def detect_craters(img, imgID):
    
    img = np.array(img)
    # show_img(img) # Test that the image comes in correctly

    # Clean up noise with morphological filter
    # Create a structuring element (e.g., a disk-shaped kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust kernel size as needed
    
    # Apply the closing operation to fill small holes
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Label the detected features
    labelled_img, num_labels = measure.label(img, background=0, return_num=True)

    # show_img(labelled_img)
    
    # Get the properties of the labeled regions (craters)
    regions = measure.regionprops(labelled_img)
    print(f"Number of craters detected: {num_labels}")
    
    # Get rid of small craters (e.g., area > threshold)
    size_threshold = 50  # Adjust this threshold as needed
    giant_regions = [region for region in regions if region.area > size_threshold]

    return labelled_img, giant_regions


def display_craters(labelled_img, region, imgID):
    plt.figure(figsize=(12, 6))

    # Draw rectangles around detected regions
    for region in regions:
        # Draw a rectangle around each region
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

        # Draw the centroid
        y0, x0 = region.centroid
        plt.plot(x0, y0, '.g', markersize=10)

    plt.title(f"crater_image_{imgID+1}")
    plt.axis('off')

    plt.show()

def display_craters_on_image(labelled_img, regions, imgID):
    """
    Draw rectangles and centroids of detected regions directly on the image.

    Parameters:
    - labelled_img: The input image (NumPy array).
    - regions: List of detected regions (with `bbox` and `centroid` attributes).
    - imgID: The image ID for reference.

    Returns:
    - modified_img: The input image with craters drawn on it.
    """
    # Make a copy of the input image to avoid modifying the original
    modified_img = labelled_img.copy()

    # show_img(modified_img)

    # Assuming modified_img is the problematic image (with CV_32S depth)
    # Convert the image to uint8
    if modified_img.dtype != np.uint8:
        modified_img = np.clip(modified_img, 0, 255).astype(np.uint8)

    # Now convert to BGR
    modified_img = cv2.cvtColor(modified_img, cv2.COLOR_GRAY2BGR)

    # show_img(modified_img)

    # Draw the rectangles and centroids
    for region in regions:
        # Draw a rectangle around each region (bbox: minr, minc, maxr, maxc)
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(modified_img, (minc, minr), (maxc, maxr), (0, 0, 255), 2)  # Red rectangle

        # Draw the centroid
        # y0, x0 = map(int, region.centroid)  # Convert centroid to integer coordinates
        # cv2.circle(modified_img, (x0, y0), 5, (0, 255, 0), -1)  # Green dot

    show_img(modified_img)
    return modified_img


def detect_depthanything_craters(img, imgID):
    
    img = np.array(img)
    show_img(img) # Test that the image comes in correctly

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply sobel filter
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient magnitude to 0-255 range
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply threshold to identify potential crater edges
    _, binary = cv2.threshold(grad_mag, 50, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to clean up the binary image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Label the binary image to identify separate regions
    labelled_img, num_labels = measure.label(closed, background=0, return_num=True)
    
    # Get the properties of the labeled regions
    regions = measure.regionprops(labelled_img)
    
    # Filter regions based on size and circularity
    min_area = 100
    max_area = 10000
    
    filtered_regions = []
    for region in regions:
        if min_area < region.area < max_area:
            # Calculate circularity
            y0, x0 = region.centroid
            r = np.sqrt(region.area / np.pi)  # Estimate radius
            
            # Add region with properties to filtered list
            filtered_regions.append(region)
    
    print(f"Number of craters detected: {len(filtered_regions)}")
    
    return labelled_img, filtered_regions

if __name__ == "__main__":
    # Specify the directory
    image_directory = "binary_images/"

    # Load images
    images = load_images_from_directory(image_directory)

    # Show the original image
    # originals = load_images_from_directory("images/")
    # for img in originals:
    #     show_img(img)

    # Print the number of images loaded
    print(f"Loaded {len(images)} binary images.")

    # Detect craters (temporarily store in /crater_detections until its working, then instead, we'll generate obstacles/grid_map)
    for i in range(len(images)): 
        labelled_img, regions = detect_craters(images[i], i)
        img = display_craters_on_image(labelled_img, regions, i)
        save_img(img, f"crater_detections/detection_image_{i}.tiff")

        # Create grid map
        print(img.shape)
        grid_size = 20
        grid_map = create_grid_map(img.shape, regions, grid_size=grid_size)

        fig = display_grid_map(grid_map, img, regions, grid_size=grid_size)
        plt.show()

    
    # ------------------------------------ DepthAnything crater detection ----------------------------------------
    image_directory = "depth_images/"

    # Load images
    images = load_images_from_directory(image_directory)

    # Print the number of images loaded
    print(f"Loaded {len(images)} depth images.")

    # Detect craters
    for i in range(len(images)): 
        labelled_img, regions = detect_depthanything_craters(images[i], i)
        img = display_craters_on_image(labelled_img, regions, i)

        # Create grid map
        print(img.shape)
        grid_size = 20
        grid_map = create_grid_map(img.shape, regions, grid_size=grid_size)

        fig = display_grid_map(grid_map, img, regions, grid_size=grid_size)
        plt.show()