"""
Samara Holmes
Spring 2025

Program to process the image using OpenCV functions
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from load_image import save_img

def sobel(img, sobel_type):
    """
    Custom function for applying a sobel filter to an image
    """
    # Convert the image to a NumPy array
    img = np.array(img)
    kernel = None
    match sobel_type:
        case "X":
            kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
        case "Y":
            kernel = np.array([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]])
    
    # Get the dimensions of the image
    height, width = img.shape

    # Create a dst image with the same size as the input image
    dst = np.zeros((height, width))

    # Apply the kernel (ignoring the border pixels)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract a 3x3 region
            region = img[i-1:i+2, j-1:j+2]
            
            # Compute the convolution of the region with the Sobel kernel
            dst[i, j] = np.sum(region * kernel)
    
    # Normalize the output to the range [0, 255]
    dst = np.clip(dst, 0, 255).astype(np.uint8)

    return dst


def threshold_img(img, threshold):
    """
    Apply a binary threshold to the image
    """
    binary_img = (img > threshold).astype(np.uint8) * 255  # Pixels > threshold set to 255, others to 0

    return binary_img

def plot_img(img, imgID):
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title(f"lunar_image_{imgID+1}")
    plt.axis('off')
    plt.show()

def process_img(img, imgID):
    img = sobel(img, "Y")
    img = sobel(img, "X")
    img = threshold_img(img, 128)
    plot_img(img, imgID)
    return img

def load_images_from_directory(directory):
    """
    Function to load all images from a given directory.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):  # Check for valid image files
            file_path = os.path.join(directory, filename)
            img = Image.open(file_path)
            images.append(img)
    return images

if __name__ == "__main__":
    # Specify the directory
    image_directory = "images/"

    # Load images
    images = load_images_from_directory(image_directory)

    # Print the number of images loaded
    print(f"Loaded {len(images)} unprocessed images.")

    for i in range(len(images)): 
        img = process_img(images[i], i)
        save_img(img, f"binary_images/lunar_image_{i+1}.tiff")