"""
Samara Holmes
Spring 2025

Program to load tiff file for Lunar dataset, and show the random crop image in a plot
"""
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, filters
import numpy as np
import random

def load_tiff(file_path):
    """
    Function to load the tiff image
    """
    # Load the TIFF file
    img = io.imread(file_path)
    print(f"Image shape: {img.shape}")

    if len(img.shape) == 3 and img.shape[0] > 1:
        if img.shape[0] == 6:
            img = np.mean(img[:3], axis=0)
        elif img.shape[0] > 3:
            img = np.mean(img, axis=0)
        else:
            img = rgb2gray(np.moveaxis(img, 0, -1))
    elif len(img.shape) > 2:
        img = rgb2gray(img)
    return img

def show_img(img):
    """
    Function to show the image using matplotlib
    """
    # Display the smoothed image
    # Normalize image values if needed
    if np.max(img) > 1:
        img = img / np.max(img)

    # Apply a Gaussian filter to smooth the image
    smoothed_image = filters.gaussian(img, sigma=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.title("Crop")
    plt.axis('off')
    plt.show()

def save_img(img, output_path):
    """
    Save image to a file
    """
    image_to_save = (img * 255).astype(np.uint8)  # Scale to 8-bit values
    Image.fromarray(image_to_save).save(output_path)
    print("Image saved")

def get_image_crop(img, size):
    # Define crop size
    crop_height, crop_width = size, size

    # Generate a random top-left corner for cropping
    start_x = random.randint(0, img.shape[1] - crop_width)
    start_y = random.randint(0, img.shape[0] - crop_height)

    # Extract the crop
    crop = img[start_y:start_y + crop_height, start_x:start_x + crop_width]

    return crop


if __name__ == "__main__":
    # Init vars
    # image_path = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Lunar_Mapping\\data\\Lunar_Clementine_NIR_cal_empcor_500m.tif"
    image_path = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Lunar_Mapping\\data\\WAC_ROI_NEARSIDE_DAWN_E300S0450_100M.tiff"
    # image_path = "C:\\Users\\samar\\Desktop\\Pattern Recogn & Comp Vision\\Pattern Recogn Repo\\Lunar_Mapping\\data\\WAC_CSHADE_E300S0450_100M_GRID.tiff"
    
    # Load the TIFF
    full_img = load_tiff(image_path)

    # Code to get just one test image
    # --------------------------------------------
    # img_crop = get_image_crop(full_img, 500)

    # show_img(img_crop)

    # save_img(img_crop, "images/test_image.tiff")
    # --------------------------------------------

    # Code to generate 20 random crops
    # --------------------------------------------
    for i in range(20):  # Loop to generate 20 random crops
        img_crop = get_image_crop(full_img, 500)

        # Show the crop
        # show_img(img_crop)

        # Save the crop to a file with a unique name
        save_img(img_crop, f"random_images/lunar_image_{i+1}.tiff")
    # --------------------------------------------

