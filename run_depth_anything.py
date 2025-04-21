"""
Samara Holmes
Spring 2025

Program to process the image using DepthAnythingV2

Pre-trained model used: Depth-Anything-V2-Large

https://github.com/DepthAnything/Depth-Anything-V2
"""

import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

from process_image import load_images_from_directory
import numpy as np
from load_image import save_img

if __name__ == "__main__":
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitl', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    # Specify the directory
    # image_directory = "random_images/"
    image_directory = "images/"

    # Load images
    images = load_images_from_directory(image_directory)

    # Print the number of images loaded
    print(f"Loaded {len(images)} unprocessed images.")

    for i in range(len(images)): 
        # Convert the image to a NumPy array
        images[i] = np.array(images[i])
        depth = model.infer_image(images[i]) # HxW raw depth map in numpy
        save_img(depth, f"depth_images/depth_image_{i}.tiff")