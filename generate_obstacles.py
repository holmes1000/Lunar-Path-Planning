"""
Samara Holmes
Spring 2025

Program to take the images and convert to a grid map
"""
import numpy as np
import matplotlib.pyplot as plt
from load_image import save_img
import matplotlib.patches as patches

def create_grid_map(img_shape, regions, grid_size=20):
    """
    Create a grid-based map from the detected craters.
    
    Parameters:
    - img_shape: Tuple (height, width) of the original image
    - regions: List of detected regions (from regionprops)
    - grid_size: Size of each grid cell in pixels
    
    Returns:
    - grid_map: 2D numpy array where 1 represents obstacle (crater) and 0 represents free space
    - grid_coords: Mapping of image coordinates to grid coordinates
    """
    
    # Calculate grid dimensions
    height, width, _ = img_shape
    grid_height = height // grid_size + (1 if height % grid_size > 0 else 0)
    grid_width = width // grid_size + (1 if width % grid_size > 0 else 0)
    
    # Initialize grid map with zeros (free space)
    grid_map = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    # Mark grid cells containing craters as obstacles (1)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        
        # Convert image coordinates to grid coordinates
        grid_minr, grid_minc = minr // grid_size, minc // grid_size
        grid_maxr, grid_maxc = maxr // grid_size + 1, maxc // grid_size + 1
        
        # Ensure grid coordinates are within bounds
        grid_maxr = min(grid_maxr, grid_height)
        grid_maxc = min(grid_maxc, grid_width)
        
        # Mark all grid cells that the crater occupies as obstacles
        for r in range(grid_minr, grid_maxr):
            for c in range(grid_minc, grid_maxc):
                grid_map[r, c] = 1
    
    return grid_map

def display_grid_map(grid_map, img, regions=None, grid_size=20):
    """
    Display the grid map with the original image and detected craters.
    
    Parameters:
    - grid_map: 2D numpy array representing the grid map
    - img: Original or labelled image
    - regions: List of detected regions (optional)
    - grid_size: Size of each grid cell in pixels
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display the original/labelled image with detected craters
    ax1.imshow(img, cmap='gray')
    if regions is not None:
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    linewidth=1, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
    ax1.set_title('Detected Craters')
    ax1.axis('off')
    
    # Display the grid map
    ax2.imshow(grid_map, cmap='binary', interpolation='none')
    ax2.set_title('Grid Map (White = Free, Black = Obstacle)')
    
    # Draw grid lines
    grid_height, grid_width = grid_map.shape
    for i in range(grid_width):
        ax2.axvline(i - 0.5, color='gray', linewidth=0.5)
    for i in range(grid_height):
        ax2.axhline(i - 0.5, color='gray', linewidth=0.5)
    
    # Set ticks to show grid coordinates
    ax2.set_xticks(np.arange(grid_width))
    ax2.set_yticks(np.arange(grid_height))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Specify the directory
    # image_directory = "crater_detections/"

    # Load images
    # images = load_images_from_directory(image_directory)

    # Print the number of images loaded
    print(f"Loaded {len(images)} crater detection images.")