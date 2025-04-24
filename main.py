"""
Samara Holmes
Spring 2025

Program to conduct the lunar crater detection and coverage path planning

Assumptions:
Robot(s) starts at bottom-left corner (0,0)
Robot(s) cannot move diagonally
Robot(s) cannot move outside the bounds of the grid
A grid cell is a meter long

This program should do image processing on a Lunar crater image, resulting in a grid map, with obstacles
From there, it should intake how many robots there are for path planning
It should then generate a coverage plan around these obstacles
It should also check to make sure there aren't any impossible routes.
The output should be a grid map with the labelled paths with path lengths and coverage percentages
"""

from load_image import save_img
from skimage import io, filters, measure, morphology
from process_image import load_images_from_directory
import numpy as np
import matplotlib.pyplot as plt
import cv2
from load_image import show_img
from generate_obstacles import create_grid_map, display_grid_map
from detect_crater import detect_craters, display_craters_on_image
from path_planner.coverage_path_planner import CoveragePathPlanner
from path_planner.cpp_utils import visualize_path, check_path_validity

if __name__ == "__main__":
    print("Running main program")

    # -------------------- Do the edge detection and crater detection ------------------------

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
        plt.show() # Area to be covered image

    # ------------------- Do Path Generation -------------------------

    # Plan coverage path
    planner = CoveragePathPlanner(grid_map, tool_size=1)
    cell_path, path_points = planner.plan_coverage_path()
    
    # Check path validity (this is for testing)
    is_valid, message = check_path_validity(planner, cell_path)
    print(f"Path validity check: {is_valid}")
    if not is_valid:
        print(f"  Error: {message}")
    
    # Visualize
    # visualize_path(planner, cell_path)
    # planner.visualize()
    
    # Calculate path length
    path_length = 0
    for i in range(len(path_points)):
        x1, y1 = path_points[i]
        x2, y2 = path_points[(i+1) % len(path_points)]
        path_length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    print(f"Grid Map: {grid_map}")  # Grid map uses 1 as obstacles
    print(f"Cell Map: {planner.cell_map}")  # Cell map uses 0 as obstacles
    print(f"Total path length: {path_length:.2f}")
    print(f"Number of cells covered: {len(cell_path)}")
    print(f"Percent Coverage: {(len(cell_path)/(np.sum(grid_map))) * 100:.2f}%")
    print(f"Cells by type:")
    print(f"  - Trunk cells: {len(planner.trunk_cells)}")
    print(f"  - Branch cells: {len(planner.branch_cells)}")
    print(f"  - Subsidiary cells: {len(planner.subsidiary_cells)}")
    print(f"  - Cells exempt from coverage: {grid_map.size - (len(planner.subsidiary_cells) + len(planner.branch_cells) + len(planner.trunk_cells))}")
    print(f"  - Percentage of cells exempt from coverage: { (grid_map.size - ((len(planner.subsidiary_cells) + len(planner.branch_cells) + len(planner.trunk_cells)))) / grid_map.size * 100}")