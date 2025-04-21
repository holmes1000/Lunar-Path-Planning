"""
Samara Holmes
Spring 2025

Initial coverage path planning program

Assumptions:

Robot starts at bottom-left corner (0,0)
Robot cannot move diagonally

"""
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import sys
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)
from matplotlib.patches import Patch

def visualize_path(planner, path=None):
    """Visualize the region, cells, and coverage path."""
    plt.figure(figsize=(12, 10))
    
    # Create a colored visualization of the cell classifications
    visualization_map = np.zeros((planner.height, planner.width, 3))
    
    # Set colors according to the requested scheme
    # Dark purple for cells exempt from coverage (value 0 in region_map)
    dark_purple = np.array([0.3, 0.0, 0.5])  # RGB for dark purple
    yellow = np.array([1.0, 1.0, 0.0])  # RGB for yellow (trunk cells)
    green = np.array([0.0, 0.8, 0.0])   # RGB for green (branch cells)
    green_blue = np.array([0.0, 0.7, 0.7])  # RGB for green-blue (subsidiary cells)
    
    # Fill the visualization map with the appropriate colors
    for i in range(planner.height):
        for j in range(planner.width):
            # if planner.grid_map[i, j] == 0:  # Exempt from coverage
            #     visualization_map[i, j] = dark_purple
            # elif (i, j) in planner.trunk_cells:
            #     visualization_map[i, j] = yellow
            # elif (i, j) in planner.branch_cells:
            #     visualization_map[i, j] = green
            # elif (i, j) in planner.subsidiary_cells:
            #     visualization_map[i, j] = green_blue
            # else:
            #     # Any other cell (should be rare)
            #     visualization_map[i, j] = np.array([0.9, 0.9, 0.9])  # Light gray
            if (i, j) in planner.trunk_cells:
                visualization_map[i, j] = yellow
            elif (i, j) in planner.branch_cells:
                visualization_map[i, j] = green
            elif (i, j) in planner.subsidiary_cells:
                visualization_map[i, j] = green_blue
            elif planner.grid_map[i, j] == 0:  # Area to be covered but not classified
                visualization_map[i, j] = np.array([0.9, 0.9, 0.9])
            else:  # grid_map[i, j] == 1, exempt from coverage
                visualization_map[i, j] = dark_purple
    
    # Plot the colored map
    plt.imshow(visualization_map, interpolation='nearest', 
              extent=[-0.5, planner.width-0.5, planner.height-0.5, -0.5])
    
    # Add grid for each cell with dotted black lines
    for i in range(planner.height + 1):
        plt.axhline(y=i-0.5, color='black', linestyle=':', alpha=0.7, linewidth=0.8)
    for j in range(planner.width + 1):
        plt.axvline(x=j-0.5, color='black', linestyle=':', alpha=0.7, linewidth=0.8)
    
    # Add highlighted grid for mega-cells (2x2)
    for i in range(0, planner.height + 1, 2):
        plt.axhline(y=i-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
    for j in range(0, planner.width + 1, 2):
        plt.axvline(x=j-0.5, color='black', linestyle='--', alpha=0.5, linewidth=1.2)
    
    # Plot path if provided
    if path:
        # Extract path points that go through cell centers
        path_x = []
        path_y = []
        for cell in path:
            # Get exact center of the cell
            i, j = cell
            path_y.append(i)
            path_x.append(j)
        
        # Plot the path
        plt.plot(path_x, path_y, 'r-', linewidth=2)  # Red path for better visibility
        
        # Mark path nodes
        plt.plot(path_x, path_y, 'ro', markersize=3, alpha=0.5)  # Small red circles at cell centers
        
        # Mark start point
        plt.plot(path_x[0], path_y[0], 'ks', markersize=8)  # Start point as black square
    
    plt.title('Coverage Path Planning', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.xticks(range(0, planner.width))
    plt.yticks(range(0, planner.height))
    
    # Add a legend
    legend_elements = [
        Patch(facecolor=yellow, label='Trunk Cells'),
        Patch(facecolor=green, label='Branch Cells'),
        Patch(facecolor=green_blue, label='Subsidiary Cells'),
        Patch(facecolor=dark_purple, label='Exempt from Coverage'),
        Patch(facecolor='red', label='Coverage Path')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def check_path_validity(planner, path):
    """Check if the path is valid (no diagonal moves, no exempt cells)."""
    if not path:
        return True, "Path is empty"
    
    for i in range(1, len(path)):
        prev_cell = path[i-1]
        current_cell = path[i]
        
        # Check if cells are valid
        pi, pj = prev_cell
        ci, cj = current_cell
        
        # Check if cells are in bounds
        if (pi < 0 or pi >= planner.height or pj < 0 or pj >= planner.width or
            ci < 0 or ci >= planner.height or cj < 0 or cj >= planner.width):
            return False, f"Cell out of bounds: ({pi},{pj}) -> ({ci},{cj})"
        
        # Check if cells are exempt from coverage
        if planner.grid_map[pi, pj] == 0 or planner.grid_map[ci, cj] == 0:
            return False, f"Path includes exempt cell: ({pi},{pj}) -> ({ci},{cj})"
        
        # Check for diagonal movement
        if abs(ci - pi) + abs(cj - pj) > 1:
            return False, f"Diagonal movement detected: ({pi},{pj}) -> ({ci},{cj})"
    
    return True, "Path is valid"