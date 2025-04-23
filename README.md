# Author
Samara Holmes

# Background
This program is an extension on the following papers:

Deep Learning Crater Detection for Lunar Terrain Relative Navigation - https://www.researchgate.net/profile/Lena-Downes/publication/338399036_Deep_Learning_Crater_Detection_for_Lunar_Terrain_Relative_Navigation/links/5efbbce7299bf18816f5ea9d/Deep-Learning-Crater-Detection-for-Lunar-Terrain-Relative-Navigation.pdf

An Efficient Coverage Method for Irregularly Shaped Terrains - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10801856

It's also similar in concept to my WPI capstone project: https://nestlab-bae-mqp-2024.github.io/

# Abstract
Navigating the Moon’s surface is challenging due to its irregular shapes and features such as craters. This paper introduces a method to conduct coverage path planning on lunar terrains. It proposes an effective method to detect craters and combines this with a path planning strategy for complete coverage. The result is a low-redundancy path planner for robots traversing between craters on the Moon.

# Objective
Using real Lunar terrain data, generate maps to conduct coverage path planning for Lunar shaped terrains.

## Phase 1: Loading the data (load_image.py)
1. Load a Lunar terrain dataset (in this case, a TIFF image)
2. Get a crop from the TIFF
3. Do some post processing on the TIFF (gaussian? something for smoothing/noise reduction?)
4. Save the crop to /images

## Phase 2: Image Processing and Crater Detection (process_image.py and detect_crater.py)
1. Load the crop from /images
2. Do image processing (sobel filters, binary thresholding, morphological filter)
3. Save those images to /binary_images
4. Detect the craters (from images in /binary_images)
5. Generate DepthAnything images (from images in /random_images) and grid maps for comparison

## Phase 3: Prepare the data for path planning (generate_obstacles.py)
1. Take the crater detections and turn them into obstacles to be used for the coverage method (combine visuals from detect_crater.py and run_depth_anything.py)
2. Split the image and define 'areas to be covered' vs 'obstacles'
3. Generate a grid map

## Phase 4: Conduct coverage path planning for one robot
1. Use the algorithm from [2] to create workflow
2. Conduct cell classification
3. Look for Hamiltonian cycles
4. Look for Hamiltonian paths
5. Combine
6. Generate the total coverage rate along with the path length for the robot
7. Save path plots to with coverage rates and path lengths to /paths/single_robot

## Phase 5: Conduct coverage path planning for multiple robots (Not implemented yet)
1. 
2. 
3. Generate the total coverage rate along with the path length for each robot
4. Save path plots to with coverage rates and path lengths to /paths/multi_robot

## Phase X: Create a UNet Model (Not implemented yet)
1. Load the data, and generate several randomly cropped images
2. Annotate the data using CVAT
3. Generate a prediction mask
4. Train against the annotations for crater detections
5. Feed this into the coverage algorithm

## Phase Y: Make the data more advanced
1. Factor in the depths of the craters
    - For certain craters, you can go through the middle

## Phase Z: Try with other models and compare
1. Try a YOLO model for crater detection (Implemented)
2. Use DeepMoon for crater detection
3. Create crater detection graph comparison


# Instructions
1. conda create --name crater python=3.12.9
2. pip install -r requirements.txt
3. run main.py


# Datasets
C:\Users\samar\Desktop\Pattern Recogn & Comp Vision\Pattern Recogn Repo\Lunar_Mapping\data\WAC_ROI_NEARSIDE_DAWN_E300S0450_100M.tiff
https://wms.lroc.asu.edu/lroc/view_rdr/WAC_ROI_NEARSIDE_DAWN


# Citations


# Presentation
https://youtu.be/iT9bRe6xMaM



