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
Using real Lunar terrain data, generate maps to conduct coverage path planning for lunar shaped terrains.

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
E. J. Speyerer, R. V. Wagner, M. S. Robinson, et
al., “Pre-flight and On-orbit Geometric Calibration of
the Lunar Reconnaissance Orbiter Camera,” en, Space
Science Reviews, vol. 200, no. 1-4, pp. 357–392, Apr.
2016, ISSN: 0038-6308, 1572-9672. DOI: 10 . 1007 /
s11214- 014- 0073- 3. [Online]. Available: http://link.
springer.com/10.1007/s11214-014-0073-3 (visited on
04/05/2025).
[2] Y. Tang, Q. Wu, C. Zhu, and L. Chen, “An Efficient
Coverage Method for Irregularly Shaped Terrains,” in
2024 IEEE/RSJ International Conference on Intelligent
Robots and Systems (IROS), ISSN: 2153-0866, Oct.
2024, pp. 7641–7647. DOI: 10.1109/IROS58592.2024.
10801856. [Online]. Available: https://ieeexplore.ieee.
org/document/10801856 (visited on 04/05/2025).
[3] Y. Bandyopadhyay, “Lunar Crater Detection Using
YOLOv8 Deep Learning,” en, Mar. 2024. [Online].
Available: https://eartharxiv.org/repository/view/6828/
(visited on 04/05/2025).
[4] M. Sinha, S. Paul, M. Ghosh, S. N. Mohanty, and
R. M. Pattanayak, “Automated Lunar Crater Identifi-
cation with Chandrayaan-2 TMC-2 Images using Deep
Convolutional Neural Networks,” en, Scientific Reports,
vol. 14, no. 1, p. 8231, Apr. 2024, ISSN: 2045-2322.
DOI: 10.1038/s41598- 024- 58438- 4. [Online]. Avail-
able: https : / / www. nature . com / articles / s41598 - 024 -
58438-4 (visited on 04/05/2025).
[5] M. Chen, D. Liu, K. Qian, J. Li, M. Lei, and Y. Zhou,
“Lunar Crater Detection Based on Terrain Analysis
and Mathematical Morphology Methods Using Digital
Elevation Models,” IEEE Transactions on Geoscience
and Remote Sensing, vol. 56, no. 7, pp. 3681–3692,
Jul. 2018, ISSN: 1558-0644. DOI: 10.1109/TGRS.2018.
2806371. [Online]. Available: https://ieeexplore.ieee.
org/document/8310940 (visited on 04/05/2025).
[6] X. Lin, Z. Zhu, X. Yu, et al., “Lunar Crater Detection
on Digital Elevation Model: A Complete Workflow
Using Deep Learning and Its Application,” en, Remote
Sensing, vol. 14, no. 3, p. 621, Jan. 2022, ISSN: 2072-
4292. DOI: 10 . 3390 / rs14030621. [Online]. Available:
https://www.mdpi.com/2072-4292/14/3/621 (visited on
04/05/2025).
[7] A. Silburt, M. Ali-Dib, C. Zhu, et al., Lunar Crater
Identification via Deep Learning, arXiv:1803.02192,
Nov. 2018. DOI: 10.48550/arXiv.1803.02192. [Online].
Available: http://arxiv.org/abs/1803.02192 (visited on
04/08/2025).
[8] L. M. Downes, T. J. Steiner, and J. P. How, “Deep
Learning Crater Detection for Lunar Terrain Relative
Navigation,” en, Other repository, Jan. 2020. [Online].
Available: https : / / dspace . mit . edu / handle / 1721 . 1 /
137175.2 (visited on 04/05/2025).
[9] I. Giannakis, A. Bhardwaj, L. Sam, and G. Leontidis,
Deep learning universal crater detection using Segment
Anything Model (SAM), 2023. DOI: 10.48550/ARXIV.
2304.07764. [Online]. Available: https://arxiv.org/abs/
2304.07764 (visited on 04/08/2025).
[10] L. Yang, B. Kang, Z. Huang, et al., Depth Anything V2,
en, Jun. 2024. [Online]. Available: https://arxiv.org/abs/
2406.09414v2 (visited on 04/20/2025).
[11] Y. Idikut, C. Cummings, and S. Holmes, “Multi-Robot
Persistent Coverage Under Fuel and Stochastic Failure
Constraints,” Undergraduate Major Qualifying Project,
Worcester Polytechnic Institute, 2024.
[12] D. Mitchell, M. Corah, N. Chakraborty, K. Sycara, and
N. Michael, “Multi-robot long-term persistent coverage
with fuel constrained robots,” in 2015 IEEE Interna-
tional Conference on Robotics and Automation (ICRA),
ISSN: 1050-4729, May 2015, pp. 1093–1099. DOI: 10.
1109/ICRA.2015.7139312. [Online]. Available: https://
ieeexplore.ieee.org/abstract/document/7139312 (visited
on 04/05/2025).
[13] R. Mannadiar and I. Rekleitis, “Optimal coverage of a
known arbitrary environment,” in 2010 IEEE Interna-
tional Conference on Robotics and Automation, ISSN:
1050-4729, May 2010, pp. 5525–5530. DOI: 10.1109/
ROBOT. 2010 . 5509860. [Online]. Available: https : / /
ieeexplore . ieee . org / document / 5509860 (visited on
04/05/2025).
[14] G. Fevgas, T. Lagkas, V. Argyriou, and P. Sarigiannidis,
“Coverage Path Planning Methods Focusing on En-
ergy Efficient and Cooperative Strategies for Unmanned
Aerial Vehicles,” en, Sensors, vol. 22, no. 3, p. 1235,
Feb. 2022, ISSN: 1424-8220. DOI: 10.3390/s22031235.
[Online]. Available: https://www.mdpi.com/1424-8220/
22/3/1235 (visited on 04/07/2025).
[15] Martian/Lunar Crater Detection Dataset, en. [Online].
Available: https://www.kaggle.com/datasets/lincolnzh/
martianlunar - crater - detection - dataset (visited on
04/20/2025)

# Presentation
https://youtu.be/iT9bRe6xMaM



