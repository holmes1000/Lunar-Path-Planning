"""
Samara Holmes
Spring 2025

Program to process the image using YOLOv8 pre-trained model

Pre-trained model: https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset
"""

from process_image import load_images_from_directory
import numpy as np
from load_image import save_img
import torch
# from ultralytics import YOLO
import yolov5
import os
from generate_obstacles import create_grid_map, display_grid_map
from skimage import io, filters, measure, morphology
from detect_crater import detect_craters, display_craters_on_image
import matplotlib.pyplot as plt
from load_image import show_img

def convert_to_regionprops(detections, image_shape):
    """Convert YOLOv5 detections to skimage regionprops"""
    # Create an empty labeled image
    label_image = np.zeros(image_shape[:2], dtype=np.int32)
    
    # For each detection, create a mask and add it to the label image
    for i, detection in enumerate(detections):
        # Get coordinates (x1, y1, x2, y2, confidence, class)
        x1, y1, x2, y2 = map(int, detection[:4])
        
        # Create a binary mask for this detection
        label_image[y1:y2, x1:x2] = i + 1  # Use i+1 as the label (0 is background)
    
    # Get regionprops for the labeled image
    regions = measure.regionprops(label_image)
    
    # Add YOLOv5 specific properties to each region
    for i, region in enumerate(regions):
        # Add confidence and class info from YOLOv5
        region.confidence = detections[i][4]
        region.class_id = int(detections[i][5])
    
    return regions

if __name__ == "__main__":
    # Specify the directory
    image_directory = "images/"

    # Load images
    images = load_images_from_directory(image_directory)

    # Print the number of images loaded
    print(f"Loaded {len(images)} unprocessed images.")

    # model = torch.load("crater.pt", weights_only=False)
    # model.eval()

    # Check if CUDA is available and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # Load a pretrained YOLO11n model
    # model = YOLO("crater.pt")
    model = yolov5.load("crater.pt")
    model.to(device)

    output_directory = "yolo_images/"

    # Process each image for crater detection
    for i, img in enumerate(images):
        # Convert PIL Image to numpy array if needed
        if hasattr(img, 'convert'):
            img_np = np.array(img)
        else:
            img_np = img
            
        # Run detection
        results = model(img_np)
        
        # Save results
        # results.save(output_path)  # Save the results with annotations
        # save_img(results, f"yolo_images/detection_image_{i}.tiff")

        detections = results.xyxy[0].cpu().numpy()  # Get boxes in xyxy format

        labelled_img, regions = detect_craters(img_np, i)

        regions = convert_to_regionprops(detections, img_np.shape)

        print("converted detections to regions")

        # Convert detections to regions
        img = display_craters_on_image(labelled_img, regions, i)
        show_img(img)
        save_img(img, f"yolo_images/detection_result_{i}.png")

        # Create grid map
        print(img.shape)
        grid_size = 20
        grid_map = create_grid_map(img.shape, regions, grid_size=grid_size)

        fig = display_grid_map(grid_map, img, regions, grid_size=grid_size)
        plt.show()

        
        # Print detection info
        print(detections)
        print(f"Processed image {i+1}/{len(images)}")
        print(f"Detected {len(results.xyxy[0])} craters in image {i}")