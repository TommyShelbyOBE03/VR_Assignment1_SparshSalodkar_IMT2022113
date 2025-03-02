# VR_Assignment1_SparshSalodkar_IMT2022113

VR Assignment 1 - Coin detection and image stitching

Description -

This project implements computer vision techniques to detect, segment, and count Indian coins in an image, and stitches multiple overlapping images into a panorama. The coin detection script uses edge detection and Hough Circle Transform, while the panorama stitching script leverages OpenCV's Stitcher class with post-processing. All code is written in Python, organized in a GitHub repository, and includes detailed documentation for setup and execution.

Overview - 

This repository contains the solutions for the Computer Vision Assignment due on March 2, 2025. It includes two Python scripts as per the assignment requirements:

1. Coin Detection and Counting (coin_detection_segmentation.py): Detects and counts coins in an image (`coins2.jpeg`) using edge detection and Hough Circle Transform.
2. Panorama Stitching ('panorama_creation.py'): Stitches multiple overlapping images into a panorama using OpenCV's `Stitcher` class, with post-processing to refine the output.

SETUP INSTRUCTIONS -

Clone the repository - 

    git clone <repository-url>
    cd VR_Assignment1_SparshSalodkar_IMT2022113

Run the following command on bash to install the required dependencies - 

    pip install -r requirements.txt

For coin segmentation run the coin_detection_segmentation.py (Make sure to change the name of the image file and the format in the code when wish to test for a different image, and make sure that the image is in the folder "images") The results for detection and segmentation will be saved in a directory called results. 

    python3 coin_detection_segmentation.py

For the panorama stitching part run the panorama_creation.py script ( Make sure that the images selected are placed in the "panoImages" folder in the main directory), The results will get saved in a directory called "stitching_output" that'll be created along the way. If not enough keypoints are found, the stitched image does not get created.

    python3 panorama_creation.py

