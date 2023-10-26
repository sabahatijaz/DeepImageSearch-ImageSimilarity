# DeepImageSearch-ImageSimilarity

mage Similarity based on Color and Structural Features

Welcome to the Image Similarity repository! This repository provides a robust solution for comparing images based on both color and structural features. By leveraging advanced computer vision techniques, this system can accurately assess the similarity between images and provide meaningful insights.
Directory Structure

    .env: Configuration file for environment variables.
    DeepImageSearch.py: Utility functions for loading and searching images.
    image_segment.py: Module for image segmentation.
    Image_Similarity.py: Main module for comparing image similarity.
    requirements.txt: List of dependencies required for the project.

Installation

Before using the module, ensure you have Python installed on your system. Install the required dependencies using the following command:

bash

pip install -r requirements.txt

Usage
1. Segment Images (Optional but Recommended)

Before comparing images, it is recommended to segment them based on the region of interest. To segment images, run the following command:

bash

python image_segment.py

This command will segment images in the input folder based on the segmentation token specified in the configuration file (.env). The segmented images will be saved in the output folder.
2. Compare Image Similarity

Use the following command to compare the similarity between a reference segmented image and a set of images in a folder:

bash

python Image_Similarity.py

The system will output an ensemble score for each image in the specified folder, indicating the similarity between the images. The ensemble score is a combination of color and structural similarity scores.
Configuration

Before running the application, configure the following parameters in the .env file:

    IMAGE_RESOLUTION: Resolution for resizing images before comparison.
    SEGMENTATION_TOKEN: Object to detect for image segmentation.
    COLOR_SIMILARITY_THRESHOLD: Threshold for color similarity comparison.
    STRUCTURAL_SIMILARITY_THRESHOLD: Threshold for structural similarity comparison.

Example Usage

python

from Image_Similarity import ensemble_method

# Specify the reference segmented image path, input image folder, and output segmented image folder
reference_image_path = 'referenceSegmented.png'
input_image_folder = 'ImageSimilarity/filtered-20230911T122855Z-001/filtered'
output_segmented_folder = 'ImageSimilarity/filtered-20230911T122855Z-001/filteredSegmented'

# Get the ensemble similarity scores for the images in the input folder
ensemble_result = ensemble_method(reference_image_path, input_image_folder, output_segmented_folder)

# Print the ensemble similarity scores
print(ensemble_result)

Output

The output will be a dictionary containing image paths as keys and their corresponding ensemble similarity scores as values. Higher ensemble scores indicate greater similarity between the images.
