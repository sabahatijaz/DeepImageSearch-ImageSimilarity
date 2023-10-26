import cv2
import numpy as np
import os

from PIL.Image import Image

from DeepImageSearch import Load_Data, Search_Setup
from decouple import config

from image_segment import prediction


def compare_color_similarity(roi1, roi2, threshold1=5):
    """
    Compare the color similarity of two ROIs.

    Args:
        roi1: A numpy array representing the first ROI.
        roi2: A numpy array representing the second ROI.
        threshold: A threshold for color similarity comparison.

    Returns:
        (bool, float): A tuple containing a boolean indicating whether the ROIs have the same color
        and the absolute color difference value.
    """
    # Convert ROIs to grayscale if needed
    roi1_gray = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    roi2_gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    # Define a lower threshold for excluding black color (adjust as needed)
    black_threshold = 10

    # Exclude black color pixels from the ROIs
    roi1_no_black = roi1[roi1_gray > black_threshold]
    roi2_no_black = roi2[roi2_gray > black_threshold]

    # Check if there are any remaining pixels after excluding black color
    if len(roi1_no_black) == 0 or len(roi2_no_black) == 0:
        return False, float('inf')  # Return False and infinite difference if no pixels remain

    # Calculate the mean pixel values in the ROIs
    mean_color_roi1 = np.mean(roi1_no_black, axis=0)
    mean_color_roi2 = np.mean(roi2_no_black, axis=0)

    # Calculate the absolute color difference
    color_difference = np.linalg.norm(mean_color_roi1 - mean_color_roi2)

    # Determine if the ROIs have the same color
    is_same_color = (color_difference < threshold1)

    return is_same_color, color_difference

def scale_values(value, thresh=40):
    """
    Scale a value based on a threshold.

    Args:
        value: The input value to scale.
        thresh: The threshold value for scaling.

    Returns:
        float: The scaled value.
    """
    if value < thresh:
        scaled_value = 0.5 + (thresh - value) * (1 - 0.5) / (thresh - 0)
    else:
        scaled_value = 0.5 - (value - thresh) * (0.5 - 0) / (100 - thresh)  # Adjust the upper limit as needed

    return scaled_value

def scale_values2(value, thresh=1.0):
    """
    Scale a value based on a threshold.

    Args:
        value: The input value to scale.
        thresh: The threshold value for scaling.

    Returns:
        float: The scaled value.
    """
    if value < thresh:
        scaled_value = 0.5 + (thresh - value) * (thresh - 0.5) / (thresh - 0.0)
    else:
        scaled_value = 0.5 - (value - thresh) * (0.5 - 0.0) / (10.0 - thresh)  # Adjust the upper limit as needed

    return scaled_value

def color_sim_check(ref_image_path, image_dir):
    """
    Compare the color similarity between a reference image and a set of images in a directory.

    Args:
        ref_image_path (str): The path to the reference segmented image.
        image_dir (str): The directory containing the images to compare.

    Returns:
        dict: A dictionary containing image paths as keys and similarity scores as values.
    """
    # Load the reference segmented image and resize it to 512x512
    resolution=config("IMAGE_RESOLUTION")
    reference_image = cv2.imread(ref_image_path)
    reference_image = cv2.resize(reference_image, (resolution, resolution))

    # Create a dictionary to store image paths and their similarity scores
    similarity_scores = {}

    # Iterate over images in the folder
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            # Load the current image from the folder and resize it to 512x512
            current_image = cv2.imread(os.path.join(image_dir, filename))
            current_image = cv2.resize(current_image, (resolution, resolution))

            # Compare the color similarity
            is_same_color, color_difference = compare_color_similarity(reference_image, current_image)

            # Store the similarity score in the dictionary
            image_path = os.path.join(image_dir, filename)
            similarity_scores[image_path] = {
                'Same Color': is_same_color,
                'Color Difference': color_difference
            }

    return similarity_scores

def transform_dictionary(input_dict):
    """
    Transforms a dictionary with scores as keys and image paths as values
    into a new dictionary with image paths as keys and scaled scores as values.

    Args:
        input_dict (dict): The original dictionary.

    Returns:
        dict: The transformed dictionary.
    """
    new_dict = {}
    for score, image_path in input_dict.items():
        scaled_score = scale_values2(score, 1.0)
        new_dict[image_path] = scaled_score
    return new_dict

def structural_sim_check(ref_image_path, image_dir):
    """
    Calculate structural similarity scores between a reference image and a set of images in a directory.

    Args:
        ref_image_path (str): The path to the reference segmented image.
        image_dir (str): The directory containing the images to compare.

    Returns:
        dict: A dictionary containing image paths as keys and similarity scores as values.
    """
    # Load images from a folder
    image_list = Load_Data().from_folder([image_dir])

    # Set up the search engine
    st = Search_Setup(image_list=image_list, model_name='vit_base_patch16_224_in21k', image_count=len(image_list), pretrained=True)

    # Index the images
    st.run_index()

    # Get similar images
    original_dict = st.get_similar_images(image_path=ref_image_path, number_of_images=len(image_list))

    # Transform the original dictionary to scale similarity scores
    new_dict = transform_dictionary(original_dict)

    return new_dict


def process_images(reference_image_path,input_folder_path, output_folder_path):
    inp_img = Image.open(reference_image_path)
    word_mask=config("SEGMENTATION_TOKEN")
    segmented_image = prediction(init_image=inp_img, word_mask=word_mask)
    cv2.imwrite(reference_image_path, segmented_image)
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Iterate through images in the specified folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            # Path to current input image
            input_image_path = os.path.join(input_folder_path, filename)
            inp_img=Image.open(input_image_path)
            # Perform segmentation using the 'prediction' function
            segmented_image = prediction(init_image=inp_img,word_mask=word_mask)

            # Path to save segmented image in the output folder
            output_image_path = os.path.join(output_folder_path, filename)

            # Save the segmented image with the same name in the output folder
            cv2.imwrite(output_image_path, segmented_image)

    return reference_image_path,output_folder_path

def ensemble_method(ref_image_path, image_dir,segmented_image_dir):
    """
    Apply an ensemble method to combine color and structural similarity scores.

    Args:
        ref_image_path (str): The path to the reference segmented image.
        image_dir (str): The directory containing the images to compare.

    Returns:
        dict: A dictionary containing image paths as keys and ensemble scores as values.
    """
    ref_image_path,segmented_image_dir=process_images(ref_image_path, image_dir,segmented_image_dir)
    # Call the color similarity method
    cl_sim = color_sim_check(ref_image_path, segmented_image_dir)

    # Call the structural similarity method
    st_sim = structural_sim_check(ref_image_path, segmented_image_dir)

    # Create an empty dictionary to store the ensemble results
    ensemble_result = {}
    COLOR_SIM_WEIGHT=config("color_similarity_threshold")
    STRUCTURAL_SIM_WEIGHT=config("structural_similarity_threshold")

    # Iterate through the keys (image paths) in st_sim (or cl_sim, as both have the same keys)
    for image_path in st_sim:
        # Combine the scores with the specified weightage
        v = scale_values(cl_sim[image_path]['Color Difference'], 40)
        ensemble_score = (STRUCTURAL_SIM_WEIGHT * st_sim[image_path]) + (COLOR_SIM_WEIGHT * v)

        # Set the ensemble score in the result dictionary
        ensemble_result[image_path] = ensemble_score

    return ensemble_result

if __name__ == "__main__":
    ensemble_result = ensemble_method('refrenceSegmented.png', 'ImageSimilarity/filtered-20230911T122855Z-001/filtered','ImageSimilarity/filtered-20230911T122855Z-001/filteredSegmented')
    print(ensemble_result)
