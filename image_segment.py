import os
import torch
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from segment_anything import SamPredictor
from typing import List
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import supervision as sv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
HOME = os.getcwd()
print("HOME:", HOME)
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                             model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = "vit_h"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)




def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """
            Segments objects in the given image using the provided bounding boxes.

            Args:
                image (np.ndarray): Input image.
                xyxy (np.ndarray): Bounding boxes in xyxy format.

            Returns:
                np.ndarray: Segmented masks for the objects.
    """
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)




def enhance_class_name(class_names: List[str]) -> List[str]:
    """
            Enhances class names by adding a common prefix.

            Args:
                class_names (List[str]): List of class names.

            Returns:
                List[str]: Enhanced class names.
    """
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def create_mask(image, objec):
    """
            Creates a segmentation mask for the specified object in the given image.

            Args:
                image (np.ndarray): Input image.
                object_name (str): Object to be segmented.

            Returns:
                np.ndarray: Segmentation mask for the specified object.
    """

    CLASSES = [objec]
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    # load image
    image = np.array(image)
    # cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} "
        for class_id
        in list(detections.class_id)]



    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} "
        for class_id
        in list(detections.class_id)]


    titles = [
        CLASSES[class_id]
        for class_id
        in detections.class_id
    ]

    images_list = list()
    for i, id in enumerate(detections.class_id):
        if titles[i] == objec:
            mid = id
            pil_im = Image.fromarray(detections.mask[i])
            pil_im.save('image' + str(i) + '.png')
            images_list.append(pil_im)

    desired_mask = ""
    print(len(images_list))
    for i in range(len(images_list)):
        if i == 0:
            desired_mask = np.array(images_list[0])
            desired_mask = desired_mask.astype(float)
        else:
            temp_mask = np.array(images_list[i])
            temp_mask = temp_mask.astype(float)
            desired_mask = cv2.bitwise_or(temp_mask, desired_mask)

    desired_mask = np.repeat(desired_mask[:, :, np.newaxis], 3, axis=2)
    mask3d = desired_mask
    mask3d[desired_mask > 0] = 255.0
    mask3d[mask3d > 255] = 255.0
    cv2.imwrite("desired_mask.jpg", mask3d)
    img = cv2.imread('desired_mask.jpg', 0)
    print(img.shape)
    return img


def prediction(init_image, word_mask):
    """
    Perform object detection, segmentation, and create a final segmentation mask.

    Args:
        init_image PIL: Input image.
        word_mask (str): Object to detect.

    Returns:
        np.ndarray: Final segmentation mask.
    """
    mask = create_mask(init_image, word_mask)
    np_img = np.array(init_image)
    bitwiseAnd = cv2.bitwise_and(np_img, np_img, mask=mask)
    return bitwiseAnd