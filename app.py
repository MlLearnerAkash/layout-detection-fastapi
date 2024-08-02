from PIL import Image
import io
import pandas as pd
import numpy as np
import torch
from typing import Optional
import layoutparser as lp
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from detectron2 import *
# Initialize the models
model_sample_model = YOLO("./models/doclaynet_chkpt/best_doclaynet.pt")


def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
                        imgsz=image_size, 
                        source=input_image, 
                        conf=conf,
                        save=save, 
                        augment=augment,
                        flipud= 0.0,
                        fliplr= 0.0,
                        mosaic = 0.0,
                        device = [0 if torch.cuda.is_available() else "cpu"]
                        )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


################################# BBOX Func #####################################

def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """
    add a bounding box on the image

    Args:
    image (Image): input image
    predict (pd.DataFrame): predict from model

    Returns:
    Image: image whis bboxs
    """
    # Create an annotator object
    annotator = Annotator(np.array(image))

    # sort predict by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)
    #Adding Layoyt parser
    # lp_model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config', 
    #                              extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    #                              label_map={0: "Fig", 1: "Title", 2: "List", 3:"Table", 4:"Text"})
    # lp_model= lp.AutoLayoutModel('lp://EfficientDete/PubLayNet')
    # iterate over the rows of predict dataframe
    areas = []
    for i, row in predict.iterrows():
        # create the text to be displayed on image
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        # get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        # height =row['ymax']-row['ymin']
        # width = row['xmax']- row['xmin']
        # areas.append(height*width)
        # add the bounding box and text on the image
        annotator.box_label(bbox, text, color=colors(row['class'], True))
        # crop_img = np.array(image)[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax']),:]
        # layout = lp_model.detect(crop_img)
        # print([(block.score, block.type) for block in layout._blocks])
        # lp.draw_box(crop_img, layout, box_width=3).save(f"lp_det_{i}.png", format="PNG")

    # area_covered = sum(areas)
    # area_covered_ratio = area_covered/1050625
    # print(area_covered_ratio)
    # convert the annotated image to PIL image
    return Image.fromarray(annotator.result())

def add_filtered_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
    """
    add a bounding box on the image

    Args:
    image (Image): input image
    predict (pd.DataFrame): predict from model

    Returns:
    Image: image whis bboxs
    """
    # Create an annotator object
    annotator = Annotator(np.array(image))

    # sort predict by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)
    # predict = predict[predict['confidence']<0.8]
    # iterate over the rows of predict dataframe

    areas = []
    for i, row in predict.iterrows():
        # create the text to be displayed on image
        text = f"{row['name']}: {int(row['confidence']*100)}%"
        # get the bounding box coordinates
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        height =row['ymax']-row['ymin']
        width = row['xmax']- row['xmin']
        areas.append(height*width)
        # add the bounding box and text on the image
        annotator.box_label(bbox, text, color=colors(row['class'], True))
        crop_img = np.array(image)[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax']),:]
        #NOTE: To use Detectron
        if True:
            running_dectron(crop_img, f"crop_{i}")

    running_dectron(np.array(image), f"original_image.png")
    return Image.fromarray(annotator.result())

################################# Models #####################################


def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.15,
    )
    return predict
############################################################ Detectron ###############################################################
from detectron2.config import get_cfg
import argparse

def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file("/home/akash/ws/layout-segmentation-yolo/detectron2/configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    # cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.freeze()
    return cfg

def running_dectron(img, file_name):
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../detectron2')))
    # from detectron2 import detectron2
    from demo.predictor import VisualizationDemo
    from detectron2.utils.logger import setup_logger
    import multiprocessing as mp

    from detectron2.data import MetadataCatalog
    logger = setup_logger()
    MetadataCatalog.get("dla_val").thing_classes = ['text', 'title', 'list', 'table', 'figure']
    mp.set_start_method("spawn", force=True)

    cfg = setup_cfg()

    demo = VisualizationDemo(cfg)
    predictions, visualized_output = demo.run_on_image(img)
    # print(predictions)
    visualized_output.save(file_name)
    
    logger.info(
                "{}: detected ".format(
                    predictions["instances"]
                )
            )