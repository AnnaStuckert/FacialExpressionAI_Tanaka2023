# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Apr  5 19:21:38 2022

# @author: tanakayudai

# Generate a mask for a mouse.
# """

# # Import modules

# # assert torch.__version__.startswith("1.8")
# import json
# import os
# import random

# # from google.colab.patches import cv2_imshow
# # when running locally, use below instead
# import cv2
# import detectron2
# import matplotlib.pyplot as plt
# import numpy as np

# #!pip install pyyaml==5.1
# import torch
# from detectron2.utils.logger import setup_logger
# from PIL import Image

# #!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
# # the code did not work for mac, trying with this instead pip install 'git+https://github.com/facebookresearch/detectron2.git'


# setup_logger()
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer

# # normal
# PATH = "/Users/annastuckert/Downloads/Tanaka_et_al_2023/normal"
# PATH_processed = "/Users/annastuckert/Downloads/Tanaka_et_al_2023/normal_mask"

# NAME = "normal"

# im_rows = int(600 * 0.33)
# im_cols = int(700 * 0.33)


# # These elements were just left as open a = (for some reason)
# # pix =
# # a =
# # dif =
# # p =

# f = os.listdir(PATH + "/")
# photo_num = len(f)


# def rgb_to_gray(src):
#     b, g, r = src[:, :, 0], src[:, :, 1], src[:, :, 2]

#     return np.array(0.2989 * r + 0.5870 * g + 0.1140 * b, dtype="uint8")


# im = cv2.imread(PATH + "/" + NAME + ".bmp")

# # Recognize a mouse with a panoptic segmentation model
# cfg = get_cfg()
# cfg.merge_from_file(
#     model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
# )
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
#     "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
# )
# predictor = DefaultPredictor(cfg)
# panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

# TARGET = []
# CLASSES = []
# MASKS = []

# panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
# im_ps = v.get_image()[:, :, ::-1]
# target = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes.index("cat")
# TARGET.append(target)
# outputs = predictor(im)
# classes = np.asarray(outputs["instances"].to("cpu").pred_classes)
# CLASSES.append(classes)
# masks = np.asarray(outputs["instances"].to("cpu").pred_masks)[
#     CLASSES[0] == TARGET[0]
# ].astype("uint8")
# MASKS.append(masks)
# im_tra = MASKS[0].transpose(1, 2, 0)[:, :, 0]
# im_tra_inv = -(im_tra - 1)
# im_mask_inv = im_tra_inv.reshape(im_rows, im_cols, 1)
# hige = im * im_mask_inv
# im_mask = im_tra.reshape(im_rows, im_cols, 1)
# im_masked = im * im_mask
# back = np.zeros((im_rows, im_cols, 3))
# back += [200, 167, 122][::-1]
# im_invmask = im_mask * 255
# im_invmask = cv2.bitwise_not(im_invmask)
# im_invmask = im_invmask / 255
# im_invmask = im_invmask.reshape(im_rows, im_cols, 1)
# back_invmasked = back * im_invmask
# mouse = im_masked + back_invmasked
# cv2.imwrite("PATH", mouse)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:21:38 2022

Updated script to generate masks for mouse images.
Handles sequential files and uses CPU for processing.

@author: tanakayudai
"""

# Import modules
import os

import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

# Initialize logger
setup_logger()

# Paths
PATH = "/Users/annastuckert/Downloads/Tanaka_et_al_2023/normal"
PATH_processed = "/Users/annastuckert/Downloads/Tanaka_et_al_2023/normal_mask"

# Ensure output directory exists
os.makedirs(PATH_processed, exist_ok=True)

# Image dimensions (resize factors)
im_rows = int(600 * 0.33)
im_cols = int(700 * 0.33)

# Configure Detectron2 to use CPU
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
)
cfg.MODEL.DEVICE = "cpu"  # Force CPU usage

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Process all files in the directory
for file_name in sorted(os.listdir(PATH)):
    # Check if the file matches the expected pattern and is a .bmp file
    if file_name.startswith("normal_") and file_name.endswith(".bmp"):
        # Full path to the input file
        input_path = os.path.join(PATH, file_name)
        output_path = os.path.join(PATH_processed, file_name)

        # Read the image
        im = cv2.imread(input_path)
        if im is None:
            print(f"Warning: Could not read image: {input_path}")
            continue

        # Perform panoptic segmentation
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]

        # Visualize the segmentation
        visualizer = Visualizer(
            im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        visualized_output = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to("cpu"), segments_info
        )
        im_ps = visualized_output.get_image()[:, :, ::-1]

        # Save the processed image
        cv2.imwrite(output_path, im_ps)
        print(f"Processed and saved: {output_path}")
