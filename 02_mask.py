#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:21:38 2022

@author: tanakayudai

Generate a mask for a mouse.
"""

# Import modules
!pip install pyyaml==5.1
import torch
assert torch.__version__.startswith("1.8")
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
import numpy as np
from PIL import Image
import os, json, cv2, random
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

PATH = '*****'
PATH_processed = '*****'

NAME = "*****"

im_rows = int(600*0.33)
im_cols = int(700*0.33)

pix =
a =
dif =
p =

f = os.listdir(PATH + "/")
photo_num =  len(f)

def rgb_to_gray(src):
  b, g, r = src[:,:,0], src[:,:,1], src[:,:,2]

  return np.array(0.2989 * r + 0.5870 * g + 0.1140 * b, dtype='uint8')

im = cv2.imread(PATH + "/" + NAME+ ".bmp")

# Recognize a mouse with a panoptic segmentation model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)
panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)

TARGET = []
CLASSES = []
MASKS = []

panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
im_ps = v.get_image()[:, :, ::-1]
target = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes.index("cat")
TARGET.append(target)
outputs = predictor(im)
classes = np.asarray(outputs["instances"].to("cpu").pred_classes)
CLASSES.append(classes)
masks = np.asarray(outputs["instances"].to("cpu").pred_masks)[CLASSES[0]==TARGET[0]].astype("uint8")
MASKS.append(masks)
im_tra = MASKS[0].transpose(1,2,0)[:,:,0]
im_tra_inv = -(im_tra -1)
im_mask_inv = im_tra_inv.reshape(im_rows, im_cols, 1)
hige = im * im_mask_inv
im_mask = im_tra.reshape(im_rows, im_cols, 1)
im_masked = im * im_mask
back = np.zeros((im_rows, im_cols, 3))
back += [200,167,122][::-1]
im_invmask = im_mask *255
im_invmask = cv2.bitwise_not(im_invmask)
im_invmask = im_invmask/255
im_invmask = im_invmask.reshape(im_rows, im_cols,1)
back_invmasked = back * im_invmask
mouse = im_masked + back_invmasked
cv2.imwrite("PATH", mouse)
