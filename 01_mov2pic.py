#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:21:38 2022

@author: tanakayudai

Convert a multi-tiff file to a series of BMP format files.
"""

# Import modules
!pip install gputil
!pip install psutil
!pip install humanize
import psutil
import humanize
import os
import GPUtil as GPU
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# Get filenames of the input movies
BASENAME = []
BASENAME1 = "******"
BASENAME2 = "******"
"......."
BASENAME.append(BASENAME1)
BASENAME.append(BASENAME2)
"......."

# Set the path
PATH = 'drive/My Drive/data/'
BASENAME = "******"

# Set the size of photos
im_rows = 600
im_cols = 700
im_rows = int(im_rows*0.33) 
im_cols = int(im_cols*0.33)

# Split a multi-page image to multiple single-page images and save the converted images as BMP files.
def open_multitiff(filename):
    img_pil = Image.open(filename)
    photo_num =img_pil.n_frames
    img = []
    count = 0
    while count < photo_num:
        img_pil.seek(count)
        img_tmp = np.asarray(img_pil)
        img.append(img_tmp)
        count += 1

    return img

def save_all_frames(video_path, dir_path, basename, ext='bmp'):
    cap = open_multitiff(video_path)
    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    for i in range (0, len(cap)):
      photo = cap[i]
      photo = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)  
      photo_resize = cv2.resize(photo, dsize=None, fx=0.33, fy=0.33)
      num_004d = "{0:04d}".format(i)
      print(num_004d)
      img = dir_path + "/" + basename + "_" + str(num_004d) + ".bmp"
      cv2.imwrite(img, photo_resize)

for vol in range(0,5):
  NAME = BASENAME + str(vol)
  save_all_frames(PATH + NAME + '.tif', '/content/drive/My Drive/data/' + NAME + "_new", NAME)

