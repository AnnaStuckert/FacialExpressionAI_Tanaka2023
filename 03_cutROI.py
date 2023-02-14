#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:21:38 2022

@author: tanakayudai

Crop the indicated ROI in input images.
"""

# Import modules
import cv2
import glob,os

# Set the path
path_from = "******"
path_to = "******"

# Set the size of photos
top = 30
bottom = 162
left = 4
right = 229

# List all file names in a folder
f = os.listdir(path_from)

# Get the length of the list
photo_num = len(f)

for i in range (1, photo_num+1, 1):
      # Format numbers into 4-digit (0001) format
      zero_i = "{0:04d}".format(i)
      # Name files
      new_name = name +  "_" + zero_i
      img = cv2.imread(path_from + "/" + new_name + ".bmp")
      img1 = img[top : bottom, left : right]
      cv2.imwrite(path_to + "/" + new_name + ".tif", img1)

      print("cutting No." + zero_i + " image...")
