#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:21:38 2022

@author: tanakayudai

Convert images to NumPy arrays.
"""

# Import module
import os

# Set the path
path = "******"
# List all file names in a folder
index0 = os.listdir(path)
# Get list length (number of files)
count0 = len(index0)

# Read image files and convert to Numpy format
outfile_name = "******"
outfile= "******"

im_rows = bottom - top
im_cols = right - left
x = [] #image date
y = [] #label date

def main():
    # Read the folder for each image
    glob_files(path_train_normal, 0)
    glob_files(path_train_pain, 1)
    glob_files(path_train_tickle, 2)
    # Save image to file
    np.savez(outfile, x=x, y=y)

def glob_files(path, label):
  # List all file names in a folder
  index = os.listdir(path)
  # Get list length (number of files)
  count = len(index)
  # Check the number of files
  files = glob.glob(path + "/*.tif")
    
  # Process each file
  num = 0
  for f in files:
    if num >= count : break
    num += 1
    # Read image files
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((im_cols, im_rows))
    img = np.asarray(img)
    x.append(img)
    y.append(label)

if __name__ == '__main__':
    main()
