""" https # //robotcar-dataset.robots.ox.ac.uk/

    Code to parse the data stored in the oxford robotcar dataset


"""

import argparse
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime as dt
from image import load_image
from camera_model import CameraModel
from sys import exit
import pandas as pd

# ---------- User Parameters -------------------------------------------------------------------------------------------
imageDir = "Dataset/"   # The folder to look for images
modelDir = "Models/"    # Contains the camera models
scale = 1.0     # Factor to scale the images before displaying

# ---------- Check dir structure ---------------------------------------------------------------------------------------
neededDirs = ["gps", "ldmrs", "lms_front", "lms_rear", "mono_left", "mono_rear", "mono_right", "stereo", "Models"]
if not os.path.isdir(imageDir):
    print("Error: Dataset dir doesn't exist")
    exit(2)

if not os.path.isdir(imageDir):
    print("Error: Models dir doesn't exist")
    exit(2)

directory = os.listdir(imageDir)
print(f"Checking dir structure of {imageDir} -> |", end='')
for i in neededDirs:
    if i in directory:
        print(f"{i}|", end='')
    else:
        print(f"{i} <- Not found")
        exit(2)
print()

# ---------- Start of main body ----------------------------------------------------------------------------------------
views = ["mono_left/", "mono_rear/", "mono_right/", "stereo/left/", "stereo/right/", "stereo/centre/"]  # Which views to show

itemList = []
for j in views:
    itemList.append(os.listdir(imageDir + j))
print(itemList)
