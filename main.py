""" https # //robotcar-dataset.robots.ox.ac.uk/

    Code to parse the data stored in the oxford robotcar dataset


"""

import os
from sys import exit
from datasetFunctions import load_stereo
import matplotlib.pyplot as plt


# ---------- User Parameters -------------------------------------------------------------------------------------------
imageDir = "Dataset/stereo/"           # The folder to look for images
modelDir = "Dataset/Models/"    # Contains the camera models
scale = 1.0     # Factor to scale the images before displaying

# ---------- Check dir structure ---------------------------------------------------------------------------------------
# neededDirs = ["gps", "ldmrs", "lms_front", "lms_rear", "mono_left", "mono_rear", "mono_right", "stereo", "Models"]
# if not os.path.isdir(imageDir):
#     print("Error: Dataset dir doesn't exist")
#     exit(2)
#
# if not os.path.isdir(modelDir):
#     print("Error: Models dir doesn't exist")
#     exit(2)
#
# directory = os.listdir(imageDir)
# print(f"Checking dir structure of {imageDir} -> |", end='')
# for i in neededDirs:
#     if i in directory:
#         print(f"{i}|", end='')
#     else:
#         print(f"{i} <- Not found")
#         exit(2)
# print()

# ---------- Start of main body ----------------------------------------------------------------------------------------
views = ["left/", "centre/", "right/"]

itemList = []
for count, i in enumerate(views):
    itemList.append(os.listdir(imageDir + i))
    itemList[count] = [imageDir + views[count] + i for i in itemList[count] if i[0] != '.']

imageDirs = zip(itemList[0], itemList[1], itemList[2])

for i in imageDirs:
    images = load_stereo(i, modelDir)
