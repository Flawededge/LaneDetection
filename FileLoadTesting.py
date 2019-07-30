""" https # //robotcar-dataset.robots.ox.ac.uk/

    Code to parse the data stored in the oxford robotcar dataset


"""

from pathlib import Path
import loadImages
import numpy as np
import cv2 as cv
import os
from copy import copy
from timeit import default_timer as timer  # For benchmarking loading images
import tkinter


# ---------- User Parameters -------------------------------------------------------------------------------------------
imageDir = "F:/TestSet/"            # The folder to look for images
modelDir = "Models/"                # Contains the camera models
storeDir = f"{imageDir}processed/"  # Relative dir in which to store processed files
storeTyp = ".jpg"                   # The image format to store processed images in
scale = 1.0                         # Factor to scale the images before displaying

display = True  # Whether to display the images using imshow from opencv
waitTime = 1    # If display, then what is the delay between images (0 to wait for key press)

# ---------- Check dir structure ---------------------------------------------------------------------------------------

# # Get the paths of everything
views = [[Path(f"{imageDir}stereo/left/"), Path(f"{storeDir}/left")],
         [Path(f"{imageDir}stereo/centre/"), Path(f"{storeDir}/centre")],
         [Path(f"{imageDir}stereo/right/"), Path(f"{storeDir}/right")]]

gpsP = Path(f"{imageDir}gps/")
lmsP = Path(f"{imageDir}lms_front/")

error = False

if not Path(storeDir).is_dir():
    os.mkdir(storeDir)

for i in views:
    if not i[0].is_dir():
        print(f"Error: '{i}' is not valid path")
        error = True

    if not i[1].is_dir():
        os.mkdir(i[1])

if error:
    print("Error occurred - Exiting")
    exit()
else:
    print("Paths are correct!")

# Get the models of the views which are going to be used
models = [loadImages.CameraModel(modelDir, str(views[0])),
          loadImages.CameraModel(modelDir, str(views[1])),
          loadImages.CameraModel(modelDir, str(views[2]))]

with open(imageDir + "stereo.timestamps") as timestampFile:
    timestamps = timestampFile.readlines()
    print(f"Now processing {len(timestamps)} images:")
    for cnt, line in enumerate(timestamps):
        start = timer()  # Start timer for benchmarking

        time = line.split()[0]  # Get the line
        print(f"{cnt}| {time} -> ", end='')

        # Load from image and process it
        print("Disk -> ", end='')
        # Load the images
        images = []
        for view, model in zip(views, models):
            cur_view = [copy(view[0]).joinpath(time + ".png"), copy(view[1]).joinpath(time + storeTyp)]
            if cur_view[1].exists():
                images.append(cv.imread(str(cur_view[1])))
            else:  # Load the original image with model and convert it to a cv image
                images.append((loadImages.load_stereo(str(cur_view[0]), model)))

                # Image is loaded in a 0-255 float, where cv works in 0-1 float, so divide by 255
                # Also converting the image from a PIL based image to a cv2 based image
                images[-1] = (np.array(images[-1]))[:, :, ::-1].astype(np.uint8)

                # Store the image in the processed folder
                cv.imwrite(str(cur_view[1]), images[-1])

        # The image is loaded
        end = timer()
        print(f"Loaded in {end-start:.5f}s -> ", end='')
        # Display the images if applicable
        if display:
            cv.imshow("Left", images[0])
            cv.imshow("Center", images[1])
            cv.imshow("Right", images[2])
            print("Image showing -> ", end='')
            cv.waitKey(waitTime)

        print("Next")
