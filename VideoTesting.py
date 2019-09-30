# https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking

# Select which file to use
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
import numpy as np
import processes
import os
from copy import copy
from BaseDetectionLib import loading

# /---------- User configuration ----------\ ---------------------------------------------------------------------------
# Define which folder the videos are in and which index to use
path = Path("D:\Golf-cart-dataset\My Videos")
number = 1

# 0 = off, 1 = Only output, 2 = Show steps
reliableLaneMarkings = 1
SDCLane = 1

# CV View windows
windowSize = [960, 640]  # The pixel size of each window
# windowSize = [480, 320]  # The pixel size of each window
grid = 4  # How many windows fit across a screen
startX = 0
titleOffset = 33

# /---------- Process setup -------------\ -----------------------------------------------------------------------------
# # Reliable lane markings
# The x, y points for the IPM. Top left, top right, bottom right, bottom left
ipm_points = np.float32([(384, 238), (501, 231), (740, 304), (200, 330)])

# SDC Lane Detection
roiRect = (200, 245, 800, 290)  # A rectangle to crop down to (x, y, x, y) top left, bottom right
sizeMultiplier = 1.6  # Multiply the size of the tiny windows to fit on the screen
top_bot_extend = (230, 380)

# /---------- Variable setup ------------\ -----------------------------------------------------------------------------
# # File
files = [i for i in path.iterdir()]
[print(f"{cnt}|\t{i}") for cnt, i in enumerate(files)]

targetFile = files[number]
print(f"\nLoading '{targetFile.parts[-1]}' at {targetFile}")

# # Video stream
cap = cv.VideoCapture(str(targetFile))  # Get the video capture stream
progress = range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))

# # Pre-Create windows
windowList = [["Combined", None]]  # Contains a list of window names

# # Reliable Lane Markings
if reliableLaneMarkings:
    windowList.append(["Reliable: Output", None])
if reliableLaneMarkings == 2:
    [windowList.append(i) for i in [["Reliable: Perspective", None],
                                    ["Reliable: PerspecGray", None],
                                    ["Reliable: PerspThresh", None],
                                    ["Reliable: PerspecMask", None],
                                    ["Reliable:  outputMask", None]
                                    ]]

# # SDC Lane Detection
# Window size for the cropped ones
sdc_size = (int(sizeMultiplier * (roiRect[2] - roiRect[0])), int(sizeMultiplier * (roiRect[3] - roiRect[1])))
if SDCLane:
    windowList.append(["SDC: Output", None])
if SDCLane == 2:
    [windowList.append(i) for i in [["SDC: mask_image", sdc_size],
                                    ["SDC: mask_rgb", sdc_size],
                                    ["SDC: mask_hsv", sdc_size],
                                    ["SDC: mask_sobel_x", sdc_size],
                                    ["SDC: mask_sobel_y", sdc_size],
                                    ["SDC: full_mask", sdc_size],
                                    ["SDC: filtered_mask", sdc_size],
                                    ["SDC: hough_lines", sdc_size]
                                    ]]

# # Create windows
for cnt, i in enumerate(windowList):
    cv.namedWindow(i[0], cv.WINDOW_GUI_EXPANDED)  # Build a named window which can be resized
    if i[1] is None:
        cv.resizeWindow(i[0], windowSize[0], windowSize[1])
    else:
        cv.resizeWindow(i[0], i[1][0], i[1][1])
    cv.moveWindow(i[0], (windowSize[0] * int(cnt % grid)) + startX, (windowSize[1] + titleOffset) * int(cnt / grid))

# # Frame processing loop
for i in tqdm(progress):
    ret, frame = cap.read()  # Get frame
    frame = cv.resize(frame, (960, 540))  # Scale the frame to fixed size so variable resolution input is possible

    passthrough = frame.copy()

    # # Reliable lane markings
    if reliableLaneMarkings == 2:
        tmp, passthrough = loading.reliable_lane_markings(frame.copy(), ipm_points,
                                                          passthrough_image=passthrough, progress_display=True)
        cv.imshow("Reliable: Output", tmp)
    elif reliableLaneMarkings:
        tmp, passthrough = loading.reliable_lane_markings(frame.copy(), ipm_points,
                                                          passthrough_image=passthrough, progress_display=False)
        cv.imshow("Reliable: Output", tmp)

    # SDC Lane detection
    if SDCLane == 2:
        tmp, passthrough = loading.sdc_lane_detection(frame.copy(), roiRect, top_bot_extend,
                                                      apply_roi=True, passthrough_image=passthrough, progress_display=True)
        cv.imshow("SDC: Output", tmp)
    elif SDCLane:
        tmp, passthrough = loading.sdc_lane_detection(frame.copy(), roiRect, top_bot_extend,
                                                      apply_roi=True, passthrough_image=passthrough, progress_display=False)
        cv.imshow("SDC: Output", tmp)

    cv.imshow("Combined", passthrough)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
