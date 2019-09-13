# https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking

# Select which file to use
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
import numpy as np
import processes
import os
from copy import copy

# Define which folder the videos are in and which index to use
path = Path("D:\Golf-cart-dataset\My Videos")
number = 1

files = [i for i in path.iterdir()]
[print(f"{cnt}|\t{i}") for cnt, i in enumerate(files)]

targetFile = files[number]
print(f"\nLoading '{targetFile.parts[-1]}' at {targetFile}")

cap = cv.VideoCapture(str(targetFile))  # Get the video capture stream
progress = range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
cv.namedWindow('frame', cv.WINDOW_GUI_EXPANDED)  # Build a named window which can be resized
cv.resizeWindow('frame', 960, 540)
cv.namedWindow('t', cv.WINDOW_GUI_EXPANDED)
cv.resizeWindow('t', 960, 540)
cv.moveWindow('frame', 0, 0)
cv.moveWindow('t', 0, 540)

# Generate the ROI image
shape = (540, 960)  # Y, X cause numpy is a heathen
width = 400
height = 80
offsetOffBottom = 220
roi_clip = np.zeros(shape, dtype=np.uint8)  # Create a blank image for the ROI clipping
cv.rectangle(roi_clip, (int((shape[1] - width) / 2), shape[0] - offsetOffBottom - height),
             (int((shape[1] + width) / 2), shape[0] - offsetOffBottom), 255, -1)
# cv.imshow('clip', roi_clip)  # Have a look at the clip if necessary

# while cap.isOpened():
for i in tqdm(progress):
    ret, frame = cap.read()
    frame = cv.resize(frame, (960, 540))

    img_hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    # img_hsv[:, :, 0] = cv.equalizeHist(img_hsv[:, :, 0])
    # img_hsv[:, :, 1] = cv.equalizeHist(img_hsv[:, :, 1])
    # img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])
    new_frame = frame.copy()
    # new_frame[:, :, 0] = cv.equalizeHist(frame[:, :, 0])
    # new_frame[:, :, 1] = cv.equalizeHist(frame[:, :, 1])
    # new_frame[:, :, 2] = cv.equalizeHist(frame[:, :, 2])
    new_frame = cv.GaussianBlur(new_frame, (5, 5), 0)
    mask_white = cv.inRange(frame[:, :, :], (110, 110, 110), (180, 180, 180))

    # Clip to the ROI, as it will make some of the binary functions run faster
    mask_white = mask_white & roi_clip

    # mask_white = cv.morphologyEx(mask_white, cv.MORPH_CLOSE, (3, 3))
    mask_white = cv.medianBlur(mask_white, 5)

    canny_edges = cv.Canny(mask_white, 50, 255)
    # rho and theta are the distance and angular resolution of the grid in Hough space
    # same values as quiz
    rho = 2
    theta = np.pi / 180
    # threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 30
    min_line_len = 60
    max_line_gap = 180
    # my hough values started closer to the values in the quiz, but got bumped up considerably for the challenge video

    canny_edges = cv.cvtColor(canny_edges, cv.COLOR_GRAY2BGR)
    line_image = processes.hough_lines(mask_white, rho, theta, threshold, min_line_len, max_line_gap, frame)
    # result = processes.weighted_img(canny_edges, frame, α=0.8, β=1., λ=0.)

    cv.imshow('frame', frame)
    cv.imshow('t', canny_edges)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
