import cv2 as cv
import numpy as np
import pandas as pd


# ////////// Binary Pipeline //////////////////////////////////////////////////////////////////////////////////////////
def sobel_filter(img):
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    ksize = 3
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    grad_x = cv.Sobel(img, ddepth, 1, 0, ksize, scale, delta, cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(img, ddepth, 0, 1, ksize, scale, delta, cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    binary_output = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # Return the result
    return binary_output


def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    binary_output = img
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    binary_output = img
    return binary_output


def hls_select(img, sthresh=(0, 255),lthresh=()):
    binary_output = img
    return binary_output


def red_select(img, thresh=(0, 255)):
    binary_output = img
    return binary_output


def binary_pipeline(img):
    # Apply a gaussian blur to remove some small noise
    gaussian_blur = cv.GaussianBlur(img, (3, 3), 0)

    # Sobel processing
    final_binary = sobel_filter(gaussian_blur)

    return final_binary
