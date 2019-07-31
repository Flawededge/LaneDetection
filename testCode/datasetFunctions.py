# # Dataset Functions
#
# This .py file handles the conversion from the dataset
#   folder to OpenCV images
#
# #

import re
import cv2 as cv
import csv
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

BAYER_STEREO = 'gbrg'
BAYER_MONO = 'rggb'


def load_stereo(paths, model_path):
    """
    :param paths: left, centre then right image directories
    :type paths: List of Strings
    :param model_path: Path to camera model to undistort image
    :type model_path: String
    :returns cv2 image list
    """
    models = ["stereo_wide_left.txt", "stereo_wide_left.txt", "stereo_wide_right.txt"]
    img = []
    for img, curModel in zip(paths, models):
        with open(model_path+curModel) as mod:
            mod = mod.readlines()

            cur = cv.imread(img)
            cur = cv.undistort(cur, )
            img.append(cur)

    return img
