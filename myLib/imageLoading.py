from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from myLib.qtThreading import Worker

""" imageLoading.py
Contains functions relating to loading images

preload_images (filename, number, totalNumber, 

"""


def preload_sequencer(stereo_path, store_path, timestamp_list, camera_models, threadpool,
                      counter_function=None, call1=None, call2=None):
    """
    :param stereo_path: The path to the stereo folder
    :type stereo_path: Path
    :param store_path: The path to the folder for items to be stored to
    :param timestamp_list: A list of the timestamps
    :type timestamp_list: list of string
    :param camera_models: A list of left, centre and right camera model classes
    :param threadpool: The threadpool
    :param counter_function: Function to be called when an image finishes loading
    :param call1: Used for compatibility with qThreading
    :param call2: Used for compatibility with qThreading
    :return: None
    """
    for image in timestamp_list:
        image += ".png"
        if (not (store_path / "left" / image).is_file()) or (not (store_path / "centre" / image).is_file()) or (not (store_path / "right" / image).is_file()):
            worker = Worker(preload_images, stereo_path, image, store_path, camera_models)
            worker.signals.finished.connect(counter_function)
            threadpool.start(worker)
        else:
            counter_function()
            print(f"{str(Image)} Already exists")


def preload_images(stereo_path, image_name, store_path, camera_models, call1=None, call2=None):
    """ Loads the left, right and centre image from the stereo folder
    :param stereo_path: The path to the stereo folder
    :type stereo_path: Path
    :param image_name: The image name, with .png at the end
    :type image_name: str
    :param store_path: The stereo path to store processed images
    :type store_path: Path
    :param camera_models: A list of 3 camera models for left, centre and right respectively
    :param call1: Used for compatibility with qThreading
    :param call2: Used for compatibility with qThreading
    :return: None
    """

    # Demosaic and undistort each of the image channels
    pattern = 'grbg'

    with Image.open(str(stereo_path/"left"/image_name)) as left:
        left = demosaic(left, pattern)
        left = camera_models[0].undistort(left)

    with Image.open(str(stereo_path / "centre" / image_name)) as centre:
        centre = demosaic(centre, pattern)
        centre = camera_models[1].undistort(centre)

    with Image.open(str(stereo_path / "right" / image_name)) as right:
        right = demosaic(right, pattern)
        right = camera_models[2].undistort(right)

    # Swap the blue and red channels
    # left = cv2.cvtColor(left.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # centre = cv2.cvtColor(centre.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # right = cv2.cvtColor(right.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Write the files to the output folder
    cv2.imwrite(str(store_path / "left" / image_name), left)
    cv2.imwrite(str(store_path / "centre" / image_name), centre)
    cv2.imwrite(str(store_path / "right" / image_name), right)

