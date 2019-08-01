from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from myLib.qtThreading import Worker
import os
import re
import numpy as np
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from scipy.ndimage import map_coordinates
import cv2
from PyQt5.QtCore import QThread

""" imageLoading.py
Contains functions relating to loading images

preload_images (filename, number, totalNumber, 

"""


# ---------- Threaded image loading ------------------------------------------------------------------------------------
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


# ---------- Camera model class ----------------------------------------------------------------------------------------
class CameraModel:
    """Provides intrinsic parameters and undistortion LUT for a camera.

    Attributes:
        camera (str): Name of the camera.
        camera sensor (str): Name of the sensor on the camera for multi-sensor cameras.
        focal_length (tuple[float]): Focal length of the camera in horizontal and vertical axis, in pixels.
        principal_point (tuple[float]): Principal point of camera for pinhole projection model, in pixels.
        G_camera_image (:obj: `numpy.matrixlib.defmatrix.matrix`): Transform from image frame to camera frame.
        bilinear_lut (:obj: `numpy.ndarray`): Look-up table for undistortion of images, mapping pixels in an undistorted
            image to pixels in the distorted image

    """

    def __init__(self, models_dir, images_dir):
        """Loads a camera model from disk.

        Args:
            models_dir (str): directory containing camera model files.
            images_dir (str): directory containing images for which to read camera model.

        """
        self.camera = None
        self.camera_sensor = None
        self.focal_length = None
        self.principal_point = None
        self.G_camera_image = None
        self.bilinear_lut = None

        self.__load_intrinsics(str(models_dir), str(images_dir))
        self.__load_lut(str(models_dir), str(images_dir))

    def project(self, xyz, image_size):
        """Projects a pointcloud into the camera using a pinhole camera model.

        Args:
            xyz (:obj: `numpy.ndarray`): 3xn array, where each column is (x, y, z) point relative to camera frame.
            image_size (tuple[int]): dimensions of image in pixels

        Returns:
            numpy.ndarray: 2xm array of points, where each column is the (u, v) pixel coordinates of a point in pixels.
            numpy.array: array of depth values for points in image.

        Note:
            Number of output points m will be less than or equal to number of input points n, as points that do not
            project into the image are discarded.

        """
        if xyz.shape[0] == 3:
            xyz = np.stack((xyz, np.ones((1, xyz.shape[1]))))
        xyzw = np.linalg.solve(self.G_camera_image, xyz)

        # Find which points lie in front of the camera
        in_front = [i for i in range(0, xyzw.shape[1]) if xyzw[2, i] >= 0]
        xyzw = xyzw[:, in_front]

        uv = np.vstack((self.focal_length[0] * xyzw[0, :] / xyzw[2, :] + self.principal_point[0],
                        self.focal_length[1] * xyzw[1, :] / xyzw[2, :] + self.principal_point[1]))

        in_img = [i for i in range(0, uv.shape[1])
                  if 0.5 <= uv[0, i] <= image_size[1] and 0.5 <= uv[1, i] <= image_size[0]]

        return uv[:, in_img], np.ravel(xyzw[2, in_img])

    def undistort(self, image):
        """Undistorts an image.

        Args:
            image (:obj: `numpy.ndarray`): A distorted image. Must be demosaiced - ie. must be a 3-channel RGB image.

        Returns:
            numpy.ndarray: Undistorted version of image.

        Raises:
            ValueError: if image size does not match camera model.
            ValueError: if image only has a single channel.

        """
        if image.shape[0] * image.shape[1] != self.bilinear_lut.shape[0]:
            raise ValueError('Incorrect image size for camera model')

        lut = self.bilinear_lut[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))

        if len(image.shape) == 1:
            raise ValueError('Undistortion function only works with multi-channel images')

        undistorted = np.rollaxis(np.array([map_coordinates(image[:, :, channel], lut, order=1)
                                for channel in range(0, image.shape[2])]), 0, 3)

        return undistorted.astype(image.dtype)

    def __get_model_name(self, images_dir):
        self.camera = re.search('(stereo|mono_(left|right|rear))', images_dir).group(0)
        if self.camera == 'stereo':
            self.camera_sensor = re.search('(left|centre|right)', images_dir).group(0)
            if self.camera_sensor == 'left':
                return 'stereo_wide_left'
            elif self.camera_sensor == 'right':
                return 'stereo_wide_right'
            elif self.camera_sensor == 'centre':
                return 'stereo_narrow_left'
            else:
                raise RuntimeError('Unknown camera model for given directory: ' + images_dir)
        else:
            return self.camera

    def __load_intrinsics(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir)
        intrinsics_path = os.path.join(models_dir, model_name + '.txt')

        with open(intrinsics_path) as intrinsics_file:
            vals = [float(x) for x in next(intrinsics_file).split()]
            self.focal_length = (vals[0], vals[1])
            self.principal_point = (vals[2], vals[3])

            G_camera_image = []
            for line in intrinsics_file:
                G_camera_image.append([float(x) for x in line.split()])
            self.G_camera_image = np.array(G_camera_image)

    def __load_lut(self, models_dir, images_dir):
        model_name = self.__get_model_name(images_dir)
        lut_path = os.path.join(models_dir, model_name + '_distortion_lut.bin')

        lut = np.fromfile(lut_path, np.double)
        lut = lut.reshape([2, lut.size // 2])
        self.bilinear_lut = lut.transpose()


def load_stereo(image_path: str, model: CameraModel = None):
    """ Loads an image and processes it with a model (if given)

    :param image_path: The path to an image
    :type image_path: String
    :param model: A camera model path to undistort the image and apply LUT
    :type model: CameraModel class
    :param view: A pass-through variable
    :param outName: Another pass-through variable
    :type outName: str
    :returns cv2 image
    """

    pattern = 'gbrg'
    with Image.open(str(image_path)) as img:
        img = demosaic(img, pattern)

        if model:
            # Apply the model
            img = model.undistort(img)
        return img
