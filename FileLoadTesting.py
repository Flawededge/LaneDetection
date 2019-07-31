""" https # //robotcar-dataset.robots.ox.ac.uk/

    Code to parse the data stored in the oxford robotcar dataset


"""

from pathlib import Path
from testCode import loadImages
import numpy as np
import cv2
import os
from PyQt5.QtCore import *
import time
import traceback, sys

# ---------- User Parameters -------------------------------------------------------------------------------------------
imageDir = "F:/Datasets/2014-05-06-12-54-54/"            # The folder to look for images
modelDir = f"{imageDir}Models/"                # Contains the camera models
storeDir = f"{imageDir}processed/"  # Relative dir in which to store processed files
scale = 1.0                         # Factor to scale the images before displaying

display = True  # Whether to display the images using imshow from opencv
waitTime = 1    # If display, then what is the delay between images (0 to wait for key press)

# ---------- Check dir structure ---------------------------------------------------------------------------------------


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything

    progress
        `int` indicating % progress

    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)

class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        # self.kwargs['progress_callback'] = self.signals.progress
        # self.kwargs['latestResult'] = self.signals.latestResult

    @pyqtSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


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
    timestamps = [i.split()[0] for i in timestamps]


def loadAndSaveImage(filename, number, totalNumber):
    tmp1 = loadImages.load_stereo(views[0][0] / filename, models[0])
    tmp2 = loadImages.load_stereo(views[1][0] / filename, models[1])
    tmp3 = loadImages.load_stereo(views[2][0] / filename, models[2])

    tmp1 = cv2.cvtColor(tmp1.astype(np.uint8), cv2.COLOR_BGR2RGB)
    tmp2 = cv2.cvtColor(tmp2.astype(np.uint8), cv2.COLOR_BGR2RGB)
    tmp3 = cv2.cvtColor(tmp3.astype(np.uint8), cv2.COLOR_BGR2RGB)

    cv2.imwrite(str(views[0][1] / filename), tmp1)
    cv2.imwrite(str(views[1][1] / filename), tmp2)
    cv2.imwrite(str(views[2][1] / filename), tmp3)

    print(f"{number}/{totalNumber}\t{filename}")

    # cv2.imshow("left", tmp1)
    # cv2.imshow("centre", tmp2)
    # cv2.imshow("right", tmp3)


threadpool = QThreadPool()
threadpool.setMaxThreadCount(10)  # make sure this is less than your logical processors
print("Multithreading with maximum %d threads" % threadpool.maxThreadCount())

amount = len(timestamps)
for cnt, filename in enumerate(timestamps):
    filename += ".png"
    worker = Worker(loadAndSaveImage, filename, cnt+1, amount)
    threadpool.start(worker)

while True:
    time.sleep(5)
