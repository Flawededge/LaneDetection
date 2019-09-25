from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
import traceback
import sys

import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt
from scipy import signal
from collections import deque

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Global figure for intra frame pyplot window use
plt.ion()
fig = plt.figure()

rollingLeft = deque(maxlen=20)
rollingRight = deque(maxlen=20)


def reliable_lane_markings(image, ipm_points, progress_display=False, ipm_size=200):
    global line1
    """ Applies the reliable_lane_markings process

    :param image: The input image
    :param ipm_points:
    :param progress_display:
    :param ipm_size:
    :return:
    """

    # Make the output perspective points
    ipm_output = np.float32([  # Setup the output transform
        (0, 0),  # Top left
        (ipm_size, 0),  # Top right
        (ipm_size, ipm_size),  # Bottom right
        (0, ipm_size)  # Bottom left
    ])

    # Perspective transform
    M = cv.getPerspectiveTransform(ipm_points, ipm_output)  # Get transform
    perspective = cv.warpPerspective(image, M, (int(ipm_size * 1), int(ipm_size * 1)))  # Apply it
    perspective = cv.bilateralFilter(perspective, 11, 180, 120)

    # Get the lane lines
    gray_perspective = cv.cvtColor(perspective, cv.COLOR_BGR2HSV)[:, :, 2]
    gray_perspective = cv.adaptiveThreshold(gray_perspective, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, -1)
    gray_perspective = cv.dilate(gray_perspective, (7, 7))
    gray_perspective = cv.bilateralFilter(gray_perspective, 15, 100, 100)

    thresh_perspective = cv.medianBlur(gray_perspective, 7)
    thresh_perspective = cv.dilate(thresh_perspective, (3, 3))
    thresh_perspective = cv.morphologyEx(thresh_perspective, cv.MORPH_CLOSE, (7, 7), iterations=20)

    # Calculate the weight of each column
    weights = np.array([np.average(i) for i in np.transpose(thresh_perspective)])

    if progress_display:  # For pyplot display
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(weights)
        # plt.show()
        fig.canvas.draw()
        # fig.canvas.flush_events()

        # Get the index of the peaks from the graph
    peaks, _ = signal.find_peaks(weights)

    # Append the new average to the average
    if len((weights[peaks[peaks < ipm_size/2]])) and len((weights[peaks[peaks > ipm_size/2]])):
        rollingLeft.append(peaks[np.argmax(weights[peaks[peaks < ipm_size/2]])])
        rollingRight.append(peaks[np.argmax(weights[peaks[peaks > ipm_size/2]]) + len(peaks[peaks < ipm_size/2])])

    maxLeft = int(np.average(rollingLeft))  # Get the new average
    maxRight = int(np.average(rollingRight))

    # Draw the lines to show
    cv.line(perspective, (maxLeft, 0), (maxLeft, int(ipm_size)), (255, 0, 0), 1)
    cv.line(perspective, (maxRight, 0), (maxRight, int(ipm_size)), (255, 0, 0), 1)

    # # Inverse perspective transform
    # Make the mask to be inverted
    mask_perspective = np.zeros_like(perspective)
    cv.rectangle(mask_perspective, (maxLeft, 0), (maxRight, int(ipm_size)), (0, 0, 255), -1)
    cv.line(mask_perspective, (maxLeft, 0), (maxLeft, int(ipm_size)), (255, 0, 0), 5)
    cv.line(mask_perspective, (maxRight, 0), (maxRight, int(ipm_size)), (255, 0, 0), 5)

    # Do the transform
    M = cv.getPerspectiveTransform(ipm_output, ipm_points)
    output_mask = cv.warpPerspective(mask_perspective, M, (image.shape[1], image.shape[0]))  # Apply it
    image = cv.addWeighted(image, 1, output_mask, 0.5, 0)

    if progress_display:
        cv.imshow("Reliable: Perspective", perspective)
        cv.imshow("Reliable: PerspecGray", gray_perspective)
        cv.imshow("Reliable: PerspThresh", thresh_perspective)
        cv.imshow("Reliable: PerspecMask", mask_perspective)
        cv.imshow("Reliable:  outputMask", output_mask)

    cv.line(image, tuple(ipm_points[0]), tuple(ipm_points[1]), (191, 66, 245), 2)
    cv.line(image, tuple(ipm_points[1]), tuple(ipm_points[2]), (191, 66, 245), 2)
    cv.line(image, tuple(ipm_points[2]), tuple(ipm_points[3]), (191, 66, 245), 2)
    cv.line(image, tuple(ipm_points[3]), tuple(ipm_points[0]), (191, 66, 245), 2)
    return image


def sdc_lane_detection(image, roi_clip, progress_display=False):
    img_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    new_image = image.copy()
    new_image = cv.GaussianBlur(new_image, (5, 5), 0)
    mask_white = cv.inRange(image[:, :, :], (110, 110, 110), (180, 180, 180))

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
    # line_image = cv.hough(mask_white, rho, theta, threshold, min_line_len, max_line_gap, image)
    # result = processes.weighted_img(canny_edges, image, α=0.8, β=1., λ=0.)

    if progress_display:
        pass

    return image


def neural_lane(image, progress_display=False):
    return image


class WorkerSignals(QObject):
    '''
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

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

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


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.counter = 0

        layout = QVBoxLayout()

        self.l = QLabel("Start")
        b = QPushButton("DANGER!")
        b.pressed.connect(self.oh_no)

        layout.addWidget(self.l)
        layout.addWidget(b)

        w = QWidget()
        w.setLayout(layout)

        self.setCentralWidget(w)

        self.show()

        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def progress_fn(self, n):
        print("%d%% done" % n)

    def execute_this_fn(self, progress_callback):
        for n in range(0, 5):
            time.sleep(1)
            progress_callback.emit(n * 100 / 4)

        return "Done."

    def print_output(self, s):
        print(s)

    def thread_complete(self):
        print("THREAD COMPLETE!")

    def oh_no(self):
        # Pass the function to execute
        worker = Worker(self.execute_this_fn)  # Any other args, kwargs are passed to the run function
        worker.signals.result.connect(self.print_output)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)

    def recurring_timer(self):
        self.counter += 1
        self.l.setText("Counter: %d" % self.counter)
