import sys
from PyQt5.QtWidgets import *
from gui import Ui_MainWindow as Window
import os
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import processing
import cv2
import processing
from importlib import reload  # reload(processing)
import time
import threading
import traceback, sys



# Start of the main UI class. Contains all of the GUI functions
class Ui(QMainWindow):
    currentFolder = Path()
    directories = []
    cameraModels = None
    fileList = []
    imageQueue = []

    def __init__(self):
        # GUI basic setup stuff
        super().__init__()
        self.ui = Window()
        self.ui.setupUi(self)
        self.show()

        # Multithreading setup
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Set up directory structure
        self.directories = [
            [self.ui.g_f_gps, "gps"],
            [self.ui.g_f_lms_front, "lms_front"],
            [self.ui.g_f_models, "Models"],
            [self.ui.g_f_stereo, "stereo"],
            [self.ui.g_f_left, "stereo/left"],
            [self.ui.g_f_centre, "stereo/centre"],
            [self.ui.g_f_right, "stereo/right"],
            [self.ui.g_f_lmsTime, "lms_front.timestamps"],
            [self.ui.g_f_stereoTime, "stereo.timestamps"]
        ]

        # GUI connections
        self.ui.browseButton.clicked.connect(self.browse_folder)
        self.ui.folderLine.returnPressed.connect(self.manual_line_change)

    def get_timestamps(self):
        """ get_timestamps()
        Grabs the timestamps from the current folder → stereo timestamps file and displays it

        Also grabs the camera models from the folder and assigns them to self.cameraModels

        :return: None
        """
        container = self.ui.listWidget
        container.clear()
        with open(self.currentFolder/"stereo.timestamps") as file:
            self.fileList = [i.split()[0] for i in file.readlines()]
            container.addItems([f"#{cnt:05}|{i}" for cnt, i in enumerate(self.fileList)])

        self.cameraModels = [processing.CameraModel(self.currentFolder/"Models", self.currentFolder/"stereo/left"),
                  processing.CameraModel(self.currentFolder/"Models", self.currentFolder/"stereo/centre"),
                  processing.CameraModel(self.currentFolder/"Models", self.currentFolder/"stereo/right")]

    def preload_images(self):
        """ Intended to preload all of the images in other threads to increase speed

        :return:
        """

        threads =

'''
    def preload_display(self, incoming):
        view, img, name = incoming
        print(f"Image recieved\t{name[:-10]}\t{view}")
        cv2.imwrite(name, img)
        # cv2.imshow(view, img/255)
        image = QImage(img, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        pix = QPixmap(image)
        self.ui.imageDisp.setPixmap(pix)
        qApp.processEvents()
        self.ui.progressBar.setValue(self.ui.progressBar.value() + 1)

    def preload_queue(self):
        self.ui.progressBar.setMaximum(len(self.fileList)*3)
        for cnt, img in enumerate(self.fileList):
            print(cnt)
            for channel, model in zip(["/left/", "/centre/", "/right/"], self.cameraModels):
                # Create the process image thread
                worker = Worker(processing.load_stereo, self.currentFolder/("stereo"+channel+img+".png"), model, channel, str(self.currentFolder / ("preprocessed"+channel+img+".png")))  # Any other args, kwargs are passed to the run function
                worker.signals.result.connect(self.preload_display)
                self.threadpool.start(worker)  # Start the thread

    def preload_create_queue_thread(self):
        worker = Worker(self.preload_queue())  # Create the thread
        self.threadpool.start(worker)  # Start the thread
'''
    def check_folder_structure(self):
        error = False
        for checkBox, directory in self.directories:
            if (self.currentFolder / directory).is_file() or (self.currentFolder / directory).is_dir():
                checkBox.setChecked(True)
            else:
                error = True
                checkBox.setChecked(False)

        if (self.currentFolder / "preprocessed").is_dir():
            self.ui.g_f_processed.setChecked(True)
        else:
            os.mkdir(str(self.currentFolder / "preprocessed"))
            self.ui.g_f_processed.setChecked(True)
            self.get_timestamps()  # TODO: Start image loading stream

        views = ["preprocessed/left", "preprocessed/centre", "preprocessed/right"]
        for i in views:
            if not (self.currentFolder / i).is_dir():
                os.mkdir(str(self.currentFolder/i))

        if error:
            self.ui.listWidget.setEnabled(False)
            self.ui.imageControls.setEnabled(False)
        else:
            self.ui.listWidget.setEnabled(True)
            self.ui.imageControls.setEnabled(True)
            self.ui.g_f_good.setChecked(True)
            self.get_timestamps()
            self.preload_create_queue_thread()

    def browse_folder(self):
        self.currentFolder = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        self.ui.folderLine.setText(str(self.currentFolder))
        self.check_folder_structure()

    def manual_line_change(self):
        print("Enter pressed")
        self.currentFolder = Path(self.ui.folderLine.text())
        self.check_folder_structure()


# Start of the actual main 'loop'. Creates the gui and sits there
app = QApplication(sys.argv)
w = Ui()
w.show()
sys.exit(app.exec_())
