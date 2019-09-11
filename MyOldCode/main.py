from gui import Ui_MainWindow as Window
from pathlib import Path
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
# from testCode import processing
import sys
import os
from tkinter.filedialog import askdirectory
from tkinter import Tk
from myLib import imageLoading, qtThreading


# Start of the main UI class. Contains all of the GUI functions
class Ui(QMainWindow):
    currentFolder = Path()
    directories = []
    cameraModels = None
    fileList = []
    imageQueue = []
    fileDialog = False

    def __init__(self):
        # GUI basic setup stuff
        super().__init__()
        self.ui = Window()
        self.ui.setupUi(self)
        self.show()

        # Multithreading setup
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(int(self.threadpool.maxThreadCount()*0.75))
        print("Multithreading with maximum %d threads" % int(self.threadpool.maxThreadCount()*0.75))

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

        self.cameraModels = [imageLoading.CameraModel(self.currentFolder / "Models", self.currentFolder / "stereo/left"),
                             imageLoading.CameraModel(self.currentFolder / "Models", self.currentFolder / "stereo/centre"),
                             imageLoading.CameraModel(self.currentFolder / "Models", self.currentFolder / "stereo/right")]

    def update_progress(self):
        self.ui.progressBar.setValue(self.ui.progressBar.value()+1)

    def display_image(self, image):  # Accepts cv2 images only
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        scene = QGraphicsScene()
        scene.addPixmap(QPixmap(image))

        self.ui.imageView.setScene(scene)
        self.ui.imageView.show()

    def start_preload(self):
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setMaximum(len(self.fileList))
        worker = qtThreading.Worker(imageLoading.preload_sequencer, self.currentFolder/"stereo",
                                    self.currentFolder/"preprocessed", self.fileList, self.cameraModels,
                                    self.threadpool, self.update_progress)
        self.threadpool.start(worker)

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
            self.start_preload()

    def browse_folder(self):
        if self.fileDialog:
            return

        self.fileDialog = True
        Tk().withdraw()
        tmp = Path(askdirectory())
        self.fileDialog = False

        if str(tmp) == '.':  # Check if the file dialog was closed
            return

        self.currentFolder = tmp
        self.ui.folderLine.setText(str(self.currentFolder))
        self.check_folder_structure()

    def manual_line_change(self):
        self.currentFolder = Path(self.ui.folderLine.text())
        self.check_folder_structure()


# Start of the actual main 'loop'. Creates the gui and sits there
app = QApplication(sys.argv)
w = Ui()
w.show()
sys.exit(app.exec_())
