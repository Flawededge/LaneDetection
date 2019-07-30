import sys
from PyQt5.QtWidgets import *
from gui import Ui_MainWindow as window
import os
from PyQt5.QtWidgets import QFileDialog


# Start of the main UI class. Contains all of the GUI functions
class Ui(QMainWindow):
    currentFolder = None
    directories = []

    def __init__(self):
        super().__init__()
        self.ui = window()
        self.ui.setupUi(self)
        self.show()

        self.directories = [
            [self.ui.g_f_gps        , "/gps"],
            [self.ui.g_f_lms_front  , "/lms_front"],
            [self.ui.g_f_models     , "/Models"],
            [self.ui.g_f_stereo     , "/stereo"],
            [self.ui.g_f_left       , "/stereo/left"],
            [self.ui.g_f_centre     , "/stereo/centre"],
            [self.ui.g_f_right      , "/stereo/right"],
            [self.ui.g_f_lmsTime    , "lms_front.timestamps"],
            [self.ui.g_f_stereoTime , "stereo.timestamps"],
            [self.ui.g_f_processed, "preprocessed"]  # TODO: Add in preprocessed check
        ]

        self.ui.browseButton.clicked.connect(self.browse_folder)
        self.ui.folderLine.returnPressed.connect(self.manual_line_change)


    def check_folder_structure(self):
        if not

    def browse_folder(self):
        self.currentFolder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.ui.folderLine.setText(self.currentFolder)

    def manual_line_change(self):
        print("Enter pressed")
        self.currentFolder = self.ui.folderLine.text()


# Start of the actual main 'loop'. Creates the gui and sits there
app = QApplication(sys.argv)
w = Ui()
w.show()
sys.exit(app.exec_())
