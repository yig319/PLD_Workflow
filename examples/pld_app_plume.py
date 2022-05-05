import shutil
import sys
import json
import datetime
import os, glob, h5py
import numpy as np
import matplotlib.pyplot as plt
from datafed.CommandLib import API

from PyQt5.QtGui import QFont 
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import * 
sys.path.append('C:/Image/PLD_Workflow/')
from pld_functions import PLD_Form


if __name__ == "__main__":
    app = QApplication(sys.argv)
    custom_font = QFont("Times", 10)

    app.setFont(custom_font, "QGroupBox")
    app.setFont(custom_font, "QComboBox")
    app.setFont(custom_font, "QLabel")
    app.setFont(custom_font, "QLineEdit")
    app.setFont(custom_font, "QPlainTextEdit")

    window = PLD_Form(version="plume")
    window.show()
    app.exec_()