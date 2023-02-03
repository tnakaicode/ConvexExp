import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import time
import shutil
import subprocess
import csv
import random
import argparse
from datetime import date, datetime
from linecache import getline, clearcache

basename = os.path.dirname(__file__) + "/"

from PyQt5 import Qt, QtCore, QtGui, QtWidgets, sip
from PyQt5.QtWidgets import qApp, QWidget, QApplication
from PyQt5.QtWidgets import QMenu, QAction, QFileDialog
from PyQt5.QtWidgets import QFormLayout, QHBoxLayout, QVBoxLayout, QGridLayout, QBoxLayout
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtWidgets import QListWidget, QListWidgetItem
from PyQt5.QtWidgets import QComboBox, QLabel, QLineEdit, QGroupBox, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3, gp_Trsf
from OCC.Core.gp import gp_Pln
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCCUtils.Common import vertex2pnt
from OCCUtils.Construct import make_box, make_line, make_wire, make_edge, make_face
from OCCUtils.Construct import make_plane, make_polygon
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import dir_to_vec, vec_to_dir

sys.path.append(os.path.join("./"))
from src.convex import CovExp
from src.base import plot2d, Error_Handling, create_tempnum
from src.base_qtApp import MainWindow

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)


class OCCView(CovExp):

    def __init__(self, touch=False, file=False):
        CovExp.__init__(self, touch=False)


class MainWidget(QtWidgets.QWidget, plot2d):

    def __init__(self, parent=None):
        plot2d.__init__(self)
        QtWidgets.QWidget.__init__(self, parent)
        self.layout = QHBoxLayout(self)

        self.file = ""
        self.text_size = [100, 25]
        self.edit_size = [75, 25]

        self.view = OCCView(touch=True)
        self.display = self.view.display

        # Date Title
        self.titl_labl = QLabel('Title')
        self.titl_edit = QLineEdit()
        self.titl_edit.setText(self.date_text + " " + self.rootname)
        self.titl_edit.setMinimumWidth(200)

        # Split Button
        self.splt_butt = QPushButton('Split Box', self)
        self.splt_butt.clicked.connect(lambda: self.calc_split())

        self.splt_snum = QLineEdit('3', self)
        self.splt_seed = QLineEdit('11', self)

        # Erase Button
        self.eras_butt = QPushButton('Erase All', self)
        self.eras_butt.clicked.connect(self.display.EraseAll)

        # Touch Button
        self.toch_butt = QPushButton('Touch', self)
        self.toch_butt.clicked.connect(lambda: self.view.set_touch())

        # Save Screen Button
        self.scrn_butt = QPushButton('Save Screen', self)
        self.scrn_butt.clicked.connect(self.save_screen)

        # Text size
        self.splt_butt.setFixedSize(*self.text_size)
        self.splt_snum.setFixedSize(*self.text_size)
        self.splt_seed.setFixedSize(*self.text_size)
        self.toch_butt.setFixedSize(*self.text_size)
        self.eras_butt.setFixedSize(*self.text_size)
        self.scrn_butt.setFixedSize(*self.text_size)

        self.create_group1()
        self.create_group2()

    def create_group1(self):
        self.topLeftGroupBox = QGroupBox("Group 1")
        layout = QGridLayout()
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(10)

        n = 0
        layout.addWidget(self.titl_labl, n, 0)
        layout.addWidget(self.titl_edit, n, 1)

        n += 1
        layout.addWidget(self.splt_butt, n, 1)

        n += 1
        layout.addWidget(self.splt_snum, n, 1)
        layout.addWidget(self.splt_seed, n, 2)

        n += 1
        layout.addWidget(self.toch_butt, n, 1)

        n += 1
        layout.addWidget(self.eras_butt, n, 1)
        layout.addWidget(self.scrn_butt, n, 2)

        self.topLeftGroupBox.setFixedWidth(layout.columnCount() * 95)
        self.topLeftGroupBox.setFixedHeight((n + 1) * 37)
        self.topLeftGroupBox.setLayout(layout)
        self.layout.addWidget(self.topLeftGroupBox, 0)

    def create_group2(self):
        self.topRightGroupBox = QGroupBox(self.view.windowTitle())
        self.topRightGroupBox.setMinimumWidth(550)

        layout = QGridLayout(self)
        layout.addWidget(self.view)

        self.topRightGroupBox.setLayout(layout)
        self.layout.addWidget(self.topRightGroupBox, 1)

    def calc_split(self):
        self.display.EraseAll()

        num = self.text2int(self.splt_snum.text(), 1)
        seed = self.text2int(self.splt_seed.text(), None)

        self.view.init_base(self.view.base)
        self.view.split_run(num, seed)
        sol_exp = TopExp_Explorer(self.view.splitter.Shape(), TopAbs_SOLID)
        sol_exp.Next()

        self.view.prop_soild(sol_exp.Current())
        self.display.DisplayShape(sol_exp.Current(), transparency=0.5)
        self.display.DisplayShape(self.view.base, transparency=0.8)
        self.display.FitAll()

    def save_screen(self):
        # call from MainWindow => disappear QTimer
        # call from MainWidget Class (link to PushBotton)
        screen = QApplication.primaryScreen()
        print()
        print(screen.geometry())
        print(QApplication.topLevelWidgets()[0].geometry())
        print(QApplication.topLevelWindows()[0].geometry())
        res = QApplication.topLevelWindows()[0].geometry()
        screenshot = screen.grabWindow(QApplication.desktop().winId(),
                                       res.x(), res.y(), res.width(), res.height())
        screenshot.save(create_tempnum(self.tempname, "", ".png"),
                        'png', quality=100)


class App (MainWindow):

    def __init__(self, *args):

        # Don't create a new QApplication,
        # it would unhook the Events set by Traits on the existing QApplication.
        # Simply use the '.instance()' method to retrieve the existing one.
        self.app = QtWidgets.QApplication.instance()
        if not self.app:  # create QApplication if it doesnt exist
            self.app = QtWidgets.QApplication(sys.argv)
        MainWindow.__init__(self)
        plt.close("all")
        self.set_canvas(MainWidget(), "Qt")
        self.set_settingfile("./temp_setting/" + self.canva.rootname + ".ini")

        self.add_menu("File")
        self.add_function_to_menu("File", self.openAct)
        self.add_function_to_menu("File", self.exitAct)
        self.add_function_to_menu("File", self.get_settigfile)
        self.add_function_to_menu("File", self.plot_close)
        self.add_function_to_menu("File", self.canva.save_screen)
        self.add_function_to_menu("File", self.canva.open_tempdir)
        self.add_function_to_menu("File", self.canva.open_newtempdir)
        plt.close("all")

    def plot_close(self):
        plt.close("all")

    def start_app(self):
        self.show()
        self.raise_()
        sys.exit(self.app.exec_())


if __name__ == '__main__':
    argvs = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", dest="dir", default="./")
    parser.add_argument("--pxyz", dest="pxyz",
                        default=[0.0, 0.0, 0.0], type=float, nargs=3)
    opt = parser.parse_args()
    print(opt)

    print("Qt version:", QtCore.QT_VERSION_STR)
    print("PyQt version:", QtCore.PYQT_VERSION_STR)
    print("SIP version:", sip.SIP_VERSION_STR)

    obj = App()
    obj.start_app()
