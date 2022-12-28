import sys
import os
import datetime
import csv
from linecache import getline, clearcache
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from PyQt5.QtCore import QTimer, QObject, QStringListModel, QSettings
from PyQt5.QtWidgets import QMenu, QAction, qApp, QFileDialog, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QPushButton
from PyQt5.QtWidgets import QComboBox


def create_dir(name="temp_setting"):
    os.makedirs(name, exist_ok=True)
    if os.path.isdir(name):
        os.makedirs(name, exist_ok=True)
        fp = open(name + "not_ignore.txt", "w")
        fp.close()
        print("make {}".format(name))
    else:
        print("already exist {}".format(name))
    return name


def check_callable(_callable):
    if not callable(_callable):
        raise AssertionError("The function supplied is not callable")


class qtBaseViewer(QtWidgets.QWidget):
    ''' The base Qt Widget
    '''

    def __init__(self, parent=None):
        super(qtBaseViewer, self).__init__(parent)

        # enable Mouse Tracking
        self.setMouseTracking(True)

        # Strong focus
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        self.setAttribute(QtCore.Qt.WA_NativeWindow)
        self.setAttribute(QtCore.Qt.WA_PaintOnScreen)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)

        self.setAutoFillBackground(False)
        self.createCursors()

    def paintEngine(self):
        return None

    @property
    def qApp(self):
        # reference to QApplication instance
        return self._qApp

    @qApp.setter
    def qApp(self, value):
        self._qApp = value

    def createCursors(self):
        module_pth = os.path.abspath(os.path.dirname(__file__))
        icon_pth = os.path.join(module_pth, "icons")

        _CURSOR_PIX_ROT = QtGui.QPixmap(
            os.path.join(icon_pth, "cursor-rotate.png"))
        _CURSOR_PIX_PAN = QtGui.QPixmap(
            os.path.join(icon_pth, "cursor-pan.png"))
        _CURSOR_PIX_ZOOM = QtGui.QPixmap(
            os.path.join(icon_pth, "cursor-magnify.png"))
        _CURSOR_PIX_ZOOM_AREA = QtGui.QPixmap(
            os.path.join(icon_pth, "cursor-magnify-area.png"))

        self._available_cursors = {
            "arrow": QtGui.QCursor(QtCore.Qt.ArrowCursor),  # default
            "pan": QtGui.QCursor(_CURSOR_PIX_PAN),
            "rotate": QtGui.QCursor(_CURSOR_PIX_ROT),
            "zoom": QtGui.QCursor(_CURSOR_PIX_ZOOM),
            "zoom-area": QtGui.QCursor(_CURSOR_PIX_ZOOM_AREA),
        }

        self._current_cursor = "arrow"


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args):
        QtWidgets.QMainWindow.__init__(self, *args)
        self.set_canvas(qtBaseViewer(self), "Qt")

        self.menu_bar = self.menuBar()
        self._menus = {}
        self._menu_methods = {}

        # place the window in the center of the screen,
        # at half the screen size
        self.centerOnScreen()
        self.setting = None

        self.openAct = QAction("&Open...", self, shortcut="Ctrl+O",
                               triggered=self.open)
        self.exitAct = QAction("E&xit", self, shortcut="Ctrl+Q",
                               triggered=self.close)
        self.aboutQtAct = QAction("About &Qt", self,
                                  triggered=qApp.aboutQt)

        # self.createMenus()

        # show time
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.time_draw)
        self.timer.start(500)  # msec

    def set_canvas(self, wi, name="Qt"):
        self.canva = wi
        self.setWindowTitle(name)
        self.setCentralWidget(self.canva)

        self.layout = QHBoxLayout(self.canva)

    def set_settingfile(self, filename):
        self.setting = QSettings(filename, QSettings.IniFormat)
        self.setting.setFallbacksEnabled(False)
        self.move(self.setting.value("pos", self.pos()))
        self.resize(self.setting.value("size", self.size()))
        font = self.font()
        font.setPointSize(self.setting.value("font", 9, int))
        self.setFont(font)

    def get_settigfile(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', "./temp_setting/",
                                                  'Ini (*.ini)', options=options)
        print(fileName)
        if fileName != "":
            self.set_settingfile(fileName)

    def time_draw(self):
        d = datetime.datetime.today()
        daystr = d.strftime("%Y-%m-%d %H:%M:%S")
        self.statusBar().showMessage(daystr)

    def centerOnScreen(self, ratio=[5, 2]):
        '''Centers the window on the screen.'''
        resolution = QtWidgets.QApplication.desktop().screenGeometry()
        x, y = ratio
        self.wx = resolution.width()
        self.wy = resolution.height()
        self.move(int(self.wx / x), int(self.wy / x))
        self.resize(int(self.wx / y), int(self.wy / y))

    def add_menu(self, menu_name):
        _menu = self.menu_bar.addMenu("&" + menu_name)
        self._menus[menu_name] = _menu

    def add_function_to_menu(self, menu_name, _callable):
        # check_callable(_callable)
        if callable(_callable):
            _action = QtWidgets.QAction(
                _callable.__name__.replace('_', ' ').lower(), self)
            # if not, the "exit" action is now shown...
            _action.setMenuRole(QtWidgets.QAction.NoRole)
            _action.triggered.connect(_callable)
            self._menus[menu_name].addAction(_action)
        elif isinstance(_callable, QAction):
            _action = _callable
            # if not, the "exit" action is now shown...
            _action.setMenuRole(QtWidgets.QAction.NoRole)
            self._menus[menu_name].addAction(_action)
        else:
            raise ValueError('the menu item %s does not exist' % menu_name)

    def createMenus(self):

        self.add_menu("File")
        self.add_function_to_menu("File", self.openAct)
        self.add_function_to_menu("File", self.exitAct)

        self.viewMenu = QMenu("&View", self)

        self.helpMenu = QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def open(self):
        options = QFileDialog.Options()
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        print(fileName)

    def closeEvent(self, e):
        if self.setting != None:
            # Write window size and position to config file
            self.setting.setValue("size", self.size())
            self.setting.setValue("pos", self.pos())
            self.setting.setValue("font", self.font().pointSize())

        e.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication.instance()
    if not app:  # create QApplication if it doesnt exist
        app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    def app_print():
        print("ok")

    win.add_menu("Menu")
    win.add_function_to_menu("Menu", app_print)
    win.add_function_to_menu("Menu", win.exitAct)
    win.add_menu("Help")
    win.add_function_to_menu("Help", win.aboutQtAct)
    win.raise_()
    sys.exit(app.exec_())
