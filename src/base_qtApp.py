import sys
import os
import datetime
import csv
from linecache import getline, clearcache
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from PyQt5.QtCore import QTimer, QObject, QStringListModel
from PyQt5.QtWidgets import QMenu, QAction, qApp, QFileDialog, QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QPushButton
from PyQt5.QtWidgets import QComboBox


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
        self.canva = qtBaseViewer(self)
        self.setWindowTitle("Qt")
        self.setCentralWidget(self.canva)

        self.layout = QtWidgets.QVBoxLayout(self.canva)

        self.menu_bar = self.menuBar()
        self._menus = {}
        self._menu_methods = {}

        # place the window in the center of the screen, at half the
        # screen size
        self.centerOnScreen()

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

    def time_draw(self):
        d = datetime.datetime.today()
        daystr = d.strftime("%Y-%m-%d %H:%M:%S")
        self.statusBar().showMessage(daystr)

    def centerOnScreen(self):
        '''Centers the window on the screen.'''
        resolution = QtWidgets.QApplication.desktop().screenGeometry()
        self.wx = resolution.width()
        self.wy = resolution.height()
        self.move(int(self.wx / 5), int(self.wy / 5))
        self.resize(int(self.wx / 2), int(self.wy / 2))

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
        # fileName, _ = QFileDialog.getSaveFileName(self, 'QFileDialog.getOpenFileName()', '',
        #                                          'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)
        print(fileName)


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
