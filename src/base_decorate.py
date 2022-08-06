import numpy as np
import sys
import traceback
from PyQt5 import QtCore
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QApplication, QWidget


def Error_Handling(func):
    def Try_Function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except TypeError as e:
            print(f"{func.__name__} wrong data types. ", e)
        except Exception as e:
            print(f"{func.__name__} has any Error. ", e)
            print(traceback.format_exc())
    return Try_Function


@Error_Handling
def Mean(a, b):
    print((a + b) / 2)


@Error_Handling
def Square(sq):
    print(sq * sq)


@Error_Handling
def Divide(l, b):
    print(b / l)


@Error_Handling
def CallPi():
    print(np.pi)


Mean(4, 5)  # 4.5
Square(14)  # 196
Divide(8, 4)  # 0.5
Square("three")
# Square wrong data types.  can't multiply sequence by non-int of type 'str'

Divide("two", "one")
# Divide wrong data types.  unsupported operand type(s) for /: 'str' and 'str'

Mean("six", "five")
# Mean wrong data types.  unsupported operand type(s) for /: 'str' and 'int'

Mean("six")
# Mean wrong data types.  Mean() missing 1 required positional argument: 'b'

CallPi()
# CallPi has any Error.  name 'pi' is not defined


class Window(QWidget):
    def __init__(self):
        super().__init__()
        b1 = QPushButton('1-f1')
        b1.clicked.connect(self.f1)
        b2 = QPushButton('2-f2')
        b2.clicked.connect(self.f2)
        b3 = QPushButton('3-f1(lambda)')
        b3.clicked.connect(lambda: self.f1())
        b4 = QPushButton('4-f2(lambda)')
        b4.clicked.connect(lambda: self.f2())
        b5 = QPushButton('5-f3')
        b5.clicked.connect(self.f3)
        layout = QVBoxLayout(self)
        layout.addWidget(b1)
        layout.addWidget(b2)
        layout.addWidget(b3)
        layout.addWidget(b4)
        layout.addWidget(b5)
        self.setLayout(layout)

    @QtCore.pyqtSlot()
    @Error_Handling
    def f1(self):
        print(self.sender().text())
        return self.name

    @QtCore.pyqtSlot()
    @Error_Handling
    def f2(self):
        print(self.sender().text())
        return self.name

    @QtCore.pyqtSlot()
    def f3(self):
        button = self.sender()
        print(button.text())
        print("f3")

app = QApplication([])
window = Window()
window.show()
window.raise_()
sys.exit(app.exec_())
