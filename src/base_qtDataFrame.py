import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from PyQt5 import QtWidgets, QtGui, QtCore, Qt, sip
from PyQt5.QtCore import QAbstractTableModel
from PyQt5.QtWidgets import QTableView, QTableWidget


class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df.copy()
        self.bolds = dict()

    def toDataFrame(self):
        return self._df.copy()

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal:
            if role == QtCore.Qt.DisplayRole:
                try:
                    return self._df.columns.tolist()[section]
                except (IndexError,):
                    return QtCore.QVariant()
            elif role == QtCore.Qt.FontRole:
                return self.bolds.get(section, QtCore.QVariant())
        elif orientation == QtCore.Qt.Vertical:
            if role == QtCore.Qt.DisplayRole:
                try:
                    # return self.df.index.tolist()
                    return self._df.index.tolist()[section]
                except (IndexError,):
                    return QtCore.QVariant()
        return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname,
                             ascending=order == QtCore.Qt.AscendingOrder,
                             inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()

    def setFont(self, section, font):
        self.bolds[section] = font
        self.headerDataChanged.emit(
            QtCore.Qt.Horizontal, 0, self.columnCount())


class csvModel(QAbstractTableModel):
    # QTableWidget.setModel() is a private method

    def __init__(self, df=pd.DataFrame()):
        QAbstractTableModel.__init__(self)
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parnet=None):
        return self._df.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return str(self._df.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self._df.columns[col]
        return None


if __name__ == "__main__":
    argvs = sys.argv
    print("Qt version:", QtCore.QT_VERSION_STR)
    print("PyQt version:", QtCore.PYQT_VERSION_STR)
    print("SIP version:", sip.SIP_VERSION_STR)

    app = QtWidgets.QApplication.instance()
    if not app:  # create QApplication if it doesnt exist
        app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QWidget()

    df = pd.DataFrame({'site_codes': ['01', '02', '03', '04'],
                       'status': ['open', 'open', 'open', 'closed'],
                       'number': [100, 200, 300, 0],
                       'Location': ['east', 'north', 'south', 'east'],
                       'data_quality': ['poor', 'moderate', 'high', 'high']},
                      index=range(1, 5))

    table = QTableView(None)
    table.setModel(PandasModel(df))
    print(table.model()._df)

    table = QTableWidget(None)
    # table.setModel(df)
    # QTableWidget.setModel() is a private method
