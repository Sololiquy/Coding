import math
import os
import io
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication,QGroupBox, QRadioButton, QFileDialog, QPushButton, QHBoxLayout, QTableWidget, QMainWindow, QMessageBox,QVBoxLayout, QTabWidget, QLabel, QWidget, QTableWidgetItem,QComboBox
import sys

class TabWidget(QDialog):
    def __init__(self, data):
        super(TabWidget, self).__init__()
        self.data = data
        self.firstTab = FirstTab(self.data)
        self.showMaximized()

        # create Header 
        FilterLayout = QHBoxLayout()
        FilterLayout.addWidget(self.createHeader1a(), 1) 
        FilterLayout.addWidget(self.createHeader2a(), 4)

        # create Tab
        tabwidget = QTabWidget()
        tabwidget.addTab(self.firstTab, "Tab 1")

        vbox = QVBoxLayout()
        vbox.addLayout(FilterLayout)
        vbox.addWidget(tabwidget)

        self.setLayout(vbox)

    def createHeader1a(self):  #Import 
        HeaderBox = QGroupBox("Import Data")

        inputfilebtn = QPushButton("Import")
        inputfilebtn.clicked.connect(self.on_pushButtonLoad_clicked)

        # importrow1
        importrow1layout = QHBoxLayout()
        importrow1layout.addWidget(inputfilebtn)

        HeaderLayout = QVBoxLayout()
        HeaderLayout.addLayout(importrow1layout)
        HeaderBox.setLayout(HeaderLayout)
        HeaderBox.setFlat(True)

        return HeaderBox

    def createHeader2a(self): #Filter

        HeaderBox = QGroupBox("Filter Data")

        rightlayout = QHBoxLayout()
        # range slider bar to filter column data for plotting
        label4 = QLabel(self)
        label4.setText("Filter range:")
        rightlayout.addWidget(label4)
        self.slider1 = QLabeledRangeSlider(Qt.Horizontal)
        self.slider1.setRange(5, 500)
        self.slider1.setValue((150, 300))
        rightlayout.addWidget(self.slider1)

        HeaderBox.setLayout(rightlayout)
        HeaderBox.setFlat(True)  #

        return HeaderBox

    #import and return file
    def getfile(self):
        option = QFileDialog.Options()
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "CSV files (*.csv)", options=option)
        return pd.read_csv(fname[0])

    @QtCore.pyqtSlot()
    def on_pushButtonLoad_clicked(self):
        importedfile = self.getfile()
        if importedfile is None:
            return
        self.firstTab.MRChart(importedfile)

        global database
        database = importedfile

    def getslider1value(self):
        return self.slider1.value


class FirstTab(QWidget):
    def __init__(self, data):
        super(FirstTab, self).__init__() 
        self.data = data
        self.tabwidget = TabWidget(self.data)# issue here. Attempting to # access TabWidget class

        # Grid layout of entire tab
        layout = QGridLayout()
        layout.addWidget(self.infrastructure(self.data), 3, 0)
        layout.setRowStretch(4, 3)
        layout.setColumnStretch(0, 1)
        self.setLayout(layout)

    def MRChart(self, importedfile):  # pie chart
        if self.radioButton1.isChecked():
            fig = go.Pie(labels=importedfile[self.radioButton1.label])
        elif self.radioButton2.isChecked():
            fig = go.Pie(labels=importedfile[self.radioButton2.label])

        layout = go.Layout(autosize=True, legend=dict(orientation="h", xanchor='center', x=0.5))
        fig = go.Figure(data=fig, layout=layout)
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        self.browser.setHtml(fig.to_html(include_plotlyjs='cdn'))

    def infrastructure(self, importedfile):
        groupBox = QGroupBox("Plot")
        self.browser = QtWebEngineWidgets.QWebEngineView(self)

        right = QVBoxLayout()

        # Change/update plot (MRChart) depending on what Radio button is selected
        self.radioButton1 = QRadioButton("Label 1")
        self.radioButton1.label = "Column_label_1"
        self.radioButton1.toggled.connect(lambda: self.MRChart(database))
        right.addWidget(self.radioButton1)

        self.radioButton2 = QRadioButton("Label 2")
        self.radioButton2.setChecked(True)
        self.radioButton2.label = "Column_label_2"
        self.radioButton2.toggled.connect(lambda: self.MRChart(database))
        right.addWidget(self.radioButton2)

        middleright = QHBoxLayout()
        middleright.addWidget(self.browser)
        middleright.addLayout(right)
        groupBox.setLayout(middleright)
        groupBox.setFlat(True)

        print(self.tabwidget.getslider1value())# attempting to print slider value here

        return groupBox

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tabwidget = TabWidget(data=None)
    tabwidget.show()
    app.exec()
