import math
import os
import io
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton, QHBoxLayout, QTableWidget, QMainWindow, QMessageBox,QVBoxLayout, QTabWidget, QLabel, QWidget, QTableWidgetItem,QComboBox
import sys



class PopulationForecastingApp(QMainWindow):
    def __init__(self):
        super(PopulationForecastingApp, self).__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(200, 200, 800, 500)
        self.setFixedSize(565, 500)
        self.setWindowTitle("Forcasting Populasi Penduduk")
        self.centralWidget = MainWidget(self)
        self.setCentralWidget(self.centralWidget)
        self.show()


class MainWidget(QWidget):
    def __init__(self, parent):
        super(MainWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.tabWidget = QTabWidget()

        self.inputDataTab = InputDataTab(self)
        self.trainingDataTab = TrainingDataTab(self)
        self.forecastingTab = ForecastingDataTab(self)

        self.tabWidget.addTab(self.inputDataTab, "Input Data")
        self.tabWidget.addTab(self.trainingDataTab, "Training Data")
        self.tabWidget.addTab(self.forecastingTab, "Forecasting Data")

        self.layout.addWidget(self.tabWidget)
        self.setLayout(self.layout)


class InputDataTab(QWidget):
    def __init__(self, parent):
        super(InputDataTab, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.tableView = QTableWidget()
        self.tableView.horizontalHeader().hide()
        self.tableView.verticalHeader().hide()

        self.clearButton = QPushButton("Clear")
        self.clearButton.clicked.connect(self.clearTable)

        self.openButton = QPushButton("Open")
        self.openButton.clicked.connect(self.openFile)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.openButton)
        buttonLayout.addStretch(1)

        self.layout.addWidget(self.tableView)
        self.layout.addLayout(buttonLayout)
        self.setLayout(self.layout)

    def clearTable(self):
        self.tableView.clearContents()
        self.tableView.setRowCount(0)
        QMessageBox.warning(self, "Warning", "Data di kosongkan")

    def openFile(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)", options=options)
        if filename:
            self.loadData(filename)
            TrainingDataTab(self).getAtributeFilename(filename)

    def loadData(self, filename):
        self.data = pd.read_csv(filename)
        self.createTableRawData(self.data)

    def createTableRawData(self, data):
        row, column = data.shape
        self.tableView.setRowCount(row)
        self.tableView.setColumnCount(column)

        for i in range(row):
            for j in range(column):
                item = QTableWidgetItem(str(data.iloc[i, j]))
                self.tableView.setItem(i, j, item)


class TrainingDataTab(QWidget):
    def __init__(self, parent):
        super(TrainingDataTab, self).__init__(parent)
        self.filename = None  
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.tableProperties = QTableWidget(8, 2)
        self.tableProperties.horizontalHeader().hide()
        self.tableProperties.verticalHeader().hide()
        self.tableProperties.setColumnWidth(1, 300)
        self.tableProperties.setItem(0, 0, QTableWidgetItem('Jumlah Data'))
        self.tableProperties.setItem(0, 1, QTableWidgetItem('0'))

        self.resetButton = QPushButton("Reset")
        self.resetButton.clicked.connect(self.test)

        self.trainButton = QPushButton("Training Data")
        self.trainButton.clicked.connect(self.trainingData)

        self.modelSelect = QComboBox()
        self.modelSelect.addItems(['Model 1', 'Model 2', 'Model 3', 'Model 4'])

        self.epochSelect = QComboBox()
        self.epochSelect.addItems(['500', '1000', '1500', '2000'])

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.resetButton)
        buttonLayout.addWidget(self.trainButton)
        buttonLayout.addWidget(self.modelSelect)
        buttonLayout.addWidget(self.epochSelect)
        buttonLayout.addStretch(1)

        self.layout.addWidget(self.tableProperties)
        self.layout.addLayout(buttonLayout)
        self.setLayout(self.layout)

    def resetModel(self):
        QMessageBox.warning(self, "Warning", "Model telah direset")

    def getAtributeFilename(self, filename):
        data = pd.read_csv(filename)
        self.filename = filename
        self.jumlahDataTableProperties(data)
        print(filename)

    def test(self):
        print(self.filename)

    def jumlahDataTableProperties(self,data):
        JumlahData = len(data.axes[0])
        self.tableProperties.setItem(0,0, QTableWidgetItem('Jumlah Data'))
        self.tableProperties.setItem(0,1, QTableWidgetItem(str(JumlahData)))
    
    def praProcessData(self,):
        selectedFeatures = ['Kelahiran', 'Kematian', 'Migrasi']
        rawData = self.data[selectedFeatures]
        selectedFeatures = ['Total Populasi']
        population = self.data[selectedFeatures]
        selectedFeatures = ['Tahun']
        year = self.data[selectedFeatures]

        x_train, x_test, y_train, y_test = train_test_split(rawData, population, test_size=0.22, shuffle=False)

        # Convert to array
        self.x_train = np.array(x_train)
        self.dataTest = np.array(y_test)
        self.year = np.array(year)
        self.population = np.array(population)

        self.normalisasiData()

    def normalisasiData(self,x_train):
        self.scaler = MinMaxScaler()
        self.scalerPopulation = MinMaxScaler()
        self.scalerPopulation.fit_transform(self.population)
        self.scaledTraininedData = self.scaler.fit_transform(self.x_train)

    def trainingData(self):
        if self.filename is None:
            QMessageBox.warning(self, "Warning", "Silahkan masukkan data terlebih dahulu")
        else:
            self.selectedModel()
            x = self.epochSelect.currentText()
            x = int(x)
            SGDmodification = SGD(learning_rate=0.01)
            self.autoencoder.compile(optimizer=SGDmodification, loss='mean_squared_error')

            early_stopping = EarlyStopping(monitor='loss', patience=10)
            self.autoencoder.fit(self.scaledTraininedData, self.scaledTraininedData, epochs=x, batch_size=32, verbose=1, callbacks=[early_stopping])

            self.modelPerformanceCheck()
            
            self.autoencoder.save('Model/default.h5')

    def selectedModel(self):
        x = self.modelSelect.currentText()
        input_node = self.scaledTraininedData.shape[1]
        self.autoencoder = tf.keras.backend.clear_session()
        if x == 'Model 1':
            self.autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(input_node,)),
                tf.keras.layers.Dense(128, activation='linear'),
                tf.keras.layers.Dense(64, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(16, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(64, activation='linear'),
                tf.keras.layers.Dense(128, activation='linear'),
                tf.keras.layers.Dense(input_node, activation='linear')
            ])
        elif x == 'Model 2':
            self.autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(input_node,)),
                tf.keras.layers.Dense(64, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(16, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(64, activation='linear'),
                tf.keras.layers.Dense(input_node, activation='linear')
            ])
        elif x == 'Model 3':
            self.autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(input_node,)),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(16, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(input_node, activation='linear')
            ])
        elif x == 'Model 4':
            self.autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(input_node,)),
                tf.keras.layers.Dense(16, activation='linear'),
                tf.keras.layers.Dense(input_node, activation='linear')
            ])
        self.autoencoder.summary()

    def modelPerformanceCheck(self):
        Prediction_population = []
        yearPredict = len(self.dataTest)
        tempPopulation = self.dataTest[0]
        tempTrainData = self.x_train[-1].reshape(1, -1)
        # Prediksi Populasi
        for x in range(yearPredict):
            tempTrainData = self.autoencoder.predict(tempTrainData, verbose=0)
            tempPopulation = tempTrainData[0,0] - tempTrainData[0,1] + tempTrainData[0,2] + tempPopulation
            Prediction_population.append(tempPopulation)

        Prediction_population = np.array(Prediction_population)

        # MinMaxScaling
        dataTestScaler = self.scalerPopulation.transform(self.dataTest.reshape(-1,1))
        Prediction_populationScaler = self.scalerPopulation.transform(Prediction_population.reshape(-1,1))

        # Perhitungan MSE
        MSEtest = mean_squared_error(dataTestScaler, Prediction_populationScaler)
        MAEtest = mean_absolute_error(dataTestScaler, Prediction_populationScaler)
        Rtest = r2_score(dataTestScaler, Prediction_populationScaler)
        print('ok')
        self.akurasiTableProperties(MAEtest,MSEtest,Rtest)

class ForecastingDataTab(QWidget):
    def __init__(self, parent):
        super(ForecastingDataTab, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout(self)

        self.graphLabel = QLabel()
        self.graphLabel.setAlignment(Qt.AlignCenter)

        self.loadButton = QPushButton("Load")
        self.loadButton.clicked.connect(self.loadModel)

        self.forecastingButton = QPushButton("Forecasting Data")
        self.forecastingButton.clicked.connect(self.forecastingData)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.loadButton)
        buttonLayout.addWidget(self.forecastingButton)
        buttonLayout.addStretch(1)

        self.layout.addWidget(self.graphLabel)
        self.layout.addLayout(buttonLayout)
        self.setLayout(self.layout)

    def loadModel(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)", options=options)
        if filename and filename.endswith('.h5'):
            self.autoencoder = load_model(filename)
            # Add logic to update UI or perform actions after loading the model
        elif filename and not filename.endswith('.h5'):
            QMessageBox.warning(self, "Warning", "Model dipilih tidak kompatibel")

    def forecastingData(self):
        # Add forecasting logic here
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PopulationForecastingApp()
    window.show()
    sys.exit(app.exec_())
