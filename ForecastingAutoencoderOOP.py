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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap,QIntValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QPushButton, QHBoxLayout, QTableWidget, QMainWindow, QMessageBox,QVBoxLayout, QTabWidget, QLabel, QWidget, QTableWidgetItem,QComboBox, QLineEdit
import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(200, 200, 800, 500)
        self.setFixedSize(565, 500)
        self.setWindowTitle("Forcasting Populasi Penduduk")
        self.tab_widget = Main(self)
        self.setCentralWidget(self.tab_widget)
  
        self.show()

class Main(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.model = QtGui.QStandardItemModel(self)

        # Tabscreen
        self.tabs = QTabWidget()
        self.tabs.resize(300, 200)

        # Atributs
        self.filename = None
        self.modelpath = None
        self.autoencoder = None
        self.model = None
        self.epoch = None
        self.futureYear = None

        # Tabs
        self.tab_inputData()
        self.tab_trainingData()
        self.tab_forcsating()

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

# TAB MENU
    def tab_inputData(self):
        containerInput = QWidget()  
        containerInput.layout = QVBoxLayout()
        containerButton = QHBoxLayout()

        self.tableView = QTableWidget()
        self.tableView.horizontalHeader().hide()
        self.tableView.verticalHeader().hide()

        ButtonOpen = QPushButton("Clear")
        ButtonOpen.clicked.connect(self.clearTable)
        ButtonOpen.setFixedWidth(60)

        ButtonSecond = QPushButton("Open")
        ButtonSecond.clicked.connect(self.openFile)
        ButtonSecond.setFixedWidth(100)

        containerButton.addWidget(ButtonOpen)
        containerButton.addWidget(ButtonSecond)
        containerButton.addStretch(1)

        containerInput.layout.addWidget(self.tableView)
        containerInput.layout.addLayout(containerButton)
        containerInput.setLayout(containerInput.layout)

        self.tabs.addTab(containerInput, "Input Data")

    def tab_trainingData(self):
        containerTraining = QWidget()  
        containerTraining.layout = QVBoxLayout()
        containerButton = QHBoxLayout()

        self.tableProperties = QTableWidget()
        self.tableProperties.horizontalHeader().hide()
        self.tableProperties.verticalHeader().hide()
        self.tableProperties.setRowCount(8)
        self.tableProperties.setColumnCount(2)
        self.tableProperties.setColumnWidth(1,300)
        self.tableProperties.setItem(0,0, QTableWidgetItem('Jumlah Data'))
        self.tableProperties.setItem(0,1, QTableWidgetItem('0'))

        self.modelSelect = QComboBox()
        listModel = ['Model 1','Model 2','Model 3','Model 4']
        self.modelSelect.addItems(listModel)
        self.model = self.modelSelect.currentText()

        self.epochSelect = QComboBox()
        listModel = ['500','1000','1500','2000']
        self.epochSelect.addItems(listModel)
        self.epoch = self.epochSelect.currentText()

        ButtonClear = QPushButton("Check")
        ButtonClear.clicked.connect(self.check) 
        ButtonClear.setFixedWidth(60)

        ButtonSecond = QPushButton("Training Data")
        ButtonSecond.clicked.connect(self.trainingData)
        ButtonSecond.setFixedWidth(100)

        containerButton.addWidget(ButtonClear)
        containerButton.addWidget(ButtonSecond)
        containerButton.addWidget(self.modelSelect)
        containerButton.addWidget(self.epochSelect)
        containerButton.addStretch(1)

        containerTraining.layout.addWidget(self.tableProperties)
        containerTraining.layout.addLayout(containerButton)
        containerTraining.setLayout(containerTraining.layout)

        self.tabs.addTab(containerTraining, "Training Data")

    def tab_forcsating(self):
        containerForcasting = QWidget()  
        containerForcasting.layout = QVBoxLayout()
        containerButton = QHBoxLayout()
    
        self.tableGraphView = QTableWidget()
        self.tableGraphView.horizontalHeader().hide()
        self.tableGraphView.verticalHeader().hide()

        ButtonOpen = QPushButton("Load")
        ButtonOpen.clicked.connect(self.loadModel) 
        ButtonOpen.setFixedWidth(60)

        ButtonSecond = QPushButton("Forcasting Data")
        ButtonSecond.clicked.connect(self.forcastingData)
        ButtonSecond.setFixedWidth(100)

        self.YearPrediction = QLineEdit(self)
        validator = QIntValidator()
        self.YearPrediction.setValidator(validator)

        containerButton.addWidget(ButtonOpen)
        containerButton.addWidget(ButtonSecond)
        containerButton.addWidget(self.YearPrediction)
        containerButton.addStretch(1)

        containerForcasting.layout.addWidget(self.tableGraphView)
        containerForcasting.layout.addLayout(containerButton)
        containerForcasting.setLayout(containerForcasting.layout)

        self.tabs.addTab(containerForcasting, "Forecasting Data")

    def check(self):
        print(self.filename)

    def trainingData(self):
        if self.filename is not None:
            aa = Autoencoder()
            aa.trainingData(self.dataTrain, self.model, self.epoch)
            self.modelpath = 'default.h5'
            self.modelPerformanceCheck('default.h5')
        else:
            QMessageBox.warning(self, "Warning", "Silahkan masukkan data terlebih dahulu.")

    def forcastingData(self):
        self.futureYear = self.YearPrediction.text()
        print(self.futureYear)
        if self.modelpath is None and self.filename is None:
            QMessageBox.warning(self, "Warning", "Silahkan masukkan data dan training data terlebih dahulu")
        elif self.filename is None:
            QMessageBox.warning(self, "Warning", "Silahkan masukkan data terlebih dahulu")
        elif self.modelpath is None:
            QMessageBox.warning(self, "Warning", "Silahkan training data terlebih dahulu")
        else:
            aa = Forcasting()
            aa.forcastingData(self.modelpath, self.year, self.dataTrain, self.population, self.futureYear)
            self.graphShow()
            
    def graphShow(self):
        graphImageLabel = QLabel()
        path = 'Graph.png'
        graphImage = QPixmap(path)
        graphImageLabel.setPixmap(graphImage.scaled(500, 500, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.tableGraphView.setRowCount(1)
        self.tableGraphView.setColumnCount(1)
        self.tableGraphView.setCellWidget(0, 0, graphImageLabel)
        self.tableGraphView.resizeRowsToContents()
        self.tableGraphView.resizeColumnsToContents()

# FILE PROCESS
    def openFile(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)", options=options)
        if filename:
            self.data = pd.read_csv(filename)
            self.filename = filename
            self.splitData()
            self.createTableRawData()
            self.jumlahDataTableProperties()

# TABLE PROCESS
    def createTableRawData(self):
        tempData = pd.read_csv(self.filename, header = None)
        Row = len(tempData.axes[0])
        Column = len(tempData.axes[1])
        self.tableView.setRowCount(Row)
        self.tableView.setColumnCount(Column)
        
        array = tempData.values

        for x in range(Column):
            for y in range(Row):
                self.tableView.setItem(y,x, QTableWidgetItem(array[y,x]))

    def jumlahDataTableProperties(self):
        JumlahData = len(self.data.axes[0])

        self.tableProperties.setItem(0,0, QTableWidgetItem('Jumlah Data'))
        self.tableProperties.setItem(0,1, QTableWidgetItem(str(JumlahData)))

    def akurasiTableProperties(self,MAE,MSE,MAPE,Rscore):
        self.tableProperties.setItem(2,0, QTableWidgetItem('MSE'))
        self.tableProperties.setItem(2,1, QTableWidgetItem(str(MSE)))
        self.tableProperties.setItem(3,0, QTableWidgetItem('MAE'))
        self.tableProperties.setItem(3,1, QTableWidgetItem(str(MAE)))
        self.tableProperties.setItem(4,0, QTableWidgetItem('MAPE'))
        self.tableProperties.setItem(4,1, QTableWidgetItem(str(MAPE)))
        self.tableProperties.setItem(5,0, QTableWidgetItem('R2'))
        self.tableProperties.setItem(5,1, QTableWidgetItem(str(Rscore)))

    def clearTable(self):
        self.tableView.clearContents()
        self.tableView.setRowCount(0)
        self.tableProperties.setItem(0,1, QTableWidgetItem('0'))
        self.filename = None
        QMessageBox.warning(self, "Warning", "Data di kosongkan")

    def splitData(self):
        selectedFeatures = ['Kelahiran', 'Kematian', 'Migrasi']
        rawData = self.data[selectedFeatures]
        selectedFeatures = ['Total Populasi']
        population = self.data[selectedFeatures]
        selectedFeatures = ['Tahun']
        year = self.data[selectedFeatures]

        x_train, x_test, y_train, y_test = train_test_split(rawData, population, test_size=0.22, shuffle=False)

        # Convert to array
        self.dataTrain = np.array(x_train)
        self.dataTest = np.array(y_test)
        self.year = np.array(year)
        self.population = np.array(population)

    def modelPerformanceCheck(self, path):
        self.autoencoder = load_model(str(path))
        Prediction_population = []
        yearPredict = len(self.dataTest)
        tempPopulation = self.dataTest[0]
        tempTrainData = self.dataTrain[-1].reshape(1, -1)
        # Prediksi Populasi
        for x in range(yearPredict):
            tempTrainData = self.autoencoder.predict(tempTrainData, verbose=0)
            tempPopulation = tempTrainData[0,0] - tempTrainData[0,1] + tempTrainData[0,2] + tempPopulation
            Prediction_population.append(tempPopulation)

        Prediction_population = np.array(Prediction_population)

        # MinMaxScaling
        self.scalerPopulation = MinMaxScaler()
        self.scalerPopulation.fit_transform(self.population)
        dataTestScaler = self.scalerPopulation.transform(self.dataTest.reshape(-1,1))
        Prediction_populationScaler = self.scalerPopulation.transform(Prediction_population.reshape(-1,1))

        # Perhitungan MSE
        MSEtest = mean_squared_error(dataTestScaler, Prediction_populationScaler)
        MAEtest = mean_absolute_error(dataTestScaler, Prediction_populationScaler)
        MAPEtest = mean_absolute_percentage_error(dataTestScaler, Prediction_populationScaler)
        Rtest = r2_score(dataTestScaler, Prediction_populationScaler)
        print('ok')
        self.akurasiTableProperties(MAEtest,MSEtest,MAPEtest, Rtest)
        self.autoencoder = tf.keras.backend.clear_session()

    def loadModel(self):
        options = QFileDialog.Options()
        modelPath, _ = QFileDialog.getOpenFileName(None, "Open File", "", "All Files (*)", options=options)
        if modelPath and modelPath.endswith('.h5'):
            self.modelpath = modelPath
            self.modelPerformanceCheck(modelPath)
        elif modelPath and not modelPath.endswith('.h5'):
            QMessageBox.warning(self, "Warning", "Model dipilih tidak kompetibel")

class Autoencoder():
    def trainingData(self, dataTrain, model, epoch):
        self.selectedModel(dataTrain, model)
        scaler = MinMaxScaler()
        scaledDataTrain = scaler.fit_transform(dataTrain)
        epoch = int(epoch)
        SGDmodification = SGD(learning_rate=0.01)

        self.autoencoder.compile(optimizer=SGDmodification, loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='loss', patience=10)
        self.autoencoder.fit(scaledDataTrain, scaledDataTrain, epochs=epoch, batch_size=32, verbose=1, callbacks=[early_stopping])
        
        self.autoencoder.save('default.h5')
    
    def selectedModel(self, dataTrain, model):
        input_node = dataTrain.shape[1]
        self.autoencoder = tf.keras.backend.clear_session()
        if model == 'Model 1':
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
        elif model == 'Model 2':
            self.autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(input_node,)),
                tf.keras.layers.Dense(64, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(16, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(64, activation='linear'),
                tf.keras.layers.Dense(input_node, activation='linear')
            ])
        elif model == 'Model 3':
            self.autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(input_node,)),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(16, activation='linear'),
                tf.keras.layers.Dense(32, activation='linear'),
                tf.keras.layers.Dense(input_node, activation='linear')
            ])
        elif model == 'Model 4':
            self.autoencoder = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(input_node,)),
                tf.keras.layers.Dense(16, activation='linear'),
                tf.keras.layers.Dense(input_node, activation='linear')
            ])
        self.autoencoder.summary()

class Forcasting():
    def forcastingData(self, modelPath, year, dataTrain, population, yearPrediction):
        print(yearPrediction)
        yearPrediction = int(yearPrediction)
        self.autoencoder = load_model(str(modelPath))
        Prediction_population = []
        self.predictYears = []
        tempYear = year[0].reshape(1, -1)

        # Prediksi Tahun Dimasukkan
        currentYear = len(dataTrain)
        Prediction_population.append(population[0])
        tempPopulation = population[0]
        self.predictYears = np.append(self.predictYears, tempYear)
        for x in range(currentYear-1):
            tempTrainData = self.autoencoder.predict(dataTrain[x].reshape(1, -1), verbose=0)
            tempPopulation = tempTrainData[0,0] - tempTrainData[0,1] + tempTrainData[0,2] + tempPopulation
            Prediction_population.append(tempPopulation)

            self.predictYears = np.append(self.predictYears, tempYear + 1 + x)

        
        tempTrainData = dataTrain[-1].reshape(1, -1)
        tempYear = self.predictYears[-1].reshape(1, -1)
    
        # Prediksi Tahun Kedepan
        for x in range(yearPrediction):
            tempTrainData = self.autoencoder.predict(tempTrainData, verbose=0)
            tempPopulation = tempTrainData[0,0] - tempTrainData[0,1] + tempTrainData[0,2] + tempPopulation

            if tempPopulation < 0:
                tempPopulation = 0

            Prediction_population.append(tempPopulation)

            self.predictYears = np.append(self.predictYears, tempYear + 1 + x)

        self.Prediction_population = np.array(Prediction_population)
        self.predictYears = np.array(self.predictYears)

        self.graphForcasting(year, population)

    def graphForcasting(self, year, population):
        plt.clf()
        x_axis = year.flatten()
        y_axis = population.flatten()
        plt.plot(x_axis,y_axis, label = 'Real Data')

        x_axis = self.predictYears.flatten()
        y_axis = self.Prediction_population.flatten()
        plt.plot(x_axis, y_axis, label = 'Forecasting Data')

        plt.xlabel('Tahun') 
        plt.ylabel('Jumlah Penduduk')
        plt.legend()
        
        fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        path = 'Graph.png'
        img.save(path)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())