import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QPushButton,
    QHBoxLayout,
    QTableWidget,
    QMainWindow,
    QVBoxLayout,
    QTabWidget,
    QTableWidgetItem,
    QWidget,
    QMessageBox,
)


class DataProcessor:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.selected_features = ['Kelahiran', 'Kematian', 'Migrasi']
        self.x = self.data[self.selected_features].values
        self.scaler = MinMaxScaler()
        self.x_normalized = self.scaler.fit_transform(self.x)
        self.y = self.data['Total Populasi'].values.reshape(-1, 1)
        self.year = self.data['Tahun'].values.reshape(-1, 1)

    def get_normalized_data(self):
        return self.x_normalized, self.y


class AutoencoderModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.Dense(64, activation='linear'),
            tf.keras.layers.Dense(32, activation='linear'),
            tf.keras.layers.Dense(64, activation='linear'),
            tf.keras.layers.Dense(self.input_dim, activation='linear')
        ])
        model.compile(optimizer='sgd', loss='mean_squared_error')
        return model

    def train_model(self, x_train, x_test):
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(x_train):
            x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
            self.model.fit(x_train_fold, x_train_fold, epochs=1000, batch_size=32, verbose=2,
                           validation_data=(x_test_fold, x_test_fold))

    def save_model(self, filename):
        self.model.save(filename)

    def predict(self, data):
        return self.model.predict(data, verbose=0)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(200, 200, 800, 500)
        self.setFixedSize(565, 500)
        self.setWindowTitle("Test")
        self.tab_widget = Main(self)
        self.setCentralWidget(self.tab_widget)
        self.show()


class Main(QWidget):
    def __init__(self, parent):
        super(Main, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.data_processor = None
        self.autoencoder_model = None
        self.initUI()

    def initUI(self):
        self.tabs = QTabWidget()
        self.tab_inputData()
        self.tab_trainingData()
        self.tab_forecastingData()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def tab_inputData(self):
        # Implement tab_inputData logic here

    def tab_trainingData(self):
        # Implement tab_trainingData logic here

    def tab_forecastingData(self):
        # Implement tab_forecastingData logic here


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
