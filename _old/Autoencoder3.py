import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
SGD = tf.keras.optimizers.SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Dataset
filedataset = pd.read_csv("Data\DataPenduduk.csv")
dataset = filedataset.values[1:,1:5]

# Praproses data
scaler = MinMaxScaler()
dataNormalisasi = scaler.fit_transform(dataset)
x = dataNormalisasi[:,:3]
y = dataNormalisasi[:,3:4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model Autoencoder
input_node = x.shape[1]
autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_node,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(input_node, activation='linear')
])

# Training data
autoencoder.compile(optimizer=SGD(learning_rate=0.1), loss='mean_squared_error')
history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=1, verbose=2, validation_data=(x_test, x_test))

autoencoder.save('Model/test.h5')
# test = tf.keras.models.load_model('test.h5')

plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss')
plt.show()

# Load the trained autoencoder model
Loadautoencoder = tf.keras.models.load_model('Model/test.h5')

# Forecasting using the autoencoder
forecasted_data = Loadautoencoder.predict(x_test)

# Inverse transform the forecasted data to the original scale
forecasted_data_original_scale = scaler.inverse_transform(forecasted_data)

# Print the forecasted data
print("Forecasted Population Data:")
print(forecasted_data_original_scale)