import numpy as np
# import pandas as pd
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Load the data and preprocess it
population_data = np.loadtxt("diabetes.csv", delimiter=",")
scaled_data = (population_data - np.mean(population_data)) / np.std(population_data)

# Define the autoencoder model
inputs = tf.keras.layers.Input(shape=(scaled_data.shape[1],))
print(inputs)
encoded = tf.keras.layers.Dense(64, activation="relu")(inputs)
print(encoded)
encoded = tf.keras.layers.Dense(32, activation="relu")(encoded)
print(encoded)
decoded = tf.keras.layers.Dense(64, activation="relu")(encoded)
print(decoded)
decoded = tf.keras.layers.Dense(scaled_data.shape[1], activation="linear")(decoded)
print(decoded)
autoencoder = tf.keras.models.Model(inputs, decoded)
print(autoencoder)

print('\n')
# Compile and train the model
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(scaled_data, scaled_data, epochs=3, batch_size=32)

forecast_steps = 3
# Use the model to make forecasts
forecasts = autoencoder.predict(scaled_data[-forecast_steps:])
print(forecasts)