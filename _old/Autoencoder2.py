import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Your data
data = np.array([
    [2825601, 1504880, 88615],
    [2926316, 1522031, 89835],
    [3035413, 1517167, 90599],
    [3146081, 1526178, 45573],
    [3256857, 1533493, 45843],
    [3359348, 1542415, 46214],
    [3475206, 1544495, 46805],
    [3588819, 1555754, 47713],
    [3700504, 1574678, 48938],
    [3810960, 1577821, 49945],
    [3929479, 1592968, 50818]
])

# Split the data into input and output
input_data = data[:-1]  # Use all columns except the last one for input
output_data = data[1:, 0]  # Use the first column for output

# Normalize the input data using MinMaxScaler
scaler = MinMaxScaler()
input_data = scaler.fit_transform(input_data)

# Build the sequential autoencoder model
model = Sequential([
    Dense(2, activation='relu', input_shape=(input_data.shape[1],)),
    Dense(input_data.shape[1], activation='linear')
])

# Compile the model
optimizer = SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the autoencoder
model.fit(input_data, input_data, epochs=1000, batch_size=1, verbose=2)

# Reconstruct the input data
reconstructed_data = model.predict(input_data)

# Inverse transform to denormalize the reconstructed data
reconstructed_data = scaler.inverse_transform(reconstructed_data)

# Calculate MSE for each data point
mse_values = [mean_squared_error(input_data[i], reconstructed_data[i]) for i in range(len(input_data))]

print("MSE Values:", mse_values)

# Plot the MSE graph
plt.plot(range(len(input_data)), mse_values, marker='o')
plt.xlabel("Data Point")
plt.ylabel("Mean Squared Error")
plt.title("MSE between Original and Reconstructed Data")
plt.show()

