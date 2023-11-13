import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate some example time series data
np.random.seed(0)
n_samples = 500
time = np.linspace(0, 8, n_samples)
y = np.sin(time)
y_noisy = y + 0.3*np.random.normal(size=n_samples)

# Apply some simple feature extraction (taking the last 10 points as features)
n_features = 10
X = np.zeros((n_samples - n_features, n_features))
for i in range(n_samples - n_features):
    X[i] = y_noisy[i:i+n_features]

# Use MinMaxScaler to scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply RBM for feature extraction
rbm = BernoulliRBM(n_components=2)
X_transformed = rbm.fit_transform(X_scaled)

# Reshape the data for LSTM
X_transformed = X_transformed.reshape((X_transformed.shape[0], X_transformed.shape[1], 1))

# Split into train and test sets
n_train = 400
X_train, X_test = X_transformed[:n_train], X_transformed[n_train:]
y_train, y_test = y[n_features:n_train+n_features], y[n_train+n_features:]

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_transformed.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X_train, y_train, epochs=200, verbose=2)

# Predict
y_pred = model.predict(X_test)

# Plot
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()