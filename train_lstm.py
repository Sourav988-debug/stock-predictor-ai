import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load scaled data
scaled_data = np.load("data/scaled_data.npy")

# Train-test split (80%)
split = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split]

# Create time-series sequences
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_sequences(train_data, time_step)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save model
model.save("models/lstm_model.h5")

print("✅ LSTM model trained and saved successfully!")