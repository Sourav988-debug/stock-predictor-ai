import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import math

# ----------------------------
# Load clean data
# ----------------------------
df = pd.read_csv("data/clean_data.csv", index_col=0)
close_prices = df['Close'].values

# ----------------------------
# Linear Regression Evaluation
# ----------------------------
with open("models/linear_model.pkl", "rb") as f:
    linear_model = pickle.load(f)

X = np.arange(len(df)).reshape(-1, 1)
split = int(len(df) * 0.8)

X_test = X[split:]
y_test = close_prices[split:]

linear_predictions = linear_model.predict(X_test)
linear_rmse = math.sqrt(mean_squared_error(y_test, linear_predictions))

# ----------------------------
# LSTM Evaluation
# ----------------------------
scaled_data = np.load("data/scaled_data.npy")

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[['Close']])

test_data = scaled_data[split - 60:]

def create_sequences(data, step=60):
    X, y = [], []
    for i in range(step, len(data)):
        X.append(data[i-step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_test_lstm, y_test_lstm = create_sequences(test_data)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

lstm_model = load_model("models/lstm_model.h5")
lstm_predictions = lstm_model.predict(X_test_lstm)

# Inverse scale LSTM predictions
lstm_predictions = scaler.inverse_transform(lstm_predictions)
y_test_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

lstm_rmse = math.sqrt(mean_squared_error(y_test_lstm, lstm_predictions))

# ----------------------------
# Results
# ----------------------------
print("\n📊 MODEL COMPARISON RESULTS")
print("-" * 40)
print(f"Linear Regression RMSE : {linear_rmse:.4f}")
print(f"LSTM RMSE              : {lstm_rmse:.4f}")

if lstm_rmse < linear_rmse:
    print("✅ LSTM performs better than Linear Regression")
else:
    print("⚠️ Linear Regression performs better (unexpected)")