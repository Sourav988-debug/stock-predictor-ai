import os
from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------
# Create Flask app FIRST
# -----------------------
app = Flask(__name__)

# -----------------------
# Load data
# -----------------------
df = pd.read_csv("data/clean_data.csv", index_col=0)
close_prices = df['Close'].values

# -----------------------
# Load models
# -----------------------
with open("models/linear_model.pkl", "rb") as f:
    linear_model = pickle.load(f)

lstm_model = load_model("models/lstm_model.h5")

# -----------------------
# Scaler (for LSTM)
# -----------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df[['Close']])

# -----------------------
# ROUTES
# -----------------------

@app.route("/")
def home():
    return jsonify({"status": "Stock Prediction API is running"})

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/history")
def history():
    return jsonify({
        "dates": df.index.astype(str).tolist(),
        "prices": close_prices.tolist()
    })

@app.route("/predict/linear")
def predict_linear():
    future_days = 252 * 10  # 10 years
    X_future = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    predictions = linear_model.predict(X_future)

    return jsonify({
        "model": "Linear Regression",
        "future_prices": predictions.tolist()
    })

@app.route("/predict/lstm")
def predict_lstm():
    time_step = 60
    data_scaled = scaler.transform(df[['Close']])

    last_sequence = data_scaled[-time_step:]
    last_sequence = last_sequence.reshape(1, time_step, 1)

    future_predictions = []
    days = 252 * 10  # 10 years

    for _ in range(days):
        pred = lstm_model.predict(last_sequence, verbose=0)
        future_predictions.append(pred[0][0])
        last_sequence = np.append(
            last_sequence[:, 1:, :],
            [[[pred[0][0]]]],
            axis=1
        )

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

    return jsonify({
        "model": "LSTM",
        "future_prices": future_predictions.flatten().tolist()
    })

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)