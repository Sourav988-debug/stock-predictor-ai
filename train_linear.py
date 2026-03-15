import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Load clean data
df = pd.read_csv("data/clean_data.csv", index_col=0)

# Create time index as feature
X = np.arange(len(df)).reshape(-1, 1)
y = df['Close'].values

# Train-test split (80-20, no shuffle)
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("models/linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Predict next 10 years (approx 2520 trading days)
future_days = 252 * 10
future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
future_predictions = model.predict(future_X)

print("✅ Linear Regression model trained successfully!")
print("Last known price:", y[-1])
print("Predicted price after 10 years:", future_predictions[-1])