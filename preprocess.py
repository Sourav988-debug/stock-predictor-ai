import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load CSV (Date is already the index)
df = pd.read_csv("data/GOOGL.csv", index_col=0)

# Convert index to datetime safely
df.index = pd.to_datetime(df.index, errors='coerce')

# Drop rows where index could not be converted to datetime
df = df[~df.index.isna()]

# Use Close price
data = df[['Close']]

# Forward fill missing values
data = data.ffill()

# Save clean data
data.to_csv("data/clean_data.csv")

# Scale for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Save scaled data
np.save("data/scaled_data.npy", scaled_data)

print("✅ Preprocessing completed successfully!")
print(data.head())