import yfinance as yf
import pandas as pd
import time

ticker = "GOOGL"

# Small delay to avoid rate limiting
time.sleep(2)

data = yf.download(
    tickers=ticker,
    start="2015-01-01",
    end="2025-01-01",
    interval="1d",
    auto_adjust=False,
    progress=False,
    threads=False
)

# Check if data is empty
if data.empty:
    print("❌ Download failed. Try again after 10–15 minutes.")
else:
    data.to_csv("data/GOOGL.csv")
    print("✅ Data downloaded successfully!")
    print(data.head())