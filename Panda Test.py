import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch stock data for a specific ticker (e.g., Apple - AAPL)
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")

# Display the first few rows of the data
print("Stock Data:")
print(data.head())

# Calculate a 20-day moving average
data["20-Day MA"] = data["Close"].rolling(window=20).mean()

# Calculate daily returns
data["Daily Return"] = data["Close"].pct_change()

# Plot the closing price and 20-day moving average
plt.figure(figsize=(12, 6))
plt.plot(data["Close"], label="Closing Price", color="blue")
plt.plot(data["20-Day MA"], label="20-Day Moving Average", color="orange")
plt.title(f"{ticker} Stock Price and 20-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Display summary statistics for daily returns
print("\nSummary Statistics for Daily Returns:")
print(data["Daily Return"].describe())