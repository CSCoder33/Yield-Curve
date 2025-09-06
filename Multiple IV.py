import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

# Black-Scholes formula for option pricing
def black_scholes(S, K, T, r, sigma, option_type="put"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Function to calculate implied volatility
def implied_volatility(option_price, S, K, T, r):
    try:
        return brentq(
            lambda sigma: black_scholes(S, K, T, r, sigma) - option_price,
            1e-5,
            5.0,
        )
    except ValueError:
        return np.nan

# Function to fetch and process options data for a given ticker
def process_options_data(ticker, selected_expiration):
    stock = yf.Ticker(ticker)
    S = stock.history(period="1d")["Close"].iloc[-1]  # Current stock price
    options_chain = stock.option_chain(selected_expiration)
    puts = options_chain.puts

    # Calculate time to expiration in years
    T = (pd.to_datetime(selected_expiration) - pd.Timestamp.today()).days / 365.0

    # Define the strike price range (10% below to 10% above the current price)
    strike_min = S * 0.9
    strike_max = S * 1.1
    puts = puts[(puts["strike"] >= strike_min) & (puts["strike"] <= strike_max)]

    # Debug: Print available strikes
    print(f"\nAvailable strikes for {ticker}:")
    print(puts["strike"].tolist())

    # Filter out illiquid options
    puts = puts[(puts["volume"] > 0) & (puts["openInterest"] > 0)]

    # Calculate % in/out of the money
    puts["percent_moneyness"] = ((puts["strike"] - S) / S) * 100

    # Use mid-price for IV calculation
    puts["midPrice"] = (puts["bid"] + puts["ask"]) / 2
    puts["Implied Volatility"] = puts.apply(
        lambda row: implied_volatility(row["midPrice"], S, row["strike"], T, 0.05),
        axis=1,
    )

    # Remove outliers in implied volatility
    puts = puts[puts["Implied Volatility"] < 1]  # Example threshold

    # Sort the data by % moneyness
    valid_puts = puts.dropna(subset=["Implied Volatility"]).sort_values(by="percent_moneyness")

    return valid_puts

# Fetch expiration dates for AAPL and SPY
aapl = yf.Ticker("AAPL")
spy = yf.Ticker("SPY")
aapl_expirations = aapl.options
spy_expirations = spy.options

# Select the 7th expiration date for both tickers
selected_expiration_aapl = aapl_expirations[6]
selected_expiration_spy = spy_expirations[6]

# Process options data for AAPL and SPY
aapl_data = process_options_data("AAPL", selected_expiration_aapl)
spy_data = process_options_data("SPY", selected_expiration_spy)

# Debug: Check processed data
print("\nProcessed AAPL Data:")
print(aapl_data[["percent_moneyness", "strike", "midPrice", "Implied Volatility"]].head())

print("\nProcessed SPY Data:")
print(spy_data[["percent_moneyness", "strike", "midPrice", "Implied Volatility"]].head())

# Interpolation for smooth curves
x_aapl = aapl_data["percent_moneyness"]
y_aapl = aapl_data["Implied Volatility"]
spline_aapl = make_interp_spline(x_aapl, y_aapl)
x_aapl_smooth = np.linspace(x_aapl.min(), x_aapl.max(), 500)
y_aapl_smooth = spline_aapl(x_aapl_smooth)

x_spy = spy_data["percent_moneyness"]
y_spy = spy_data["Implied Volatility"]
spline_spy = make_interp_spline(x_spy, y_spy)
x_spy_smooth = np.linspace(x_spy.min(), x_spy.max(), 500)
y_spy_smooth = spline_spy(x_spy_smooth)

# Plot the implied volatility skew
plt.figure(figsize=(12, 6))
plt.plot(x_aapl_smooth, y_aapl_smooth, label="AAPL IV (Smoothed)", color="blue")
plt.plot(x_spy_smooth, y_spy_smooth, label="SPY IV (Smoothed)", color="orange")
plt.title("Implied Volatility Skew: AAPL vs SPY")
plt.xlabel("% In/Out of the Money")
plt.ylabel("Implied Volatility")
plt.legend()
plt.grid()
plt.show()

# Display the first few rows of processed data
print("\nAAPL Puts with Implied Volatility:")
print(aapl_data[["percent_moneyness", "strike", "midPrice", "Implied Volatility"]].head())

print("\nSPY Puts with Implied Volatility:")
print(spy_data[["percent_moneyness", "strike", "midPrice", "Implied Volatility"]].head())