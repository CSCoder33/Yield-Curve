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

# Fetch options data for a specific ticker
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Get the current stock price
S = stock.history(period="1d")["Close"].iloc[-1]

# Get the risk-free rate (e.g., 5% annualized)
r = 0.05

# Get the expiration dates for the options
expirations = stock.options
print(f"Available Expiration Dates for {ticker}:")
print(expirations)

# Select the 7th expiration date
if len(expirations) >= 7:
    selected_expiration = expirations[6]  # 7th expiration (index 6)
else:
    raise ValueError("Not enough expiration dates available to select the 7th one.")
print(f"\nFetching options chain for expiration date: {selected_expiration}")

# Fetch the options chain for the selected expiration date
options_chain = stock.option_chain(selected_expiration)

# Extract puts
puts = options_chain.puts

# Debug: Check all available strikes
print("\nAvailable Put Strikes (Before Filtering):")
print(puts["strike"].tolist())

# Calculate time to expiration in years
T = (pd.to_datetime(selected_expiration) - pd.Timestamp.today()).days / 365.0

# Define the strike price range (15% below to 15% above the current price)
strike_min = S * 0.85
strike_max = S * 1.15
print(f"\nFiltering strikes between {strike_min:.2f} and {strike_max:.2f}")

# Filter puts to include only strikes within the range
puts = puts[(puts["strike"] >= strike_min) & (puts["strike"] <= strike_max)]

# Debug: Check the filtered strikes
print("\nFiltered Put Strikes:")
print(puts["strike"].tolist())

# Calculate implied volatility for puts
puts["Implied Volatility"] = puts.apply(
    lambda row: implied_volatility(
        row["lastPrice"], S, row["strike"], T, r
    ),
    axis=1,
)

# Filter out rows with NaN implied volatility
valid_puts = puts.dropna(subset=["Implied Volatility"])

# Sort the data by strike price to ensure proper interpolation
valid_puts = valid_puts.sort_values(by="strike")

# Interpolation for a smooth curve
x = valid_puts["strike"]
y = valid_puts["Implied Volatility"]

# Create a spline function for smooth interpolation
spline = make_interp_spline(x, y)
x_smooth = np.linspace(x.min(), x.max(), 500)  # Generate 500 points for a smooth curve
y_smooth = spline(x_smooth)

# Plot the implied volatility skew
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, y_smooth, label="Puts IV (Smoothed)", color="blue")
plt.title(f"Implied Volatility Skew for {ticker} (Expiration: {selected_expiration})")
plt.xlabel("Strike Price")
plt.ylabel("Implied Volatility")
plt.legend()
plt.grid()
plt.xlim([strike_min, strike_max])  # Restrict x-axis to 15% below and above the current price
plt.show()

# Display the first few rows of puts with IV
print("\nPuts with Implied Volatility:")
print(valid_puts[["strike", "lastPrice", "Implied Volatility"]].head())