import pandas as pd
import yfinance as yf

# Fetch options data for a specific ticker (e.g., Apple - AAPL)
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Get the expiration dates for the options
expirations = stock.options
print(f"Available Expiration Dates for {ticker}:")
print(expirations)

# Select the first expiration date
selected_expiration = expirations[0]
print(f"\nFetching options chain for expiration date: {selected_expiration}")

# Fetch the options chain for the selected expiration date
options_chain = stock.option_chain(selected_expiration)

# Separate calls and puts
calls = options_chain.calls
puts = options_chain.puts

# Display the first few rows of calls and puts
print("\nCalls:")
print(calls.head())