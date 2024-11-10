import yfinance as yf

# Define the ticker symbol for NIFTY50
nifty_ticker = "^NSEI"

# Download historical data for NIFTY50 from Yahoo Finance
nifty_data = yf.download(nifty_ticker, start="2010-01-01", end="2023-12-31", interval="1mo")

# Display the first few rows of the data to verify
print(nifty_data.head())

# Save the data to a CSV file
nifty_data.to_csv("NIFTY50_data.csv")
