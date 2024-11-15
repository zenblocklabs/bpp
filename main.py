### 1 Fetch Historical Price Data with yfinance
import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Define ticker symbol and date range
BTC_SYMBOL = "BTC-USD"
start_date = "2024-05-01"
end_date = "2024-11-01"

# Download historical data for Bitcoin
bitcoin_data = yf.download(BTC_SYMBOL, start=start_date, end=end_date)

# Add a 'timestamp' column with the datetime index
bitcoin_data['timestamp'] = bitcoin_data.index

# Add a 'date' column formatted as YYYY-MM-DD
bitcoin_data['date'] = bitcoin_data.index.date

# Rename columns to match specified output
bitcoin_data = bitcoin_data.rename(columns={
    "Adj Close": "adj close",
    "Close": "close",
    "High": "high",
    "Low": "low",
    "Open": "open",
    "Volume": "volume"
})

# Reorder columns as requested
bitcoin_data = bitcoin_data[['date', 'adj close', 'close', 'high', 'low', 'open', 'volume', 'timestamp']]

# Save to CSV
bitcoin_data.to_csv("bitcoin_data.csv", index=False)




### 2 Data Cleaning and Preparation for Exploration

## 1: Load CSV Data for Cleaning
# Load the Saved CSV Files
bitcoin_data = pd.read_csv("bitcoin_data.csv", parse_dates=['date'], index_col="date") if 'date' in pd.read_csv("bitcoin_data.csv", nrows=0).columns else pd.read_csv("bitcoin_data.csv", parse_dates=True)
print("Bitcoin Data Columns:", pd.read_csv("bitcoin_data.csv", nrows=0).columns)

# Basic Data Overview
print(bitcoin_data.info())  # Check for data types and null values

## 2: Handle Missing Values (NaNs)
# Check for Missing Values
print(bitcoin_data.isnull().sum())

# Fill Missing Values
bitcoin_data.ffill(inplace=True)

## 3: Handle Outliers and Extreme Values
# Visual Inspection
plt.plot(bitcoin_data['close'], label='Bitcoin Close Price')
plt.plot(bitcoin_data['open'], label='Bitcoin Open Price')
plt.legend()
plt.show()

# Ensure 'Close' column is numeric
bitcoin_data['close'] = pd.to_numeric(bitcoin_data['close'], errors='coerce')

# Fill any NaN values that may have resulted from the conversion
bitcoin_data['close'] = bitcoin_data['close'].ffill()

# Quantile-based Filtering with .loc to avoid chained assignment issues
lower_limit = bitcoin_data['close'].quantile(0.01)
upper_limit = bitcoin_data['close'].quantile(0.99)

# Use direct assignment with .loc to avoid chained assignment
bitcoin_data.loc[:, 'close'] = bitcoin_data['close'].clip(lower=lower_limit, upper=upper_limit)
