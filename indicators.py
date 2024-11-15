import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch stock data
ticker = 'BTC-USD'
data = stock_data = yf.download(ticker, start='2022-01-01', end='2024-01-01')

### Relative Strength Index (RSI)

def calculate_rsi(df, period=14, column='Close'):
    """
    Calculate the Relative Strength Index (RSI) for a given column in the DataFrame.

    Parameters:
    - df: DataFrame containing stock data.
    - period: The look-back period for calculating RSI (default is 14).
    - column: The column to apply RSI to (e.g., 'Close').

    Returns:
    - Series with RSI values.
    """
    delta = df[column].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

### Bollinger Bands

def calculate_bollinger_bands(df, window=20, column='Close'):
    """
    Calculate Bollinger Bands for a given column in the DataFrame.

    Parameters:
    - df: DataFrame containing stock data.
    - window: The period for the moving average (default is 20).
    - column: The column to apply Bollinger Bands to (e.g., 'Close').

    Returns:
    - DataFrame with columns for the moving average, upper band, and lower band.
    """
    ma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()
    upper_band = ma + (std * 2)
    lower_band = ma - (std * 2)

    df[f'BB_MA_{window}'] = ma
    df[f'BB_Upper_{window}'] = upper_band
    df[f'BB_Lower_{window}'] = lower_band
    return df

### Visualization Functions

def plot_bollinger_bands(df, ticker, window=20):
    """
    Plot the price with Bollinger Bands and Moving Average.

    Parameters:
    - df: DataFrame containing stock data and Bollinger Band calculations.
    - ticker: Stock ticker symbol as a string.
    - window: Period for the moving average used in Bollinger Bands (default is 20).
    """
    plt.figure(figsize=(14, 8))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df[f'BB_MA_{window}'], label=f'{window}-Day MA', color='orange')
    plt.plot(df[f'BB_Upper_{window}'], label='Upper Bollinger Band', color='green')
    plt.plot(df[f'BB_Lower_{window}'], label='Lower Bollinger Band', color='red')
    plt.fill_between(df.index, df[f'BB_Upper_{window}'], df[f'BB_Lower_{window}'], color='gray', alpha=0.2)
    plt.title(f"{ticker} Price with Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_rsi(df, ticker):
    """
    Plot the RSI with overbought and oversold levels.

    Parameters:
    - df: DataFrame containing stock data and RSI calculations.
    - ticker: Stock ticker symbol as a string.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f"{ticker} Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.show()

def plot_ma_vs_price(df, ticker, ma_windows=[20, 50]):
    """
    Plot Close price vs. Moving Averages.

    Parameters:
    - df: DataFrame containing stock data.
    - ticker: Stock ticker symbol as a string.
    - ma_windows: List of periods for moving averages to plot (default is [20, 50]).
    """
    plt.figure(figsize=(14, 8))
    plt.plot(df['Close'], label='Close Price', color='blue')

    for window in ma_windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        plt.plot(df[f'MA_{window}'], label=f'{window}-Day MA')

    plt.title(f"{ticker} Price vs. Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Function to calculate MACD
def calculate_macd(stock_data):
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    return stock_data

# Function to calculate SMA 50
def calculate_sma_50(stock_data):
    stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
    return stock_data

# Function to calculate SMA 200
def calculate_sma_200(stock_data):
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    return stock_data

# Function to calculate ATR (Average True Range)
def calculate_atr(stock_data):
    stock_data['High_Low'] = stock_data['High'] - stock_data['Low']
    stock_data['High_Close'] = (stock_data['High'] - stock_data['Close'].shift()).abs()
    stock_data['Low_Close'] = (stock_data['Low'] - stock_data['Close'].shift()).abs()
    tr = stock_data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    stock_data['ATR'] = tr.rolling(window=14).mean()
    return stock_data



# Apply each indicator function
stock_data = calculate_macd(stock_data)
stock_data = calculate_sma_50(stock_data)
stock_data = calculate_sma_200(stock_data)
stock_data = calculate_atr(stock_data)

# Plotting
plt.figure(figsize=(14, 10))

# Plot Close Price, SMA 50, and SMA 200
plt.subplot(3, 1, 1)
plt.plot(stock_data['Close'], label='Close Price', color='blue')
plt.plot(stock_data['SMA_50'], label='SMA 50', color='orange')
plt.plot(stock_data['SMA_200'], label='SMA 200', color='green')
plt.title(f'{ticker} Close Price with SMA 50 and SMA 200')
plt.legend()

# Plot MACD and Signal Line
plt.subplot(3, 1, 2)
plt.plot(stock_data['MACD'], label='MACD', color='purple')
plt.plot(stock_data['Signal_Line'], label='Signal Line', color='red')
plt.title('MACD and Signal Line')
plt.legend()

# Plot ATR
plt.subplot(3, 1, 3)
plt.plot(stock_data['ATR'], label='ATR', color='brown')
plt.title('Average True Range (ATR)')
plt.legend()

plt.tight_layout()
plt.show()


# Calculate indicators
data['RSI'] = calculate_rsi(data)
data = calculate_bollinger_bands(data)
plot_bollinger_bands(data, ticker)
plot_rsi(data, ticker)
plot_ma_vs_price(data, ticker, ma_windows=[20, 50])