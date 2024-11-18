import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch Bitcoin historical price data from CoinGecko API
def fetch_bitcoin_data():
    print("Fetching Bitcoin price data...")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "365"}  # Data for the last 365 days???
    response = requests.get(url, params=params)
    data = response.json()
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    return prices

# Preprocess the data
def preprocess_data(prices):
    print("Preprocessing data...")
    prices['lag_1'] = prices['price'].shift(1)  # Lagged feature
    prices['lag_2'] = prices['price'].shift(2)  # Another lagged feature
    prices.dropna(inplace=True)  # Drop rows with missing values
    return prices

# Train a Linear Regression model
def train_model(data):
    print("Training model...")
    # Features and labels
    X = data[['lag_1', 'lag_2']]
    y = data['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Model RMSE: {rmse}")
    
    return model, X_test, y_test, y_pred

# Plot actual vs predicted prices
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual Price", color='blue')
    plt.plot(y_pred, label="Predicted Price", color='orange')
    plt.title("Bitcoin Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

# Main program
def main():
    # Step 1: Fetch data
    prices = fetch_bitcoin_data()
    
    # Step 2: Preprocess data
    prices = preprocess_data(prices)
    
    # Step 3: Train the model
    model, X_test, y_test, y_pred = train_model(prices)
    
    # Step 4: Plot results
    plot_results(y_test, y_pred)
    
    # Predict future price
    recent_data = prices[['lag_1', 'lag_2']].iloc[-1].values.reshape(1, -1)
    future_price = model.predict(recent_data)
    print(f"Predicted Bitcoin Price for the Next Day: ${future_price[0]:.2f}")

# Run the program
if __name__ == "__main__":
    main()
