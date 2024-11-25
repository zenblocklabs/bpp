import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, GRU # type: ignore

# Fetch Bitcoin historical price data from CoinGecko API
def fetch_bitcoin_data():
    print("Fetching Bitcoin price data...")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "365"}  # Data for the last 365 days
    response = requests.get(url, params=params)
    data = response.json()
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    return prices

# Preprocess the data for GRU model
def preprocess_data(prices, look_back=5):
    print("Preprocessing data for GRU...")
    prices['price'] = prices['price'].astype(float)  # Ensure the price is float
    data = prices['price'].values
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:i + look_back])  # Last 'look_back' prices as features
        y.append(data[i + look_back])   # The next price as the label
    x = np.array(x)
    y = np.array(y)
    return x, y, data

# Create GRU model
def create_gru_model(input_shape):
    print("Building GRU model...")
    model = Sequential()
    model.add(GRU(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))  # Output layer for price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main workflow
if __name__ == "__main__":
    # Fetch and preprocess data
    prices = fetch_bitcoin_data()
    look_back = 10  # Number of timesteps for GRU
    x, y, data = preprocess_data(prices, look_back)
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Reshape data for GRU model (samples, timesteps, features)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    # Build and train GRU model
    model = create_gru_model(input_shape=(x_train.shape[1], x_train.shape[2]))
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Evaluate model
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Predict next day's price
    last_sequence = data[-look_back:].reshape(1, look_back, 1)  # Use the last 'look_back' days
    next_day_prediction = model.predict(last_sequence)[0][0]
    print(f"Predicted Bitcoin price for the next day: ${next_day_prediction:.2f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='orange')
    plt.legend()
    plt.title('GRU Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    # plt.show()
