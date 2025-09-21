from key import POLYGONKEY
from consts import POLYGONURL1
import sys
import json
import requests
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"src/main.py Usage: python src/main.py btc (or eth, doge, etc...)")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    from_date = "2025-01-01"
    to_date = "2025-09-19"
    # Get OHLC data from Polygon
    response = requests.get(
        f"{POLYGONURL1}{ticker}USD/range/1/day/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=120&apiKey={POLYGONKEY}"
    )
    # Parse JSON
    json_data = response.json()  # or json.loads(response.text)
    # Extract close prices
    close_prices = [item["c"] for item in json_data["results"]]
    close_prices = close_prices[::-1]
    print("\nsrc/main.py :: Close prices:", close_prices)
    # Prepare sequences for LSTM (sequence of 3 -> next value)
    X, y = [], []
    seq_length = 3
    for i in range(len(close_prices) - seq_length):
        X.append(close_prices[i:i+seq_length])
        y.append(close_prices[i+seq_length])
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)
    # Build LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # Train
    model.fit(X, y, epochs=200, verbose=0)
    # Predict next value
    x_input = np.array(close_prices[-seq_length:]).reshape(1, seq_length, 1)
    predicted_next = model.predict(x_input, verbose=0)[0][0]
    print(f"src/main.py :: Predicted next close value for {ticker}: {predicted_next}")
