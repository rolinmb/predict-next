import numpy as np
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, Dense # pyright: ignore[reportMissingImports]

data = [10, 12, 13, 15, 16, 18]
if __name__ == "__main__":
    # Prepare data (sequence of 3 -> next value)
    X, y = [], []
    seq_length = 3
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

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
    x_input = np.array(data[-seq_length:]).reshape(1, seq_length, 1)
    predicted_next = model.predict(x_input, verbose=0)[0][0]
    print("Predicted next value:", predicted_next)
