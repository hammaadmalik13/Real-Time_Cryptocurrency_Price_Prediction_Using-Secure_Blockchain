import math
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Sequential

CLOSE_INDEX = 3
WINDOW_SIZE = 60
FEATURES = ["open", "high", "low", "close", "volume"]


def create_sequences(data, window_size=WINDOW_SIZE):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(data[i][CLOSE_INDEX])
    return np.array(X), np.array(y)


def inverse_close(scaler, scaled_close_values, feature_count=len(FEATURES)):
    dummy = np.zeros((len(scaled_close_values), feature_count))
    dummy[:, CLOSE_INDEX] = np.asarray(scaled_close_values).flatten()
    return scaler.inverse_transform(dummy)[:, CLOSE_INDEX]


if __name__ == "__main__":
    df = pd.read_csv("data/btc_data.csv")
    data = df[FEATURES]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, WINDOW_SIZE)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("X_train shape:", X_train.shape)

    model = Sequential([
        Input(shape=(WINDOW_SIZE, len(FEATURES))),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
    )

    predictions = model.predict(X_test)
    predictions_actual = inverse_close(scaler, predictions)
    y_test_actual = inverse_close(scaler, y_test)

    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    print("RMSE:", rmse)

    plt.figure()
    plt.plot(y_test_actual, label="Actual Price")
    plt.plot(predictions_actual, label="Predicted Price")
    plt.legend()
    plt.title("Bitcoin Price Prediction (LSTM)")
    plt.savefig("bit coin price prediction.png")
    plt.show()

    os.makedirs("models", exist_ok=True)
    model.save("models/btc_lstm_model.keras")
    joblib.dump(scaler, "models/scaler.save")
    print("Model and scaler saved to models/")
