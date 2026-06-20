import pandas as pd
import numpy as np


def add_technical_indicators(df):
    """Add RSI, EMA, MACD, and volatility columns to OHLCV data."""
    df = df.copy()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["ema"] = df["close"].ewm(span=20, adjust=False).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    df["volatility"] = df["close"].rolling(20).std()

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    import math
    import os
    import joblib
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input

    CLOSE_INDEX = 3
    WINDOW_SIZE = 60

    df = pd.read_csv("data/btc_data.csv")
    df = add_technical_indicators(df)

    features = df[[
        "open", "high", "low", "close", "volume",
        "rsi", "ema", "macd", "volatility",
    ]]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    def create_sequences(data, window_size=WINDOW_SIZE):
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i - window_size:i])
            y.append(data[i][CLOSE_INDEX])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, WINDOW_SIZE)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        Input(shape=(WINDOW_SIZE, X_train.shape[2])),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

    predictions = model.predict(X_test)

    def inverse_close(scaled_close_values, feature_count):
        dummy = np.zeros((len(scaled_close_values), feature_count))
        dummy[:, CLOSE_INDEX] = scaled_close_values.flatten()
        return scaler.inverse_transform(dummy)[:, CLOSE_INDEX]

    predictions_actual = inverse_close(predictions, features.shape[1])
    y_test_actual = inverse_close(y_test.reshape(-1, 1), features.shape[1])

    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    print("Advanced Model RMSE:", rmse)

    plt.figure()
    plt.plot(y_test_actual, label="Actual")
    plt.plot(predictions_actual, label="Predicted")
    plt.legend()
    plt.title("Advanced LSTM with Technical Indicators")
    plt.savefig("Bit_coin_predictions.png")
    plt.show()

    os.makedirs("models", exist_ok=True)
    model.save("models/crypto_lstm_model.keras")
    joblib.dump(scaler, "models/feature_scaler.save")
    print("Model saved successfully!")
