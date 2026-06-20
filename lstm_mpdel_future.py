import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

MODEL_PATH = "models/btc_lstm_model.keras"
SCALER_PATH = "models/scaler.save"
WINDOW_SIZE = 60
CLOSE_INDEX = 3


def forecast_and_trade(model, scaled_data, scaler, df,
                       window_size=WINDOW_SIZE, future_days=90, threshold=0.02):
    predictions = []
    signals = []

    current_price = df["close"].values[-1]
    current_sequence = scaled_data[-window_size:].copy()

    for _ in range(future_days):
        input_data = np.reshape(current_sequence, (1, window_size, 5))
        next_pred = model.predict(input_data, verbose=0)
        predicted_scaled_close = next_pred[0][0]
        predictions.append(predicted_scaled_close)

        dummy = np.zeros((1, 5))
        dummy[0, CLOSE_INDEX] = predicted_scaled_close
        predicted_price = scaler.inverse_transform(dummy)[0][CLOSE_INDEX]

        change_pct = (predicted_price - current_price) / current_price

        if change_pct > threshold:
            signal = "BUY"
        elif change_pct < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        signals.append(signal)
        current_price = predicted_price

        new_row = np.zeros((1, 5))
        new_row[0, CLOSE_INDEX] = predicted_scaled_close
        current_sequence = np.vstack((current_sequence[1:], new_row))

    dummy_all = np.zeros((future_days, 5))
    dummy_all[:, CLOSE_INDEX] = predictions
    future_prices = scaler.inverse_transform(dummy_all)[:, CLOSE_INDEX]

    return future_prices, signals


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run lstm_model.py first to train and save the model."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"{SCALER_PATH} not found. Run preprocessing.py or lstm_model.py first."
        )

    df = pd.read_csv("data/btc_data.csv")
    features = df[["open", "high", "low", "close", "volume"]]

    scaler = joblib.load(SCALER_PATH)
    scaled_data = scaler.transform(features)

    model = load_model(MODEL_PATH)

    future_prices, signals = forecast_and_trade(
        model, scaled_data, scaler, df
    )

    print("\n3-Month Forecast with Trading Signals:\n")
    for i in range(len(future_prices)):
        print(f"Day {i + 1}: Price={future_prices[i]:.2f} | Signal={signals[i]}")

    plt.figure(figsize=(14, 6))
    plt.plot(future_prices, label="Forecast Price")

    for i, signal in enumerate(signals):
        if signal == "BUY":
            plt.scatter(i, future_prices[i], marker="^", color="green")
        elif signal == "SELL":
            plt.scatter(i, future_prices[i], marker="v", color="red")

    plt.title("90-Day Forecast with Trading Signals")
    plt.xlabel("Future Days")
    plt.ylabel("Predicted Price")
    plt.legend()
    plt.grid()
    plt.savefig("Bit_coin_predictions.png")
    plt.show()

    capital = 10000
    position = 0

    for i, signal in enumerate(signals):
        if signal == "BUY" and capital > 0:
            position = capital / future_prices[i]
            capital = 0
        elif signal == "SELL" and position > 0:
            capital = position * future_prices[i]
            position = 0

    final_value = capital if capital > 0 else position * future_prices[-1]

    print("\nSimulation Result")
    print("Initial Capital: 10000")
    print(f"Final Portfolio Value: {final_value:.2f}")
