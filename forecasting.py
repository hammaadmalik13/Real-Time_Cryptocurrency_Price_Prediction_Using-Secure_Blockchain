from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from crypto_lstm.config import DEFAULT_FORECAST_STEPS, DEFAULT_SIGNAL_THRESHOLD, WINDOW_SIZE
from crypto_lstm.preprocessing import inverse_target_values, target_index_for


@dataclass(frozen=True)
class ForecastPoint:
    step: int
    predicted_price: float
    signal: str


def signal_from_change(
    predicted_price: float,
    current_price: float,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
) -> str:
    change_pct = (predicted_price - current_price) / current_price
    if change_pct > threshold:
        return "BUY"
    if change_pct < -threshold:
        return "SELL"
    return "HOLD"


def forecast_prices_and_signals(
    model,
    scaled_data: np.ndarray,
    scaler,
    price_frame: pd.DataFrame,
    feature_names: list[str],
    *,
    window_size: int = WINDOW_SIZE,
    steps: int = DEFAULT_FORECAST_STEPS,
    threshold: float = DEFAULT_SIGNAL_THRESHOLD,
) -> list[ForecastPoint]:
    values = np.asarray(scaled_data)
    if values.ndim != 2:
        raise ValueError("scaled_data must be a 2D array.")
    if len(values) < window_size:
        raise ValueError("Not enough scaled rows to build the forecast window.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    target_index = target_index_for(feature_names)
    feature_count = len(feature_names)
    current_sequence = values[-window_size:].copy()
    current_price = float(price_frame["close"].iloc[-1])
    forecast_points: list[ForecastPoint] = []

    for step in range(1, steps + 1):
        input_data = current_sequence.reshape(1, window_size, feature_count)
        scaled_prediction = float(np.asarray(model.predict(input_data, verbose=0)).reshape(-1)[0])
        predicted_price = float(
            inverse_target_values(
                scaler,
                [scaled_prediction],
                target_index=target_index,
                feature_count=feature_count,
            )[0]
        )

        signal = signal_from_change(predicted_price, current_price, threshold)
        forecast_points.append(
            ForecastPoint(
                step=step,
                predicted_price=predicted_price,
                signal=signal,
            )
        )

        next_row = current_sequence[-1].copy()
        next_row[target_index] = scaled_prediction
        current_sequence = np.vstack([current_sequence[1:], next_row])
        current_price = predicted_price

    return forecast_points


def simulate_portfolio(
    points: Iterable[ForecastPoint],
    initial_capital: float = 10_000,
) -> float:
    capital = float(initial_capital)
    position = 0.0
    last_price = None

    for point in points:
        last_price = point.predicted_price
        if point.signal == "BUY" and capital > 0:
            position = capital / point.predicted_price
            capital = 0.0
        elif point.signal == "SELL" and position > 0:
            capital = position * point.predicted_price
            position = 0.0

    if position > 0 and last_price is not None:
        return position * last_price
    return capital


def format_forecast_table(points: Iterable[ForecastPoint]) -> str:
    lines = ["Step | Predicted Price | Signal", "-----|-----------------|--------"]
    for point in points:
        lines.append(f"{point.step:>4} | {point.predicted_price:>15.2f} | {point.signal}")
    return "\n".join(lines)

