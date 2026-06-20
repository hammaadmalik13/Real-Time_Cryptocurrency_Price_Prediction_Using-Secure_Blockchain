from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_lstm.config import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PLOT_PATH,
    DEFAULT_SCALER_PATH,
    WINDOW_SIZE,
)
from crypto_lstm.features import prepare_feature_frame
from crypto_lstm.preprocessing import (
    create_sequences,
    fit_transform_features,
    inverse_target_values,
    save_scaler,
    split_train_test,
    target_index_for,
)


@dataclass(frozen=True)
class TrainResult:
    rmse: float
    samples: int
    train_samples: int
    test_samples: int
    feature_names: list[str]
    model_path: str
    scaler_path: str
    metadata_path: str
    plot_path: str | None


def build_lstm_model(window_size: int, feature_count: int):
    from tensorflow.keras.layers import Dense, Input, LSTM
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(window_size, feature_count)),
            LSTM(64, return_sequences=True),
            LSTM(64),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def load_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def plot_prediction_comparison(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Price")
    plt.plot(predicted_prices, label="Predicted Price")
    plt.title("Bitcoin Price Prediction (LSTM)")
    plt.xlabel("Test Step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def train_lstm_model(
    df: pd.DataFrame,
    *,
    include_indicators: bool = False,
    window_size: int = WINDOW_SIZE,
    train_ratio: float = 0.8,
    epochs: int = 20,
    batch_size: int = 32,
    validation_split: float = 0.1,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    plot_path: str | Path | None = DEFAULT_PLOT_PATH,
) -> TrainResult:
    from sklearn.metrics import mean_squared_error

    feature_frame, feature_names = prepare_feature_frame(df, include_indicators)
    target_index = target_index_for(feature_names)

    scaler, scaled_data = fit_transform_features(feature_frame[feature_names])
    X, y = create_sequences(scaled_data, window_size, target_index)
    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio)

    model = build_lstm_model(window_size, len(feature_names))
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )

    predictions = model.predict(X_test, verbose=0)
    predicted_prices = inverse_target_values(
        scaler,
        predictions,
        target_index=target_index,
        feature_count=len(feature_names),
    )
    actual_prices = inverse_target_values(
        scaler,
        y_test,
        target_index=target_index,
        feature_count=len(feature_names),
    )
    rmse = math.sqrt(mean_squared_error(actual_prices, predicted_prices))

    model_output_path = Path(model_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_output_path)

    scaler_output_path = save_scaler(scaler, scaler_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_indicators": include_indicators,
        "window_size": window_size,
        "train_ratio": train_ratio,
        "epochs": epochs,
        "batch_size": batch_size,
        "feature_names": feature_names,
        "target_feature": "close",
        "target_index": target_index,
        "rmse": rmse,
        "rows": int(len(feature_frame)),
        "samples": int(len(X)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    metadata_output_path = save_metadata(metadata_path, metadata)

    plot_output_path = None
    if plot_path:
        plot_output_path = plot_prediction_comparison(actual_prices, predicted_prices, plot_path)

    return TrainResult(
        rmse=rmse,
        samples=int(len(X)),
        train_samples=int(len(X_train)),
        test_samples=int(len(X_test)),
        feature_names=feature_names,
        model_path=str(model_output_path),
        scaler_path=str(scaler_output_path),
        metadata_path=str(metadata_output_path),
        plot_path=str(plot_output_path) if plot_output_path else None,
    )


def result_as_dict(result: TrainResult) -> dict[str, Any]:
    return asdict(result)
