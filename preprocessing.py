from __future__ import annotations

from pathlib import Path

import numpy as np

from crypto_lstm.config import CLOSE_FEATURE, WINDOW_SIZE


def create_sequences(
    data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    target_index: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create rolling LSTM windows and target values from scaled feature data."""
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("Expected a 2D array shaped as rows x features.")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if len(values) <= window_size:
        raise ValueError("Not enough rows to create at least one sequence.")
    if target_index < 0 or target_index >= values.shape[1]:
        raise ValueError("target_index is outside the feature array.")

    X, y = [], []
    for index in range(window_size, len(values)):
        X.append(values[index - window_size:index])
        y.append(values[index, target_index])

    return np.asarray(X), np.asarray(y)


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    split_index = int(len(X) * train_ratio)
    if split_index == 0 or split_index == len(X):
        raise ValueError("train_ratio leaves one split empty.")

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


def fit_transform_features(feature_frame):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_frame)
    return scaler, scaled_data


def inverse_target_values(
    scaler,
    scaled_values,
    target_index: int,
    feature_count: int,
) -> np.ndarray:
    dummy = np.zeros((len(np.asarray(scaled_values).reshape(-1)), feature_count))
    dummy[:, target_index] = np.asarray(scaled_values).reshape(-1)
    return scaler.inverse_transform(dummy)[:, target_index]


def target_index_for(feature_names: list[str], target_feature: str = CLOSE_FEATURE) -> int:
    if target_feature not in feature_names:
        raise ValueError(f"{target_feature!r} is not present in feature_names.")
    return feature_names.index(target_feature)


def save_scaler(scaler, path: str | Path) -> Path:
    import joblib

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)
    return output_path

