from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser()


DATA_DIR = path_from_env("DATA_DIR", PROJECT_ROOT / "data")
MODELS_DIR = path_from_env("MODELS_DIR", PROJECT_ROOT / "models")
REPORTS_DIR = path_from_env("REPORTS_DIR", PROJECT_ROOT / "reports")

DEFAULT_DATA_PATH = path_from_env("BTC_DATA_PATH", DATA_DIR / "btc_data.csv")
DEFAULT_MODEL_PATH = path_from_env("MODEL_PATH", MODELS_DIR / "btc_lstm_model.keras")
DEFAULT_SCALER_PATH = path_from_env("SCALER_PATH", MODELS_DIR / "scaler.save")
DEFAULT_METADATA_PATH = path_from_env("MODEL_METADATA_PATH", MODELS_DIR / "model_metadata.json")
DEFAULT_PLOT_PATH = path_from_env("PLOT_PATH", REPORTS_DIR / "btc_lstm_predictions.png")

BASE_FEATURES = ["open", "high", "low", "close", "volume"]
TECHNICAL_FEATURES = ["rsi", "ema", "macd", "volatility"]

CLOSE_FEATURE = "close"
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "60"))
DEFAULT_FORECAST_STEPS = int(os.getenv("FORECAST_STEPS", "90"))
DEFAULT_SIGNAL_THRESHOLD = float(os.getenv("SIGNAL_THRESHOLD", "0.02"))

