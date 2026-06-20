from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from crypto_lstm.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_FORECAST_STEPS,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    DEFAULT_SIGNAL_THRESHOLD,
    WINDOW_SIZE,
)
from crypto_lstm.data import load_price_data
from crypto_lstm.features import prepare_feature_frame
from crypto_lstm.forecasting import ForecastPoint, forecast_prices_and_signals
from crypto_lstm.model import load_metadata


app = FastAPI(
    title="Crypto LSTM Prediction API",
    version="0.1.0",
    description="Forecast BTC prices with a trained LSTM model.",
)


class ForecastRequest(BaseModel):
    steps: int = Field(DEFAULT_FORECAST_STEPS, ge=1, le=365)
    threshold: float = Field(DEFAULT_SIGNAL_THRESHOLD, ge=0, le=1)


class ForecastItem(BaseModel):
    step: int
    predicted_price: float
    signal: str


class ForecastResponse(BaseModel):
    generated_at_utc: str
    model_metadata: dict[str, Any]
    forecast: list[ForecastItem]


def _artifact_status() -> dict[str, bool]:
    return {
        "data_exists": Path(DEFAULT_DATA_PATH).exists(),
        "model_exists": Path(DEFAULT_MODEL_PATH).exists(),
        "scaler_exists": Path(DEFAULT_SCALER_PATH).exists(),
        "metadata_exists": Path(DEFAULT_METADATA_PATH).exists(),
    }


@lru_cache(maxsize=1)
def _load_runtime_artifacts():
    missing = [
        str(path)
        for path in [DEFAULT_MODEL_PATH, DEFAULT_SCALER_PATH]
        if not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing runtime artifact(s): "
            + ", ".join(missing)
            + ". Train the model before serving forecasts."
        )

    from tensorflow.keras.models import load_model
    import joblib

    model = load_model(DEFAULT_MODEL_PATH)
    scaler = joblib.load(DEFAULT_SCALER_PATH)
    metadata = load_metadata(DEFAULT_METADATA_PATH)
    return model, scaler, metadata


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Crypto LSTM Prediction API"}


@app.get("/health")
def health() -> dict[str, Any]:
    status = _artifact_status()
    ready = status["data_exists"] and status["model_exists"] and status["scaler_exists"]
    return {"status": "ready" if ready else "setup_required", **status}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        model, scaler, metadata = _load_runtime_artifacts()
        df = load_price_data(DEFAULT_DATA_PATH)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    include_indicators = bool(metadata.get("include_indicators", False))
    window_size = int(metadata.get("window_size", WINDOW_SIZE))
    feature_frame, feature_names = prepare_feature_frame(df, include_indicators)

    try:
        scaled_data = scaler.transform(feature_frame[feature_names])
        points = forecast_prices_and_signals(
            model,
            scaled_data,
            scaler,
            feature_frame,
            feature_names,
            window_size=window_size,
            steps=request.steps,
            threshold=request.threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ForecastResponse(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        model_metadata=metadata,
        forecast=[
            ForecastItem(
                step=point.step,
                predicted_price=point.predicted_price,
                signal=point.signal,
            )
            for point in points
        ],
    )
