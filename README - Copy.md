# Crypto LSTM Price Prediction

Bitcoin price forecasting project using Binance OHLCV data, an LSTM model, and a FastAPI service for publishing predictions.

This rebuild turns the original notebook-style scripts into a publishable Python project:

- reusable package code in `src/crypto_lstm`
- command-line tools for data collection, training, and forecasting
- a FastAPI app for deployment
- tests for feature preparation and sequence generation
- Docker support for API publishing

> Forecasts are experimental and are not financial advice.

## Project Structure

```text
src/crypto_lstm/      Reusable package code
scripts/              Local script wrappers
tests/                Lightweight unit tests
data/                 Local datasets, ignored by git
models/               Trained model artifacts, ignored by git
Dockerfile            API deployment image
requirements.txt      Runtime dependencies
pyproject.toml        Packaging and CLI metadata
```

The older root-level scripts are still present for reference. New development should use the package and CLI commands.

## Fresh Setup

Use Python 3.10 or 3.11 for the smoothest TensorFlow install.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Run The Pipeline

Collect recent BTC/USDT candles:

```powershell
crypto-lstm-collect --symbol BTCUSDT --interval 1h --limit 1000
```

Train the baseline LSTM:

```powershell
crypto-lstm-train --epochs 20
```

Train with technical indicators:

```powershell
crypto-lstm-train --epochs 30 --include-indicators
```

Generate a 90-step forecast:

```powershell
crypto-lstm-forecast --steps 90
```

## Run The API

Train the model first so `models/btc_lstm_model.keras`, `models/scaler.save`, and `models/model_metadata.json` exist.

```powershell
uvicorn crypto_lstm.api:app --reload
```

Open:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/forecast`

Example request:

```json
{
  "steps": 30,
  "threshold": 0.02
}
```

## Publish With Docker

Build the image:

```powershell
docker build -t crypto-lstm-api .
```

Run the API:

```powershell
docker run -p 8000:8000 crypto-lstm-api
```

For Render, Railway, Fly.io, or similar platforms, use this start command:

```text
uvicorn crypto_lstm.api:app --host 0.0.0.0 --port $PORT
```

Make sure trained artifacts are available in the deployment. Common options:

- train locally and copy `models/btc_lstm_model.keras`, `models/scaler.save`, and `models/model_metadata.json` into the deployment image
- train inside CI before building the image
- store artifacts in cloud storage and download them during release

## Publish Checklist

- Do not commit `venv/`, `.venv/`, cached files, generated plots, raw datasets, or large model binaries unless your hosting plan expects them.
- Add a license before making the repository public.
- Add model evaluation notes to your project page: dataset window, interval, RMSE, and limitations.
- Re-train and re-publish artifacts whenever you change features, window size, or model architecture.

## Environment Variables

The defaults work locally, but these paths can be overridden:

```text
DATA_DIR
MODELS_DIR
BTC_DATA_PATH
MODEL_PATH
SCALER_PATH
MODEL_METADATA_PATH
WINDOW_SIZE
FORECAST_STEPS
SIGNAL_THRESHOLD
```

