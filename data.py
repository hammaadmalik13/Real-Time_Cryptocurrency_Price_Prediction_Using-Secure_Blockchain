from __future__ import annotations

from pathlib import Path

import pandas as pd

from crypto_lstm.config import BASE_FEATURES, DEFAULT_DATA_PATH


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download OHLCV candles from Binance and return a normalized dataframe."""
    import requests

    if limit < 1 or limit > 1000:
        raise ValueError("Binance supports a limit between 1 and 1000 candles.")

    response = requests.get(
        BINANCE_KLINES_URL,
        params={"symbol": symbol.upper(), "interval": interval, "limit": limit},
        timeout=timeout,
    )
    response.raise_for_status()

    raw_rows = response.json()
    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(raw_rows, columns=columns)
    df = df[["timestamp", *BASE_FEATURES]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    for column in BASE_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return df.sort_values("timestamp").reset_index(drop=True)


def save_price_data(df: pd.DataFrame, path: str | Path = DEFAULT_DATA_PATH) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def load_price_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} was not found. Run data collection first.")

    df = pd.read_csv(input_path)
    missing_columns = {"timestamp", *BASE_FEATURES}.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for column in BASE_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["timestamp", *BASE_FEATURES])
    return df.sort_values("timestamp").reset_index(drop=True)
