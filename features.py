from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_lstm.config import BASE_FEATURES, TECHNICAL_FEATURES


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, EMA, MACD, and volatility columns to OHLCV data."""
    output = df.copy()

    for column in BASE_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    delta = output["close"].diff()
    average_gain = delta.clip(lower=0).rolling(14).mean()
    average_loss = -delta.clip(upper=0).rolling(14).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    output["rsi"] = 100 - (100 / (1 + relative_strength))
    output.loc[average_loss == 0, "rsi"] = 100

    output["ema"] = output["close"].ewm(span=20, adjust=False).mean()

    ema12 = output["close"].ewm(span=12, adjust=False).mean()
    ema26 = output["close"].ewm(span=26, adjust=False).mean()
    output["macd"] = ema12 - ema26

    output["volatility"] = output["close"].rolling(20).std()

    return output.dropna(subset=[*BASE_FEATURES, *TECHNICAL_FEATURES]).reset_index(drop=True)


def prepare_feature_frame(
    df: pd.DataFrame,
    include_indicators: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a numeric feature dataframe and the feature order used by the model."""
    feature_names = [*BASE_FEATURES]
    output = df.copy()

    if include_indicators:
        output = add_technical_indicators(output)
        feature_names.extend(TECHNICAL_FEATURES)

    missing_columns = set(feature_names).difference(output.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing feature columns: {missing}")

    for column in feature_names:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    output = output.dropna(subset=feature_names).reset_index(drop=True)
    if output.empty:
        raise ValueError("No usable rows remain after feature preparation.")

    return output, feature_names

