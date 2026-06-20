from __future__ import annotations

import numpy as np
import pandas as pd

from crypto_lstm.config import BASE_FEATURES, TECHNICAL_FEATURES
from crypto_lstm.features import add_technical_indicators, prepare_feature_frame


def make_price_frame(rows: int = 80) -> pd.DataFrame:
    close = np.linspace(100, 140, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h"),
            "open": close - 1,
            "high": close + 2,
            "low": close - 3,
            "close": close,
            "volume": np.linspace(10, 20, rows),
        }
    )


def test_add_technical_indicators_adds_expected_columns() -> None:
    output = add_technical_indicators(make_price_frame())

    for column in TECHNICAL_FEATURES:
        assert column in output.columns
    assert output[[*BASE_FEATURES, *TECHNICAL_FEATURES]].notna().all().all()


def test_prepare_feature_frame_can_include_indicators() -> None:
    output, feature_names = prepare_feature_frame(make_price_frame(), include_indicators=True)

    assert feature_names == [*BASE_FEATURES, *TECHNICAL_FEATURES]
    assert not output.empty

