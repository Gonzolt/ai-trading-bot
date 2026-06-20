from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=160, freq="D", tz="UTC")
    trend = np.linspace(100, 132, len(index))
    wave = np.sin(np.linspace(0, 16, len(index))) * 2
    close = trend + wave
    return pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000_000, 1_300_000, len(index)),
        },
        index=index,
    )
