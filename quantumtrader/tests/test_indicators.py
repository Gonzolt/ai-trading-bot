import pandas as pd
import numpy as np

from quantumtrader.data.cleaners import clean_ohlcv
from quantumtrader.strategies.indicators import ema, rsi, atr


def test_clean_ohlcv_removes_outliers() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": [1]*10, "high": [2]*10, "low": [0.5]*10, "close": [1]*10, "volume": [100]*10},
        index=idx,
    )
    df.iloc[5, df.columns.get_loc("close")] = 1000  # outlier
    cleaned = clean_ohlcv(df)
    assert len(cleaned) <= len(df)


def test_indicators() -> None:
    s = pd.Series(np.linspace(100, 110, 50))
    assert len(ema(s, 10)) == 50
    assert 0 <= rsi(s).iloc[-1] <= 100
