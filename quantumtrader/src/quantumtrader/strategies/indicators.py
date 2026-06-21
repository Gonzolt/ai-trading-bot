from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + relative_strength))
    value = value.mask((loss == 0) & (gain > 0), 100)
    value = value.mask((gain == 0) & (loss > 0), 0)
    return value


def macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    histogram = line - signal
    return line, signal, histogram


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    tr = true_range(df)
    atr_value = tr.rolling(period, min_periods=period).sum()
    plus_di = 100 * plus_dm.rolling(period, min_periods=period).sum() / atr_value
    minus_di = 100 * minus_dm.rolling(period, min_periods=period).sum() / atr_value
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.rolling(period, min_periods=period).mean()


def bollinger_bands(
    series: pd.Series, window: int = 20, stddev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std()
    upper = mid + stddev * sigma
    lower = mid - stddev * sigma
    zscore = (series - mid) / sigma.replace(0, np.nan)
    return lower, mid, upper, zscore


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    clean = returns.dropna()
    if clean.empty or clean.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(annualization) * clean.mean() / clean.std(ddof=0))
