from __future__ import annotations

from trading_bot.strategies.indicators import adx, atr, macd, rsi


def test_indicators_return_expected_shapes(sample_ohlcv):
    assert atr(sample_ohlcv).dropna().iloc[-1] > 0
    assert 0 <= rsi(sample_ohlcv["close"]).dropna().iloc[-1] <= 100
    line, signal, histogram = macd(sample_ohlcv["close"])
    assert len(line) == len(sample_ohlcv)
    assert len(signal) == len(sample_ohlcv)
    assert len(histogram) == len(sample_ohlcv)
    assert 0 <= adx(sample_ohlcv).dropna().iloc[-1] <= 100
