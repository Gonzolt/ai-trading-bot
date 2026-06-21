from __future__ import annotations

import pandas as pd

from quantumtrader.config import load_settings
from quantumtrader.models import StrategySignal
from quantumtrader.strategies.base import Strategy
from quantumtrader.strategies.indicators import atr, macd, rsi


class MomentumStrategy(Strategy):
    """RSI recovery + MACD histogram confirmation. Stop 1.5×ATR."""

    name = "momentum"

    def __init__(self) -> None:
        settings = load_settings()
        self.rsi_period = int(settings.get("strategy.momentum.rsi_period", 14))
        self.stop_multiple = float(settings.get("strategy.momentum.atr_stop_multiple", 1.5))

    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        if len(asset_df) < 60:
            return StrategySignal(symbol, self.name, "hold", None, None)
        close = asset_df["close"]
        rsi_values = rsi(close, self.rsi_period)
        _, _, histogram = macd(close)
        atr_value = atr(asset_df).iloc[-1]
        price = float(close.iloc[-1])

        rsi_cross_up = rsi_values.iloc[-2] <= 30 and rsi_values.iloc[-1] > 30
        hist_positive = histogram.iloc[-2] <= 0 and histogram.iloc[-1] > 0
        if rsi_cross_up and hist_positive:
            return StrategySignal(
                symbol,
                self.name,
                "buy",
                price - self.stop_multiple * atr_value,
                None,
                confidence=min(1.0, float(rsi_values.iloc[-1]) / 100),
            )
        if rsi_values.iloc[-1] > 70 or histogram.iloc[-1] < 0:
            return StrategySignal(symbol, self.name, "sell", None, None, confidence=0.6)
        return StrategySignal(symbol, self.name, "hold", None, None)
