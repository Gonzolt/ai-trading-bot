from __future__ import annotations

import pandas as pd

from quantumtrader.config import load_settings
from quantumtrader.models import StrategySignal
from quantumtrader.strategies.base import Strategy
from quantumtrader.strategies.indicators import adx, atr, ema


class TrendFollowingStrategy(Strategy):
    """EMA crossover with ADX trend filter. Stop 2×ATR, target 3×ATR."""

    name = "trend_following"

    def __init__(self) -> None:
        settings = load_settings()
        self.fast = int(settings.get("strategy.trend_following.fast_ema", 20))
        self.slow = int(settings.get("strategy.trend_following.slow_ema", 50))
        self.adx_min = float(settings.get("strategy.trend_following.adx_min", 25))
        self.stop_multiple = float(settings.get("strategy.trend_following.atr_stop_multiple", 2.0))
        self.target_multiple = float(settings.get("strategy.trend_following.atr_target_multiple", 3.0))

    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        if len(asset_df) < max(self.slow, 60):
            return StrategySignal(symbol, self.name, "hold", None, None)
        close = asset_df["close"]
        fast_ema = ema(close, self.fast)
        slow_ema = ema(close, self.slow)
        trend_strength = adx(asset_df).iloc[-1]
        atr_value = atr(asset_df).iloc[-1]
        price = float(close.iloc[-1])

        crossed_up = fast_ema.iloc[-2] <= slow_ema.iloc[-2] and fast_ema.iloc[-1] > slow_ema.iloc[-1]
        crossed_down = fast_ema.iloc[-2] >= slow_ema.iloc[-2] and fast_ema.iloc[-1] < slow_ema.iloc[-1]
        if crossed_up and trend_strength > self.adx_min:
            return StrategySignal(
                symbol,
                self.name,
                "buy",
                price - self.stop_multiple * atr_value,
                price + self.target_multiple * atr_value,
                confidence=min(1.0, float(trend_strength) / 50),
            )
        if crossed_down:
            return StrategySignal(symbol, self.name, "sell", None, None, confidence=0.5)
        return StrategySignal(symbol, self.name, "hold", None, None)
