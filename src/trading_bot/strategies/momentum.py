from __future__ import annotations

import pandas as pd

from trading_bot.config import load_settings
from trading_bot.models import StrategySignal
from trading_bot.strategies.base import Strategy
from trading_bot.strategies.indicators import atr, macd, rsi


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(self) -> None:
        settings = load_settings()
        self.rsi_period = int(settings.get("strategy.momentum.rsi_period", 14))
        self.stop_multiple = float(settings.get("strategy.momentum.atr_stop_multiple", 1.5))
        self.reward_risk = float(settings.get("strategy.momentum.reward_risk", 2.0))

    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        if len(asset_df) < 60:
            return StrategySignal(symbol, self.name, "hold", None, None)
        close = asset_df["close"]
        rsi_value = rsi(close, self.rsi_period)
        _, _, histogram = macd(close)
        atr_value = float(atr(asset_df).iloc[-1])
        price = float(close.iloc[-1])

        exited_oversold = rsi_value.iloc[-2] < 30 <= rsi_value.iloc[-1]
        macd_flip = histogram.iloc[-2] <= 0 < histogram.iloc[-1]
        if exited_oversold and macd_flip:
            risk = self.stop_multiple * atr_value
            return StrategySignal(
                symbol,
                self.name,
                "buy",
                price - risk,
                price + self.reward_risk * risk,
                confidence=0.65,
            )
        if histogram.iloc[-2] >= 0 > histogram.iloc[-1]:
            return StrategySignal(symbol, self.name, "sell", None, None, confidence=0.5)
        return StrategySignal(symbol, self.name, "hold", None, None)
