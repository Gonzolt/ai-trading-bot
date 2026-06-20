from __future__ import annotations

import pandas as pd

from trading_bot.config import load_settings
from trading_bot.models import StrategySignal
from trading_bot.strategies.base import Strategy
from trading_bot.strategies.indicators import bollinger_bands


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(self) -> None:
        settings = load_settings()
        self.window = int(settings.get("strategy.mean_reversion.window", 20))
        self.stddev = float(settings.get("strategy.mean_reversion.stddev", 2.0))
        self.z_entry = float(settings.get("strategy.mean_reversion.z_entry", -2.0))

    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        if len(asset_df) < self.window + 5:
            return StrategySignal(symbol, self.name, "hold", None, None)
        lower, mid, upper, zscore = bollinger_bands(asset_df["close"], self.window, self.stddev)
        price = float(asset_df["close"].iloc[-1])
        band_width = float(upper.iloc[-1] - lower.iloc[-1])
        if zscore.iloc[-1] < self.z_entry and price <= lower.iloc[-1]:
            return StrategySignal(
                symbol,
                self.name,
                "buy",
                price - 2 * band_width,
                float(mid.iloc[-1]),
                confidence=min(1.0, abs(float(zscore.iloc[-1])) / 4),
            )
        if price >= mid.iloc[-1]:
            return StrategySignal(symbol, self.name, "sell", None, None, confidence=0.5)
        return StrategySignal(symbol, self.name, "hold", None, None)
