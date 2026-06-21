from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from quantumtrader.models import StrategySignal


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str

    @abstractmethod
    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        """Generate a buy/sell/hold signal from OHLCV history."""
        raise NotImplementedError
