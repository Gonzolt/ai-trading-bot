from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from trading_bot.models import StrategySignal


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        raise NotImplementedError
