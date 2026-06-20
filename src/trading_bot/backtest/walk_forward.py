from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.backtest.backtester import VectorizedBacktester
from trading_bot.strategies.base import Strategy


@dataclass(frozen=True)
class WalkForwardReport:
    in_sample_sharpe: float
    validation_sharpe: float
    test_sharpe: float
    passed: bool


def walk_forward_validate(
    symbol: str,
    df: pd.DataFrame,
    strategy: Strategy,
    min_oos_ratio: float = 0.50,
) -> WalkForwardReport:
    n = len(df)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)
    backtester = VectorizedBacktester()
    train = backtester.run(symbol, df.iloc[:train_end], strategy).metrics
    validation = backtester.run(symbol, df.iloc[train_end:validation_end], strategy).metrics
    test = backtester.run(symbol, df.iloc[validation_end:], strategy).metrics
    is_sharpe = train["sharpe"]
    oos_floor = abs(is_sharpe) * min_oos_ratio
    passed = validation["sharpe"] >= oos_floor and test["sharpe"] >= oos_floor
    return WalkForwardReport(is_sharpe, validation["sharpe"], test["sharpe"], passed)
