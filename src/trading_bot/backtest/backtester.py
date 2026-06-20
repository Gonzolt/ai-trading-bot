from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.backtest.metrics import performance_metrics
from trading_bot.strategies.base import Strategy


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


class VectorizedBacktester:
    def __init__(self, initial_cash: float = 100000, commission_bps: float = 1.0) -> None:
        self.initial_cash = initial_cash
        self.commission_bps = commission_bps

    def run(self, symbol: str, df: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        cash = self.initial_cash
        quantity = 0.0
        equity_values: list[float] = []
        equity_index = []
        trades = []

        for idx in range(80, len(df)):
            window = df.iloc[: idx + 1]
            price = float(window["close"].iloc[-1])
            signal = strategy.generate_signal(symbol, window)

            if signal.signal == "buy" and quantity == 0:
                quantity = cash / price
                commission = quantity * price * self.commission_bps / 10000
                cash -= quantity * price + commission
                trades.append({"time": window.index[-1], "side": "buy", "price": price, "pnl": 0.0})
            elif signal.signal == "sell" and quantity > 0:
                proceeds = quantity * price
                commission = proceeds * self.commission_bps / 10000
                pnl = proceeds + cash - self.initial_cash - commission
                cash += proceeds - commission
                trades.append({"time": window.index[-1], "side": "sell", "price": price, "pnl": pnl})
                quantity = 0.0

            equity_values.append(cash + quantity * price)
            equity_index.append(window.index[-1])

        equity = pd.Series(equity_values, index=equity_index, name="equity")
        trades_df = pd.DataFrame(trades)
        return BacktestResult(equity, trades_df, performance_metrics(equity, trades_df))
