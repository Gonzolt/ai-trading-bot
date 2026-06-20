from __future__ import annotations

from trading_bot.models import StrategySignal
from trading_bot.strategies.base import Strategy
from trading_bot.backtest.backtester import VectorizedBacktester


class BuyHoldExitStrategy(Strategy):
    name = "test"

    def generate_signal(self, symbol, asset_df):
        if len(asset_df) == 81:
            return StrategySignal(symbol, self.name, "buy", None, None)
        if len(asset_df) == 140:
            return StrategySignal(symbol, self.name, "sell", None, None)
        return StrategySignal(symbol, self.name, "hold", None, None)


def test_backtester_produces_metrics(sample_ohlcv):
    result = VectorizedBacktester().run("AAA", sample_ohlcv, BuyHoldExitStrategy())
    assert not result.equity_curve.empty
    assert "sharpe" in result.metrics
