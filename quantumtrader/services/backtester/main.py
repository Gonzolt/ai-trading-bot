from __future__ import annotations

import asyncio
import json
import logging

import pandas as pd

from quantumtrader.backtest.backtester import VectorizedBacktester
from quantumtrader.backtest.walk_forward import walk_forward_validate
from quantumtrader.config import load_settings
from quantumtrader.database import db
from quantumtrader.strategies.mean_reversion import MeanReversionStrategy
from quantumtrader.strategies.momentum import MomentumStrategy
from quantumtrader.strategies.trend import TrendFollowingStrategy
from quantumtrader.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class BacktesterService:
    """Scheduled backtesting and walk-forward optimization."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.strategies = [
            TrendFollowingStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
        ]
        self.backtester = VectorizedBacktester(
            initial_cash=float(self.settings.get("backtest.initial_cash", 100000)),
            commission_bps=float(self.settings.get("backtest.commission_bps", 1.0)),
        )

    async def _load_history(self, symbol: str) -> pd.DataFrame:
        rows = await db.fetch(
            "SELECT time, open, high, low, close, volume FROM ohlcv_1d WHERE symbol = $1 ORDER BY time",
            symbol,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time")

    async def run_backtests(self) -> None:
        symbols = self.settings.get("universe.symbols", [])
        min_oos = float(self.settings.get("backtest.walk_forward.min_oos_sharpe_ratio", 0.50))
        for symbol in symbols:
            df = await self._load_history(symbol)
            if len(df) < 200:
                continue
            for strategy in self.strategies:
                result = self.backtester.run(symbol, df, strategy)
                wf = walk_forward_validate(symbol, df, strategy, min_oos)
                await db.execute(
                    """
                    INSERT INTO backtest_runs
                    (symbol, strategy, sharpe, sortino, max_drawdown, win_rate, profit_factor, calmar, passed_walk_forward, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                    """,
                    symbol,
                    strategy.name,
                    result.metrics["sharpe"],
                    result.metrics["sortino"],
                    result.metrics["max_drawdown"],
                    result.metrics["win_rate"],
                    result.metrics["profit_factor"],
                    result.metrics["calmar"],
                    wf.passed,
                    json.dumps({"validation_sharpe": wf.validation_sharpe, "test_sharpe": wf.test_sharpe}),
                )
                logger.info(
                    "backtest %s/%s sharpe=%.2f passed_wf=%s",
                    symbol,
                    strategy.name,
                    result.metrics["sharpe"],
                    wf.passed,
                )

    async def run_loop(self) -> None:
        while True:
            try:
                await self.run_backtests()
            except Exception:
                logger.exception("backtest run failed")
            await asyncio.sleep(604800)


async def main() -> None:
    configure_logging()
    await BacktesterService().run_loop()


if __name__ == "__main__":
    asyncio.run(main())
