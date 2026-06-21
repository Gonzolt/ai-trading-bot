from __future__ import annotations

import asyncio
import json
import logging

import pandas as pd
import redis.asyncio as aioredis

from quantumtrader.config import load_settings
from quantumtrader.database import db
from quantumtrader.strategies.mean_reversion import MeanReversionStrategy
from quantumtrader.strategies.ml_predictor import MLPredictorStrategy
from quantumtrader.strategies.momentum import MomentumStrategy
from quantumtrader.strategies.trend import TrendFollowingStrategy
from quantumtrader.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class SignalEngineService:
    """Runs all strategies on selected assets and publishes signals."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.strategies = [
            TrendFollowingStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            MLPredictorStrategy(),
        ]
        self.redis: aioredis.Redis | None = None

    async def _load_history(self, symbol: str) -> pd.DataFrame:
        rows = await db.fetch(
            """
            SELECT time, open, high, low, close, volume
            FROM ohlcv_1d WHERE symbol = $1 ORDER BY time
            """,
            symbol,
        )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df.set_index("time")

    async def _get_selected_symbols(self) -> list[str]:
        rows = await db.fetch(
            """
            SELECT symbol FROM asset_rankings
            WHERE ranking_date = (SELECT max(ranking_date) FROM asset_rankings)
            ORDER BY rank LIMIT 15
            """
        )
        if rows:
            return [r["symbol"] for r in rows]
        return self.settings.get("universe.symbols", [])

    async def run_cycle(self) -> None:
        if self.redis is None:
            self.redis = aioredis.from_url(self.settings.redis_url)
        symbols = await self._get_selected_symbols()
        for symbol in symbols:
            df = await self._load_history(symbol)
            if df.empty:
                continue
            for strategy in self.strategies:
                signal = strategy.generate_signal(symbol, df)
                if signal.signal == "hold":
                    continue
                await db.execute(
                    """
                    INSERT INTO signals (symbol, strategy, signal, confidence, stop, target)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    symbol,
                    signal.strategy,
                    signal.signal,
                    signal.confidence,
                    signal.stop,
                    signal.target,
                )
                if self.redis:
                    await self.redis.publish("signals", json.dumps({
                        "symbol": symbol,
                        "strategy": signal.strategy,
                        "signal": signal.signal,
                        "confidence": signal.confidence,
                    }))
                logger.info("signal %s %s %s", symbol, signal.strategy, signal.signal)

    async def run_loop(self) -> None:
        while True:
            try:
                await self.run_cycle()
            except Exception:
                logger.exception("signal cycle failed")
            await asyncio.sleep(300)


async def main() -> None:
    configure_logging()
    await SignalEngineService().run_loop()


if __name__ == "__main__":
    asyncio.run(main())
