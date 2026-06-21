from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pandas as pd

from quantumtrader.config import load_settings
from quantumtrader.database import db
from quantumtrader.selection.scoring import AssetScorer
from quantumtrader.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class AssetScorerService:
    """Daily composite ranking of all assets."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.scorer = AssetScorer()

    async def _load_all_history(self) -> dict[str, pd.DataFrame]:
        symbols = self.settings.get("universe.symbols", [])
        history: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            rows = await db.fetch(
                "SELECT time, open, high, low, close, volume FROM ohlcv_1d WHERE symbol = $1 ORDER BY time",
                symbol,
            )
            if not rows:
                continue
            df = pd.DataFrame([dict(r) for r in rows])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            history[symbol] = df.set_index("time")
        return history

    async def run_scoring(self) -> None:
        history = await self._load_all_history()
        rankings = self.scorer.score(history)
        today = datetime.now(UTC).date().isoformat()
        payload = [
            {"symbol": r.symbol, "score": r.score, "rank": r.rank, "components": r.components}
            for r in rankings
        ]
        await db.save_rankings(today, payload)
        logger.info("saved %d asset rankings for %s", len(payload), today)

    async def run_loop(self) -> None:
        while True:
            try:
                await self.run_scoring()
            except Exception:
                logger.exception("asset scoring failed")
            await asyncio.sleep(86400)


async def main() -> None:
    configure_logging()
    await AssetScorerService().run_loop()


if __name__ == "__main__":
    asyncio.run(main())
