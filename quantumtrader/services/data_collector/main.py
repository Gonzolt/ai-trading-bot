from __future__ import annotations

import asyncio
import logging
import os

from motor.motor_asyncio import AsyncIOMotorClient

from quantumtrader.config import load_settings
from quantumtrader.data.cleaners import clean_ohlcv, to_db_rows
from quantumtrader.data.fetchers.alpha_vantage_fetcher import AlphaVantageFetcher
from quantumtrader.data.fetchers.finnhub_fetcher import FinnhubNewsFetcher, MT5Fetcher
from quantumtrader.data.fetchers.polygon_fetcher import PolygonFetcher
from quantumtrader.data.fetchers.yfinance_fetcher import YFinanceFetcher
from quantumtrader.database import db
from quantumtrader.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class DataCollectorService:
    """Fetches market data from multiple APIs and persists to TimescaleDB/MongoDB."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.fetchers = [
            YFinanceFetcher(),
            PolygonFetcher(),
            AlphaVantageFetcher(),
            MT5Fetcher(),
        ]
        self.news_fetcher = FinnhubNewsFetcher()
        self.mongo: AsyncIOMotorClient | None = None

    async def _get_mongo(self):
        if self.mongo is None:
            self.mongo = AsyncIOMotorClient(self.settings.mongo_url)
        return self.mongo[self.settings.mongo_db]

    async def collect_symbol(self, symbol: str) -> None:
        days = int(self.settings.get("data.history_days", 800))
        df = None
        source = "none"
        for fetcher in self.fetchers:
            try:
                candidate = await fetcher.fetch_history(symbol, days=days)
                if candidate is not None and not candidate.empty:
                    df = candidate
                    source = fetcher.source
                    break
            except Exception:
                logger.exception("fetcher %s failed for %s", fetcher.source, symbol)

        if df is None or df.empty:
            logger.warning("no data collected for %s", symbol)
            return

        cleaned = clean_ohlcv(df)
        rows = to_db_rows(cleaned)
        count = await db.upsert_ohlcv(symbol, rows, source)
        logger.info("stored %d bars for %s via %s", count, symbol, source)

        articles = await self.news_fetcher.fetch_news(symbol)
        if articles:
            mongo = await self._get_mongo()
            await mongo.news_articles.insert_many(articles)
            logger.info("stored %d news articles for %s", len(articles), symbol)

    async def run_loop(self) -> None:
        symbols = self.settings.get("universe.symbols", [])
        interval = int(self.settings.get("data.collection_interval_hours", 24)) * 3600
        while True:
            for symbol in symbols:
                try:
                    await self.collect_symbol(symbol)
                except Exception:
                    logger.exception("collection failed for %s", symbol)
            await asyncio.sleep(interval)


async def main() -> None:
    configure_logging()
    service = DataCollectorService()
    await service.run_loop()


if __name__ == "__main__":
    asyncio.run(main())
