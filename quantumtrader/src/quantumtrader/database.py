from __future__ import annotations

import json
from typing import Any

import asyncpg

from quantumtrader.config import load_settings


class Database:
    """Async PostgreSQL/TimescaleDB access layer."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            settings = load_settings()
            self._pool = await asyncpg.create_pool(settings.database_url, min_size=2, max_size=10)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def upsert_ohlcv(self, symbol: str, rows: list[dict[str, Any]], source: str) -> int:
        """Insert or update daily OHLCV bars."""
        if not rows:
            return 0
        await self.connect()
        assert self._pool is not None
        sql = """
            INSERT INTO ohlcv_1d (time, symbol, open, high, low, close, adjusted_close, volume, source)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (time, symbol) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adjusted_close = EXCLUDED.adjusted_close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
        """
        count = 0
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                for row in rows:
                    await conn.execute(
                        sql,
                        row["time"],
                        symbol,
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row.get("adjusted_close", row["close"]),
                        row.get("volume", 0.0),
                        source,
                    )
                    count += 1
        return count

    async def save_rankings(self, ranking_date: str, rankings: list[dict[str, Any]]) -> None:
        """Persist daily asset rankings."""
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM asset_rankings WHERE ranking_date = $1::date",
                    ranking_date,
                )
                for row in rankings:
                    await conn.execute(
                        """
                        INSERT INTO asset_rankings (ranking_date, symbol, score, components, rank)
                        VALUES ($1::date, $2, $3, $4::jsonb, $5)
                        """,
                        ranking_date,
                        row["symbol"],
                        row["score"],
                        json.dumps(row["components"]),
                        row["rank"],
                    )


db = Database()
