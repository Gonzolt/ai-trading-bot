from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from trading_bot.config import load_settings


class Database:
    def __init__(self, dsn: str | None = None) -> None:
        settings = load_settings()
        self._dsn = dsn or settings.database_url
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        async with self._lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=10)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        await self.connect()
        assert self._pool is not None
        return list(await self._pool.fetch(query, *args))

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        await self.connect()
        assert self._pool is not None
        return await self._pool.fetchrow(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        await self.connect()
        assert self._pool is not None
        return await self._pool.execute(query, *args)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        await self.connect()
        assert self._pool is not None
        async with self._pool.acquire() as connection:
            yield connection


db = Database()
