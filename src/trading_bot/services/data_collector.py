from __future__ import annotations

import asyncio
import logging

from trading_bot.config import load_settings
from trading_bot.data.providers import YFinanceProvider
from trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    settings = load_settings()
    provider = YFinanceProvider()
    symbols = settings.get("universe.symbols", [])
    days = int(settings.get("data.history_days", 800))
    while True:
        for symbol in symbols:
            try:
                df = await provider.history(symbol, days=days)
                logger.info("collected %s rows for %s", len(df), symbol)
            except Exception:
                logger.exception("failed collecting %s", symbol)
        await asyncio.sleep(60 * 60 * 24)


if __name__ == "__main__":
    asyncio.run(main())
