from __future__ import annotations

import asyncio
import logging

from trading_bot.risk.manager import RiskManager
from trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    manager = RiskManager()
    logger.info("risk manager online: daily_loss=%.2f%%", manager.max_daily_loss_pct * 100)
    while True:
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
