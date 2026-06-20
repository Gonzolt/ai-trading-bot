from __future__ import annotations

import asyncio
import logging

from trading_bot.execution.paper_broker import PaperBroker
from trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging()
    broker = PaperBroker()
    logger.info("paper broker online with equity %.2f", broker.equity)
    while True:
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
