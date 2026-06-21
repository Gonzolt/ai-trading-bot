from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis

from quantumtrader.config import load_settings
from quantumtrader.database import db
from quantumtrader.execution.brokers import AlpacaBroker, PaperBroker
from quantumtrader.models import Order
from quantumtrader.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class BrokerService:
    """Execution gateway – routes orders to paper or live brokers."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.paper = PaperBroker()
        self.alpaca = AlpacaBroker()
        self.redis: aioredis.Redis | None = None

    def _select_broker(self):
        if self.settings.live_trading_enabled:
            return self.alpaca
        return self.paper

    async def execute_order(self, payload: dict) -> None:
        order = Order(
            symbol=payload["symbol"],
            side=payload["side"],
            quantity=float(payload["quantity"]),
            price=float(payload["price"]),
            strategy=payload.get("strategy"),
        )
        broker = self._select_broker()
        for attempt in range(3):
            try:
                trade = await broker.submit_order(order)
                await db.execute(
                    """
                    INSERT INTO trades (symbol, strategy, side, quantity, price, commission, slippage, mode)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    trade.symbol,
                    trade.strategy,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.commission,
                    trade.slippage,
                    trade.mode,
                )
                logger.info("executed %s %s qty=%.4f", trade.side, trade.symbol, trade.quantity)
                return
            except Exception:
                logger.exception("order attempt %d failed", attempt + 1)
                await asyncio.sleep(2 ** attempt)

    async def run_loop(self) -> None:
        broker = self._select_broker()
        await broker.reconcile_positions()
        self.redis = aioredis.from_url(self.settings.redis_url)
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("approved_orders")
        logger.info("broker listening on approved_orders channel")
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                payload = json.loads(message["data"])
                await self.execute_order(payload)
            except Exception:
                logger.exception("execution failed")


async def main() -> None:
    configure_logging()
    await BrokerService().run_loop()


if __name__ == "__main__":
    asyncio.run(main())
