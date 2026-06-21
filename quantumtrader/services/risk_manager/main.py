from __future__ import annotations

import asyncio
import json
import logging

import pandas as pd
import redis.asyncio as aioredis

from quantumtrader.config import load_settings
from quantumtrader.database import db
from quantumtrader.models import Order, Position
from quantumtrader.risk.manager import RiskManager
from quantumtrader.strategies.indicators import atr
from quantumtrader.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class RiskManagerService:
    """Central risk controller – validates signals before broker execution."""

    def __init__(self) -> None:
        self.settings = load_settings()
        self.risk = RiskManager()
        self.redis: aioredis.Redis | None = None
        self.peak_equity = float(self.settings.get("broker.paper.starting_equity", 100000))
        self.start_of_day_equity = self.peak_equity

    async def _portfolio_state(self) -> tuple[float, float, list[Position]]:
        equity_row = await db.fetchrow(
            "SELECT equity, cash FROM equity_curve ORDER BY time DESC LIMIT 1"
        )
        equity = float(equity_row["equity"]) if equity_row else self.peak_equity
        cash = float(equity_row["cash"]) if equity_row else equity
        pos_rows = await db.fetch(
            "SELECT symbol, quantity, average_price, mark_price FROM positions"
        )
        positions = [
            Position(
                symbol=r["symbol"],
                quantity=float(r["quantity"]),
                average_price=float(r["average_price"]),
                mark_price=float(r["mark_price"]),
            )
            for r in pos_rows
        ]
        return equity, cash, positions

    async def process_signal(self, payload: dict) -> None:
        symbol = payload["symbol"]
        side = payload["signal"]
        if side not in ("buy", "sell"):
            return

        equity, cash, positions = await self._portfolio_state()
        self.peak_equity = max(self.peak_equity, equity)
        halt = self.risk.should_halt(equity, self.start_of_day_equity, self.peak_equity)
        if halt:
            logger.warning("trading halted: %s", halt)
            if self.redis:
                await self.redis.publish("risk_status", json.dumps({"halted": True, "reason": halt}))
            return

        rows = await db.fetch(
            "SELECT time, open, high, low, close, volume FROM ohlcv_1d WHERE symbol = $1 ORDER BY time",
            symbol,
        )
        if not rows:
            return
        df = pd.DataFrame([dict(r) for r in rows])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
        price = float(df["close"].iloc[-1])
        atr_value = float(atr(df).iloc[-1])

        order = Order(symbol=symbol, side=side, quantity=1.0, price=price, strategy=payload.get("strategy"))
        decision = self.risk.validate_order(order, equity, cash, positions, atr_value)
        if not decision.accepted:
            logger.info("order rejected: %s", decision.reason)
            return

        if self.redis:
            await self.redis.publish(
                "approved_orders",
                json.dumps({
                    "symbol": symbol,
                    "side": side,
                    "quantity": decision.quantity,
                    "price": price,
                    "strategy": payload.get("strategy"),
                }),
            )

    async def run_loop(self) -> None:
        self.redis = aioredis.from_url(self.settings.redis_url)
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("signals")
        logger.info("risk manager listening on signals channel")
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                payload = json.loads(message["data"])
                await self.process_signal(payload)
            except Exception:
                logger.exception("risk processing failed")


async def main() -> None:
    configure_logging()
    await RiskManagerService().run_loop()


if __name__ == "__main__":
    asyncio.run(main())
