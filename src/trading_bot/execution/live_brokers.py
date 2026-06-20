from __future__ import annotations

import os

from trading_bot.models import Order


class AlpacaBroker:
    def __init__(self) -> None:
        if os.getenv("ENABLE_LIVE_TRADING", "false").lower() != "true":
            raise RuntimeError("Live trading is disabled. Set ENABLE_LIVE_TRADING=true intentionally.")
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=os.getenv("ALPACA_PAPER", "true").lower() == "true",
        )

    def submit_order(self, order: Order):
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        request = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return self.client.submit_order(order_data=request)
