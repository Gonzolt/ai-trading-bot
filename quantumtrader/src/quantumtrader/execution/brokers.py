from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from datetime import UTC, datetime

from quantumtrader.config import load_settings
from quantumtrader.models import Order, Trade

logger = logging.getLogger(__name__)


class BrokerGateway(ABC):
    """Abstract execution gateway."""

    @abstractmethod
    async def submit_order(self, order: Order) -> Trade:
        raise NotImplementedError

    @abstractmethod
    async def reconcile_positions(self) -> None:
        raise NotImplementedError


class PaperBroker(BrokerGateway):
    """Simulated broker with configurable slippage and commission."""

    def __init__(self) -> None:
        settings = load_settings()
        self.commission_bps = float(settings.get("broker.paper.commission_bps", 1.0))
        self.slippage_bps = float(settings.get("broker.paper.slippage_bps", 2.0))
        self.mode = settings.trading_mode

    async def submit_order(self, order: Order) -> Trade:
        slip = order.price * self.slippage_bps / 10000 * random.uniform(0.5, 1.5)
        fill_price = order.price + slip if order.side == "buy" else order.price - slip
        commission = order.quantity * fill_price * self.commission_bps / 10000
        logger.info("paper fill %s %s @ %.4f", order.side, order.symbol, fill_price)
        return Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=abs(fill_price - order.price) * order.quantity,
            realized_pnl=0.0,
            created_at=datetime.now(UTC),
            mode=self.mode,
            strategy=order.strategy,
        )

    async def reconcile_positions(self) -> None:
        logger.info("paper broker reconciliation complete")


class AlpacaBroker(BrokerGateway):
    """Alpaca Markets execution for stocks/ETFs."""

    def __init__(self) -> None:
        import os

        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self._client = None

    def _get_client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        return self._client

    async def submit_order(self, order: Order) -> Trade:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        client = self._get_client()
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=order.symbol,
            qty=order.quantity,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        result = client.submit_order(req)
        fill_price = float(order.price)
        return Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=0.0,
            slippage=0.0,
            realized_pnl=0.0,
            created_at=datetime.now(UTC),
            mode="live" if not self.paper else "paper",
            strategy=order.strategy,
        )

    async def reconcile_positions(self) -> None:
        client = self._get_client()
        positions = client.get_all_positions()
        logger.info("Alpaca reconciliation: %d open positions", len(positions))
