from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from trading_bot.config import load_settings
from trading_bot.models import Order, Position, Trade


@dataclass
class PaperBroker:
    starting_equity: float | None = None
    commission_bps: float | None = None
    slippage_bps: float | None = None
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)

    def __post_init__(self) -> None:
        settings = load_settings()
        self.starting_equity = float(
            self.starting_equity or settings.get("broker.paper.starting_equity", 100000)
        )
        self.commission_bps = float(
            self.commission_bps or settings.get("broker.paper.commission_bps", 1.0)
        )
        self.slippage_bps = float(self.slippage_bps or settings.get("broker.paper.slippage_bps", 2.0))
        self.cash = self.starting_equity

    def mark_to_market(self, marks: dict[str, float]) -> float:
        for symbol, price in marks.items():
            if symbol in self.positions:
                self.positions[symbol].mark_price = price
        return self.equity

    @property
    def equity(self) -> float:
        return self.cash + sum(position.market_value for position in self.positions.values())

    def submit_order(self, order: Order) -> Trade:
        slip = order.price * self.slippage_bps / 10000
        fill_price = order.price + slip if order.side == "buy" else order.price - slip
        notional = order.quantity * fill_price
        commission = notional * self.commission_bps / 10000
        realized_pnl = 0.0

        if order.side == "buy":
            if notional + commission > self.cash:
                raise ValueError("paper broker: insufficient cash")
            current = self.positions.get(order.symbol)
            if current:
                new_qty = current.quantity + order.quantity
                current.average_price = (
                    current.average_price * current.quantity + notional
                ) / new_qty
                current.quantity = new_qty
                current.mark_price = fill_price
            else:
                self.positions[order.symbol] = Position(
                    order.symbol,
                    order.quantity,
                    fill_price,
                    fill_price,
                    strategy=order.strategy,
                    stop=order.stop,
                    target=order.target,
                )
            self.cash -= notional + commission
        else:
            current = self.positions.get(order.symbol)
            if current is None or current.quantity < order.quantity:
                raise ValueError("paper broker: insufficient position")
            realized_pnl = order.quantity * (fill_price - current.average_price) - commission
            current.quantity -= order.quantity
            current.mark_price = fill_price
            if current.quantity == 0:
                del self.positions[order.symbol]
            self.cash += notional - commission

        trade = Trade(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=abs(slip * order.quantity),
            realized_pnl=realized_pnl,
            created_at=datetime.now(timezone.utc),
            mode="paper",
            strategy=order.strategy,
        )
        self.trades.append(trade)
        return trade
