from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

SignalSide = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_class: str
    sector: str | None = None
    exchange: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategySignal:
    symbol: str
    strategy: str
    signal: SignalSide
    stop: float | None
    target: float | None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    quantity: float
    average_price: float
    mark_price: float
    sector: str | None = None
    strategy: str | None = None
    stop: float | None = None
    target: float | None = None

    @property
    def market_value(self) -> float:
        return self.quantity * self.mark_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.mark_price - self.average_price)


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    strategy: str | None = None
    stop: float | None = None
    target: float | None = None


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    commission: float
    slippage: float
    realized_pnl: float
    created_at: datetime
    mode: str = "paper"
    strategy: str | None = None


@dataclass(frozen=True)
class RiskDecision:
    accepted: bool
    reason: str
    quantity: float = 0.0


@dataclass(frozen=True)
class RankingRow:
    symbol: str
    score: float
    rank: int
    components: dict[str, float]
