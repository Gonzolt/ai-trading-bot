from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from trading_bot.config import load_settings
from trading_bot.models import Order, Position, RiskDecision


class RiskManager:
    def __init__(self) -> None:
        settings = load_settings()
        self.max_daily_loss_pct = float(settings.get("risk.max_daily_loss_pct", 0.02))
        self.risk_per_trade_pct = float(settings.get("risk.risk_per_trade_pct", 0.01))
        self.atr_position_multiple = float(settings.get("risk.atr_position_multiple", 2.0))
        self.max_position_pct = float(settings.get("risk.max_position_pct", 0.25))
        self.max_assets_per_sector = int(settings.get("risk.max_assets_per_sector", 2))
        self.max_drawdown_pct = float(settings.get("risk.max_drawdown_pct", 0.20))
        self.max_leverage = float(settings.get("risk.max_leverage", 2.0))
        self.max_correlation = float(settings.get("risk.max_correlation", 0.80))
        self.correlation_lookback_days = int(settings.get("risk.correlation_lookback_days", 60))

    def should_halt(self, equity: float, start_of_day_equity: float, peak_equity: float) -> str | None:
        if start_of_day_equity <= 0:
            return "invalid start of day equity"
        daily_return = equity / start_of_day_equity - 1
        if daily_return <= -self.max_daily_loss_pct:
            return "daily loss limit reached"
        if peak_equity > 0 and (peak_equity - equity) / peak_equity > self.max_drawdown_pct:
            return "global drawdown limit reached; switch to paper trading"
        return None

    def position_size(self, equity: float, atr_value: float, price: float) -> float:
        if equity <= 0 or atr_value <= 0 or price <= 0:
            return 0.0
        risk_budget = equity * self.risk_per_trade_pct
        quantity = risk_budget / (self.atr_position_multiple * atr_value)
        max_quantity = (equity * self.max_position_pct) / price
        return max(0.0, min(quantity, max_quantity))

    def validate_order(
        self,
        order: Order,
        equity: float,
        cash: float,
        positions: list[Position],
        atr_value: float,
        history: dict[str, pd.Series] | None = None,
    ) -> RiskDecision:
        if order.side == "sell":
            return RiskDecision(True, "sell allowed", order.quantity)

        quantity = min(order.quantity, self.position_size(equity, atr_value, order.price))
        if quantity <= 0 or math.isnan(quantity):
            return RiskDecision(False, "position size is zero")

        notional = quantity * order.price
        if notional > equity * self.max_position_pct:
            return RiskDecision(False, "max position percent exceeded")
        gross_exposure = sum(abs(position.market_value) for position in positions) + notional
        if equity > 0 and gross_exposure / equity > self.max_leverage:
            return RiskDecision(False, "max leverage exceeded")
        if notional > cash:
            return RiskDecision(False, "insufficient cash")

        sectors = Counter(position.sector for position in positions if position.sector)
        requested_sector = next(
            (position.sector for position in positions if position.symbol == order.symbol),
            None,
        )
        if requested_sector and sectors[requested_sector] >= self.max_assets_per_sector:
            return RiskDecision(False, "sector concentration limit exceeded")

        if history and order.symbol in history:
            candidate_returns = history[order.symbol].pct_change().tail(self.correlation_lookback_days)
            for position in positions:
                if position.symbol not in history:
                    continue
                other_returns = history[position.symbol].pct_change().tail(self.correlation_lookback_days)
                corr = candidate_returns.corr(other_returns)
                if pd.notna(corr) and corr > self.max_correlation:
                    return RiskDecision(False, f"correlation with {position.symbol} is {corr:.2f}")

        return RiskDecision(True, "accepted", quantity)
