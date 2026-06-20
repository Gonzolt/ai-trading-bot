from __future__ import annotations

from trading_bot.models import Order
from trading_bot.risk.manager import RiskManager


def test_daily_loss_halt_triggers():
    manager = RiskManager()
    assert manager.should_halt(equity=97_900, start_of_day_equity=100_000, peak_equity=102_000)


def test_position_size_caps_at_max_position():
    manager = RiskManager()
    size = manager.position_size(equity=100_000, atr_value=0.50, price=100)
    assert size <= 250


def test_validate_order_accepts_sized_order():
    manager = RiskManager()
    order = Order(symbol="SPY", side="buy", quantity=1_000, price=100)
    decision = manager.validate_order(order, equity=100_000, cash=100_000, positions=[], atr_value=2.0)
    assert decision.accepted
    assert decision.quantity <= 250
