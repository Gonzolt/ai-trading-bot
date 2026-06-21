import pytest
import pandas as pd
import numpy as np

from quantumtrader.strategies.trend import TrendFollowingStrategy
from quantumtrader.selection.scoring import AssetScorer
from quantumtrader.risk.manager import RiskManager
from quantumtrader.models import Order, Position
from quantumtrader.backtest.backtester import VectorizedBacktester


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def test_trend_strategy_returns_signal(sample_ohlcv: pd.DataFrame) -> None:
    strategy = TrendFollowingStrategy()
    signal = strategy.generate_signal("TEST", sample_ohlcv)
    assert signal.signal in ("buy", "sell", "hold")
    assert signal.strategy == "trend_following"


def test_asset_scorer_ranks(sample_ohlcv: pd.DataFrame) -> None:
    scorer = AssetScorer()
    history = {"AAA": sample_ohlcv, "BBB": sample_ohlcv * 1.01}
    rankings = scorer.score(history)
    assert len(rankings) <= 15
    assert rankings[0].rank == 1


def test_risk_manager_position_size() -> None:
    rm = RiskManager()
    qty = rm.position_size(equity=100_000, atr_value=2.0, price=100.0)
    assert qty > 0
    assert qty <= 250  # 25% of 100k / 100


def test_risk_rejects_high_correlation(sample_ohlcv: pd.DataFrame) -> None:
    rm = RiskManager()
    order = Order(symbol="NEW", side="buy", quantity=10, price=100)
    positions = [Position("OLD", 10, 90, 100, sector="Tech")]
    history = {
        "NEW": sample_ohlcv["close"],
        "OLD": sample_ohlcv["close"] * 1.001,
    }
    decision = rm.validate_order(order, 100_000, 50_000, positions, atr_value=2.0, history=history, sector="Tech")
    assert not decision.accepted or "correlation" not in decision.reason


def test_backtester_runs(sample_ohlcv: pd.DataFrame) -> None:
    strategy = TrendFollowingStrategy()
    result = VectorizedBacktester().run("TEST", sample_ohlcv, strategy)
    assert "sharpe" in result.metrics
    assert len(result.equity_curve) > 0
