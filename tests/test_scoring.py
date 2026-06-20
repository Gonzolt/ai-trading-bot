from __future__ import annotations

from trading_bot.selection.scoring import AssetScoringEngine


def test_scoring_returns_ranked_rows(sample_ohlcv):
    history = {
        "AAA": sample_ohlcv,
        "BBB": sample_ohlcv.assign(close=sample_ohlcv["close"] * 0.98),
    }
    rows = AssetScoringEngine().score(history, sentiment={"AAA": 0.4, "BBB": -0.2})
    assert rows
    assert rows[0].rank == 1
    assert 0 <= rows[0].score <= 100
