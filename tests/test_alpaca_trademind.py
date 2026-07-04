import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import alpaca_trademind as app


def market_frame(symbol: str, seed: int = 42, rows: int = 620) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.008, rows)
    close = 120.0 * np.exp(np.cumsum(returns))
    open_price = close * (1 + rng.normal(0, 0.0015, rows))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.0002, 0.004, rows))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.0002, 0.004, rows))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=rows, freq="D", tz="UTC"),
            "symbol": symbol,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100_000, 2_000_000, rows),
        }
    )


def test_features_and_future_labels_are_well_formed():
    featured = app.engineer_features(market_frame("AAPL"))
    assert set(app.FEATURE_COLUMNS).issubset(featured.columns)
    assert featured["target"].dropna().isin([0, 1, 2]).all()
    assert featured["target"].tail(app.FORWARD_HORIZON).isna().all()


def test_chronological_windows_and_training_only_scaler():
    frames = {
        "AAPL": market_frame("AAPL", 1),
        "MSFT": market_frame("MSFT", 2),
        "GOOGL": market_frame("GOOGL", 3),
    }
    bundle = app.build_training_dataset(frames)
    assert bundle.x_train.shape[1] == app.LOOKBACK * len(app.FEATURE_COLUMNS) == 300
    assert bundle.train_rows > bundle.validation_rows > 0
    assert set(bundle.class_counts) == {0, 1, 2}
    assert np.allclose(bundle.x_train.mean(axis=0), 0.0, atol=1e-4)


def test_cache_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app, "CACHE_DIR", tmp_path)
    original = market_frame("AAPL")
    path = app.write_cached_bars(original, "AAPL", "1Day")
    loaded = app.read_cached_bars("AAPL", "1Day")
    assert path.exists()
    assert len(loaded) == len(original)
    assert list(loaded.columns[:7]) == [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


def test_shallow_network_shape_and_backpropagation():
    model = app.ShallowTradeNet(300)
    inputs = torch.randn(16, 300)
    labels = torch.randint(0, 3, (16,))
    logits = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert logits.shape == (16, 3)
    assert torch.isfinite(loss)


def test_missing_credentials_fail_closed(monkeypatch):
    for name in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    try:
        app.api_credentials()
    except app.UserFacingError as exc:
        assert "credentials are missing" in str(exc)
    else:
        raise AssertionError("Missing credentials should fail closed")


def test_missing_massive_key_fails_but_binance_needs_no_credentials(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    try:
        app.massive_api_key()
    except app.UserFacingError as exc:
        assert "Massive API key is missing" in str(exc)
    else:
        raise AssertionError("A stock download must require a Massive key")


def test_massive_aggregate_response_is_normalised():
    payload = {
        "status": "OK",
        "results": [
            {
                "t": 1_735_689_600_000,
                "o": 100.0,
                "h": 103.0,
                "l": 99.0,
                "c": 102.0,
                "v": 123_456,
                "n": 900,
                "vw": 101.5,
            }
        ],
    }
    frame = app.massive_results_frame(payload, "AAPL")
    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "AAPL"
    assert frame.loc[0, "close"] == 102.0
    assert frame.loc[0, "timestamp"].tzinfo is not None


def test_binance_archive_supports_microsecond_timestamps():
    row = (
        "1735689600000000,42000,43000,41000,42500,12.5,1735689659999999,"
        "530000,1000,7.0,297000,0\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTCUSDT-1m-2025-01-01.csv", row)
    frame = app.binance_zip_frame(buffer.getvalue(), "BTCUSDT")
    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "BTCUSDT"
    assert frame.loc[0, "close"] == 42500.0
    assert frame.loc[0, "timestamp"].year == 2025
