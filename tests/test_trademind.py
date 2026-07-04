import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import trademind as app


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
    vector = app.model_feature_vector(featured, 100)
    assert vector is not None
    assert vector.shape == (20,)
    assert featured["target"].dropna().isin([0, 1, 2]).all()
    assert featured["target"].tail(app.FORWARD_HORIZON).isna().all()


def test_target_uses_exact_forward_horizon_not_intermediate_maximum():
    frame = market_frame("BTCUSDT")
    index = 100
    frame.loc[index, "close"] = 100.0
    frame.loc[index + 1, "close"] = 110.0
    frame.loc[index + app.FORWARD_HORIZON, "close"] = 90.0
    featured = app.engineer_features(frame, move_threshold=0.002)
    assert featured.loc[index, "target"] == 2


def test_vectorized_windows_match_single_window_feature_builder():
    frame = market_frame("AAPL")
    featured = app.engineer_features(frame).reset_index(drop=True)
    expected = app.model_feature_vector(
        featured, len(featured) - app.FORWARD_HORIZON - 1
    )
    windows, _ = app.symbol_windows(frame)
    assert expected is not None
    assert np.allclose(windows[-1], expected, rtol=1e-5, atol=1e-7)


def test_chronological_windows_and_training_only_scaler():
    frames = {
        "AAPL": market_frame("AAPL", 1),
        "MSFT": market_frame("MSFT", 2),
        "GOOGL": market_frame("GOOGL", 3),
    }
    bundle = app.build_training_dataset(frames)
    assert bundle.x_train.shape[1] == len(app.MODEL_FEATURES) == 20
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


def test_live_snapshots_are_resampled_to_model_timeframe():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:05Z", "2026-01-01T00:00:45Z", "2026-01-01T00:01:10Z"]
            ),
            "symbol": ["BTCUSDT"] * 3,
            "open": [100.0, 100.0, 101.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 98.0, 100.0],
            "close": [100.5, 101.0, 102.0],
            "volume": [10.0, 12.0, 4.0],
        }
    )
    result = app.resample_live_bars(frame, "BTCUSDT", "1Min")
    assert len(result) == 2
    assert result.iloc[0]["high"] == 102.0
    assert result.iloc[0]["low"] == 98.0
    assert result.iloc[0]["close"] == 101.0
    assert result.iloc[0]["volume"] == 12.0


def test_shallow_network_shape_and_backpropagation():
    model = app.ShallowTradeNet(20)
    inputs = torch.randn(16, 20)
    labels = torch.randint(0, 3, (16,))
    logits = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert logits.shape == (16, 3)
    assert torch.isfinite(loss)


def test_checkpoint_is_validated_and_replaced_atomically(tmp_path: Path):
    path = tmp_path / "trade_model.pt"
    model = app.ShallowTradeNet(len(app.MODEL_FEATURES))
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_size": len(app.MODEL_FEATURES),
        "feature_columns": app.MODEL_FEATURES,
        "mean": np.zeros(len(app.MODEL_FEATURES), dtype=np.float32),
        "std": np.ones(len(app.MODEL_FEATURES), dtype=np.float32),
    }
    app.save_checkpoint_atomic(checkpoint, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["input_size"] == len(app.MODEL_FEATURES)
    assert not path.with_suffix(".pt.tmp").exists()

    original = path.read_bytes()
    invalid = dict(checkpoint, input_size=999)
    with pytest.raises(app.UserFacingError, match="unexpected input size"):
        app.save_checkpoint_atomic(invalid, path)
    assert path.read_bytes() == original


def test_cashier_sector_mapping_recognises_alpaca_crypto_symbols():
    assert app.CashierAgent.position_sector("BTC/USD") == "Crypto"
    assert app.CashierAgent.position_sector("ETHUSD") == "Crypto"
    assert app.CashierAgent.position_sector("AAPL") == app.SECTOR_MAP["AAPL"]


def test_macro_f1_and_confusion_matrix_are_correct():
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 1, 2, 2, 2])
    accuracy, macro_f1, confusion = app.classification_metrics(labels, predictions)
    assert accuracy == 5 / 6
    assert 0.82 < macro_f1 < 0.83
    assert confusion == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]


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


def test_sqlite_agent_schema_uses_wal(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "trademind.db")
    with database.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    expected = {
        "live_prices",
        "research_findings",
        "asset_rankings",
        "investment_decisions",
        "portfolio_log",
        "training_log",
        "agent_logs",
    }
    tables = {
        row["name"]
        for row in database.query("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert expected.issubset(tables)


def test_database_helpers_release_windows_file_handles(tmp_path: Path):
    path = tmp_path / "releasable.db"
    database = app.AgentDatabase(path)
    database.log("test", "connection should close")
    assert database.query("SELECT COUNT(*) AS count FROM agent_logs")[0]["count"] == 1
    path.unlink()
    assert not path.exists()


def test_training_log_schema_includes_quality_and_scheduler_metrics(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "metrics.db")
    columns = {
        row["name"] for row in database.query("PRAGMA table_info(training_log)")
    }
    assert {"macro_f1", "learning_rate"}.issubset(columns)


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
