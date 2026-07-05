import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import trademind as app


def insert_decision(
    database: app.AgentDatabase,
    symbol: str,
    asset_type: str,
    *,
    price: float = 100.0,
    atr: float = 2.0,
    status: str = "pending",
    client_order_id: str = "",
) -> int:
    return database.execute(
        "INSERT INTO investment_decisions"
        "(symbol,asset_type,timestamp,signal,probability,price,atr,stop_loss,"
        "take_profit,status,client_order_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            symbol,
            asset_type,
            datetime.now(timezone.utc).isoformat(),
            "BUY",
            0.8,
            price,
            atr,
            price - 2 * atr,
            price + 3 * atr,
            status,
            client_order_id,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


class FakeTradingClient:
    def __init__(self, *, market_open: bool = True, equity: float = 100_000.0):
        self.market_open = market_open
        self.equity = equity
        self.submitted = []
        self.closed = []
        self.clock_calls = 0
        self.order = SimpleNamespace(id="order-1")

    def get_account(self):
        return SimpleNamespace(equity=str(self.equity), trading_blocked=False)

    def get_clock(self):
        self.clock_calls += 1
        return SimpleNamespace(
            is_open=self.market_open,
            next_open=datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc),
        )

    def get_all_positions(self):
        return []

    def get_orders(self, filter=None):
        return []

    def submit_order(self, order_data):
        self.submitted.append(order_data)
        return self.order

    def close_position(self, symbol):
        self.closed.append(symbol)
        return self.order


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


def test_training_windows_do_not_cross_timestamp_gaps():
    contiguous = market_frame("BTCUSDT")
    gapped = contiguous.copy()
    gapped.loc[300:, "timestamp"] += pd.Timedelta(days=30)
    contiguous_windows, _ = app.symbol_windows(contiguous)
    gapped_windows, _ = app.symbol_windows(gapped)
    assert len(gapped_windows) <= len(contiguous_windows) - app.LOOKBACK


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
    assert result.iloc[0]["volume"] == 22.0


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
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["input_size"] == len(app.MODEL_FEATURES)
    restored = app.ShallowTradeNet(loaded["input_size"])
    restored.load_state_dict(loaded["state_dict"])
    assert restored(torch.randn(1, loaded["input_size"])).shape == (1, 3)
    assert not path.with_suffix(".pt.tmp").exists()

    original = path.read_bytes()
    invalid = dict(checkpoint, input_size=999)
    with pytest.raises(app.UserFacingError, match="unexpected input size"):
        app.save_checkpoint_atomic(invalid, path)
    assert path.read_bytes() == original


def test_cashier_sector_mapping_recognises_crypto_and_database_sector(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "sectors.db")
    database.execute(
        "INSERT INTO universe_assets(symbol,asset_type,name,sector,updated_at) "
        "VALUES(?,?,?,?,?)",
        ("MSFT", app.ASSET_STOCK, "Microsoft", "Information Technology", "now"),
    )
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    assert cashier.position_sector("BTC/USD") == "Crypto"
    assert cashier.position_sector("ETHUSD") == "Crypto"
    assert cashier.position_sector("MSFT") == "Information Technology"


def test_stock_decision_waits_while_market_is_closed(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "closed.db")
    decision_id = insert_decision(database, "AAPL", app.ASSET_STOCK)
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    cashier.client = FakeTradingClient(market_open=False)
    cashier.start_equity = cashier.peak_equity = 100_000.0
    cashier.trained_symbols = {"AAPL"}
    cashier.trained_asset = app.ASSET_STOCK
    cashier.process_pending()
    decision = database.query(
        "SELECT status FROM investment_decisions WHERE id=?", (decision_id,)
    )[0]
    assert decision["status"] == "pending"
    assert not cashier.client.submitted


def test_stale_pending_decision_expires_without_submission(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "stale-decision.db")
    decision_id = insert_decision(database, "BTCUSDT", app.ASSET_CRYPTO)
    database.execute(
        "UPDATE investment_decisions SET created_at=? WHERE id=?",
        ("2020-01-01T00:00:00+00:00", decision_id),
    )
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    cashier.client = FakeTradingClient()
    cashier.start_equity = cashier.peak_equity = 100_000.0
    cashier.trained_symbols = {"BTCUSDT"}
    cashier.trained_asset = app.ASSET_CRYPTO
    cashier.process_pending()
    status = database.query(
        "SELECT status FROM investment_decisions WHERE id=?", (decision_id,)
    )[0]["status"]
    assert status == "expired"
    assert not cashier.client.submitted


def test_crypto_decision_does_not_use_stock_market_clock(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "crypto-clock.db")
    insert_decision(database, "BTCUSDT", app.ASSET_CRYPTO, price=50_000.0, atr=500.0)
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    cashier.client = FakeTradingClient(market_open=False)
    cashier.start_equity = cashier.peak_equity = 100_000.0
    cashier.trained_symbols = {"BTCUSDT"}
    cashier.trained_asset = app.ASSET_CRYPTO
    cashier.process_pending()
    assert cashier.client.clock_calls == 0
    assert len(cashier.client.submitted) == 1


def test_real_stock_sectors_do_not_collapse_into_other(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "sector-risk.db")
    for symbol, sector in (
        ("AAPL", "Information Technology"),
        ("JPM", "Financials"),
        ("XOM", "Energy"),
    ):
        database.execute(
            "INSERT INTO universe_assets(symbol,asset_type,name,sector,updated_at) "
            "VALUES(?,?,?,?,?)",
            (symbol, app.ASSET_STOCK, symbol, sector, "now"),
        )
    decision_id = insert_decision(database, "XOM", app.ASSET_STOCK)
    decision = database.query(
        "SELECT * FROM investment_decisions WHERE id=?", (decision_id,)
    )[0]
    client = FakeTradingClient()
    client.get_all_positions = lambda: [
        SimpleNamespace(symbol="AAPL"),
        SimpleNamespace(symbol="JPM"),
    ]
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    cashier.client = client
    cashier.start_equity = cashier.peak_equity = 100_000.0
    allowed, _ = cashier._risk_allows(decision, 100_000.0)
    assert allowed


def test_stock_buy_uses_whole_share_native_bracket(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "bracket.db")
    decision_id = insert_decision(database, "AAPL", app.ASSET_STOCK)
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    cashier.client = FakeTradingClient()
    cashier.start_equity = cashier.peak_equity = 100_000.0
    cashier.trained_symbols = {"AAPL"}
    cashier.trained_asset = app.ASSET_STOCK
    cashier.process_pending()
    assert len(cashier.client.submitted) == 1
    request = cashier.client.submitted[0]
    assert request.order_class == app.OrderClass.BRACKET
    assert float(request.qty).is_integer()
    assert request.stop_loss.stop_price == 96.0
    assert request.take_profit.limit_price == 106.0
    decision = database.query(
        "SELECT status,client_order_id FROM investment_decisions WHERE id=?",
        (decision_id,),
    )[0]
    assert decision["status"] == "submitted"
    assert decision["client_order_id"].startswith("korvax-")


def test_stock_bracket_rejects_sub_one_share_sizing(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "small.db")
    decision_id = insert_decision(database, "AAPL", app.ASSET_STOCK, price=1_000.0)
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    cashier.client = FakeTradingClient(equity=1_000.0)
    cashier.start_equity = cashier.peak_equity = 1_000.0
    cashier.trained_symbols = {"AAPL"}
    cashier.trained_asset = app.ASSET_STOCK
    cashier.process_pending()
    decision = database.query(
        "SELECT status,note FROM investment_decisions WHERE id=?", (decision_id,)
    )[0]
    assert decision["status"] == "rejected"
    assert "whole share" in decision["note"]
    assert not cashier.client.submitted


def test_rejected_order_is_reconciled_without_close_attempt(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "reconcile.db")
    decision_id = insert_decision(
        database,
        "BTCUSDT",
        app.ASSET_CRYPTO,
        status="submitted",
        client_order_id="korvax-test",
    )
    client = FakeTradingClient()
    client.get_order_by_client_id = lambda _client_id: SimpleNamespace(
        status=app.OrderStatus.REJECTED,
        filled_qty="0",
        reject_reason="insufficient buying power",
    )
    cashier = app.CashierAgent(database, app.queue.Queue(), app.threading.Event())
    cashier.client = client
    cashier.reconcile_orders()
    decision = database.query(
        "SELECT status,note FROM investment_decisions WHERE id=?", (decision_id,)
    )[0]
    assert decision["status"] == "rejected"
    assert "insufficient buying power" in decision["note"]
    assert not client.closed


def test_us_risk_day_rolls_at_new_york_market_open():
    before_open = datetime(2026, 1, 5, 14, 29, tzinfo=timezone.utc)
    at_open = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
    assert app.us_trading_day(before_open).isoformat() == "2026-01-02"
    assert app.us_trading_day(at_open).isoformat() == "2026-01-05"


def test_investor_skips_symbol_outside_trained_universe(tmp_path: Path):
    database = app.AgentDatabase(tmp_path / "trained-only.db")
    database.execute(
        "INSERT INTO asset_rankings VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            "TSLA",
            app.ASSET_STOCK,
            "now",
            99.0,
            1.0,
            1.0,
            30.0,
            0.01,
            0.5,
            1.0,
            250.0,
        ),
    )
    investor = app.InvestorAgent(
        database, app.queue.Queue(), app.threading.Event()
    )
    investor.checkpoint = {
        "symbols": ["AAPL"],
        "asset_class": app.ASSET_STOCK,
        "mean": [0.0] * len(app.MODEL_FEATURES),
        "std": [1.0] * len(app.MODEL_FEATURES),
        "timeframe": "1Min",
    }
    investor.model = object()
    investor.evaluate_rankings()
    assert not database.query("SELECT 1 FROM investment_decisions")


def test_universe_refresh_preserves_user_crypto_watchlist(tmp_path: Path, monkeypatch):
    database = app.AgentDatabase(tmp_path / "watchlist.db")
    watchlist = app.WatchlistState()
    watchlist.crypto = ["CUSTOMUSDT"]
    sp500 = pd.DataFrame(
        {
            "Symbol": ["AAPL"],
            "Security": ["Apple"],
            "GICS Sector": ["Information Technology"],
        }
    )

    class Response:
        def __init__(self, *, text="", payload=None):
            self.text = text
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, **_kwargs):
        if "wikipedia" in url:
            return Response(text="table")
        return Response(payload=[{"symbol": "btc", "name": "Bitcoin"}])

    monkeypatch.setattr(app, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(app.pd, "read_html", lambda _source: [sp500])
    analyst = app.AnalystAgent(
        database, app.queue.Queue(), app.threading.Event(), watchlist
    )
    analyst.refresh_universe()
    assert watchlist.crypto == ["CUSTOMUSDT"]
    sector = database.query(
        "SELECT sector FROM universe_assets WHERE symbol='AAPL'"
    )[0]["sector"]
    assert sector == "Information Technology"


def test_crypto_price_errors_use_temporary_cooldown_and_vision_host(
    tmp_path: Path, monkeypatch
):
    database = app.AgentDatabase(tmp_path / "cooldown.db")
    watchlist = app.WatchlistState()
    watchlist.stocks = []
    watchlist.crypto = ["BTCUSDT"]
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [[1_735_689_600_000, "100", "102", "99", "101", "12"]]

    def failing_get(url, **_kwargs):
        calls.append(url)
        raise RuntimeError("temporary network error")

    monkeypatch.setattr(app, "requests", SimpleNamespace(get=failing_get))
    researcher = app.ResearcherAgent(
        database, app.queue.Queue(), app.threading.Event(), watchlist
    )
    researcher.fetch_prices()
    researcher.fetch_prices()
    assert len(calls) == 1
    assert "data-api.binance.vision" in calls[0]
    researcher.crypto_retry_after["BTCUSDT"] = 0.0
    monkeypatch.setattr(
        app, "requests", SimpleNamespace(get=lambda url, **kwargs: Response())
    )
    researcher.fetch_prices()
    assert database.query("SELECT 1 FROM live_prices WHERE symbol='BTCUSDT'")


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


def test_missing_binance_month_uses_daily_archive_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(app, "CACHE_DIR", tmp_path)
    calls = []

    def fake_archive(symbol, interval, cadence, stamp, events):
        calls.append((cadence, stamp))
        if cadence == "monthly":
            return None
        timestamp = pd.Timestamp(stamp, tz="UTC")
        return pd.DataFrame(
            {
                "timestamp": [timestamp],
                "symbol": [symbol],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [10.0],
            }
        )

    monkeypatch.setattr(app, "_binance_archive", fake_archive)
    result = app.sync_binance_crypto_bars("BTCUSDT", "1Min", force_refresh=True)
    assert not result.empty
    assert any(cadence == "monthly" for cadence, _ in calls)
    assert any(cadence == "daily" for cadence, _ in calls)
