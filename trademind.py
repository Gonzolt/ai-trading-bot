"""Korvax TradeMind: CPU-only multi-agent research and paper-trading desktop app."""

from __future__ import annotations

import copy
import io
import json
import math
import os
import queue
import sqlite3
import threading
import time
import tkinter as tk
import urllib.parse
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, Literal, TypeVar
from zoneinfo import ZoneInfo

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("MKL_NUM_THREADS", "20")

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import torch
from alpaca.common.enums import Sort
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)
from dotenv import load_dotenv
from defusedxml import ElementTree as ET
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, MACD
from ta.volatility import AverageTrueRange
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    import polars as pl
except ImportError:  # setup.bat installs it; fallback keeps diagnostics usable.
    pl = None

try:
    import requests
except ImportError:
    requests = None

try:
    import finnhub
except ImportError:
    finnhub = None

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


# Editable research parameters.
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
DEFAULT_CRYPTO_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
]
DEFAULT_TIMEFRAME = "1Min"  # "1Day" or "1Min"
DEFAULT_ASSET_CLASS = "stocks"
STOCK_DAILY_HISTORY_DAYS = 2 * 365
STOCK_MINUTE_HISTORY_DAYS = 90
CRYPTO_DAILY_HISTORY_DAYS = 2 * 365
CRYPTO_MINUTE_HISTORY_DAYS = 90
MASSIVE_BASE_URL = "https://api.massive.com"
BINANCE_PUBLIC_BASE_URL = "https://data.binance.vision/data/spot"
LOOKBACK = 30
FORWARD_HORIZON = 5
MOVE_THRESHOLD = 0.002  # 0.2%
CRYPTO_MINUTE_MOVE_THRESHOLD = 0.0005  # 0.05% over five one-minute bars
STOCK_DAILY_MOVE_THRESHOLD = 0.005
CRYPTO_DAILY_MOVE_THRESHOLD = 0.02
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.05
DROPOUT_RATE = 0.20
CPU_THREADS = 20
VALIDATION_FRACTION = 0.20
MIN_EPOCH_DISPLAY_SECONDS = 0.65
AUTOSAVE_EPOCHS = 10
MAX_CHART_EPOCHS = 500
RESEARCH_PRICE_SECONDS = 2
RESEARCH_NEWS_SECONDS = 5 * 60
ANALYST_SECONDS = 60
UNIVERSE_SECONDS = 60 * 60
INVESTOR_SECONDS = 5
CASHIER_SECONDS = 1
MAX_DAILY_LOSS = 0.02
MAX_GLOBAL_DRAWDOWN = 0.20
RISK_PER_TRADE = 0.01
MAX_ASSET_ALLOCATION = 0.25
MAX_POSITIONS_PER_SECTOR = 2
MAX_CRYPTO_POSITIONS = 3
TRADE_TRAINED_SYMBOLS_ONLY = True
MIN_SIGNAL_CONFIDENCE = 0.45
DECISION_COOLDOWN_SECONDS = 60
MAX_PENDING_DECISION_SECONDS = 300
MIN_ORDER_QUANTITY = 0.0001
CRYPTO_RETRY_SECONDS = 60
MAX_PRICE_AGE_SECONDS = 180
ASSET_STOCK = "stock"
ASSET_CRYPTO = "crypto"

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "market_cache"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "trade_model.pt"
FINBERT_DIR = MODEL_DIR / "finbert"
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
DB_PATH = ROOT / "trademind.db"
ENV_PATH = ROOT / ".env"

BG = "#1e1e1e"
BLACK = "#000000"
TEXT = "#d4d4d4"
ACCENT = "#555555"
DIM = "#777777"
HOVER = "#2d2d2d"
TROUGH = "#333333"
PROGRESS = "#888888"
WHITE = "#ffffff"

MODEL_FEATURES = [
    "return_1",
    "return_mean_5",
    "return_mean_10",
    "return_mean_30",
    "return_std_5",
    "return_std_20",
    "price_sma_5",
    "price_sma_10",
    "price_sma_30",
    "rsi_14",
    "macd_hist_pct",
    "atr_14_pct",
    "volume_change",
    "volume_ratio_5_20",
    "range_pct",
    "body_pct",
    "momentum_5",
    "momentum_20",
    "sharpe_20",
    "sentiment",
]
FEATURE_COLUMNS = MODEL_FEATURES
SIGNAL_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Communication",
    "AMZN": "Consumer",
    "META": "Communication",
    "NVDA": "Technology",
    "TSLA": "Consumer",
}


def normalize_asset_type(value: str) -> str:
    normalized = str(value).lower()
    if normalized in {"stock", "stocks"}:
        return ASSET_STOCK
    if normalized in {"crypto", "cryptos"}:
        return ASSET_CRYPTO
    return normalized


def us_trading_day(timestamp: datetime) -> date:
    """Map a timestamp to the US session whose risk window is currently active."""
    eastern = timestamp.astimezone(ZoneInfo("America/New_York"))
    candidate = eastern.date()
    if (eastern.hour, eastern.minute) < (9, 30):
        candidate -= timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate

load_dotenv(ENV_PATH, override=False)


def training_move_threshold(asset_class: str, timeframe: str) -> float:
    if timeframe == "1Min":
        return CRYPTO_MINUTE_MOVE_THRESHOLD if asset_class == "crypto" else MOVE_THRESHOLD
    return (
        CRYPTO_DAILY_MOVE_THRESHOLD
        if asset_class == "crypto"
        else STOCK_DAILY_MOVE_THRESHOLD
    )


class UserFacingError(RuntimeError):
    pass


class HttpRequestError(UserFacingError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


T = TypeVar("T")


def api_credentials() -> tuple[str, str]:
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise UserFacingError(
            "Alpaca credentials are missing. Add APCA_API_KEY_ID and "
            "APCA_API_SECRET_KEY to .env, then restart the app."
        )
    return key, secret


def massive_api_key() -> str:
    key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not key:
        raise UserFacingError(
            "Massive API key is missing. Add MASSIVE_API_KEY to .env, then restart "
            "the app. A free Stocks Basic key is sufficient."
        )
    return key


def api_status_code(exc: Exception) -> int | None:
    error = getattr(exc, "error", None)
    value = getattr(error, "status_code", None)
    if value is None:
        value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(exc, "code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def call_with_backoff(
    operation: Callable[[], T],
    events: queue.Queue | None,
    context: str,
    attempts: int = 5,
) -> T:
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            status = api_status_code(exc)
            text = str(exc).lower()
            retryable = status == 429 or (status is not None and status >= 500)
            retryable = retryable or any(
                phrase in text
                for phrase in ("rate limit", "timed out", "network error", "temporarily unavailable")
            )
            if not retryable or attempt == attempts - 1:
                raise
            delay = min(30, 2 ** (attempt + 1))
            if events is not None:
                events.put(
                    {
                        "kind": "log",
                        "text": f"RETRY  | {context}  status={status or 'network'}  wait={delay}s",
                    }
                )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def timeframe_object(name: str) -> TimeFrame:
    if name == "1Min":
        return TimeFrame.Minute
    if name == "1Day":
        return TimeFrame.Day
    raise ValueError(f"Unsupported timeframe: {name}")


class SlidingWindowRateLimiter:
    """Thread-safe limiter for Massive's free five-request-per-minute allowance."""

    def __init__(self, calls: int = 5, period_seconds: float = 60.0) -> None:
        self.calls = calls
        self.period_seconds = period_seconds
        self.timestamps: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self, events: queue.Queue | None = None) -> None:
        logged = False
        while True:
            with self.lock:
                now = time.monotonic()
                while self.timestamps and now - self.timestamps[0] >= self.period_seconds:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.calls:
                    self.timestamps.append(now)
                    return
                wait_for = max(0.05, self.period_seconds - (now - self.timestamps[0]))
            if events is not None and not logged:
                events.put(
                    {
                        "kind": "log",
                        "text": f"LIMIT  | Massive free tier  wait={wait_for:.0f}s",
                    }
                )
                logged = True
            time.sleep(min(wait_for, 1.0))


MASSIVE_RATE_LIMITER = SlidingWindowRateLimiter()


def http_get_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    events: queue.Queue | None = None,
    context: str,
    missing_ok: bool = False,
) -> bytes | None:
    def request() -> bytes | None:
        if requests is None:
            raise UserFacingError("The requests package is required for HTTP downloads.")
        try:
            response = requests.get(
                url,
                headers={"User-Agent": "TradeMind/1.0", **(headers or {})},
                timeout=60,
            )
            if missing_ok and response.status_code == 404:
                return None
            if response.status_code >= 400:
                detail = response.text[:300]
                raise HttpRequestError(
                    f"{context} failed with HTTP {response.status_code}: "
                    f"{detail or response.reason}",
                    response.status_code,
                )
            return response.content
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if missing_ok and status == 404:
                return None
            if status is not None:
                raise HttpRequestError(
                    f"{context} failed with HTTP {status}: {exc}", int(status)
                ) from exc
            raise UserFacingError(f"{context} network error: {exc}") from exc

    return call_with_backoff(request, events, context)


def cache_paths(symbol: str, timeframe: str) -> tuple[Path, Path]:
    safe_symbol = symbol.replace("/", "-").upper()
    stem = f"{safe_symbol}_{timeframe}"
    return CACHE_DIR / f"{stem}.parquet", CACHE_DIR / f"{stem}.csv"


def normalise_bar_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        )
    result = frame.copy()
    if isinstance(result.index, pd.MultiIndex) or result.index.name:
        result = result.reset_index()
    result.columns = [str(column).lower() for column in result.columns]
    if "timestamp" not in result.columns:
        for candidate in ("time", "datetime", "index"):
            if candidate in result.columns:
                result = result.rename(columns={candidate: "timestamp"})
                break
    if "symbol" not in result.columns:
        result["symbol"] = symbol
    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise UserFacingError(f"Alpaca bars are missing columns: {', '.join(missing)}")
    optional = [column for column in ("trade_count", "vwap") if column in result.columns]
    result = result[required + optional].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result["symbol"] = result["symbol"].astype(str).str.upper()
    for column in ("open", "high", "low", "close", "volume") + tuple(optional):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return (
        result.dropna(subset=required)
        .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def merge_bar_frames(symbol: str, *frames: pd.DataFrame) -> pd.DataFrame:
    populated = [frame for frame in frames if frame is not None and not frame.empty]
    if not populated:
        return normalise_bar_frame(pd.DataFrame(), symbol)
    return normalise_bar_frame(pd.concat(populated, ignore_index=True), symbol)


def resample_live_bars(
    frame: pd.DataFrame, symbol: str, timeframe: str
) -> pd.DataFrame:
    """Aggregate repeated live snapshots to the model's training timeframe."""
    if frame.empty:
        return normalise_bar_frame(pd.DataFrame(), symbol)
    table = frame.copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], utc=True)
    frequency = "1min" if timeframe == "1Min" else "1D"
    grouped = (
        table.set_index("timestamp")
        .sort_index()
        .resample(frequency)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    grouped["symbol"] = symbol
    return normalise_bar_frame(grouped, symbol)


def read_cached_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    parquet_path, csv_path = cache_paths(symbol, timeframe)
    if parquet_path.exists():
        return normalise_bar_frame(pd.read_parquet(parquet_path), symbol)
    if csv_path.exists():
        return normalise_bar_frame(pd.read_csv(csv_path), symbol)
    return normalise_bar_frame(pd.DataFrame(), symbol)


def write_cached_bars(frame: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path, csv_path = cache_paths(symbol, timeframe)
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        frame.to_csv(csv_path, index=False)
        return csv_path


def bars_are_fresh(frame: pd.DataFrame, timeframe: str, now: datetime) -> bool:
    if frame.empty:
        return False
    last = frame["timestamp"].iloc[-1].to_pydatetime()
    if timeframe == "1Min":
        return now - last < timedelta(minutes=2)
    return last.date() >= now.date()


def massive_results_frame(payload: dict, symbol: str) -> pd.DataFrame:
    rows = []
    for item in payload.get("results") or []:
        rows.append(
            {
                "timestamp": pd.to_datetime(item["t"], unit="ms", utc=True),
                "symbol": symbol,
                "open": item["o"],
                "high": item["h"],
                "low": item["l"],
                "close": item["c"],
                "volume": item["v"],
                "trade_count": item.get("n"),
                "vwap": item.get("vw"),
            }
        )
    return normalise_bar_frame(pd.DataFrame(rows), symbol)


def sync_massive_stock_bars(
    symbol: str,
    timeframe: str,
    events: queue.Queue | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    symbol = symbol.upper()
    key = massive_api_key()
    cached = read_cached_bars(symbol, timeframe)
    now = datetime.now(timezone.utc)
    if not force_refresh and bars_are_fresh(cached, timeframe, now):
        if events is not None:
            events.put(
                {"kind": "log", "text": f"CACHE  | {symbol} {timeframe} rows={len(cached)} current"}
            )
        return cached

    delta = timedelta(minutes=1) if timeframe == "1Min" else timedelta(days=1)
    history_days = (
        STOCK_MINUTE_HISTORY_DAYS if timeframe == "1Min" else STOCK_DAILY_HISTORY_DAYS
    )
    start = (
        cached["timestamp"].iloc[-1].to_pydatetime() + delta
        if not cached.empty
        else now - timedelta(days=history_days)
    )
    end = now - (timedelta(minutes=15) if timeframe == "1Min" else timedelta(days=1))
    if start >= end:
        return cached

    timespan = "minute" if timeframe == "1Min" else "day"
    encoded_symbol = urllib.parse.quote(symbol, safe="")
    url = (
        f"{MASSIVE_BASE_URL}/v2/aggs/ticker/{encoded_symbol}/range/1/{timespan}/"
        f"{start.date().isoformat()}/{end.date().isoformat()}?adjusted=true&sort=asc&limit=50000"
    )
    pages: list[pd.DataFrame] = []
    page_count = 0
    while url:
        MASSIVE_RATE_LIMITER.acquire(events)
        raw = http_get_bytes(
            url,
            headers={"Authorization": f"Bearer {key}"},
            events=events,
            context=f"Massive {symbol}",
        )
        payload = json.loads((raw or b"{}").decode("utf-8"))
        status = str(payload.get("status", "")).upper()
        if status not in {"OK", "DELAYED", ""}:
            raise UserFacingError(
                f"Massive rejected {symbol}: {payload.get('error') or payload.get('message') or status}"
            )
        pages.append(massive_results_frame(payload, symbol))
        page_count += 1
        url = payload.get("next_url")

    fetched = merge_bar_frames(symbol, *pages)
    combined = merge_bar_frames(symbol, cached, fetched)
    path = write_cached_bars(combined, symbol, timeframe)
    if events is not None:
        events.put(
            {
                "kind": "log",
                "text": (
                    f"MASSIVE | {symbol} {timeframe} pages={page_count} fetched={len(fetched)} "
                    f"cached={len(combined)} file={path.name}"
                ),
            }
        )
    return combined


BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def binance_zip_frame(content: bytes, symbol: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise UserFacingError(f"Binance archive for {symbol} contains no CSV file.")
        with archive.open(names[0]) as stream:
            raw = pd.read_csv(stream, header=None, names=BINANCE_KLINE_COLUMNS)
    timestamps = pd.to_numeric(raw["open_time"], errors="coerce")
    unit = "us" if timestamps.dropna().median() >= 100_000_000_000_000 else "ms"
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit=unit, utc=True, errors="coerce"),
            "symbol": symbol,
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "volume": raw["volume"],
            "trade_count": raw["trade_count"],
        }
    )
    return normalise_bar_frame(frame, symbol)


def _binance_archive(
    symbol: str,
    interval: str,
    cadence: str,
    stamp: str,
    events: queue.Queue | None,
) -> pd.DataFrame | None:
    filename = f"{symbol}-{interval}-{stamp}.zip"
    url = f"{BINANCE_PUBLIC_BASE_URL}/{cadence}/klines/{symbol}/{interval}/{filename}"
    content = http_get_bytes(
        url,
        events=events,
        context=f"Binance {symbol} {stamp}",
        missing_ok=True,
    )
    return None if content is None else binance_zip_frame(content, symbol)


def sync_binance_crypto_bars(
    symbol: str,
    timeframe: str,
    events: queue.Queue | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    symbol = symbol.replace("/", "").replace("-", "").upper()
    cached = read_cached_bars(symbol, timeframe)
    now = datetime.now(timezone.utc)
    if not force_refresh and bars_are_fresh(cached, timeframe, now):
        if events is not None:
            events.put(
                {"kind": "log", "text": f"CACHE  | {symbol} {timeframe} rows={len(cached)} current"}
            )
        return cached

    history_days = (
        CRYPTO_MINUTE_HISTORY_DAYS if timeframe == "1Min" else CRYPTO_DAILY_HISTORY_DAYS
    )
    start = (
        cached["timestamp"].iloc[-1].to_pydatetime()
        if not cached.empty and not force_refresh
        else now - timedelta(days=history_days)
    )
    end_date = (now - timedelta(days=1)).date()
    interval = "1m" if timeframe == "1Min" else "1d"
    frames: list[pd.DataFrame] = []
    missing_archives = 0

    first_month = pd.Period(start.date(), freq="M")
    last_complete_month = pd.Period(now.date(), freq="M") - 1
    if first_month <= last_complete_month:
        for month in pd.period_range(first_month, last_complete_month, freq="M"):
            frame = _binance_archive(symbol, interval, "monthly", str(month), events)
            if frame is None:
                missing_archives += 1
                fallback_start = max(start.date(), month.start_time.date())
                fallback_end = min(end_date, month.end_time.date())
                for day in pd.date_range(fallback_start, fallback_end, freq="D"):
                    daily = _binance_archive(
                        symbol,
                        interval,
                        "daily",
                        day.date().isoformat(),
                        events,
                    )
                    if daily is None:
                        missing_archives += 1
                    else:
                        frames.append(daily)
            else:
                frames.append(frame)

    current_month_start = now.date().replace(day=1)
    daily_start = max(start.date(), current_month_start)
    if daily_start <= end_date:
        for day in pd.date_range(daily_start, end_date, freq="D"):
            frame = _binance_archive(
                symbol, interval, "daily", day.date().isoformat(), events
            )
            if frame is None:
                missing_archives += 1
            else:
                frames.append(frame)

    fetched = merge_bar_frames(symbol, *frames)
    if fetched.empty and cached.empty:
        raise UserFacingError(
            f"No Binance public archives were found for {symbol}. Use a spot symbol such as BTCUSDT."
        )
    combined = merge_bar_frames(symbol, cached, fetched)
    path = write_cached_bars(combined, symbol, timeframe)
    if events is not None:
        events.put(
            {
                "kind": "log",
                "text": (
                    f"BINANCE | {symbol} {timeframe} archives={len(frames)} missing={missing_archives} "
                    f"fetched={len(fetched)} cached={len(combined)} file={path.name}"
                ),
            }
        )
    return combined


def sync_yfinance_stock_bars(
    symbol: str,
    timeframe: str,
    events: queue.Queue | None = None,
) -> pd.DataFrame:
    if yf is None:
        raise UserFacingError(
            "Stock history needs MASSIVE_API_KEY or the yfinance package. Run setup.bat."
        )
    symbol = symbol.upper()
    cached = read_cached_bars(symbol, timeframe)
    now = datetime.now(timezone.utc)
    history_days = 7 if timeframe == "1Min" else STOCK_DAILY_HISTORY_DAYS
    start = (
        cached["timestamp"].iloc[-1].to_pydatetime()
        if not cached.empty
        else now - timedelta(days=history_days)
    )
    downloaded = yf.download(
        symbol,
        start=start.date().isoformat(),
        end=(now + timedelta(days=1)).date().isoformat(),
        interval="1m" if timeframe == "1Min" else "1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if downloaded.empty:
        if cached.empty:
            raise UserFacingError(f"Yahoo Finance returned no bars for {symbol}.")
        return cached
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)
    downloaded = downloaded.reset_index().rename(
        columns={
            "Date": "timestamp",
            "Datetime": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    downloaded["symbol"] = symbol
    fetched = normalise_bar_frame(downloaded, symbol)
    combined = merge_bar_frames(symbol, cached, fetched)
    path = write_cached_bars(combined, symbol, timeframe)
    if events is not None:
        events.put(
            {
                "kind": "log",
                "text": (
                    f"YAHOO  | {symbol} {timeframe} fetched={len(fetched)} "
                    f"cached={len(combined)} file={path.name}"
                ),
            }
        )
    return combined


def sync_training_bars(
    asset_class: str,
    symbol: str,
    timeframe: str,
    events: queue.Queue | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if asset_class == "stocks":
        if os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY"):
            return sync_massive_stock_bars(symbol, timeframe, events, force_refresh)
        return sync_yfinance_stock_bars(symbol, timeframe, events)
    if asset_class == "crypto":
        return sync_binance_crypto_bars(symbol, timeframe, events, force_refresh)
    raise UserFacingError(f"Unsupported asset class: {asset_class}")


def fetch_alpaca_execution_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: str,
    base_frame: pd.DataFrame,
    events: queue.Queue | None = None,
) -> pd.DataFrame:
    """Overlay recent free IEX bars for paper decisions without contaminating training cache."""
    now = datetime.now(timezone.utc)
    recent_start = now - (
        timedelta(days=7) if timeframe == "1Min" else timedelta(days=120)
    )
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe_object(timeframe),
        start=recent_start,
        end=now,
        adjustment=Adjustment.ALL,
        feed=DataFeed.IEX,
        sort=Sort.ASC,
    )
    response = call_with_backoff(
        lambda: client.get_stock_bars(request), events, f"Alpaca IEX bars {symbol}"
    )
    recent = normalise_bar_frame(response.df, symbol)
    return merge_bar_frames(symbol, base_frame, recent)


def engineer_features(
    frame: pd.DataFrame,
    include_target: bool = True,
    move_threshold: float = MOVE_THRESHOLD,
) -> pd.DataFrame:
    bars = frame.sort_values("timestamp").copy()
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    open_price = bars["open"].astype(float)
    volume = bars["volume"].astype(float)

    bars["log_return"] = np.log(close / close.shift(1))
    bars["volatility_20"] = bars["log_return"].rolling(20).std()
    bars["atr_14_pct"] = (
        AverageTrueRange(high=high, low=low, close=close, window=14)
        .average_true_range()
        .replace(0, np.nan)
        / close
    )
    bars["rsi_14"] = RSIIndicator(close=close, window=14).rsi() / 100.0
    bars["macd_hist_pct"] = MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9
    ).macd_diff() / close
    bars["volume_change"] = volume.pct_change().clip(-10, 10)
    bars["range_pct"] = (high - low) / close
    bars["body_pct"] = (close - open_price) / open_price.replace(0, np.nan)
    bars["sma_5_gap"] = close / close.rolling(5).mean() - 1.0
    bars["sma_10_gap"] = close / close.rolling(10).mean() - 1.0
    bars["sma_30_gap"] = close / close.rolling(30).mean() - 1.0
    bars["momentum_5"] = close.pct_change(5)
    bars["momentum_20"] = close.pct_change(20)
    bars["volume_ratio_5_20"] = (
        volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
    )
    bars["sharpe_20"] = (
        bars["log_return"].rolling(20).mean()
        / bars["log_return"].rolling(20).std().replace(0, np.nan)
        * np.sqrt(20)
    )

    if include_target:
        bars["future_return"] = close.shift(-FORWARD_HORIZON) / close - 1.0
        bars["target"] = np.select(
            [
                bars["future_return"] > move_threshold,
                bars["future_return"] < -move_threshold,
            ],
            [1, 2],
            default=0,
        ).astype(float)
        bars.loc[bars.index[-FORWARD_HORIZON:], "target"] = np.nan
    return bars.replace([np.inf, -np.inf], np.nan)


def model_feature_vector(
    featured: pd.DataFrame, end: int, sentiment: float = 0.0
) -> np.ndarray | None:
    start = end - LOOKBACK + 1
    if start < 0:
        return None
    window = featured.iloc[start : end + 1]
    returns = window["log_return"]
    volume_change = window["volume_change"]
    last = window.iloc[-1]
    values = np.asarray(
        [
            last["log_return"],
            returns.tail(5).mean(),
            returns.tail(10).mean(),
            returns.mean(),
            returns.tail(5).std(),
            returns.tail(20).std(),
            last["sma_5_gap"],
            last["sma_10_gap"],
            last["sma_30_gap"],
            last["rsi_14"],
            last["macd_hist_pct"],
            last["atr_14_pct"],
            volume_change.tail(5).mean(),
            last["volume_ratio_5_20"],
            last["range_pct"],
            last["body_pct"],
            last["momentum_5"],
            last["momentum_20"],
            last["sharpe_20"],
            float(np.clip(sentiment, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )
    return values if np.isfinite(values).all() else None


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_rows: int
    validation_rows: int
    class_counts: dict[int, int]


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float, list[list[int]]]:
    """Return accuracy, macro-F1, and a 3x3 confusion matrix."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))
    f1_scores: list[float] = []
    for class_id in range(3):
        true_positive = float(confusion[class_id, class_id])
        false_positive = float(confusion[:, class_id].sum() - true_positive)
        false_negative = float(confusion[class_id, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1_scores.append(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        )
    return accuracy, float(np.mean(f1_scores)), confusion.tolist()


def symbol_windows(
    frame: pd.DataFrame, move_threshold: float = MOVE_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    featured = engineer_features(
        frame, include_target=True, move_threshold=move_threshold
    ).reset_index(drop=True)
    returns = featured["log_return"]
    volume_change = featured["volume_change"]
    matrix = np.column_stack(
        [
            returns,
            returns.rolling(5).mean(),
            returns.rolling(10).mean(),
            returns.rolling(LOOKBACK).mean(),
            returns.rolling(5).std(),
            returns.rolling(20).std(),
            featured["sma_5_gap"],
            featured["sma_10_gap"],
            featured["sma_30_gap"],
            featured["rsi_14"],
            featured["macd_hist_pct"],
            featured["atr_14_pct"],
            volume_change.rolling(5).mean(),
            featured["volume_ratio_5_20"],
            featured["range_pct"],
            featured["body_pct"],
            featured["momentum_5"],
            featured["momentum_20"],
            featured["sharpe_20"],
            np.zeros(len(featured), dtype=np.float64),
        ]
    )
    targets = featured["target"].to_numpy(dtype=np.float64)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(targets)
    timestamps = featured["timestamp"]
    deltas = timestamps.diff().dt.total_seconds()
    normal_interval = float(deltas[deltas > 0].median()) if (deltas > 0).any() else 0.0
    if normal_interval > 0:
        for gap_index in np.flatnonzero(deltas.to_numpy() > normal_interval * 2):
            invalid_start = max(0, int(gap_index) - FORWARD_HORIZON)
            invalid_end = min(len(valid), int(gap_index) + LOOKBACK)
            valid[invalid_start:invalid_end] = False
    if not valid.any():
        return (
            np.empty((0, len(MODEL_FEATURES)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return matrix[valid].astype(np.float32), targets[valid].astype(np.int64)


def build_training_dataset(
    frames: dict[str, pd.DataFrame],
    move_threshold: float = MOVE_THRESHOLD,
) -> DatasetBundle:
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    validation_x: list[np.ndarray] = []
    validation_y: list[np.ndarray] = []
    for symbol, frame in frames.items():
        x_symbol, y_symbol = symbol_windows(frame, move_threshold)
        if len(x_symbol) < 50:
            raise UserFacingError(
                f"{symbol} has only {len(x_symbol)} usable windows; download more bars."
            )
        split = int(len(x_symbol) * (1.0 - VALIDATION_FRACTION))
        split = min(max(split, 1), len(x_symbol) - 1)
        train_x.append(x_symbol[:split])
        train_y.append(y_symbol[:split])
        validation_x.append(x_symbol[split:])
        validation_y.append(y_symbol[split:])

    x_train = np.concatenate(train_x)
    y_train = np.concatenate(train_y)
    x_validation = np.concatenate(validation_x)
    y_validation = np.concatenate(validation_y)
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_validation = ((x_validation - mean) / std).astype(np.float32)
    counts_array = np.bincount(y_train, minlength=3)
    if np.any(counts_array == 0):
        absent = [SIGNAL_NAMES[index] for index, count in enumerate(counts_array) if count == 0]
        raise UserFacingError(
            "Training data has no examples for: " + ", ".join(absent) + ". Adjust MOVE_THRESHOLD."
        )
    counts = {index: int(value) for index, value in enumerate(counts_array)}
    return DatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        mean=mean,
        std=std,
        train_rows=len(x_train),
        validation_rows=len(x_validation),
        class_counts=counts,
    )


class ShallowTradeNet(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(16, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def latest_model_input(
    frame: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    sentiment: float = 0.0,
) -> tuple[np.ndarray, pd.Timestamp, float]:
    featured = engineer_features(frame, include_target=False)
    featured = featured.reset_index(drop=True)
    vector = model_feature_vector(featured, len(featured) - 1, sentiment)
    if vector is None:
        raise UserFacingError(
            f"Need {LOOKBACK} complete feature rows; only {len(featured)} bars are available."
        )
    scaled = ((vector - mean) / std).astype(np.float32)
    last = featured.iloc[-1]
    return scaled, pd.Timestamp(last["timestamp"]), float(last["close"])


def save_checkpoint_atomic(checkpoint: dict, path: Path = MODEL_PATH) -> None:
    """Write and validate a checkpoint before atomically replacing the live model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    safe_checkpoint = dict(checkpoint)
    for field in ("mean", "std"):
        safe_checkpoint[field] = np.asarray(safe_checkpoint[field]).tolist()
    try:
        torch.save(safe_checkpoint, temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=True)
        if int(loaded.get("input_size", 0)) != len(MODEL_FEATURES):
            raise UserFacingError("Checkpoint validation failed: unexpected input size.")
        if list(loaded.get("feature_columns", [])) != MODEL_FEATURES:
            raise UserFacingError("Checkpoint validation failed: feature schema mismatch.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SQLite message bus
# ---------------------------------------------------------------------------


class ClosingSQLiteConnection(sqlite3.Connection):
    """Commit/rollback like sqlite3, then release the Windows file handle."""

    def __exit__(self, exc_type, exc_value, traceback) -> Literal[False]:
        try:
            super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()
        return False


class AgentDatabase:
    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=10.0, factory=ClosingSQLiteConnection
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS live_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL NOT NULL,
            UNIQUE(symbol, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_live_symbol_time
            ON live_prices(symbol, timestamp DESC);
        CREATE TABLE IF NOT EXISTS research_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            headline TEXT NOT NULL,
            source TEXT NOT NULL,
            label TEXT NOT NULL,
            sentiment REAL NOT NULL,
            UNIQUE(symbol, headline)
        );
        CREATE TABLE IF NOT EXISTS asset_rankings (
            symbol TEXT PRIMARY KEY,
            asset_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            score REAL NOT NULL,
            momentum_z REAL NOT NULL,
            vol_ratio_z REAL NOT NULL,
            adx REAL NOT NULL,
            atr_pct REAL NOT NULL,
            sentiment REAL NOT NULL,
            sharpe_60_z REAL NOT NULL,
            last_price REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS investment_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            signal TEXT NOT NULL,
            probability REAL NOT NULL,
            price REAL NOT NULL,
            atr REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            note TEXT NOT NULL DEFAULT '',
            client_order_id TEXT NOT NULL DEFAULT '',
            filled_quantity REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_decision_status
            ON investment_decisions(status, timestamp);
        CREATE TABLE IF NOT EXISTS portfolio_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            equity REAL NOT NULL,
            pnl REAL NOT NULL,
            order_id TEXT NOT NULL DEFAULT '',
            note TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS training_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            train_loss REAL NOT NULL,
            validation_loss REAL NOT NULL,
            validation_accuracy REAL NOT NULL,
            macro_f1 REAL NOT NULL DEFAULT 0,
            learning_rate REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS agent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            agent TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS universe_assets (
            symbol TEXT PRIMARY KEY,
            asset_type TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            sector TEXT NOT NULL DEFAULT 'Other',
            updated_at TEXT NOT NULL
        );
        """
        connection = self.connect()
        try:
            with connection:
                connection.executescript(schema)
                training_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(training_log)")
                }
                if "macro_f1" not in training_columns:
                    connection.execute(
                        "ALTER TABLE training_log ADD COLUMN macro_f1 REAL NOT NULL DEFAULT 0"
                    )
                if "learning_rate" not in training_columns:
                    connection.execute(
                        "ALTER TABLE training_log ADD COLUMN learning_rate REAL NOT NULL DEFAULT 0"
                    )
                universe_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(universe_assets)")
                }
                if "sector" not in universe_columns:
                    connection.execute(
                        "ALTER TABLE universe_assets ADD COLUMN sector TEXT NOT NULL DEFAULT 'Other'"
                    )
                decision_columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(investment_decisions)"
                    )
                }
                if "client_order_id" not in decision_columns:
                    connection.execute(
                        "ALTER TABLE investment_decisions ADD COLUMN "
                        "client_order_id TEXT NOT NULL DEFAULT ''"
                    )
                if "filled_quantity" not in decision_columns:
                    connection.execute(
                        "ALTER TABLE investment_decisions ADD COLUMN "
                        "filled_quantity REAL NOT NULL DEFAULT 0"
                    )
                if "created_at" not in decision_columns:
                    connection.execute(
                        "ALTER TABLE investment_decisions ADD COLUMN "
                        "created_at TEXT NOT NULL DEFAULT ''"
                    )
        finally:
            connection.close()

    def execute(self, sql: str, parameters: tuple = ()) -> int:
        connection = self.connect()
        try:
            with connection:
                cursor = connection.execute(sql, parameters)
                return int(cursor.lastrowid or 0)
        finally:
            connection.close()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        connection = self.connect()
        try:
            with connection:
                connection.executemany(sql, rows)
        finally:
            connection.close()

    def query(self, sql: str, parameters: tuple = ()) -> list[sqlite3.Row]:
        connection = self.connect()
        try:
            return list(connection.execute(sql, parameters).fetchall())
        finally:
            connection.close()

    def log(self, agent: str, message: str, level: str = "INFO") -> None:
        self.execute(
            "INSERT INTO agent_logs(timestamp, agent, level, message) VALUES(?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), agent, level, message),
        )

    def latest_sentiment(self, symbol: str) -> float:
        rows = self.query(
            "SELECT sentiment FROM research_findings WHERE symbol=? "
            "ORDER BY timestamp DESC LIMIT 10",
            (symbol,),
        )
        return float(np.mean([row["sentiment"] for row in rows])) if rows else 0.0


class WatchlistState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stocks = DEFAULT_SYMBOLS.copy()
        self.crypto = DEFAULT_CRYPTO_SYMBOLS.copy()

    def snapshot(self) -> tuple[list[str], list[str]]:
        with self.lock:
            return self.stocks.copy(), self.crypto.copy()

    def set_primary(self, symbols: list[str], asset_class: str) -> None:
        with self.lock:
            if asset_class == "stocks":
                self.stocks = symbols.copy()
            else:
                self.crypto = [item.replace("/", "").replace("-", "") for item in symbols]


class BaseAgent(threading.Thread):
    def __init__(
        self,
        name: str,
        database: AgentDatabase,
        events: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.agent_name = name
        self.database = database
        self.events = events
        self.stop_event = stop_event

    def log(self, message: str, level: str = "INFO") -> None:
        self.database.log(self.agent_name, message, level)
        self.events.put(
            {"kind": "log", "text": f"{self.agent_name.upper():8s}| {message}"}
        )


# ---------------------------------------------------------------------------
# Researcher agent: live prices, news, and lazy local FinBERT
# ---------------------------------------------------------------------------


class FinBERTSentiment:
    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.failed_reason: str | None = None
        self.lock = threading.Lock()

    def _load(self) -> bool:
        with self.lock:
            if self.model is not None:
                return True
            if self.failed_reason is not None:
                return False
            if AutoTokenizer is None or AutoModelForSequenceClassification is None:
                self.failed_reason = "transformers is not installed"
                return False
            try:
                FINBERT_DIR.mkdir(parents=True, exist_ok=True)
                has_cache = any(
                    FINBERT_DIR.glob("models--ProsusAI--finbert/snapshots/*")
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "ProsusAI/finbert",
                    revision=FINBERT_REVISION,
                    cache_dir=FINBERT_DIR,
                    local_files_only=has_cache,
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert",
                    revision=FINBERT_REVISION,
                    cache_dir=FINBERT_DIR,
                    local_files_only=has_cache,
                ).to("cpu")
                self.model.eval()
                return True
            except Exception as exc:
                self.failed_reason = str(exc)
                return False

    @staticmethod
    def _lexical(text: str) -> tuple[str, float]:
        lowered = text.lower()
        positive = sum(
            word in lowered
            for word in ("beat", "growth", "gain", "surge", "profit", "upgrade")
        )
        negative = sum(
            word in lowered
            for word in ("miss", "loss", "drop", "fall", "risk", "downgrade")
        )
        score = float(np.clip((positive - negative) / 3.0, -1.0, 1.0))
        return ("positive" if score > 0 else "negative" if score < 0 else "neutral", score)

    def score(self, text: str) -> tuple[str, float]:
        if not self._load():
            return self._lexical(text)
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("FinBERT reported ready without tokenizer/model state.")
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=1)[0]
        labels = {
            int(index): str(label).lower()
            for index, label in self.model.config.id2label.items()
        }
        best = int(probabilities.argmax().item())
        probability_by_label = {
            labels[index]: float(probabilities[index].item()) for index in labels
        }
        signed = probability_by_label.get("positive", 0.0) - probability_by_label.get(
            "negative", 0.0
        )
        return labels.get(best, "neutral"), float(signed)


class ResearcherAgent(BaseAgent):
    def __init__(self, database, events, stop_event, watchlist: WatchlistState) -> None:
        super().__init__("researcher", database, events, stop_event)
        self.watchlist = watchlist
        self.sentiment = FinBERTSentiment()
        self.stock_client = None
        self.missing_stock_credentials_logged = False
        self.crypto_retry_after: dict[str, float] = {}

    def run(self) -> None:
        next_prices = 0.0
        next_news = time.monotonic() + 30.0
        self.log("agent online")
        while not self.stop_event.is_set():
            now = time.monotonic()
            try:
                if now >= next_prices:
                    self.fetch_prices()
                    next_prices = now + RESEARCH_PRICE_SECONDS
                if now >= next_news:
                    self.fetch_news()
                    next_news = now + RESEARCH_NEWS_SECONDS
            except Exception as exc:
                self.log(str(exc), "ERROR")
            self.stop_event.wait(0.2)
        self.log("agent stopped")

    def _store_bar(self, symbol: str, asset_type: str, bar) -> None:
        timestamp = pd.Timestamp(getattr(bar, "timestamp", datetime.now(timezone.utc)))
        self.database.execute(
            "INSERT OR REPLACE INTO live_prices"
            "(symbol,asset_type,timestamp,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?,?)",
            (
                symbol,
                asset_type,
                timestamp.isoformat(),
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                float(bar.volume),
            ),
        )

    def fetch_prices(self) -> None:
        stocks, crypto_symbols = self.watchlist.snapshot()
        try:
            if self.stock_client is None:
                key, secret = api_credentials()
                self.stock_client = StockHistoricalDataClient(key, secret)
            request = StockLatestBarRequest(symbol_or_symbols=stocks, feed=DataFeed.IEX)
            bars = self.stock_client.get_stock_latest_bar(request)
            for symbol, bar in bars.items():
                self._store_bar(symbol, ASSET_STOCK, bar)
        except UserFacingError:
            if not self.missing_stock_credentials_logged:
                self.log("Alpaca keys missing; stock stream paused, crypto remains active", "WARN")
                self.missing_stock_credentials_logged = True
        if requests is None:
            return
        now = time.monotonic()
        for symbol in crypto_symbols:
            if now < self.crypto_retry_after.get(symbol, 0.0):
                continue
            try:
                response = requests.get(
                    "https://data-api.binance.vision/api/v3/klines",
                    params={"symbol": symbol, "interval": "1m", "limit": 1},
                    timeout=10,
                )
                response.raise_for_status()
                item = response.json()[0]
                self.database.execute(
                    "INSERT OR REPLACE INTO live_prices"
                    "(symbol,asset_type,timestamp,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        symbol,
                        ASSET_CRYPTO,
                        pd.to_datetime(item[0], unit="ms", utc=True).isoformat(),
                        float(item[1]),
                        float(item[2]),
                        float(item[3]),
                        float(item[4]),
                        float(item[5]),
                    ),
                )
            except Exception as exc:
                self.crypto_retry_after[symbol] = time.monotonic() + CRYPTO_RETRY_SECONDS
                self.log(
                    f"crypto symbol cooldown {symbol} for {CRYPTO_RETRY_SECONDS}s: {exc}",
                    "WARN",
                )

    def fetch_news(self) -> None:
        stocks, _ = self.watchlist.snapshot()
        headlines: list[tuple[str, str, str]] = []
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        if finnhub_key and finnhub is not None:
            client = finnhub.Client(api_key=finnhub_key)
            today = datetime.now(timezone.utc).date()
            for symbol in stocks:
                for item in client.company_news(
                    symbol,
                    _from=(today - timedelta(days=1)).isoformat(),
                    to=today.isoformat(),
                )[:10]:
                    headline = str(item.get("headline", "")).strip()
                    if headline:
                        headlines.append((symbol, headline, "Finnhub"))
        else:
            for symbol in stocks:
                url = (
                    "https://feeds.finance.yahoo.com/rss/2.0/headline?"
                    + urllib.parse.urlencode({"s": symbol, "region": "US", "lang": "en-US"})
                )
                content = http_get_bytes(url, context=f"Yahoo RSS {symbol}", missing_ok=True)
                if not content:
                    continue
                if len(content) > 2_000_000:
                    self.log(f"Yahoo RSS payload too large for {symbol}", "WARN")
                    continue
                try:
                    root = ET.fromstring(content)
                except ET.ParseError as exc:
                    self.log(f"Yahoo RSS parse failed for {symbol}: {exc}", "WARN")
                    continue
                for item in root.findall(".//item")[:10]:
                    headline = (item.findtext("title") or "").strip()
                    if headline:
                        headlines.append((symbol, headline, "Yahoo RSS"))
        inserted = 0
        for symbol, headline, source in headlines:
            if self.database.query(
                "SELECT 1 FROM research_findings WHERE symbol=? AND headline=? LIMIT 1",
                (symbol, headline),
            ):
                continue
            label, sentiment = self.sentiment.score(headline)
            self.database.execute(
                "INSERT OR IGNORE INTO research_findings"
                "(symbol,timestamp,headline,source,label,sentiment) VALUES(?,?,?,?,?,?)",
                (
                    symbol,
                    datetime.now(timezone.utc).isoformat(),
                    headline,
                    source,
                    label,
                    sentiment,
                ),
            )
            inserted += 1
        if inserted:
            self.log(
                f"news processed={inserted} new/{len(headlines)} fetched "
                f"model={'FinBERT' if self.sentiment.model is not None else 'lexical fallback'}"
            )


# ---------------------------------------------------------------------------
# Analyst agent: indicators and ranked composite scores
# ---------------------------------------------------------------------------


class AnalystAgent(BaseAgent):
    def __init__(self, database, events, stop_event, watchlist: WatchlistState) -> None:
        super().__init__("analyst", database, events, stop_event)
        self.watchlist = watchlist

    def run(self) -> None:
        next_analysis = 0.0
        next_universe = time.monotonic() + 30.0
        self.log("agent online")
        while not self.stop_event.is_set():
            now = time.monotonic()
            try:
                if now >= next_analysis:
                    self.rank_assets()
                    next_analysis = now + ANALYST_SECONDS
                if now >= next_universe:
                    self.refresh_universe()
                    next_universe = now + UNIVERSE_SECONDS
            except Exception as exc:
                self.log(str(exc), "ERROR")
            self.stop_event.wait(1.0)
        self.log("agent stopped")

    def _symbol_frame(self, symbol: str) -> tuple[pd.DataFrame, str] | None:
        rows = self.database.query(
            "SELECT * FROM live_prices WHERE symbol=? ORDER BY timestamp DESC LIMIT 500",
            (symbol,),
        )
        if len(rows) < 35:
            return None
        frame = pd.DataFrame([dict(row) for row in reversed(rows)])
        if pl is not None:
            frame = pl.from_pandas(frame).sort("timestamp").to_pandas()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame, str(rows[0]["asset_type"])

    def rank_assets(self) -> None:
        stocks, crypto_symbols = self.watchlist.snapshot()
        raw: list[dict] = []
        for symbol in stocks + crypto_symbols:
            result = self._symbol_frame(symbol)
            if result is None:
                continue
            frame, asset_type = result
            featured = engineer_features(frame, include_target=False).dropna()
            if featured.empty:
                continue
            last = featured.iloc[-1]
            high, low, close = frame["high"], frame["low"], frame["close"]
            adx = float(ADXIndicator(high, low, close, window=14).adx().iloc[-1])
            raw.append(
                {
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "momentum": float(last["momentum_20"]),
                    "vol_ratio": float(last["volume_ratio_5_20"]),
                    "adx": adx,
                    "atr_pct": float(last["atr_14_pct"]),
                    "sentiment": self.database.latest_sentiment(symbol),
                    "sharpe": float(last["sharpe_20"]),
                    "last_price": float(last["close"]),
                }
            )
        if not raw:
            return
        table = pd.DataFrame(raw)
        for source, target in (
            ("momentum", "momentum_z"),
            ("vol_ratio", "vol_ratio_z"),
            ("atr_pct", "atr_pct_z"),
            ("sharpe", "sharpe_60_z"),
        ):
            std = float(table[source].std(ddof=0))
            table[target] = (table[source] - table[source].mean()) / (std or 1.0)
        def sigmoid(values):
            return 1.0 / (1.0 + np.exp(-values))

        table["score"] = 100.0 * (
            0.25 * sigmoid(table["momentum_z"])
            + 0.15 * sigmoid(table["vol_ratio_z"])
            + 0.10 * np.clip(table["adx"] / 50.0, 0.0, 1.0)
            + 0.10 * sigmoid(-table["atr_pct_z"])
            + 0.15 * ((np.clip(table["sentiment"], -1, 1) + 1.0) / 2.0)
            + 0.25 * sigmoid(table["sharpe_60_z"])
        )
        top = table.sort_values("score", ascending=False).head(10)
        connection = self.database.connect()
        try:
            with connection:
                connection.execute("DELETE FROM asset_rankings")
                connection.executemany(
                    "INSERT INTO asset_rankings VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        (
                            row.symbol,
                            row.asset_type,
                            datetime.now(timezone.utc).isoformat(),
                            float(row.score),
                            float(row.momentum_z),
                            float(row.vol_ratio_z),
                            float(row.adx),
                            float(row.atr_pct),
                            float(row.sentiment),
                            float(row.sharpe_60_z),
                            float(row.last_price),
                        )
                        for row in top.itertuples()
                    ],
                )
        finally:
            connection.close()
        self.events.put({"kind": "rankings", "rows": top.to_dict("records")})
        self.log(f"ranked assets={len(top)} leader={top.iloc[0]['symbol']}")

    def refresh_universe(self) -> None:
        rows: list[tuple[str, str, str, str, str]] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            wikipedia_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            if requests is None:
                sp500 = pd.read_html(wikipedia_url)[0]
            else:
                response = requests.get(
                    wikipedia_url,
                    headers={"User-Agent": "Mozilla/5.0 KorvaxTradeMind/1.0"},
                    timeout=20,
                )
                response.raise_for_status()
                sp500 = pd.read_html(io.StringIO(response.text))[0]
            rows.extend(
                (
                    str(row["Symbol"]).replace(".", "-"),
                    ASSET_STOCK,
                    str(row["Security"]),
                    str(row["GICS Sector"]),
                    timestamp,
                )
                for _, row in sp500.iterrows()
            )
        except Exception as exc:
            self.log(f"S&P 500 universe refresh skipped: {exc}", "WARN")
        if requests is not None:
            try:
                response = requests.get(
                    "https://api.coingecko.com/api/v3/coins/markets",
                    params={
                        "vs_currency": "usd",
                        "order": "market_cap_desc",
                        "per_page": 100,
                        "page": 1,
                    },
                    timeout=20,
                )
                response.raise_for_status()
                coins = response.json()
                tradable_coins = [
                    item
                    for item in coins
                    if str(item.get("symbol", "")).lower()
                    not in {"usdt", "usdc", "dai", "usde", "fdusd", "tusd"}
                ]
                rows.extend(
                    (
                        f"{str(item['symbol']).upper()}USDT",
                        ASSET_CRYPTO,
                        str(item["name"]),
                        "Crypto",
                        timestamp,
                    )
                    for item in tradable_coins
                )
            except Exception as exc:
                self.log(f"CoinGecko universe refresh skipped: {exc}", "WARN")
        self.database.executemany(
            "INSERT OR REPLACE INTO universe_assets"
            "(symbol,asset_type,name,sector,updated_at) VALUES(?,?,?,?,?)",
            rows,
        )
        if rows:
            self.log(f"universe refreshed assets={len(rows)}")


# ---------------------------------------------------------------------------
# Investor agent: model inference and ATR-derived decisions
# ---------------------------------------------------------------------------


class InvestorAgent(BaseAgent):
    def __init__(self, database, events, stop_event) -> None:
        super().__init__("investor", database, events, stop_event)
        self.checkpoint = None
        self.model = None
        self.model_mtime_ns = 0
        self.last_missing_log = 0.0
        self.last_decision: dict[str, float] = {}
        self.skipped_untrained: set[str] = set()
        self.stock_data_client = None

    def _load_model(self) -> None:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        model = ShallowTradeNet(int(checkpoint["input_size"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.checkpoint = checkpoint
        self.model = model
        self.model_mtime_ns = MODEL_PATH.stat().st_mtime_ns

    def run(self) -> None:
        self.log("agent online")
        while not self.stop_event.is_set():
            try:
                current_mtime = MODEL_PATH.stat().st_mtime_ns
                if self.model is None or current_mtime != self.model_mtime_ns:
                    self._load_model()
                    self.log("trade model loaded/reloaded")
                self.evaluate_rankings()
            except FileNotFoundError:
                now = time.monotonic()
                if now - self.last_missing_log >= 60:
                    self.log("model missing; waiting for training", "WARN")
                    self.last_missing_log = now
            except Exception as exc:
                self.log(str(exc), "ERROR")
            self.stop_event.wait(INVESTOR_SECONDS)
        self.log("agent stopped")

    def evaluate_rankings(self) -> None:
        if self.checkpoint is None or self.model is None:
            raise RuntimeError("Investor model is not loaded.")
        rankings = self.database.query(
            "SELECT * FROM asset_rankings ORDER BY score DESC LIMIT 3"
        )
        mean = np.asarray(self.checkpoint["mean"], dtype=np.float32)
        std = np.asarray(self.checkpoint["std"], dtype=np.float32)
        timeframe = str(self.checkpoint.get("timeframe", "1Min"))
        for ranking in rankings:
            symbol = str(ranking["symbol"])
            trained_symbols = {
                str(item).upper() for item in self.checkpoint.get("symbols", [])
            }
            trained_asset = normalize_asset_type(
                str(self.checkpoint.get("asset_class", ""))
            )
            ranked_asset = normalize_asset_type(str(ranking["asset_type"]))
            if TRADE_TRAINED_SYMBOLS_ONLY and (
                symbol.upper() not in trained_symbols or ranked_asset != trained_asset
            ):
                if symbol not in self.skipped_untrained:
                    self.skipped_untrained.add(symbol)
                    self.log(
                        f"skipping untrained asset {symbol} ({ranked_asset}); "
                        f"model={trained_asset} {sorted(trained_symbols)}",
                        "WARN",
                    )
                continue
            rows = self.database.query(
                "SELECT * FROM live_prices WHERE symbol=? ORDER BY timestamp DESC LIMIT 500",
                (symbol,),
            )
            if not rows:
                continue
            if ranked_asset == ASSET_STOCK:
                if self.stock_data_client is None:
                    key, secret = api_credentials()
                    self.stock_data_client = StockHistoricalDataClient(key, secret)
                frame = fetch_alpaca_execution_bars(
                    self.stock_data_client,
                    symbol,
                    timeframe,
                    normalise_bar_frame(pd.DataFrame(), symbol),
                    self.events,
                )
            else:
                live = resample_live_bars(
                    pd.DataFrame([dict(row) for row in reversed(rows)]),
                    symbol,
                    timeframe,
                )
                frame = merge_bar_frames(
                    symbol,
                    read_cached_bars(symbol, timeframe).tail(240),
                    live,
                )
            if len(frame) < LOOKBACK + 5:
                continue
            model_input, timestamp, price = latest_model_input(
                frame,
                mean,
                std,
                0.0,  # Training sentiment is constant zero; ranking sentiment stays separate.
            )
            with torch.inference_mode():
                probabilities = torch.softmax(
                    self.model(torch.from_numpy(model_input).unsqueeze(0)), dim=1
                )[0]
            signal = int(probabilities.argmax().item())
            probability = float(probabilities[signal].item())
            if signal == 0 or probability < MIN_SIGNAL_CONFIDENCE:
                continue
            now = time.monotonic()
            if now - self.last_decision.get(symbol, 0.0) < DECISION_COOLDOWN_SECONDS:
                continue
            if self.database.query(
                "SELECT 1 FROM investment_decisions WHERE symbol=? "
                "AND status IN ('pending','submitted','partially_filled') LIMIT 1",
                (symbol,),
            ):
                continue
            featured = engineer_features(frame, include_target=False)
            atr_values = featured["atr_14_pct"].dropna()
            if atr_values.empty:
                continue
            atr = float(atr_values.iloc[-1] * price)
            stop = price - 2 * atr if signal == 1 else price + 2 * atr
            target = price + 3 * atr if signal == 1 else price - 3 * atr
            self.database.execute(
                "INSERT INTO investment_decisions"
                "(symbol,asset_type,timestamp,signal,probability,price,atr,stop_loss,"
                "take_profit,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,'pending',?)",
                (
                    symbol,
                    ranking["asset_type"],
                    timestamp.isoformat(),
                    SIGNAL_NAMES[signal],
                    probability,
                    price,
                    atr,
                    stop,
                    target,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.last_decision[symbol] = now
            self.log(f"{symbol} {SIGNAL_NAMES[signal]} probability={probability:.1%}")


# ---------------------------------------------------------------------------
# Cashier agent: paper-only execution and portfolio risk controls
# ---------------------------------------------------------------------------


class CashierAgent(BaseAgent):
    def __init__(self, database, events, stop_event) -> None:
        super().__init__("cashier", database, events, stop_event)
        self.client = None
        self.start_equity = 0.0
        self.peak_equity = 0.0
        self.equity_day = us_trading_day(datetime.now(timezone.utc))
        self.halted = False
        self.last_market_closed_log = 0.0
        self.trained_symbols: set[str] = set()
        self.trained_asset = ""
        self.last_stale_price_log: dict[str, float] = {}

    @staticmethod
    def alpaca_symbol(symbol: str, asset_type: str) -> str:
        if normalize_asset_type(asset_type) == ASSET_CRYPTO and symbol.endswith("USDT"):
            return f"{symbol[:-4]}/USD"
        return symbol

    def position_sector(self, symbol: str) -> str:
        normalized = symbol.upper().replace("/", "")
        if normalized.endswith(("USD", "USDT", "USDC")):
            return "Crypto"
        rows = self.database.query(
            "SELECT sector FROM universe_assets WHERE symbol=? LIMIT 1", (normalized,)
        )
        return str(rows[0]["sector"]) if rows else SECTOR_MAP.get(normalized, "Other")

    def _reset_daily_baseline(self, equity: float) -> None:
        today = us_trading_day(datetime.now(timezone.utc))
        if today != self.equity_day:
            self.equity_day = today
            self.start_equity = equity
            self.halted = False
            self.log(f"daily risk baseline reset equity={equity:.2f}")

    def run(self) -> None:
        self.log("agent online; endpoint=paper")
        try:
            key, secret = api_credentials()
            self.client = TradingClient(key, secret, paper=True)
            account = call_with_backoff(
                self.client.get_account, self.events, "paper account"
            )
            if bool(account.trading_blocked):
                raise UserFacingError("The Alpaca paper account is blocked from trading.")
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            self.trained_symbols = {
                str(symbol).upper() for symbol in checkpoint.get("symbols", [])
            }
            self.trained_asset = normalize_asset_type(
                str(checkpoint.get("asset_class", ""))
            )
            self.start_equity = self.peak_equity = float(account.equity)
            self.equity_day = us_trading_day(datetime.now(timezone.utc))
        except Exception as exc:
            self.log(f"paper connection failed: {exc}", "ERROR")
            self.events.put(
                {
                    "kind": "paper_error",
                    "context": "Paper trading",
                    "text": str(exc),
                }
            )
            return
        while not self.stop_event.is_set():
            try:
                self.process_pending()
                self.monitor_risk()
            except Exception as exc:
                self.log(str(exc), "ERROR")
            self.stop_event.wait(CASHIER_SECONDS)
        self.log("agent stopped")
        self.events.put({"kind": "paper_done", "text": "CASHIER | paper execution stopped"})

    def _risk_allows(self, decision: sqlite3.Row, equity: float) -> tuple[bool, str]:
        if str(decision["signal"]) == "SELL":
            return True, "risk-reducing exit"
        if self.halted:
            return False, "risk halt active"
        if equity <= self.start_equity * (1.0 - MAX_DAILY_LOSS):
            self.halted = True
            return False, "maximum daily loss reached"
        if equity <= self.peak_equity * (1.0 - MAX_GLOBAL_DRAWDOWN):
            self.halted = True
            return False, "global drawdown limit reached"
        sector = self.position_sector(str(decision["symbol"]))
        positions = call_with_backoff(
            self.client.get_all_positions, self.events, "paper positions"
        )
        same_sector = sum(
            1
            for position in positions
            if self.position_sector(str(position.symbol)) == sector
        )
        position_limit = (
            MAX_CRYPTO_POSITIONS if sector == "Crypto" else MAX_POSITIONS_PER_SECTOR
        )
        if same_sector >= position_limit:
            return False, f"sector position limit reached: {sector}"
        return True, "approved"

    def process_pending(self) -> None:
        if self.client is None:
            raise RuntimeError("Cashier paper client is not connected.")
        account = call_with_backoff(
            self.client.get_account, self.events, "paper account"
        )
        equity = float(account.equity)
        self._reset_daily_baseline(equity)
        self.peak_equity = max(self.peak_equity, equity)
        decisions = self.database.query(
            "SELECT * FROM investment_decisions WHERE status='pending' ORDER BY id LIMIT 10"
        )
        for decision in decisions:
            created_at = str(decision["created_at"] or "")
            try:
                created = pd.Timestamp(created_at)
                if created.tzinfo is None:
                    created = created.tz_localize("UTC")
                decision_age = (
                    pd.Timestamp.now(tz="UTC") - created.tz_convert("UTC")
                ).total_seconds()
            except (TypeError, ValueError):
                decision_age = float("inf")
            if decision_age > MAX_PENDING_DECISION_SECONDS:
                self.database.execute(
                    "UPDATE investment_decisions SET status='expired', note=? WHERE id=?",
                    (f"signal expired before execution age={decision_age:.0f}s", decision["id"]),
                )
                continue
            asset_type = normalize_asset_type(str(decision["asset_type"]))
            decision_symbol = str(decision["symbol"]).upper()
            if TRADE_TRAINED_SYMBOLS_ONLY and (
                decision_symbol not in self.trained_symbols
                or asset_type != self.trained_asset
            ):
                self.database.execute(
                    "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                    ("symbol or asset type is outside trained universe", decision["id"]),
                )
                self.log(f"{decision_symbol} rejected: outside trained universe", "WARN")
                continue
            if asset_type == ASSET_STOCK:
                clock = call_with_backoff(
                    self.client.get_clock, self.events, "paper market clock"
                )
                if not bool(clock.is_open):
                    now = time.monotonic()
                    if now - self.last_market_closed_log >= 60:
                        self.last_market_closed_log = now
                        self.log(
                            f"stock market closed; queued decisions wait until {clock.next_open}",
                            "WARN",
                        )
                    continue
            allowed, note = self._risk_allows(decision, equity)
            if not allowed:
                self.database.execute(
                    "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                    (note, decision["id"]),
                )
                self.log(f"{decision['symbol']} rejected: {note}", "WARN")
                continue
            symbol = self.alpaca_symbol(decision_symbol, asset_type)
            positions = call_with_backoff(
                self.client.get_all_positions, self.events, "paper positions"
            )
            normalized = symbol.upper().replace("/", "")
            matching_positions = [
                position
                for position in positions
                if str(position.symbol).upper().replace("/", "") == normalized
            ]
            open_orders = call_with_backoff(
                lambda: self.client.get_orders(
                    filter=GetOrdersRequest(
                        status=QueryOrderStatus.OPEN, symbols=[symbol]
                    )
                ),
                self.events,
                f"open paper orders {symbol}",
            )
            if open_orders:
                self.log(f"{symbol} waiting for an existing open order", "WARN")
                continue
            atr = max(float(decision["atr"]), float(decision["price"]) * 0.001)
            risk_quantity = (equity * RISK_PER_TRADE) / (2.0 * atr)
            cap_quantity = (equity * MAX_ASSET_ALLOCATION) / float(decision["price"])
            quantity = min(risk_quantity, cap_quantity)
            if asset_type == ASSET_STOCK:
                quantity = float(math.floor(quantity))
                if quantity < 1:
                    self.database.execute(
                        "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                        ("stock bracket sizing is below one whole share", decision["id"]),
                    )
                    continue
            if not math.isfinite(quantity) or quantity < MIN_ORDER_QUANTITY:
                self.database.execute(
                    "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                    ("calculated quantity is below minimum", decision["id"]),
                )
                continue
            signal = str(decision["signal"])
            client_order_id = ""
            if signal == "SELL":
                if not matching_positions:
                    self.database.execute(
                        "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                        ("no long position", decision["id"]),
                    )
                    continue
                try:
                    order = call_with_backoff(
                        lambda: self.client.close_position(symbol),
                        self.events,
                        f"close paper position {symbol}",
                    )
                    quantity = 0.0
                except APIError as exc:
                    self.database.execute(
                        "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                        (f"no long position: {exc}", decision["id"]),
                    )
                    continue
            else:
                if matching_positions:
                    self.database.execute(
                        "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                        ("position already open", decision["id"]),
                    )
                    continue
                client_order_id = f"korvax-{decision['id']}-{int(time.time())}"
                request_arguments = {
                    "symbol": symbol,
                    "qty": quantity,
                    "side": OrderSide.BUY,
                    "time_in_force": (
                        TimeInForce.GTC
                        if asset_type == ASSET_CRYPTO
                        else TimeInForce.DAY
                    ),
                    "client_order_id": client_order_id,
                }
                if asset_type == ASSET_STOCK:
                    entry = float(decision["price"])
                    stop_price = round(float(decision["stop_loss"]), 2)
                    take_profit = round(float(decision["take_profit"]), 2)
                    if not stop_price < entry < take_profit:
                        self.database.execute(
                            "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                            ("invalid stock bracket prices", decision["id"]),
                        )
                        continue
                    request_arguments.update(
                        {
                            "order_class": OrderClass.BRACKET,
                            "take_profit": TakeProfitRequest(limit_price=take_profit),
                            "stop_loss": StopLossRequest(stop_price=stop_price),
                        }
                    )
                request = MarketOrderRequest(**request_arguments)
                order = call_with_backoff(
                    lambda: self.client.submit_order(order_data=request),
                    self.events,
                    f"submit paper buy {symbol}",
                )
            order_id = str(getattr(order, "id", ""))
            final_status = "closed" if signal == "SELL" else "submitted"
            self.database.execute(
                "UPDATE investment_decisions SET status=?, note=?, client_order_id=? WHERE id=?",
                (final_status, order_id, client_order_id, decision["id"]),
            )
            pnl = equity - self.start_equity
            self.database.execute(
                "INSERT INTO portfolio_log(timestamp,symbol,action,quantity,price,equity,pnl,order_id,note) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    signal,
                    quantity,
                    float(decision["price"]),
                    equity,
                    pnl,
                    order_id,
                    "paper",
                ),
            )
            self.events.put(
                {
                    "kind": "paper_row",
                    "values": (
                        datetime.now().strftime("%Y-%m-%d %H:%M"),
                        symbol,
                        signal,
                        f"{float(decision['probability']):.1%}",
                        f"{float(decision['price']):.2f}",
                        f"{quantity:.6g}",
                        f"{equity:.2f}",
                        f"{pnl:+.2f}",
                        "CLOSED" if signal == "SELL" else "SUBMITTED",
                    ),
                }
            )
            self.log(f"paper {signal} submitted {symbol} qty={quantity:.6g}")

    def reconcile_orders(self) -> None:
        if self.client is None:
            raise RuntimeError("Cashier paper client is not connected.")
        decisions = self.database.query(
            "SELECT * FROM investment_decisions "
            "WHERE status IN ('submitted','partially_filled') AND client_order_id<>''"
        )
        terminal_rejections = {
            OrderStatus.REJECTED.value,
            OrderStatus.CANCELED.value,
            OrderStatus.EXPIRED.value,
            OrderStatus.DONE_FOR_DAY.value,
        }
        for decision in decisions:
            client_order_id = str(decision["client_order_id"])
            order = call_with_backoff(
                lambda: self.client.get_order_by_client_id(client_order_id),
                self.events,
                f"reconcile order {client_order_id}",
            )
            raw_status = getattr(order, "status", "")
            status = str(getattr(raw_status, "value", raw_status)).lower()
            filled_quantity = float(getattr(order, "filled_qty", 0) or 0)
            if status == OrderStatus.FILLED.value:
                self.database.execute(
                    "UPDATE investment_decisions SET status='filled', filled_quantity=?, note=? "
                    "WHERE id=?",
                    (filled_quantity, f"broker status={status}", decision["id"]),
                )
            elif status == OrderStatus.PARTIALLY_FILLED.value:
                self.database.execute(
                    "UPDATE investment_decisions SET status='partially_filled', "
                    "filled_quantity=?, note=? WHERE id=?",
                    (filled_quantity, f"broker status={status}", decision["id"]),
                )
            elif status in terminal_rejections:
                reason = str(getattr(order, "reject_reason", "") or status)
                self.database.execute(
                    "UPDATE investment_decisions SET status='rejected', note=? WHERE id=?",
                    (f"broker {status}: {reason}", decision["id"]),
                )
                self.log(f"order {client_order_id} reconciled as {status}", "WARN")

    def monitor_risk(self) -> None:
        if self.client is None:
            raise RuntimeError("Cashier paper client is not connected.")
        self.reconcile_orders()
        account = call_with_backoff(
            self.client.get_account, self.events, "paper account"
        )
        equity = float(account.equity)
        self._reset_daily_baseline(equity)
        self.peak_equity = max(self.peak_equity, equity)
        if equity <= self.peak_equity * (1.0 - MAX_GLOBAL_DRAWDOWN):
            self.halted = True
            self.log("20% drawdown reached; new orders halted", "ERROR")
        decisions = self.database.query(
            "SELECT * FROM investment_decisions "
            "WHERE status='filled' AND signal='BUY' AND asset_type=?",
            (ASSET_CRYPTO,),
        )
        for decision in decisions:
            latest = self.database.query(
                "SELECT timestamp,close FROM live_prices WHERE symbol=? "
                "ORDER BY timestamp DESC LIMIT 1",
                (decision["symbol"],),
            )
            if not latest:
                continue
            price_time = pd.Timestamp(latest[0]["timestamp"])
            if price_time.tzinfo is None:
                price_time = price_time.tz_localize("UTC")
            age = (
                pd.Timestamp.now(tz="UTC") - price_time.tz_convert("UTC")
            ).total_seconds()
            if age > MAX_PRICE_AGE_SECONDS:
                symbol = str(decision["symbol"])
                now = time.monotonic()
                if now - self.last_stale_price_log.get(symbol, 0.0) >= 60:
                    self.last_stale_price_log[symbol] = now
                    self.log(
                        f"stale crypto price for {symbol} age={age:.0f}s; "
                        "synthetic stop deferred",
                        "WARN",
                    )
                continue
            price = float(latest[0]["close"])
            stop = float(decision["stop_loss"])
            entry = float(decision["price"])
            atr = float(decision["atr"])
            if price >= entry + atr and stop < entry:
                stop = entry
                self.database.execute(
                    "UPDATE investment_decisions SET stop_loss=?, note='stop moved to breakeven' WHERE id=?",
                    (stop, decision["id"]),
                )
                self.log(f"{decision['symbol']} stop moved to breakeven")
            if price <= stop or price >= float(decision["take_profit"]):
                symbol = self.alpaca_symbol(
                    str(decision["symbol"]), str(decision["asset_type"])
                )
                try:
                    self.client.close_position(symbol)
                    reason = "stop" if price <= stop else "target"
                    self.database.execute(
                        "UPDATE investment_decisions SET status='closed', note=? WHERE id=?",
                        (reason, decision["id"]),
                    )
                    self.log(f"{symbol} exited at {reason} price={price:.4g}")
                except APIError as exc:
                    self.log(f"{symbol} exit check: {exc}", "WARN")


class SymbolDialog(tk.Toplevel):
    def __init__(
        self, parent: tk.Tk, symbols: list[str], timeframe: str, asset_class: str
    ) -> None:
        super().__init__(parent)
        self.title("Market Universe")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result: tuple[list[str], str, str] | None = None
        self.symbol_var = tk.StringVar(value=", ".join(symbols))
        self.timeframe_var = tk.StringVar(value=timeframe)
        self.asset_class_var = tk.StringVar(value=asset_class)

        frame = ttk.LabelFrame(
            self, text=" FREE MARKET DATA ", style="Panel.TLabelframe", padding=8
        )
        frame.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        ttk.Label(frame, text="SYMBOLS / COMMA SEPARATED", style="Meta.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        entry = tk.Entry(
            frame,
            textvariable=self.symbol_var,
            width=44,
            bg=BLACK,
            fg=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 10),
        )
        entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(3, 8))
        ttk.Label(frame, text="ASSET / DATA SOURCE", style="Meta.TLabel").grid(
            row=2, column=0, sticky="w"
        )
        asset_combo = ttk.Combobox(
            frame,
            textvariable=self.asset_class_var,
            values=("stocks", "crypto"),
            state="readonly",
            width=12,
            font=("Consolas", 9),
        )
        asset_combo.grid(row=3, column=0, sticky="w", pady=(3, 8))
        asset_combo.bind("<<ComboboxSelected>>", self._source_changed)
        ttk.Label(frame, text="STOCKS=MASSIVE  CRYPTO=BINANCE", style="Meta.TLabel").grid(
            row=3, column=1, sticky="e", padx=(12, 0)
        )
        ttk.Label(frame, text="BAR TIMEFRAME", style="Meta.TLabel").grid(
            row=4, column=0, sticky="w"
        )
        combo = ttk.Combobox(
            frame,
            textvariable=self.timeframe_var,
            values=("1Day", "1Min"),
            state="readonly",
            width=12,
            font=("Consolas", 9),
        )
        combo.grid(row=5, column=0, sticky="w", pady=(3, 8))
        ttk.Button(
            frame, text="Apply", command=self._apply, style="Flat.TButton"
        ).grid(row=6, column=0, sticky="w")
        ttk.Button(
            frame, text="Cancel", command=self.destroy, style="Flat.TButton"
        ).grid(row=6, column=1, sticky="e")
        entry.focus_set()
        self.bind("<Return>", lambda _event: self._apply())
        self.bind("<Escape>", lambda _event: self.destroy())

    def _source_changed(self, _event=None) -> None:
        current = {
            item.strip().upper()
            for item in self.symbol_var.get().replace(";", ",").split(",")
            if item.strip()
        }
        known_defaults = {frozenset(DEFAULT_SYMBOLS), frozenset(DEFAULT_CRYPTO_SYMBOLS)}
        if frozenset(current) not in known_defaults:
            return
        defaults = (
            DEFAULT_SYMBOLS
            if self.asset_class_var.get() == "stocks"
            else DEFAULT_CRYPTO_SYMBOLS
        )
        self.symbol_var.set(", ".join(defaults))

    def _apply(self) -> None:
        symbols = [
            item.strip().upper()
            for item in self.symbol_var.get().replace(";", ",").split(",")
            if item.strip()
        ]
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            messagebox.showerror("Symbols required", "Enter at least one symbol.", parent=self)
            return
        asset_class = self.asset_class_var.get()
        if any(
            not symbol.replace(".", "").replace("-", "").replace("/", "").isalnum()
            for symbol in symbols
        ):
            messagebox.showerror("Invalid symbol", "Use standard ticker symbols.", parent=self)
            return
        if asset_class == "crypto":
            symbols = [symbol.replace("/", "").replace("-", "") for symbol in symbols]
        self.result = symbols, self.timeframe_var.get(), asset_class
        self.destroy()


class NetworkSimulation:
    def __init__(self, app: "MainApp") -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Korvax TradeMind / Network Trace")
        self.window.configure(bg=BLACK)
        self.window.geometry("930x520")
        self.window.minsize(760, 430)
        self.canvas = tk.Canvas(
            self.window, bg=BLACK, highlightthickness=0, borderwidth=0
        )
        self.canvas.pack(fill="both", expand=True)
        self.input_centres: list[tuple[float, float]] = []
        self.hidden_one = (0.0, 0.0, 0.0, 0.0)
        self.hidden_two = (0.0, 0.0, 0.0, 0.0)
        self.output_centres: dict[int, tuple[float, float]] = {}
        self.output_nodes: dict[int, int] = {}
        self.packet: int | None = None
        self.status_item: int | None = None
        self.resize_job: str | None = None
        self.completed_until = 0.0
        self.canvas.bind("<Configure>", self._schedule_redraw)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(100, self.draw)
        self.window.after(200, self.pulse)

    def close(self) -> None:
        self.app.simulation_window = None
        self.window.destroy()

    def _schedule_redraw(self, _event=None) -> None:
        if self.resize_job is not None:
            self.window.after_cancel(self.resize_job)
        self.resize_job = self.window.after(80, self.draw)

    def draw(self) -> None:
        if not self.window.winfo_exists():
            return
        self.resize_job = None
        self.canvas.delete("all")
        width = max(760, self.canvas.winfo_width())
        height = max(430, self.canvas.winfo_height())
        font = ("Consolas", 9)
        small = ("Consolas", 8)

        input_x = width * 0.10
        input_labels = [f"F{i:02d}" for i in range(1, 13)]
        input_y = np.linspace(height * 0.13, height * 0.77, 12).tolist()
        self.input_centres = list(zip([input_x] * len(input_y), input_y))
        self.hidden_one = (width * 0.32, height * 0.22, width * 0.45, height * 0.68)
        self.hidden_two = (width * 0.56, height * 0.30, width * 0.68, height * 0.60)
        output_x = width * 0.87
        self.output_centres = {
            1: (output_x, height * 0.27),
            2: (output_x, height * 0.45),
            0: (output_x, height * 0.63),
        }
        h1_centre = ((self.hidden_one[0] + self.hidden_one[2]) / 2, height * 0.45)
        h2_centre = ((self.hidden_two[0] + self.hidden_two[2]) / 2, height * 0.45)

        for centre in self.input_centres:
            self.canvas.create_line(
                centre[0] + 9,
                centre[1],
                self.hidden_one[0],
                h1_centre[1],
                fill=ACCENT,
            )
        self.canvas.create_line(
            self.hidden_one[2], h1_centre[1], self.hidden_two[0], h2_centre[1], fill=ACCENT
        )
        for centre in self.output_centres.values():
            self.canvas.create_line(
                self.hidden_two[2], h2_centre[1], centre[0] - 11, centre[1], fill=ACCENT
            )

        for (x, y), label in zip(self.input_centres, input_labels):
            self.canvas.create_oval(
                x - 7, y - 7, x + 7, y + 7, fill=BG, outline=ACCENT, width=1
            )
            self.canvas.create_text(
                x - 14, y, anchor="e", text=label, fill=TEXT, font=small
            )

        self._draw_hidden_box(self.hidden_one, "DENSE 01", "20 FEATURES  ->  32", font, small)
        self._draw_hidden_box(self.hidden_two, "DENSE 02", "32  ->  16", font, small)
        self.output_nodes.clear()
        for class_id in (1, 2, 0):
            x, y = self.output_centres[class_id]
            node = self.canvas.create_oval(
                x - 10, y - 10, x + 10, y + 10, fill=TROUGH, outline=ACCENT, width=1
            )
            self.output_nodes[class_id] = node
            self.canvas.create_text(
                x + 18,
                y,
                anchor="w",
                text=f"[{class_id}]  {SIGNAL_NAMES[class_id]}",
                fill=TEXT,
                font=font,
            )

        self.canvas.create_text(
            20, 15, anchor="w", text="12 VISIBLE NODES / 20 MODEL FEATURES", fill=ACCENT, font=small
        )
        self.canvas.create_text(
            width - 20,
            15,
            anchor="e",
            text="3-CLASS LOGITS / SOFTMAX AT INFERENCE",
            fill=ACCENT,
            font=small,
        )
        self.canvas.create_line(18, height - 53, width - 18, height - 53, fill=ACCENT)
        self.status_item = self.canvas.create_text(
            20,
            height - 28,
            anchor="w",
            text=self.app.training_phase,
            fill=ACCENT,
            font=small,
        )
        self.packet = None

    def _draw_hidden_box(
        self,
        box: tuple[float, float, float, float],
        title: str,
        detail: str,
        font,
        small,
    ) -> None:
        left, top, right, bottom = box
        self.canvas.create_rectangle(
            left, top, right, bottom, fill=BG, outline=ACCENT, width=1
        )
        self.canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2 - 10,
            text=title,
            fill=TEXT,
            font=font,
        )
        self.canvas.create_text(
            (left + right) / 2,
            (top + bottom) / 2 + 12,
            text=detail,
            fill=ACCENT,
            font=small,
        )

    def pulse(self) -> None:
        if not self.window.winfo_exists():
            return
        self.window.after(500, self.pulse)
        if not self.app.training:
            if self.packet is not None:
                self.canvas.delete(self.packet)
                self.packet = None
            if self.status_item is not None:
                self.canvas.itemconfigure(
                    self.status_item,
                    text=self.app.training_phase,
                    fill=TEXT if time.monotonic() < self.completed_until else ACCENT,
                )
            return
        class_id = self.app.current_batch_label
        start = self.input_centres[self.app.current_batch % len(self.input_centres)]
        h1 = (self.hidden_one[0], (self.hidden_one[1] + self.hidden_one[3]) / 2)
        h1_out = (self.hidden_one[2], h1[1])
        h2 = (self.hidden_two[0], (self.hidden_two[1] + self.hidden_two[3]) / 2)
        h2_out = (self.hidden_two[2], h2[1])
        output = self.output_centres[class_id]
        path = [start, h1, h1_out, h2, h2_out, output]
        self.packet = self.canvas.create_oval(0, 0, 6, 6, fill=WHITE, outline="")
        if self.status_item is not None:
            self.canvas.itemconfigure(
                self.status_item,
                text=(
                    f"STATE TRAINING  |  EPOCH {self.app.current_epoch:02d}  |  "
                    f"BATCH {self.app.current_batch:04d}  |  MAJORITY {SIGNAL_NAMES[class_id]}"
                ),
                fill=TEXT,
            )
        self._animate(path, class_id, 0, 18)

    def _animate(self, path, class_id: int, frame: int, total: int) -> None:
        if not self.window.winfo_exists() or self.packet is None:
            return
        if frame > total:
            self.canvas.delete(self.packet)
            self.packet = None
            node = self.output_nodes[class_id]
            self.canvas.itemconfigure(node, fill=WHITE)
            self.window.after(260, lambda: self._dim(node))
            return
        position = frame / total * (len(path) - 1)
        segment = min(int(position), len(path) - 2)
        fraction = position - segment
        x1, y1 = path[segment]
        x2, y2 = path[segment + 1]
        x = x1 + (x2 - x1) * fraction
        y = y1 + (y2 - y1) * fraction
        self.canvas.coords(self.packet, x - 3, y - 3, x + 3, y + 3)
        self.window.after(20, self._animate, path, class_id, frame + 1, total)

    def mark_complete(self) -> None:
        self.completed_until = time.monotonic() + 5.0
        if self.status_item is not None:
            self.canvas.itemconfigure(
                self.status_item, text=self.app.training_phase, fill=TEXT
            )

    def _dim(self, node: int) -> None:
        if self.window.winfo_exists():
            try:
                self.canvas.itemconfigure(node, fill=TROUGH)
            except tk.TclError:
                pass


class MainApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Korvax TradeMind / Multi-Agent Paper Console")
        self.root.configure(bg=BG)
        self.root.geometry("1280x760")
        self.root.minsize(1040, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.events: queue.Queue = queue.Queue()
        self.database = AgentDatabase()
        self.watchlist = WatchlistState()
        self.core_agent_stop = threading.Event()
        self.cashier_stop = threading.Event()
        self.agent_threads: list[BaseAgent] = []
        self.cashier_agent: CashierAgent | None = None
        has_massive_key = bool(
            os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        )
        self.symbols = (
            DEFAULT_SYMBOLS.copy() if has_massive_key else DEFAULT_CRYPTO_SYMBOLS.copy()
        )
        self.timeframe = DEFAULT_TIMEFRAME
        self.asset_class = DEFAULT_ASSET_CLASS if has_massive_key else "crypto"
        self.watchlist.set_primary(self.symbols, self.asset_class)
        self.logs: deque[str] = deque(maxlen=200)
        self.busy = False
        self.training = False
        self.paper_running = False
        self.closing = False
        self.stop_training_event = threading.Event()
        self.operation_thread: threading.Thread | None = None
        self.train_after_download = False

        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []
        self.validation_accuracies: list[float] = []
        self.validation_f1_scores: list[float] = []
        self.training_epoch_numbers: list[int] = []
        self.current_epoch = 0
        self.current_batch = 0
        self.batches_per_epoch = 0
        self.current_batch_label = 0
        self.elapsed_seconds = 0.0
        self.started_at = 0.0
        self.latest_train_loss: float | None = None
        self.latest_validation_loss: float | None = None
        self.latest_accuracy: float | None = None
        self.latest_macro_f1: float | None = None
        self.latest_learning_rate = LEARNING_RATE
        self.latest_confusion: list[list[int]] = []
        self.latest_class_counts = {0: 0, 1: 0, 2: 0}
        self.paper_equity_points: list[float] = []
        self.training_phase = "STATE IDLE  |  NO ACTIVE TRAINING RUN"
        self.simulation_window: NetworkSimulation | None = None

        self._build_style()
        self._build_ui()
        self._seed_database_from_cache()
        self._start_core_agents()
        data_state = "present" if self._massive_credentials_present() else "missing"
        paper_state = "present" if self._credentials_present() else "missing"
        self._log(
            f"READY  | symbols={','.join(self.symbols)}  timeframe={self.timeframe}  "
            f"massive={data_state}  alpaca_paper={paper_state}"
        )
        self.root.after(100, self._poll_events)
        self.root.after(1000, self._update_clock)
        self.root.after(2000, self._redraw_chart)

    def _seed_database_from_cache(self) -> None:
        candidates = [(symbol, "stock") for symbol in DEFAULT_SYMBOLS]
        candidates.extend((symbol, "crypto") for symbol in DEFAULT_CRYPTO_SYMBOLS)
        candidates.extend(
            (symbol, "stock" if self.asset_class == "stocks" else "crypto")
            for symbol in self.symbols
        )
        inserted = 0
        for symbol, asset_type in dict.fromkeys(candidates):
            for timeframe in (self.timeframe, "1Day", "1Min"):
                try:
                    frame = read_cached_bars(symbol, timeframe).tail(500)
                except Exception as exc:
                    frame = normalise_bar_frame(pd.DataFrame(), symbol)
                    self.database.log(
                        "monitor",
                        f"cache seed skipped {symbol} {timeframe}: {exc}",
                        "WARN",
                    )
                if frame.empty:
                    continue
                rows = [
                    (
                        symbol,
                        asset_type,
                        pd.Timestamp(row.timestamp).isoformat(),
                        float(row.open),
                        float(row.high),
                        float(row.low),
                        float(row.close),
                        float(row.volume),
                    )
                    for row in frame.itertuples()
                ]
                self.database.executemany(
                    "INSERT OR REPLACE INTO live_prices"
                    "(symbol,asset_type,timestamp,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?,?)",
                    rows,
                )
                inserted += len(rows)
                break
        if inserted:
            self._log(f"DATABASE| seeded historical bars={inserted}")

    def _start_core_agents(self) -> None:
        if self.agent_threads:
            return
        self.agent_threads = [
            ResearcherAgent(
                self.database, self.events, self.core_agent_stop, self.watchlist
            ),
            AnalystAgent(
                self.database, self.events, self.core_agent_stop, self.watchlist
            ),
            InvestorAgent(self.database, self.events, self.core_agent_stop),
        ]
        for agent in self.agent_threads:
            agent.start()

    @staticmethod
    def _credentials_present() -> bool:
        try:
            api_credentials()
            return True
        except UserFacingError:
            return False

    @staticmethod
    def _massive_credentials_present() -> bool:
        try:
            massive_api_key()
            return True
        except UserFacingError:
            return False

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Consolas", 9))
        style.configure(
            "Title.TLabel", background=BG, foreground=TEXT, font=("Consolas", 11, "bold")
        )
        style.configure(
            "Meta.TLabel", background=BG, foreground=DIM, font=("Consolas", 8)
        )
        style.configure(
            "Status.TLabel", background=BG, foreground=TEXT, font=("Consolas", 9, "bold")
        )
        style.configure(
            "Panel.TLabelframe",
            background=BG,
            bordercolor=ACCENT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background=BG,
            foreground=ACCENT,
            font=("Consolas", 8, "bold"),
        )
        style.configure(
            "Flat.TButton",
            background=BG,
            foreground=TEXT,
            bordercolor=ACCENT,
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            relief="flat",
            borderwidth=1,
            padding=(7, 3),
            font=("Consolas", 9),
        )
        style.map(
            "Flat.TButton",
            background=[("active", HOVER), ("pressed", TROUGH), ("disabled", BG)],
            foreground=[("disabled", ACCENT)],
        )
        style.configure(
            "Trade.Horizontal.TProgressbar",
            troughcolor=TROUGH,
            background=PROGRESS,
            bordercolor=ACCENT,
            lightcolor=PROGRESS,
            darkcolor=PROGRESS,
            thickness=10,
        )
        style.configure(
            "Trade.TNotebook", background=BG, bordercolor=ACCENT, tabmargins=(0, 0, 0, 0)
        )
        style.configure(
            "Trade.TNotebook.Tab",
            background=BG,
            foreground=DIM,
            bordercolor=ACCENT,
            padding=(10, 4),
            font=("Consolas", 8, "bold"),
        )
        style.map(
            "Trade.TNotebook.Tab",
            background=[("selected", HOVER)],
            foreground=[("selected", TEXT)],
        )
        style.configure(
            "Trade.Treeview",
            background=BLACK,
            fieldbackground=BLACK,
            foreground=TEXT,
            rowheight=22,
            bordercolor=ACCENT,
            font=("Consolas", 8),
        )
        style.configure(
            "Trade.Treeview.Heading",
            background=BG,
            foreground=TEXT,
            bordercolor=ACCENT,
            font=("Consolas", 8, "bold"),
        )
        style.map("Trade.Treeview", background=[("selected", HOVER)])

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        top = ttk.Frame(self.root)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(5, 3))
        top.grid_columnconfigure(0, weight=1)
        ttk.Label(top, text="KORVAX TRADEMIND", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.status_var = tk.StringVar(value="STATE  IDLE")
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=1, sticky="e"
        )
        self.meta_var = tk.StringVar()
        self._refresh_meta()
        ttk.Label(top, textvariable=self.meta_var, style="Meta.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(3, 0)
        )

        left = ttk.LabelFrame(
            self.root,
            text=" EVENT LOG ",
            style="Panel.TLabelframe",
            padding=(5, 4),
        )
        left.grid(row=1, column=0, sticky="nsew", padx=(6, 3), pady=3)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.log_box = tk.Listbox(
            left,
            width=53,
            bg=BLACK,
            fg=TEXT,
            selectbackground=HOVER,
            selectforeground=WHITE,
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            font=("Consolas", 9),
            activestyle="none",
        )
        self.log_box.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        log_scroll = ttk.Scrollbar(left, orient="vertical", command=self.log_box.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=log_scroll.set)

        right = ttk.LabelFrame(
            self.root,
            text=" RESEARCH / PAPER ACCOUNT ",
            style="Panel.TLabelframe",
            padding=(4, 4),
        )
        right.grid(row=1, column=1, sticky="nsew", padx=(3, 6), pady=3)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(right, style="Trade.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew")

        training_tab = ttk.Frame(self.notebook)
        rankings_tab = ttk.Frame(self.notebook)
        paper_tab = ttk.Frame(self.notebook)
        self.notebook.add(training_tab, text="TRAINING")
        self.notebook.add(rankings_tab, text="RANKINGS")
        self.notebook.add(paper_tab, text="PAPER P&L")
        training_tab.grid_rowconfigure(0, weight=1)
        training_tab.grid_columnconfigure(0, weight=1)
        paper_tab.grid_rowconfigure(0, weight=1)
        paper_tab.grid_rowconfigure(1, weight=2)
        paper_tab.grid_columnconfigure(0, weight=1)
        rankings_tab.grid_rowconfigure(0, weight=1)
        rankings_tab.grid_columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(7.4, 5.2), dpi=100, facecolor=BG)
        self.loss_axes = self.figure.add_subplot(211)
        self.accuracy_axes = self.figure.add_subplot(212)
        self.figure.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.10, hspace=0.36)
        self._style_axes()
        self.chart = FigureCanvasTkAgg(self.figure, master=training_tab)
        chart_widget = self.chart.get_tk_widget()
        chart_widget.configure(bg=BG, highlightthickness=0)
        chart_widget.grid(row=0, column=0, sticky="nsew")

        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(
            training_tab,
            maximum=100,
            variable=self.progress_var,
            style="Trade.Horizontal.TProgressbar",
        ).grid(row=1, column=0, sticky="ew", pady=(4, 3))
        self.telemetry_var = tk.StringVar()
        self._refresh_telemetry()
        ttk.Label(training_tab, textvariable=self.telemetry_var, justify="left").grid(
            row=2, column=0, sticky="ew"
        )

        ranking_columns = ("rank", "symbol", "type", "score", "price", "sentiment", "adx")
        self.rankings_table = ttk.Treeview(
            rankings_tab,
            columns=ranking_columns,
            show="headings",
            style="Trade.Treeview",
        )
        for column, width in zip(ranking_columns, (55, 90, 75, 85, 110, 95, 80)):
            self.rankings_table.heading(column, text=column.upper())
            self.rankings_table.column(column, width=width, anchor="center", stretch=True)
        self.rankings_table.grid(row=0, column=0, sticky="nsew")

        paper_columns = (
            "time",
            "symbol",
            "signal",
            "confidence",
            "price",
            "position",
            "equity",
            "pnl",
            "action",
        )
        self.paper_figure = Figure(figsize=(7.4, 1.8), dpi=100, facecolor=BG)
        self.paper_axes = self.paper_figure.add_subplot(111)
        self.paper_axes.set_facecolor(BG)
        self.paper_axes.set_title("PAPER EQUITY", color=TEXT, fontsize=8)
        self.paper_axes.tick_params(colors=TEXT, labelsize=7)
        for spine in self.paper_axes.spines.values():
            spine.set_color(ACCENT)
        self.paper_chart = FigureCanvasTkAgg(self.paper_figure, master=paper_tab)
        self.paper_chart.get_tk_widget().configure(bg=BG, highlightthickness=0)
        self.paper_chart.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.paper_table = ttk.Treeview(
            paper_tab,
            columns=paper_columns,
            show="headings",
            style="Trade.Treeview",
        )
        widths = (125, 70, 75, 85, 85, 75, 100, 90, 170)
        for column, width in zip(paper_columns, widths):
            self.paper_table.heading(column, text=column.upper())
            self.paper_table.column(column, width=width, anchor="center", stretch=True)
        self.paper_table.grid(row=1, column=0, sticky="nsew")
        paper_scroll = ttk.Scrollbar(
            paper_tab, orient="vertical", command=self.paper_table.yview
        )
        paper_scroll.grid(row=1, column=1, sticky="ns")
        self.paper_table.configure(yscrollcommand=paper_scroll.set)

        bottom = ttk.LabelFrame(
            self.root,
            text=" CONTROL ",
            style="Panel.TLabelframe",
            padding=(4, 4),
        )
        bottom.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=6,
            pady=(3, 6),
        )
        self.symbol_button = self._button(bottom, "Load Symbols", self.load_symbols)
        self.download_button = self._button(bottom, "Download Data", self.download_data)
        self.train_button = self._button(bottom, "Start Training", self.start_training)
        self.stop_button = self._button(
            bottom, "Stop Training", self.stop_training, state="disabled"
        )
        self.simulation_button = self._button(
            bottom, "Open Simulation", self.open_simulation
        )
        self.paper_button = self._button(
            bottom, "Start Paper Trading", self.toggle_paper_trading
        )
        self.reset_button = self._button(bottom, "Reset Model", self.reset_model)

    def _button(self, parent, text: str, command, state: str = "normal") -> ttk.Button:
        button = ttk.Button(
            parent, text=text, command=command, style="Flat.TButton", state=state
        )
        button.pack(side="left", padx=(0, 3))
        return button

    def open_simulation(self) -> None:
        if (
            self.simulation_window is not None
            and self.simulation_window.window.winfo_exists()
        ):
            self.simulation_window.window.deiconify()
            self.simulation_window.window.lift()
            self.simulation_window.window.focus_force()
            return
        self.simulation_window = NetworkSimulation(self)

    def _style_axes(self) -> None:
        for axes in (self.loss_axes, self.accuracy_axes):
            axes.set_facecolor(BG)
            axes.grid(False)
            axes.spines["top"].set_visible(False)
            axes.spines["right"].set_visible(False)
            axes.spines["bottom"].set_color(ACCENT)
            axes.spines["left"].set_color(ACCENT)
            axes.tick_params(colors=TEXT, labelsize=8, width=0.7, length=3)
        self.loss_axes.set_ylabel("LOSS", color=TEXT, fontsize=8)
        self.accuracy_axes.set_ylabel("VAL ACCURACY", color=TEXT, fontsize=8)
        self.accuracy_axes.set_xlabel("EPOCH", color=TEXT, fontsize=8)
        self.accuracy_axes.set_ylim(0, 1.02)

    def _refresh_meta(self) -> None:
        cache_rows = 0
        for symbol in self.symbols:
            try:
                cache_rows += len(read_cached_bars(symbol, self.timeframe))
            except Exception as exc:
                self.logs.append(
                    f"{time.strftime('%H:%M:%S')}  META   | cache read failed {symbol}: {exc}"
                )
        if self.asset_class == "stocks":
            data_source = (
                "MASSIVE READY" if self._massive_credentials_present() else "YAHOO FALLBACK"
            )
        else:
            data_source = "BINANCE PUBLIC"
        paper_credentials = "READY" if self._credentials_present() else "KEYS MISSING"
        model = "READY" if MODEL_PATH.exists() else "MISSING"
        threshold = training_move_threshold(self.asset_class, self.timeframe)
        self.meta_var.set(
            f"ASSET {self.asset_class.upper()}  |  SYMBOLS {','.join(self.symbols)}  |  BARS {self.timeframe}  |  "
            f"CACHE {cache_rows} ROWS  |  WINDOW {LOOKBACK}  |  HORIZON {FORWARD_HORIZON}  |  "
            f"THRESHOLD {threshold:.2%}  |  NETWORK {len(MODEL_FEATURES)}-32-16-3  |  "
            f"DATA {data_source}  |  MODEL {model}  |  ALPACA PAPER {paper_credentials}"
        )

    def _log(self, message: str) -> None:
        self.logs.append(f"{time.strftime('%H:%M:%S')}  {message}")
        try:
            self.database.log("monitor", message)
        except Exception as exc:
            self.logs.append(
                f"{time.strftime('%H:%M:%S')}  LOGGING| database unavailable: {exc}"
            )
        self.log_box.delete(0, tk.END)
        for line in self.logs:
            self.log_box.insert(tk.END, line)
        self.log_box.yview_moveto(1.0)

    def _set_controls(self, busy: bool) -> None:
        self.busy = busy
        normal = "disabled" if busy else "normal"
        self.symbol_button.configure(state=normal)
        self.download_button.configure(state=normal)
        self.train_button.configure(state=normal)
        self.reset_button.configure(state=normal)
        self.stop_button.configure(state="normal" if self.training else "disabled")
        if self.paper_running:
            self.paper_button.configure(state="normal")
        elif busy:
            self.paper_button.configure(state="disabled")
        else:
            self.paper_button.configure(state="normal")

    def reset_model(self) -> None:
        if self.training or self.paper_running:
            return
        if not MODEL_PATH.exists():
            self._log("MODEL   | no checkpoint to reset")
            return
        if not messagebox.askyesno(
            "Reset model",
            "Delete the trained trade model and start over?",
            parent=self.root,
        ):
            return
        MODEL_PATH.unlink(missing_ok=True)
        for agent in self.agent_threads:
            if isinstance(agent, InvestorAgent):
                agent.model = None
                agent.checkpoint = None
        self._refresh_meta()
        self._log("MODEL   | checkpoint reset")

    def load_symbols(self) -> None:
        if self.busy or self.paper_running:
            return
        dialog = SymbolDialog(
            self.root, self.symbols, self.timeframe, self.asset_class
        )
        self.root.wait_window(dialog)
        if dialog.result is None:
            return
        self.symbols, self.timeframe, self.asset_class = dialog.result
        self.watchlist.set_primary(self.symbols, self.asset_class)
        self._refresh_meta()
        self._log(
            f"UNIVERSE | asset={self.asset_class}  symbols={','.join(self.symbols)}  "
            f"timeframe={self.timeframe}"
        )

    def download_data(self) -> None:
        if self.busy:
            return
        self.train_after_download = False
        self.status_var.set("STATE  DOWNLOADING")
        self._set_controls(True)
        self.operation_thread = threading.Thread(
            target=self._download_worker, name="free-data-download", daemon=True
        )
        self.operation_thread.start()

    def _download_worker(self) -> None:
        try:
            total = 0
            failures: list[str] = []
            for symbol in self.symbols:
                if self.stop_training_event.is_set() and self.train_after_download:
                    raise InterruptedError
                try:
                    frame = sync_training_bars(
                        self.asset_class, symbol, self.timeframe, self.events
                    )
                    total += len(frame)
                except Exception as exc:
                    failures.append(f"{symbol}: {exc}")
                    self.events.put(
                        {"kind": "log", "text": f"CACHE  | {symbol} failed: {exc}"}
                    )
            if failures and self.train_after_download:
                raise UserFacingError(
                    "Training cache could not be prepared:\n" + "\n".join(failures)
                )
            source = (
                "Massive"
                if self.asset_class == "stocks"
                and (os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY"))
                else "Yahoo"
                if self.asset_class == "stocks"
                else "Binance"
            )
            self.events.put(
                {
                    "kind": "operation_done",
                    "status": "PARTIAL" if failures else "IDLE",
                    "start_training": self.train_after_download,
                    "text": (
                        f"CACHE  | source={source} synchronization complete  "
                        f"total_rows={total} failures={len(failures)}"
                    ),
                }
            )
        except InterruptedError:
            self.events.put(
                {
                    "kind": "training_done",
                    "stopped": True,
                    "text": "STOP   | cache preparation cancelled",
                    "best_epoch": 0,
                    "epochs": 0,
                }
            )
        except Exception as exc:
            self.events.put(
                {
                    "kind": "error",
                    "context": "Data download",
                    "text": str(exc),
                }
            )

    def start_training(self) -> None:
        if self.paper_running:
            self._log("TRAIN  | blocked while paper execution is active")
            messagebox.showwarning(
                "Stop paper trading",
                "Stop paper trading before starting a training run.",
                parent=self.root,
            )
            return
        if self.training:
            self._log("TRAIN  | a training/download run is already active")
            return
        if self.busy:
            self._log("TRAIN  | another background operation is still active")
            return
        self.notebook.select(0)
        missing = [
            symbol
            for symbol in self.symbols
            if read_cached_bars(symbol, self.timeframe).empty
        ]
        if missing:
            self.train_after_download = True
            self.training = True
            self.stop_training_event.clear()
            self.training_phase = (
                "STATE PREPARING DATA  |  DOWNLOADING MISSING TRAINING CACHE"
            )
            self.status_var.set("STATE  DOWNLOADING FOR TRAINING")
            self._set_controls(True)
            self._log(
                "TRAIN  | cache missing; download queued automatically for: "
                + ",".join(missing)
            )
            self.operation_thread = threading.Thread(
                target=self._download_worker,
                name="free-data-download-for-training",
                daemon=True,
            )
            self.operation_thread.start()
            return
        self.train_losses.clear()
        self.validation_losses.clear()
        self.validation_accuracies.clear()
        self.validation_f1_scores.clear()
        self.training_epoch_numbers.clear()
        self.current_epoch = 0
        self.current_batch = 0
        self.batches_per_epoch = 0
        self.current_batch_label = 0
        self.elapsed_seconds = 0.0
        self.latest_train_loss = None
        self.latest_validation_loss = None
        self.latest_accuracy = None
        self.latest_macro_f1 = None
        self.latest_learning_rate = LEARNING_RATE
        self.latest_confusion = []
        self.stop_training_event.clear()
        self.training = True
        self.started_at = time.monotonic()
        training_symbols = tuple(self.symbols)
        training_asset_class = self.asset_class
        training_timeframe = self.timeframe
        self.training_phase = "STATE INITIALIZING  |  BUILDING FEATURE WINDOWS"
        self.status_var.set("STATE  TRAINING / INITIALIZING")
        self._log(
            f"TRAIN  | started asset={training_asset_class} timeframe={training_timeframe} "
            f"symbols={','.join(training_symbols)} mode=continuous_until_manual_stop"
        )
        self.progress_var.set(0.0)
        self._set_controls(True)
        self.operation_thread = threading.Thread(
            target=self._training_worker,
            args=(training_symbols, training_asset_class, training_timeframe),
            name="torch-training",
            daemon=True,
        )
        self.operation_thread.start()

    def stop_training(self) -> None:
        if not self.training or self.stop_training_event.is_set():
            return
        self.stop_training_event.set()
        self.training_phase = "STATE STOPPING  |  FINISHING CURRENT BATCH"
        self.stop_button.configure(state="disabled")
        self._log("STOP   | training stop queued; current batch will complete")

    def _training_worker(
        self,
        symbols: tuple[str, ...],
        asset_class: str,
        timeframe: str,
    ) -> None:
        try:
            torch.set_num_threads(CPU_THREADS)
            torch.manual_seed(42)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")
            move_threshold = training_move_threshold(asset_class, timeframe)
            frames = {
                symbol: read_cached_bars(symbol, timeframe) for symbol in symbols
            }
            self.events.put(
                {
                    "kind": "log",
                    "text": (
                        f"FEATURE | indicators={len(FEATURE_COLUMNS)}  lookback={LOOKBACK}  "
                        f"horizon={FORWARD_HORIZON}  threshold={move_threshold:.2%}  "
                        f"dropout={DROPOUT_RATE:.0%}  smoothing={LABEL_SMOOTHING:.0%}  "
                        "sampler=balanced"
                    ),
                }
            )
            bundle = build_training_dataset(frames, move_threshold)
            self.events.put(
                {
                    "kind": "training_setup",
                    "train_rows": bundle.train_rows,
                    "validation_rows": bundle.validation_rows,
                    "class_counts": bundle.class_counts,
                }
            )
            if self.stop_training_event.is_set():
                raise InterruptedError

            train_dataset = TensorDataset(
                torch.from_numpy(bundle.x_train), torch.from_numpy(bundle.y_train)
            )
            validation_dataset = TensorDataset(
                torch.from_numpy(bundle.x_validation),
                torch.from_numpy(bundle.y_validation),
            )
            generator = torch.Generator().manual_seed(42)
            counts = np.asarray(
                [bundle.class_counts[index] for index in range(3)], dtype=np.float64
            )
            sample_weights = torch.from_numpy(
                (1.0 / counts[bundle.y_train]).astype(np.float64)
            )
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=generator,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=0,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            model = ShallowTradeNet(bundle.x_train.shape[1]).to("cpu")
            loss_function = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
            )
            completed_epochs = 0
            best_epoch = 0
            best_validation_loss = float("inf")
            best_macro_f1 = -1.0
            best_confusion: list[list[int]] = []
            best_state: dict[str, torch.Tensor] | None = None

            if MODEL_PATH.exists():
                saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
                resume_compatible = (
                    list(saved.get("feature_columns", [])) == FEATURE_COLUMNS
                    and int(saved.get("input_size", 0)) == bundle.x_train.shape[1]
                    and normalize_asset_type(str(saved.get("asset_class", "")))
                    == normalize_asset_type(asset_class)
                    and str(saved.get("timeframe", "")) == timeframe
                    and list(saved.get("symbols", [])) == list(symbols)
                    and np.allclose(saved.get("mean"), bundle.mean, rtol=1e-6, atol=1e-8)
                    and np.allclose(saved.get("std"), bundle.std, rtol=1e-6, atol=1e-8)
                )
                if resume_compatible:
                    model.load_state_dict(saved["state_dict"])
                    completed_epochs = int(saved.get("epochs_completed", 0))
                    best_epoch = int(saved.get("best_epoch", completed_epochs))
                    best_validation_loss = float(
                        saved.get("best_validation_loss", float("inf"))
                    )
                    best_macro_f1 = float(saved.get("best_macro_f1", -1.0))
                    best_confusion = list(saved.get("best_confusion_matrix", []))
                    best_state = copy.deepcopy(model.state_dict())
                    resume_lr = float(saved.get("learning_rate", LEARNING_RATE))
                    optimizer.param_groups[0]["lr"] = resume_lr
                    self.events.put(
                        {
                            "kind": "log",
                            "text": (
                                f"RESUME | epoch={completed_epochs} best_epoch={best_epoch} "
                                f"macro_f1={best_macro_f1:.3f} lr={resume_lr:.2e}"
                            ),
                        }
                    )
                else:
                    self.events.put(
                        {
                            "kind": "log",
                            "text": "RESUME | saved model does not match current data; starting fresh",
                        }
                    )

            def checkpoint_payload(
                state_dict: dict[str, torch.Tensor], epochs_completed: int
            ) -> dict:
                return {
                    "state_dict": state_dict,
                    "input_size": bundle.x_train.shape[1],
                    "mean": bundle.mean,
                    "std": bundle.std,
                    "feature_columns": FEATURE_COLUMNS,
                    "lookback": LOOKBACK,
                    "forward_horizon": FORWARD_HORIZON,
                    "move_threshold": move_threshold,
                    "symbols": list(symbols),
                    "asset_class": normalize_asset_type(asset_class),
                    "training_source": (
                        "Massive/Yahoo" if asset_class == "stocks" else "Binance Public"
                    ),
                    "timeframe": timeframe,
                    "epochs_completed": epochs_completed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "best_macro_f1": best_macro_f1,
                    "best_confusion_matrix": best_confusion,
                    "label_smoothing": LABEL_SMOOTHING,
                    "dropout_rate": DROPOUT_RATE,
                    "balanced_sampling": True,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                }

            epoch = completed_epochs
            while not self.stop_training_event.is_set():
                epoch += 1
                epoch_started_at = time.monotonic()
                model.train()
                running_loss = 0.0
                seen = 0
                batches = len(train_loader)
                for batch_index, (features, labels) in enumerate(train_loader, start=1):
                    if self.stop_training_event.is_set():
                        break
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(features)
                    loss = loss_function(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    size = labels.size(0)
                    running_loss += float(loss.item()) * size
                    seen += size
                    majority = int(torch.bincount(labels, minlength=3).argmax().item())
                    self.events.put(
                        {
                            "kind": "training_progress",
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches": batches,
                            "label": majority,
                            "loss": running_loss / max(seen, 1),
                            "elapsed": time.monotonic() - self.started_at,
                        }
                    )
                if seen == 0:
                    break

                model.eval()
                validation_loss = 0.0
                validation_seen = 0
                validation_labels: list[np.ndarray] = []
                validation_predictions: list[np.ndarray] = []
                with torch.inference_mode():
                    for features, labels in validation_loader:
                        logits = model(features)
                        loss = loss_function(logits, labels)
                        size = labels.size(0)
                        validation_loss += float(loss.item()) * size
                        validation_seen += size
                        validation_labels.append(labels.numpy())
                        validation_predictions.append(logits.argmax(dim=1).numpy())
                train_loss = running_loss / seen
                val_loss = validation_loss / max(validation_seen, 1)
                accuracy, macro_f1, confusion = classification_metrics(
                    np.concatenate(validation_labels),
                    np.concatenate(validation_predictions),
                )
                scheduler.step(val_loss)
                learning_rate = float(optimizer.param_groups[0]["lr"])
                completed_epochs = epoch
                self.events.put(
                    {
                        "kind": "epoch_metrics",
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "confusion": confusion,
                        "learning_rate": learning_rate,
                    }
                )
                self.database.execute(
                    "INSERT INTO training_log"
                    "(run_id,timestamp,epoch,train_loss,validation_loss,validation_accuracy,"
                    "macro_f1,learning_rate) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        datetime.now(timezone.utc).isoformat(),
                        epoch,
                        train_loss,
                        val_loss,
                        accuracy,
                        macro_f1,
                        learning_rate,
                    ),
                )
                remaining_display_time = (
                    MIN_EPOCH_DISPLAY_SECONDS
                    - (time.monotonic() - epoch_started_at)
                )
                if remaining_display_time > 0:
                    self.stop_training_event.wait(remaining_display_time)
                if macro_f1 > best_macro_f1 + 1e-4 or (
                    abs(macro_f1 - best_macro_f1) <= 1e-4
                    and val_loss < best_validation_loss - 1e-4
                ):
                    best_validation_loss = val_loss
                    best_macro_f1 = macro_f1
                    best_confusion = confusion
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                if epoch % AUTOSAVE_EPOCHS == 0 and best_state is not None:
                    save_checkpoint_atomic(
                        checkpoint_payload(best_state, completed_epochs), MODEL_PATH
                    )
                    self.events.put(
                        {
                            "kind": "log",
                            "text": (
                                f"AUTOSAVE | epoch={epoch} best_epoch={best_epoch} "
                                f"f1={best_macro_f1:.3f} val={best_validation_loss:.5f}"
                            ),
                        }
                    )

            if best_state is not None:
                model.load_state_dict(best_state)

            if completed_epochs == 0:
                self.events.put(
                    {
                        "kind": "training_done",
                        "stopped": True,
                        "text": "STOP   | no complete epoch; existing model was preserved",
                        "best_epoch": 0,
                        "epochs": 0,
                    }
                )
                return

            checkpoint = checkpoint_payload(model.state_dict(), completed_epochs)
            save_checkpoint_atomic(checkpoint, MODEL_PATH)
            stopped = self.stop_training_event.is_set()
            self.events.put(
                {
                    "kind": "training_done",
                    "stopped": stopped,
                    "text": (
                        f"SAVE   | trade_model.pt  epochs={completed_epochs}  "
                        f"best_epoch={best_epoch}  stopped={stopped}  "
                        f"elapsed={time.monotonic() - self.started_at:.2f}s"
                    ),
                    "best_epoch": best_epoch,
                    "epochs": completed_epochs,
                }
            )
        except InterruptedError:
            self.events.put(
                {"kind": "training_done", "stopped": True, "text": "STOP   | before training"}
            )
        except Exception as exc:
            self.events.put(
                {"kind": "error", "context": "Training", "text": str(exc)}
            )

    def toggle_paper_trading(self) -> None:
        if self.paper_running:
            self.cashier_stop.set()
            self.paper_button.configure(state="disabled")
            self._log("CASHIER | paper stop requested")
            return
        if self.busy or self.training:
            return
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model required",
                "Train and save models/trade_model.pt first.",
                parent=self.root,
            )
            return
        try:
            api_credentials()
        except UserFacingError as exc:
            messagebox.showerror("Alpaca credentials", str(exc), parent=self.root)
            return
        confirmed = messagebox.askyesno(
            "Start paper trading",
            "This enables the Investor -> Cashier pipeline and may submit simulated "
            "stock or crypto orders to Alpaca's PAPER endpoint.\n\n"
            "Risk limits: 1% equity risk/trade, 2% daily loss, 20% drawdown.\n\nContinue?",
            parent=self.root,
        )
        if not confirmed:
            return
        self.cashier_stop.clear()
        self.paper_running = True
        self.status_var.set("STATE  MULTI-AGENT PAPER")
        self._set_controls(True)
        self.paper_button.configure(text="Stop Paper Trading", state="normal")
        self.notebook.select(2)
        self.cashier_agent = CashierAgent(
            self.database, self.events, self.cashier_stop
        )
        self.cashier_agent.start()

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event.get("kind")
                if kind == "log":
                    self._log(event["text"])
                elif kind == "operation_done":
                    self.status_var.set(f"STATE  {event['status']}")
                    self._set_controls(False)
                    self._log(event["text"])
                    self._seed_database_from_cache()
                    self._refresh_meta()
                    if event.get("start_training"):
                        self.train_after_download = False
                        self.training = False
                        self.root.after(100, self.start_training)
                elif kind == "training_setup":
                    self.status_var.set("STATE  TRAINING")
                    self.training_phase = (
                        "STATE TRAINING CONTINUOUSLY  |  PRESS STOP TO FINISH  |  "
                        f"{event['train_rows']} TRAIN / {event['validation_rows']} VALIDATION"
                    )
                    self.latest_class_counts = dict(event["class_counts"])
                    self._log(
                        f"SPLIT  | train={event['train_rows']}  validation={event['validation_rows']}  "
                        f"hold={self.latest_class_counts[0]}  buy={self.latest_class_counts[1]}  "
                        f"sell={self.latest_class_counts[2]}"
                    )
                elif kind == "training_progress":
                    self.current_epoch = int(event["epoch"])
                    self.current_batch = int(event["batch"])
                    self.batches_per_epoch = int(event["batches"])
                    self.current_batch_label = int(event["label"])
                    self.latest_train_loss = float(event["loss"])
                    self.elapsed_seconds = float(event["elapsed"])
                    epoch_fraction = self.current_batch / max(self.batches_per_epoch, 1)
                    self.progress_var.set(100.0 * epoch_fraction)
                    if self.current_batch in (1, self.batches_per_epoch):
                        self._log(
                            f"BATCH  | epoch={self.current_epoch:02d}  "
                            f"batch={self.current_batch:04d}/{self.batches_per_epoch:04d}  "
                            f"loss={self.latest_train_loss:.5f}  "
                            f"class={SIGNAL_NAMES[self.current_batch_label]}"
                        )
                elif kind == "epoch_metrics":
                    self.training_epoch_numbers.append(int(event["epoch"]))
                    self.train_losses.append(float(event["train_loss"]))
                    self.validation_losses.append(float(event["validation_loss"]))
                    self.validation_accuracies.append(float(event["accuracy"]))
                    self.validation_f1_scores.append(float(event["macro_f1"]))
                    self.training_epoch_numbers = self.training_epoch_numbers[
                        -MAX_CHART_EPOCHS:
                    ]
                    self.train_losses = self.train_losses[-MAX_CHART_EPOCHS:]
                    self.validation_losses = self.validation_losses[-MAX_CHART_EPOCHS:]
                    self.validation_accuracies = self.validation_accuracies[
                        -MAX_CHART_EPOCHS:
                    ]
                    self.validation_f1_scores = self.validation_f1_scores[
                        -MAX_CHART_EPOCHS:
                    ]
                    self.latest_train_loss = self.train_losses[-1]
                    self.latest_validation_loss = self.validation_losses[-1]
                    self.latest_accuracy = self.validation_accuracies[-1]
                    self.latest_macro_f1 = self.validation_f1_scores[-1]
                    self.latest_learning_rate = float(event["learning_rate"])
                    self.latest_confusion = list(event["confusion"])
                    self._log(
                        f"EPOCH  | {event['epoch']:04d}/CONTINUOUS  "
                        f"train={self.latest_train_loss:.5f}  "
                        f"val={self.latest_validation_loss:.5f}  "
                        f"accuracy={self.latest_accuracy:.2%}  "
                        f"macro_f1={self.latest_macro_f1:.3f}  "
                        f"lr={self.latest_learning_rate:.2e}"
                    )
                    if int(event["epoch"]) % 5 == 0:
                        hold_row, buy_row, sell_row = self.latest_confusion
                        self._log(
                            "CONFUSION | actual rows H/B/S -> "
                            f"H={hold_row} B={buy_row} S={sell_row}"
                        )
                elif kind == "training_done":
                    self.training = False
                    if event["stopped"]:
                        saved_epoch = event.get("best_epoch", "--")
                        self.status_var.set(
                            f"STATE  MODEL SAVED / STOPPED / BEST EPOCH {saved_epoch}"
                        )
                        self.training_phase = (
                            "STATE COMPLETE  |  STOPPED BY USER  |  "
                            f"BEST EPOCH {saved_epoch}"
                        )
                    else:
                        self.progress_var.set(100.0)
                        self.status_var.set(
                            f"STATE  MODEL READY / BEST EPOCH {event.get('best_epoch', '--')}"
                        )
                        self.training_phase = (
                            "STATE COMPLETE  |  MODEL READY  |  "
                            f"BEST EPOCH {event.get('best_epoch', '--')}"
                        )
                    self._set_controls(False)
                    self._log(event["text"])
                    if not event["stopped"]:
                        self._log(
                            "TRAIN  | COMPLETE; the best checkpoint is active"
                        )
                    if self.simulation_window is not None:
                        self.simulation_window.mark_complete()
                    self._refresh_meta()
                elif kind == "rankings":
                    for item in self.rankings_table.get_children():
                        self.rankings_table.delete(item)
                    for rank, row in enumerate(event["rows"], start=1):
                        self.rankings_table.insert(
                            "",
                            "end",
                            values=(
                                rank,
                                row["symbol"],
                                row["asset_type"],
                                f"{float(row['score']):.1f}",
                                f"{float(row['last_price']):.4g}",
                                f"{float(row['sentiment']):+.2f}",
                                f"{float(row['adx']):.1f}",
                            ),
                        )
                elif kind == "paper_row":
                    self.paper_table.insert("", 0, values=event["values"])
                    try:
                        self.paper_equity_points.append(float(event["values"][6]))
                        self.paper_equity_points = self.paper_equity_points[-250:]
                    except (TypeError, ValueError):
                        pass
                    children = self.paper_table.get_children()
                    if len(children) > 250:
                        self.paper_table.delete(children[-1])
                elif kind == "paper_done":
                    self.paper_running = False
                    self.cashier_agent = None
                    self.status_var.set("STATE  IDLE")
                    self.paper_button.configure(text="Start Paper Trading")
                    self._set_controls(False)
                    self._log(event["text"])
                elif kind == "paper_error":
                    self.paper_running = False
                    self.status_var.set("STATE  ERROR")
                    self.paper_button.configure(text="Start Paper Trading")
                    self._set_controls(False)
                    self._log(f"ERROR  | {event['context']}: {event['text']}")
                    messagebox.showerror(event["context"], event["text"], parent=self.root)
                elif kind == "error":
                    self.training = False
                    self.train_after_download = False
                    self.training_phase = (
                        f"STATE ERROR  |  {event['context'].upper()}: {event['text']}"
                    )
                    self.status_var.set("STATE  ERROR")
                    self._set_controls(False)
                    self._log(f"ERROR  | {event['context']}: {event['text']}")
                    messagebox.showerror(event["context"], event["text"], parent=self.root)
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self._poll_events)

    def _redraw_chart(self) -> None:
        self.loss_axes.clear()
        self.accuracy_axes.clear()
        self._style_axes()
        epochs = self.training_epoch_numbers
        if epochs:
            self.loss_axes.plot(
                epochs, self.train_losses, color=WHITE, linewidth=0.9, label="train"
            )
            self.loss_axes.plot(
                epochs,
                self.validation_losses,
                color=TEXT,
                linewidth=0.9,
                linestyle="--",
                marker="o",
                markersize=2.5,
                label="validation",
            )
            self.accuracy_axes.plot(
                epochs,
                self.validation_accuracies,
                color=WHITE,
                linewidth=0.9,
                marker="o",
                markersize=2.5,
                label="accuracy",
            )
            self.accuracy_axes.plot(
                epochs,
                self.validation_f1_scores,
                color=TEXT,
                linewidth=0.9,
                linestyle="--",
                label="macro-F1",
            )
            legend = self.loss_axes.legend(loc="upper right", frameon=False, fontsize=8)
            for label in legend.get_texts():
                label.set_color(TEXT)
            score_legend = self.accuracy_axes.legend(
                loc="lower right", frameon=False, fontsize=8
            )
            for label in score_legend.get_texts():
                label.set_color(TEXT)
        else:
            self.loss_axes.text(
                0.5,
                0.5,
                "NO TRAINING SESSION",
                transform=self.loss_axes.transAxes,
                ha="center",
                va="center",
                color=ACCENT,
                fontsize=9,
                family="monospace",
            )
            self.accuracy_axes.text(
                0.5,
                0.5,
                "VALIDATION ACCURACY",
                transform=self.accuracy_axes.transAxes,
                ha="center",
                va="center",
                color=ACCENT,
                fontsize=8,
                family="monospace",
            )
        self.chart.draw_idle()
        self.paper_axes.clear()
        self.paper_axes.set_facecolor(BG)
        self.paper_axes.set_title("PAPER EQUITY", color=TEXT, fontsize=8)
        self.paper_axes.tick_params(colors=TEXT, labelsize=7)
        for spine in self.paper_axes.spines.values():
            spine.set_color(ACCENT)
        if self.paper_equity_points:
            self.paper_axes.plot(
                range(1, len(self.paper_equity_points) + 1),
                self.paper_equity_points,
                color=WHITE,
                linewidth=0.9,
            )
        else:
            self.paper_axes.text(
                0.5,
                0.5,
                "NO PAPER FILLS",
                transform=self.paper_axes.transAxes,
                ha="center",
                va="center",
                color=ACCENT,
                fontsize=8,
                family="monospace",
            )
        self.paper_chart.draw_idle()
        if self.root.winfo_exists():
            self.root.after(2000, self._redraw_chart)

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "--:--:--"
        if seconds < 60:
            return f"{max(0.0, seconds):05.1f}s"
        hours, remainder = divmod(max(0, int(seconds)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _metric(value: float | None, digits: int = 5) -> str:
        return f"{value:.{digits}f}" if value is not None else "--"

    def _refresh_telemetry(self) -> None:
        if self.training and self.started_at:
            self.elapsed_seconds = time.monotonic() - self.started_at
        self.telemetry_var.set(
            f"EPOCH {self.current_epoch:04d}/CONT  |  "
            f"BATCH {self.current_batch:04d}/{self.batches_per_epoch:04d}  |  "
            f"ELAPSED {self._duration(self.elapsed_seconds)}  |  END MANUAL STOP\n"
            f"TRAIN LOSS {self._metric(self.latest_train_loss)}  |  "
            f"VAL LOSS {self._metric(self.latest_validation_loss)}  |  "
            f"VAL ACC {self._metric(self.latest_accuracy, 3)}  |  "
            f"MACRO F1 {self._metric(self.latest_macro_f1, 3)}\n"
            f"LABELS HOLD={self.latest_class_counts[0]}  BUY={self.latest_class_counts[1]}  "
            f"SELL={self.latest_class_counts[2]}  |  ADAMW  |  "
            f"LR {self.latest_learning_rate:.2E}  |  CPU"
        )

    def _update_clock(self) -> None:
        self._refresh_telemetry()
        if self.root.winfo_exists():
            self.root.after(1000, self._update_clock)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        if self.training:
            self.stop_training_event.set()
            self.status_var.set("STATE  STOPPING / SAVING MODEL BEFORE EXIT")
            self.training_phase = "STATE STOPPING  |  SAVING BEST MODEL BEFORE EXIT"
        self.cashier_stop.set()
        self._finish_close_when_safe()

    def _finish_close_when_safe(self) -> None:
        if self.operation_thread is not None and self.operation_thread.is_alive():
            self.root.after(100, self._finish_close_when_safe)
            return
        self.core_agent_stop.set()
        self.root.destroy()


def main() -> None:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    torch.set_num_threads(CPU_THREADS)
    root = tk.Tk()
    if os.getenv("TRADEMIND_START_MINIMIZED") == "1":
        root.iconify()
    MainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
