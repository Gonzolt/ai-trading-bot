"""Alpaca TradeMind Trainer: CPU market-data research and paper execution UI."""

from __future__ import annotations

import copy
import io
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable, TypeVar

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
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from dotenv import load_dotenv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Editable research parameters.
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
DEFAULT_CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_TIMEFRAME = "1Day"  # "1Day" or "1Min"
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
STOCK_DAILY_MOVE_THRESHOLD = 0.005
CRYPTO_DAILY_MOVE_THRESHOLD = 0.02
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
CPU_THREADS = 20
PAPER_QUANTITY = 1.0
PAPER_POLL_SECONDS = 60
VALIDATION_FRACTION = 0.20
EARLY_STOPPING_PATIENCE = 5

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "market_cache"
MODEL_PATH = ROOT / "trade_model.pt"
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

FEATURE_COLUMNS = [
    "log_return",
    "volatility_20",
    "atr_14_pct",
    "rsi_14",
    "macd_hist_pct",
    "volume_change",
    "range_pct",
    "body_pct",
    "sma_10_gap",
    "sma_30_gap",
]
SIGNAL_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

load_dotenv(ENV_PATH, override=False)


def training_move_threshold(asset_class: str, timeframe: str) -> float:
    if timeframe == "1Min":
        return MOVE_THRESHOLD
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
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TradeMind/1.0", **(headers or {})},
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise HttpRequestError(
                f"{context} failed with HTTP {exc.code}: {detail or exc.reason}",
                exc.code,
            ) from exc
        except urllib.error.URLError as exc:
            raise UserFacingError(f"{context} network error: {exc.reason}") from exc

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
        if not cached.empty
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


def sync_training_bars(
    asset_class: str,
    symbol: str,
    timeframe: str,
    events: queue.Queue | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if asset_class == "stocks":
        return sync_massive_stock_bars(symbol, timeframe, events, force_refresh)
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
    bars["sma_10_gap"] = close / close.rolling(10).mean() - 1.0
    bars["sma_30_gap"] = close / close.rolling(30).mean() - 1.0

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
        bars.loc[bars["future_return"].isna(), "target"] = np.nan
    return bars.replace([np.inf, -np.inf], np.nan)


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


def symbol_windows(
    frame: pd.DataFrame, move_threshold: float = MOVE_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    featured = engineer_features(
        frame, include_target=True, move_threshold=move_threshold
    )
    featured = featured.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    values = featured[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = featured["target"].to_numpy()
    windows: list[np.ndarray] = []
    labels: list[int] = []
    for end in range(LOOKBACK - 1, len(featured)):
        if np.isnan(targets[end]):
            continue
        start = end - LOOKBACK + 1
        windows.append(values[start : end + 1].reshape(-1))
        labels.append(int(targets[end]))
    if not windows:
        return (
            np.empty((0, LOOKBACK * len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64)


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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def latest_model_input(
    frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray
) -> tuple[np.ndarray, pd.Timestamp, float]:
    featured = engineer_features(frame, include_target=False)
    featured = featured.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if len(featured) < LOOKBACK:
        raise UserFacingError(
            f"Need {LOOKBACK} valid feature rows; only {len(featured)} are available."
        )
    window = featured[FEATURE_COLUMNS].tail(LOOKBACK).to_numpy(dtype=np.float32).reshape(-1)
    scaled = ((window - mean) / std).astype(np.float32)
    last = featured.iloc[-1]
    return scaled, pd.Timestamp(last["timestamp"]), float(last["close"])


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
    def __init__(self, app: "AlpacaTradeMindApp") -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("Alpaca TradeMind / Network Trace")
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
        self.canvas.bind("<Configure>", self._schedule_redraw)
        self.window.after(100, self.draw)
        self.window.after(500, self.pulse)

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
        input_labels = ["PRICE", "RETURNS", "VOLATILITY", "MOMENTUM", "VOLUME"]
        input_y = [height * value for value in (0.20, 0.32, 0.44, 0.56, 0.68)]
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

        self._draw_hidden_box(self.hidden_one, "DENSE 01", "300 INPUTS  ->  32", font, small)
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
            20, 15, anchor="w", text="FEATURE GROUPS / 30-BAR WINDOW", fill=ACCENT, font=small
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
            text="STATE IDLE  |  FORWARD TRACE 2.0 s",
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
        self.window.after(2000, self.pulse)
        if not self.app.training:
            if self.packet is not None:
                self.canvas.delete(self.packet)
                self.packet = None
            if self.status_item is not None:
                self.canvas.itemconfigure(
                    self.status_item, text="STATE IDLE  |  FORWARD TRACE 2.0 s", fill=ACCENT
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
        self._animate(path, class_id, 0, 50)

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
        self.window.after(25, self._animate, path, class_id, frame + 1, total)

    def _dim(self, node: int) -> None:
        if self.window.winfo_exists():
            try:
                self.canvas.itemconfigure(node, fill=TROUGH)
            except tk.TclError:
                pass


class AlpacaTradeMindApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TradeMind Trainer / Free Data + Alpaca Paper")
        self.root.configure(bg=BG)
        self.root.geometry("1280x760")
        self.root.minsize(1040, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.events: queue.Queue = queue.Queue()
        has_massive_key = bool(
            os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        )
        self.symbols = (
            DEFAULT_SYMBOLS.copy() if has_massive_key else DEFAULT_CRYPTO_SYMBOLS.copy()
        )
        self.timeframe = DEFAULT_TIMEFRAME
        self.asset_class = DEFAULT_ASSET_CLASS if has_massive_key else "crypto"
        self.logs: deque[str] = deque(maxlen=18)
        self.busy = False
        self.training = False
        self.paper_running = False
        self.stop_training_event = threading.Event()
        self.stop_paper_event = threading.Event()
        self.operation_thread: threading.Thread | None = None
        self.paper_thread: threading.Thread | None = None
        self.train_after_download = False

        self.train_losses: list[float] = []
        self.validation_losses: list[float] = []
        self.validation_accuracies: list[float] = []
        self.current_epoch = 0
        self.current_batch = 0
        self.batches_per_epoch = 0
        self.current_batch_label = 0
        self.elapsed_seconds = 0.0
        self.started_at = 0.0
        self.latest_train_loss: float | None = None
        self.latest_validation_loss: float | None = None
        self.latest_accuracy: float | None = None
        self.latest_class_counts = {0: 0, 1: 0, 2: 0}

        self._build_style()
        self._build_ui()
        data_state = "present" if self._massive_credentials_present() else "missing"
        paper_state = "present" if self._credentials_present() else "missing"
        self._log(
            f"READY  | symbols={','.join(self.symbols)}  timeframe={self.timeframe}  "
            f"massive={data_state}  alpaca_paper={paper_state}"
        )
        self.root.after(100, self._poll_events)
        self.root.after(1000, self._update_clock)
        self.root.after(2000, self._redraw_chart)

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
        ttk.Label(top, text="TRADEMIND TRAINER", style="Title.TLabel").grid(
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
        paper_tab = ttk.Frame(self.notebook)
        self.notebook.add(training_tab, text="TRAINING")
        self.notebook.add(paper_tab, text="PAPER P&L")
        training_tab.grid_rowconfigure(0, weight=1)
        training_tab.grid_columnconfigure(0, weight=1)
        paper_tab.grid_rowconfigure(0, weight=1)
        paper_tab.grid_columnconfigure(0, weight=1)

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
        self.paper_table.grid(row=0, column=0, sticky="nsew")
        paper_scroll = ttk.Scrollbar(
            paper_tab, orient="vertical", command=self.paper_table.yview
        )
        paper_scroll.grid(row=0, column=1, sticky="ns")
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
            bottom, "Open Simulation", lambda: NetworkSimulation(self)
        )
        self.paper_button = self._button(
            bottom, "Start Paper Trading", self.toggle_paper_trading
        )

    def _button(self, parent, text: str, command, state: str = "normal") -> ttk.Button:
        button = ttk.Button(
            parent, text=text, command=command, style="Flat.TButton", state=state
        )
        button.pack(side="left", padx=(0, 3))
        return button

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
            except Exception:
                pass
        if self.asset_class == "stocks":
            data_source = (
                "MASSIVE READY" if self._massive_credentials_present() else "MASSIVE KEY MISSING"
            )
        else:
            data_source = "BINANCE PUBLIC"
        paper_credentials = "READY" if self._credentials_present() else "KEYS MISSING"
        model = "READY" if MODEL_PATH.exists() else "MISSING"
        threshold = training_move_threshold(self.asset_class, self.timeframe)
        self.meta_var.set(
            f"ASSET {self.asset_class.upper()}  |  SYMBOLS {','.join(self.symbols)}  |  BARS {self.timeframe}  |  "
            f"CACHE {cache_rows} ROWS  |  WINDOW {LOOKBACK}  |  HORIZON {FORWARD_HORIZON}  |  "
            f"THRESHOLD {threshold:.2%}  |  NETWORK {LOOKBACK * len(FEATURE_COLUMNS)}-32-16-3  |  "
            f"DATA {data_source}  |  MODEL {model}  |  ALPACA PAPER {paper_credentials}"
        )

    def _log(self, message: str) -> None:
        self.logs.append(f"{time.strftime('%H:%M:%S')}  {message}")
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
        self.stop_button.configure(state="normal" if self.training else "disabled")
        if self.paper_running:
            self.paper_button.configure(state="normal")
        elif busy:
            self.paper_button.configure(state="disabled")
        else:
            self.paper_button.configure(state="normal")

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
        self._refresh_meta()
        self._log(
            f"UNIVERSE | asset={self.asset_class}  symbols={','.join(self.symbols)}  "
            f"timeframe={self.timeframe}"
        )

    def download_data(self) -> None:
        if self.busy:
            return
        self.train_after_download = False
        if self.asset_class == "stocks":
            try:
                massive_api_key()
            except UserFacingError as exc:
                messagebox.showerror("Massive API key", str(exc), parent=self.root)
                return
        self.status_var.set("STATE  DOWNLOADING")
        self._set_controls(True)
        self.operation_thread = threading.Thread(
            target=self._download_worker, name="free-data-download", daemon=True
        )
        self.operation_thread.start()

    def _download_worker(self) -> None:
        try:
            total = 0
            for symbol in self.symbols:
                frame = sync_training_bars(
                    self.asset_class, symbol, self.timeframe, self.events
                )
                total += len(frame)
            self.events.put(
                {
                    "kind": "operation_done",
                    "status": "IDLE",
                    "start_training": self.train_after_download,
                    "text": (
                        f"CACHE  | source={'Massive' if self.asset_class == 'stocks' else 'Binance'} "
                        f"synchronization complete  total_rows={total}"
                    ),
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
        if self.busy or self.paper_running:
            return
        missing = [
            symbol
            for symbol in self.symbols
            if read_cached_bars(symbol, self.timeframe).empty
        ]
        if missing:
            if self.asset_class == "stocks":
                try:
                    massive_api_key()
                except UserFacingError as exc:
                    messagebox.showerror("Massive API key", str(exc), parent=self.root)
                    return
            self.train_after_download = True
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
        self.current_epoch = 0
        self.current_batch = 0
        self.batches_per_epoch = 0
        self.current_batch_label = 0
        self.elapsed_seconds = 0.0
        self.latest_train_loss = None
        self.latest_validation_loss = None
        self.latest_accuracy = None
        self.stop_training_event.clear()
        self.training = True
        self.started_at = time.monotonic()
        self.status_var.set("STATE  ENGINEERING")
        self.progress_var.set(0.0)
        self._set_controls(True)
        self.operation_thread = threading.Thread(
            target=self._training_worker, name="torch-training", daemon=True
        )
        self.operation_thread.start()

    def stop_training(self) -> None:
        if not self.training or self.stop_training_event.is_set():
            return
        self.stop_training_event.set()
        self.stop_button.configure(state="disabled")
        self._log("STOP   | training stop queued; current batch will complete")

    def _training_worker(self) -> None:
        try:
            torch.set_num_threads(CPU_THREADS)
            move_threshold = training_move_threshold(
                self.asset_class, self.timeframe
            )
            frames = {
                symbol: read_cached_bars(symbol, self.timeframe)
                for symbol in self.symbols
            }
            self.events.put(
                {
                    "kind": "log",
                    "text": (
                        f"FEATURE | indicators={len(FEATURE_COLUMNS)}  lookback={LOOKBACK}  "
                        f"horizon={FORWARD_HORIZON}  threshold={move_threshold:.2%}"
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
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                generator=generator,
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            counts = np.asarray(
                [bundle.class_counts[index] for index in range(3)], dtype=np.float32
            )
            class_weights = counts.sum() / (3.0 * counts)
            model = ShallowTradeNet(bundle.x_train.shape[1]).to("cpu")
            loss_function = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32)
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
            )
            completed_epochs = 0
            best_epoch = 0
            best_validation_loss = float("inf")
            best_state: dict[str, torch.Tensor] | None = None
            stale_epochs = 0
            for epoch in range(1, EPOCHS + 1):
                if self.stop_training_event.is_set():
                    break
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
                correct = 0
                with torch.inference_mode():
                    for features, labels in validation_loader:
                        logits = model(features)
                        loss = loss_function(logits, labels)
                        size = labels.size(0)
                        validation_loss += float(loss.item()) * size
                        validation_seen += size
                        correct += int((logits.argmax(dim=1) == labels).sum().item())
                train_loss = running_loss / seen
                val_loss = validation_loss / max(validation_seen, 1)
                accuracy = correct / max(validation_seen, 1)
                completed_epochs = epoch
                self.events.put(
                    {
                        "kind": "epoch_metrics",
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": val_loss,
                        "accuracy": accuracy,
                    }
                )
                if val_loss < best_validation_loss - 1e-4:
                    best_validation_loss = val_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= EARLY_STOPPING_PATIENCE:
                        self.events.put(
                            {
                                "kind": "log",
                                "text": (
                                    f"EARLY  | no validation improvement for "
                                    f"{EARLY_STOPPING_PATIENCE} epochs  best_epoch={best_epoch}"
                                ),
                            }
                        )
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            checkpoint = {
                "state_dict": model.state_dict(),
                "input_size": bundle.x_train.shape[1],
                "mean": bundle.mean,
                "std": bundle.std,
                "feature_columns": FEATURE_COLUMNS,
                "lookback": LOOKBACK,
                "forward_horizon": FORWARD_HORIZON,
                "move_threshold": move_threshold,
                "symbols": self.symbols.copy(),
                "asset_class": self.asset_class,
                "training_source": "Massive" if self.asset_class == "stocks" else "Binance Public",
                "timeframe": self.timeframe,
                "epochs_completed": completed_epochs,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            torch.save(checkpoint, MODEL_PATH)
            stopped = self.stop_training_event.is_set()
            self.events.put(
                {
                    "kind": "training_done",
                    "stopped": stopped,
                    "text": (
                        f"SAVE   | trade_model.pt  epochs={completed_epochs}  "
                        f"best_epoch={best_epoch}  stopped={stopped}"
                    ),
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
            self.stop_paper_event.set()
            self.paper_button.configure(state="disabled")
            self._log("PAPER  | stop requested")
            return
        if self.busy or self.training:
            return
        if self.asset_class != "stocks":
            messagebox.showerror(
                "Stock paper trading only",
                "This safety-tested paper execution loop currently supports US stocks only. "
                "Crypto data can be downloaded and trained, but crypto orders are disabled.",
                parent=self.root,
            )
            return
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model required", "Train and save trade_model.pt first.", parent=self.root
            )
            return
        try:
            api_credentials()
        except UserFacingError as exc:
            messagebox.showerror("Alpaca credentials", str(exc), parent=self.root)
            return
        confirmed = messagebox.askyesno(
            "Start paper trading",
            "This starts a PAPER-ONLY loop and may submit simulated market orders.\n\n"
            f"Symbol: {self.symbols[0]}\nQuantity: {PAPER_QUANTITY:g} share\n"
            f"Polling: {PAPER_POLL_SECONDS} seconds\n\nContinue?",
            parent=self.root,
        )
        if not confirmed:
            return
        self.stop_paper_event.clear()
        self.paper_running = True
        self.status_var.set("STATE  PAPER TRADING")
        self._set_controls(True)
        self.paper_button.configure(text="Stop Paper Trading", state="normal")
        self.notebook.select(1)
        self.paper_thread = threading.Thread(
            target=self._paper_worker, name="alpaca-paper", daemon=True
        )
        self.paper_thread.start()

    def _position_or_none(self, client: TradingClient, symbol: str):
        try:
            return call_with_backoff(
                lambda: client.get_open_position(symbol), self.events, f"position {symbol}"
            )
        except APIError as exc:
            status = api_status_code(exc)
            if status == 404 or "position does not exist" in str(exc).lower():
                return None
            raise

    def _paper_worker(self) -> None:
        try:
            key, secret = api_credentials()
            data_client = StockHistoricalDataClient(key, secret)
            trading_client = TradingClient(key, secret, paper=True)
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if list(checkpoint.get("feature_columns", [])) != FEATURE_COLUMNS:
                raise UserFacingError("trade_model.pt uses a different feature schema; retrain it.")
            if int(checkpoint.get("lookback", 0)) != LOOKBACK:
                raise UserFacingError("trade_model.pt uses a different lookback; retrain it.")
            model = ShallowTradeNet(int(checkpoint["input_size"]))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            mean = np.asarray(checkpoint["mean"], dtype=np.float32)
            std = np.asarray(checkpoint["std"], dtype=np.float32)
            timeframe = str(checkpoint["timeframe"])
            if str(checkpoint.get("asset_class", "stocks")) != "stocks":
                raise UserFacingError(
                    "This model was trained on crypto. Alpaca paper execution is currently enabled for US-stock models only."
                )
            symbol = self.symbols[0]
            trained_symbols = [str(item).upper() for item in checkpoint.get("symbols", [])]
            if symbol not in trained_symbols:
                raise UserFacingError(
                    f"{symbol} was not in the model's training universe: "
                    + ", ".join(trained_symbols)
                )
            account = call_with_backoff(trading_client.get_account, self.events, "account")
            if bool(account.trading_blocked):
                raise UserFacingError("The Alpaca paper account is blocked from trading.")
            starting_equity = float(account.equity)
            last_processed: pd.Timestamp | None = None
            market_closed_logged = False
            self.events.put(
                {
                    "kind": "log",
                    "text": (
                        f"PAPER  | endpoint=paper  symbol={symbol}  timeframe={timeframe}  "
                        f"qty={PAPER_QUANTITY:g}"
                    ),
                }
            )
            while not self.stop_paper_event.is_set():
                clock = call_with_backoff(trading_client.get_clock, self.events, "market clock")
                if not bool(clock.is_open):
                    if not market_closed_logged:
                        self.events.put(
                            {
                                "kind": "log",
                                "text": f"PAPER  | market closed  next_open={clock.next_open}",
                            }
                        )
                        market_closed_logged = True
                    self.stop_paper_event.wait(PAPER_POLL_SECONDS)
                    continue
                market_closed_logged = False
                base_bars = read_cached_bars(symbol, timeframe)
                bars = fetch_alpaca_execution_bars(
                    data_client, symbol, timeframe, base_bars, self.events
                )
                model_input, timestamp, price = latest_model_input(bars, mean, std)
                if last_processed is not None and timestamp <= last_processed:
                    self.stop_paper_event.wait(PAPER_POLL_SECONDS)
                    continue
                last_processed = timestamp
                with torch.inference_mode():
                    logits = model(torch.from_numpy(model_input).unsqueeze(0))
                    probabilities = torch.softmax(logits, dim=1)[0]
                    signal = int(probabilities.argmax().item())
                    confidence = float(probabilities[signal].item())

                position = self._position_or_none(trading_client, symbol)
                position_side = str(getattr(position, "side", "")).lower()
                is_long = position is not None and "long" in position_side
                position_qty = abs(float(position.qty)) if is_long else 0.0
                action = "NONE"
                open_orders = call_with_backoff(
                    lambda: trading_client.get_orders(
                        filter=GetOrdersRequest(
                            status=QueryOrderStatus.OPEN, symbols=[symbol]
                        )
                    ),
                    self.events,
                    f"open orders {symbol}",
                )
                if position is not None and not is_long:
                    action = "SKIP NON-LONG POSITION"
                elif open_orders:
                    action = "WAIT OPEN ORDER"
                elif signal == 1 and position_qty == 0:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=PAPER_QUANTITY,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=f"trademind-buy-{int(time.time())}",
                    )
                    call_with_backoff(
                        lambda: trading_client.submit_order(order_data=order),
                        self.events,
                        "paper buy",
                    )
                    action = f"BUY {PAPER_QUANTITY:g}"
                elif signal == 2 and position_qty > 0:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=position_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                        client_order_id=f"trademind-sell-{int(time.time())}",
                    )
                    call_with_backoff(
                        lambda: trading_client.submit_order(order_data=order),
                        self.events,
                        "paper sell",
                    )
                    action = f"SELL {position_qty:g}"
                account = call_with_backoff(trading_client.get_account, self.events, "account")
                equity = float(account.equity)
                pnl = equity - starting_equity
                self.events.put(
                    {
                        "kind": "paper_row",
                        "values": (
                            timestamp.strftime("%Y-%m-%d %H:%M"),
                            symbol,
                            SIGNAL_NAMES[signal],
                            f"{confidence:.1%}",
                            f"{price:.2f}",
                            f"{position_qty:g}",
                            f"{equity:.2f}",
                            f"{pnl:+.2f}",
                            action,
                        ),
                    }
                )
                self.events.put(
                    {
                        "kind": "log",
                        "text": (
                            f"PAPER  | {symbol} signal={SIGNAL_NAMES[signal]} "
                            f"confidence={confidence:.1%} action={action}"
                        ),
                    }
                )
                self.stop_paper_event.wait(PAPER_POLL_SECONDS)
            self.events.put({"kind": "paper_done", "text": "PAPER  | loop stopped"})
        except Exception as exc:
            self.events.put(
                {"kind": "paper_error", "context": "Paper trading", "text": str(exc)}
            )

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
                    self._refresh_meta()
                    if event.get("start_training"):
                        self.train_after_download = False
                        self.root.after(100, self.start_training)
                elif kind == "training_setup":
                    self.status_var.set("STATE  TRAINING")
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
                    self.progress_var.set(
                        100.0 * self.current_batch / max(self.batches_per_epoch, 1)
                    )
                elif kind == "epoch_metrics":
                    self.train_losses.append(float(event["train_loss"]))
                    self.validation_losses.append(float(event["validation_loss"]))
                    self.validation_accuracies.append(float(event["accuracy"]))
                    self.latest_train_loss = self.train_losses[-1]
                    self.latest_validation_loss = self.validation_losses[-1]
                    self.latest_accuracy = self.validation_accuracies[-1]
                    self._log(
                        f"EPOCH  | {event['epoch']:02d}/{EPOCHS:02d}  "
                        f"train={self.latest_train_loss:.5f}  "
                        f"val={self.latest_validation_loss:.5f}  "
                        f"accuracy={self.latest_accuracy:.2%}"
                    )
                elif kind == "training_done":
                    self.training = False
                    self.status_var.set("STATE  STOPPED" if event["stopped"] else "STATE  FINISHED")
                    self._set_controls(False)
                    self._log(event["text"])
                    self._refresh_meta()
                elif kind == "paper_row":
                    self.paper_table.insert("", 0, values=event["values"])
                    children = self.paper_table.get_children()
                    if len(children) > 250:
                        self.paper_table.delete(children[-1])
                elif kind == "paper_done":
                    self.paper_running = False
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
        epochs = list(range(1, len(self.train_losses) + 1))
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
            )
            legend = self.loss_axes.legend(loc="upper right", frameon=False, fontsize=8)
            for label in legend.get_texts():
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
        if self.root.winfo_exists():
            self.root.after(2000, self._redraw_chart)

    @staticmethod
    def _duration(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(seconds):
            return "--:--:--"
        hours, remainder = divmod(max(0, int(seconds)), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _metric(value: float | None, digits: int = 5) -> str:
        return f"{value:.{digits}f}" if value is not None else "--"

    def _refresh_telemetry(self) -> None:
        if self.training and self.started_at:
            self.elapsed_seconds = time.monotonic() - self.started_at
        total_batches = EPOCHS * self.batches_per_epoch
        completed = 0
        if self.current_epoch and self.batches_per_epoch:
            completed = (self.current_epoch - 1) * self.batches_per_epoch + self.current_batch
        eta = None
        if self.training and completed > 0 and total_batches > completed:
            eta = self.elapsed_seconds / completed * (total_batches - completed)
        self.telemetry_var.set(
            f"EPOCH {self.current_epoch:02d}/{EPOCHS:02d}  |  "
            f"BATCH {self.current_batch:04d}/{self.batches_per_epoch:04d}  |  "
            f"ELAPSED {self._duration(self.elapsed_seconds)}  |  ETA {self._duration(eta)}\n"
            f"TRAIN LOSS {self._metric(self.latest_train_loss)}  |  "
            f"VAL LOSS {self._metric(self.latest_validation_loss)}  |  "
            f"VAL ACC {self._metric(self.latest_accuracy, 3)}  |  "
            f"BATCH CLASS {SIGNAL_NAMES[self.current_batch_label]}\n"
            f"LABELS HOLD={self.latest_class_counts[0]}  BUY={self.latest_class_counts[1]}  "
            f"SELL={self.latest_class_counts[2]}  |  OPTIMIZER ADAMW  |  LR {LEARNING_RATE:.2E}  |  CPU"
        )

    def _update_clock(self) -> None:
        self._refresh_telemetry()
        if self.root.winfo_exists():
            self.root.after(1000, self._update_clock)

    def close(self) -> None:
        if self.training:
            self.stop_training_event.set()
        if self.paper_running:
            self.stop_paper_event.set()
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
    AlpacaTradeMindApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
