from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.config import load_settings
from trading_bot.models import StrategySignal
from trading_bot.strategies.base import Strategy
from trading_bot.strategies.indicators import atr, macd, rsi

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, sentiment: pd.Series | None = None) -> pd.DataFrame:
    close = df["close"]
    volume = df["volume"].replace(0, np.nan)
    _, _, histogram = macd(close)
    features = pd.DataFrame(index=df.index)
    features["return_1d"] = close.pct_change()
    features["return_5d"] = close.pct_change(5)
    features["volatility_20d"] = features["return_1d"].rolling(20).std()
    features["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    features["rsi_14"] = rsi(close, 14)
    features["macd_histogram"] = histogram
    features["atr_pct"] = atr(df) / close
    features["sentiment"] = sentiment.reindex(df.index).ffill().fillna(0) if sentiment is not None else 0.0
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def train_lightgbm_model(df: pd.DataFrame, model_path: str | Path) -> Path:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("Install lightgbm to train the ML strategy.") from exc

    features = engineer_features(df)
    target = (df["close"].pct_change().shift(-1).reindex(features.index) > 0).astype(int)
    dataset = lgb.Dataset(features, label=target)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "learning_rate": 0.03,
        "num_leaves": 16,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }
    model = lgb.train(params, dataset, num_boost_round=120)
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output))
    return output


class MLDirectionStrategy(Strategy):
    name = "ml"

    def __init__(self, model_path: str | None = None) -> None:
        settings = load_settings()
        self.enter_probability = float(settings.get("strategy.ml.probability_enter", 0.60))
        self.exit_probability = float(settings.get("strategy.ml.probability_exit", 0.40))
        self.model_path = Path(model_path or settings.get("strategy.ml.model_path", "models/lightgbm_direction.txt"))
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            return None
        try:
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(self.model_path))
            return self._model
        except Exception:
            logger.exception("Could not load LightGBM model from %s", self.model_path)
            return None

    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        model = self._load_model()
        if model is None or len(asset_df) < 80:
            return StrategySignal(symbol, self.name, "hold", None, None, metadata={"model_loaded": False})
        features = engineer_features(asset_df)
        if features.empty:
            return StrategySignal(symbol, self.name, "hold", None, None)
        probability = float(model.predict(features.tail(1))[0])
        price = float(asset_df["close"].iloc[-1])
        atr_value = float(atr(asset_df).iloc[-1])
        if probability > self.enter_probability:
            return StrategySignal(
                symbol,
                self.name,
                "buy",
                price - 2 * atr_value,
                price + 3 * atr_value,
                confidence=probability,
                metadata={"probability_up": probability},
            )
        if probability < self.exit_probability:
            return StrategySignal(
                symbol,
                self.name,
                "sell",
                None,
                None,
                confidence=1 - probability,
                metadata={"probability_up": probability},
            )
        return StrategySignal(
            symbol,
            self.name,
            "hold",
            None,
            None,
            confidence=probability,
            metadata={"probability_up": probability},
        )
