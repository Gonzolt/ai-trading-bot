from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quantumtrader.config import load_settings
from quantumtrader.models import StrategySignal
from quantumtrader.strategies.base import Strategy
from quantumtrader.strategies.indicators import macd, rsi

logger = logging.getLogger(__name__)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for the LightGBM classifier."""
    close = df["close"]
    volume = df["volume"]
    returns = close.pct_change()
    _, _, histogram = macd(close)
    rsi_values = rsi(close)
    features = pd.DataFrame(
        {
            "ret_1": returns,
            "ret_5": close.pct_change(5),
            "ret_20": close.pct_change(20),
            "vol_ratio": volume / volume.rolling(20).mean(),
            "rsi": rsi_values,
            "macd_hist": histogram,
        },
        index=df.index,
    )
    return features.dropna()


class MLPredictorStrategy(Strategy):
    """LightGBM direction classifier. Retrained nightly via Celery."""

    name = "ml_predictor"

    def __init__(self) -> None:
        settings = load_settings()
        self.buy_threshold = float(settings.get("strategy.ml_predictor.probability_buy", 0.60))
        self.sell_threshold = float(settings.get("strategy.ml_predictor.probability_sell", 0.40))
        model_rel = settings.get("strategy.ml_predictor.model_path", "models/lightgbm_direction.txt")
        self.model_path = Path(settings.root_dir) / model_rel
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            logger.warning("ML model not found at %s; strategy will hold", self.model_path)
            return
        try:
            import lightgbm as lgb

            self._model = lgb.Booster(model_file=str(self.model_path))
        except Exception:
            logger.exception("failed loading LightGBM model")

    def train(self, df: pd.DataFrame, sentiment: float = 0.0) -> None:
        """Train classifier on historical data (called by nightly Celery job)."""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
        except ImportError:
            logger.error("lightgbm/sklearn not installed")
            return

        features = _build_features(df)
        if len(features) < 100:
            return
        labels = (df.loc[features.index, "close"].pct_change().shift(-1) > 0).astype(int)
        features["sentiment"] = sentiment
        aligned = features.join(labels.rename("label")).dropna()
        if aligned.empty:
            return
        x_train, x_test, y_train, y_test = train_test_split(
            aligned.drop(columns=["label"]),
            aligned["label"],
            test_size=0.2,
            shuffle=False,
        )
        train_set = lgb.Dataset(x_train, label=y_train)
        params = {"objective": "binary", "metric": "auc", "verbosity": -1}
        booster = lgb.train(params, train_set, num_boost_round=100)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(self.model_path))
        self._model = booster
        logger.info("ML model trained; test size=%d", len(x_test))

    def generate_signal(self, symbol: str, asset_df: pd.DataFrame) -> StrategySignal:
        if self._model is None or len(asset_df) < 80:
            return StrategySignal(symbol, self.name, "hold", None, None)
        features = _build_features(asset_df)
        if features.empty:
            return StrategySignal(symbol, self.name, "hold", None, None)
        row = features.iloc[[-1]]
        prob = float(self._model.predict(row)[0])
        if prob > self.buy_threshold:
            return StrategySignal(symbol, self.name, "buy", None, None, confidence=prob)
        if prob < self.sell_threshold:
            return StrategySignal(symbol, self.name, "sell", None, None, confidence=1 - prob)
        return StrategySignal(symbol, self.name, "hold", None, None, confidence=abs(prob - 0.5))
