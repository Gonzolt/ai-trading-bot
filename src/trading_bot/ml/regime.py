from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeDetector:
    """Gaussian HMM regime detector over returns and realized volatility."""

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components
        self.model = None

    def _features(self, close: pd.Series) -> np.ndarray:
        returns = close.pct_change().fillna(0)
        volatility = returns.rolling(20).std().fillna(0)
        return np.column_stack([returns.to_numpy(), volatility.to_numpy()])

    def fit(self, close: pd.Series) -> "RegimeDetector":
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise RuntimeError("Install hmmlearn to use market regime detection.") from exc

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        self.model.fit(self._features(close))
        return self

    def predict_latest(self, close: pd.Series) -> str:
        if self.model is None:
            return "unknown"
        states = self.model.predict(self._features(close))
        latest = int(states[-1])
        return {0: "calm", 1: "trend", 2: "stress"}.get(latest, f"regime_{latest}")
