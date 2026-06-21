from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_regime(returns: pd.Series, n_states: int = 3) -> str:
    """Gaussian HMM regime detection on returns/volatility."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        return "unknown"

    clean = returns.dropna()
    if len(clean) < 60:
        return "unknown"

    vol = clean.rolling(5).std().fillna(clean.std())
    features = np.column_stack([clean.values, vol.values])
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    try:
        model.fit(features)
        state = int(model.predict(features)[-1])
    except Exception:
        logger.exception("HMM fit failed")
        return "unknown"

    labels = ["low_vol", "trending", "high_vol"]
    return labels[min(state, len(labels) - 1)]
