from __future__ import annotations

import numpy as np
import pandas as pd

from quantumtrader.config import load_settings
from quantumtrader.models import RankingRow
from quantumtrader.strategies.indicators import adx, atr, sharpe_ratio


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _percentile_0_100(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.rank(pct=True).fillna(0.5) * 100


class AssetScorer:
    """Daily composite asset ranking (0–100) for universe selection."""

    def __init__(self) -> None:
        settings = load_settings()
        self.weights = settings.get("selection.weights", {})
        self.max_assets = int(settings.get("universe.max_selected_assets", 15))

    def score(
        self,
        history: dict[str, pd.DataFrame],
        sentiment: dict[str, float] | None = None,
    ) -> list[RankingRow]:
        sentiment = sentiment or {}
        raw = []
        for symbol, df in history.items():
            if len(df) < 80:
                continue
            close = df["close"]
            volume = df["volume"]
            returns = close.pct_change()
            atr_pct = (atr(df).iloc[-1] / close.iloc[-1]) if close.iloc[-1] else np.nan
            raw.append(
                {
                    "symbol": symbol,
                    "momentum_20d": close.pct_change(20).iloc[-1],
                    "volume_growth_5d": volume.tail(5).mean() / volume.tail(20).mean(),
                    "adx": adx(df).iloc[-1],
                    "atr_pct": atr_pct,
                    "sentiment": (sentiment.get(symbol, 0.0) + 1) / 2,
                    "sharpe_60d": sharpe_ratio(returns.tail(60)),
                }
            )

        if not raw:
            return []

        frame = pd.DataFrame(raw).set_index("symbol")
        components = pd.DataFrame(index=frame.index)
        components["momentum_20d"] = _percentile_0_100(_zscore(frame["momentum_20d"]))
        components["volume_growth_5d"] = _percentile_0_100(_zscore(frame["volume_growth_5d"]))
        components["adx"] = frame["adx"].clip(0, 100).fillna(0)
        components["atr_inverse"] = _percentile_0_100(-_zscore(frame["atr_pct"]))
        components["sentiment"] = (frame["sentiment"].clip(0, 1) * 100).fillna(50)
        components["sharpe_60d"] = _percentile_0_100(_zscore(frame["sharpe_60d"]))

        weighted = pd.Series(0.0, index=components.index)
        for key, weight in self.weights.items():
            mapped_key = "atr_inverse" if key == "atr_inverse" else key
            weighted += components[mapped_key] * float(weight)

        ranked = weighted.sort_values(ascending=False).head(self.max_assets)
        rows: list[RankingRow] = []
        for rank, (symbol, score) in enumerate(ranked.items(), start=1):
            rows.append(
                RankingRow(
                    symbol=symbol,
                    score=float(round(score, 4)),
                    rank=rank,
                    components={
                        key: float(round(value, 4)) for key, value in components.loc[symbol].items()
                    },
                )
            )
        return rows
