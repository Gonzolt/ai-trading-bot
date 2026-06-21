from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def sortino_ratio(returns: pd.Series, annualization: int = 252) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std(ddof=0)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(annualization) * returns.mean() / downside_std)


def performance_metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, float]:
    """Compute Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor, Calmar."""
    returns = equity.pct_change().dropna()
    sharpe = 0.0
    if not returns.empty and returns.std(ddof=0) != 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std(ddof=0))
    sortino = sortino_ratio(returns) if not returns.empty else 0.0
    drawdown = max_drawdown(equity)
    calmar = 0.0
    if drawdown != 0:
        years = max(len(equity) / 252, 1 / 252)
        annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
        calmar = float(annual_return / abs(drawdown))

    win_rate = 0.0
    profit_factor = 0.0
    if trades is not None and not trades.empty and "pnl" in trades:
        wins = trades[trades["pnl"] > 0]["pnl"].sum()
        losses = abs(trades[trades["pnl"] < 0]["pnl"].sum())
        win_rate = float((trades["pnl"] > 0).mean())
        profit_factor = float(wins / losses) if losses else float("inf")

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "calmar": calmar,
    }
