from __future__ import annotations

import logging
import os

from celery import Celery
from celery.schedules import crontab

from quantumtrader.config import load_settings

settings = load_settings()
logger = logging.getLogger(__name__)

celery_app = Celery("quantumtrader", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.timezone = settings.get("app.timezone", "UTC")
celery_app.conf.beat_schedule = {
    "nightly-ml-retrain": {
        "task": "quantumtrader.scheduler.retrain_ml_models",
        "schedule": crontab(hour=2, minute=0),
    },
    "weekly-backtest": {
        "task": "quantumtrader.scheduler.run_weekly_backtest",
        "schedule": crontab(hour=3, minute=0, day_of_week=0),
    },
    "daily-asset-scoring": {
        "task": "quantumtrader.scheduler.run_asset_scoring",
        "schedule": crontab(hour=6, minute=0),
    },
}


@celery_app.task(name="quantumtrader.scheduler.retrain_ml_models")
def retrain_ml_models() -> str:
    """Nightly LightGBM retraining."""
    import asyncio

    import pandas as pd

    from quantumtrader.database import db
    from quantumtrader.strategies.ml_predictor import MLPredictorStrategy

    async def _run() -> None:
        strategy = MLPredictorStrategy()
        symbols = settings.get("universe.symbols", [])
        for symbol in symbols:
            rows = await db.fetch(
                "SELECT time, open, high, low, close, volume FROM ohlcv_1d WHERE symbol = $1 ORDER BY time",
                symbol,
            )
            if not rows:
                continue
            df = pd.DataFrame([dict(r) for r in rows])
            df["time"] = pd.to_datetime(df["time"], utc=True)
            strategy.train(df.set_index("time"))

    asyncio.run(_run())
    return "ml retrain complete"


@celery_app.task(name="quantumtrader.scheduler.run_weekly_backtest")
def run_weekly_backtest() -> str:
    from services.backtester.main import BacktesterService

    import asyncio

    asyncio.run(BacktesterService().run_backtests())
    return "weekly backtest complete"


@celery_app.task(name="quantumtrader.scheduler.run_asset_scoring")
def run_asset_scoring() -> str:
    from services.asset_scorer.main import AssetScorerService

    import asyncio

    asyncio.run(AssetScorerService().run_scoring())
    return "asset scoring complete"
