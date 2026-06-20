from __future__ import annotations

import logging
from datetime import timedelta

from celery import Celery

from trading_bot.config import load_settings

logger = logging.getLogger(__name__)
settings = load_settings()

celery_app = Celery(
    "trading_bot_scheduler",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.beat_schedule = {
    "nightly-model-retraining": {
        "task": "trading_bot.services.scheduler.retrain_models",
        "schedule": timedelta(days=1),
    },
    "weekly-walk-forward": {
        "task": "trading_bot.services.scheduler.weekly_walk_forward",
        "schedule": timedelta(days=7),
    },
}
celery_app.conf.timezone = "UTC"


@celery_app.task(name="trading_bot.services.scheduler.retrain_models")
def retrain_models() -> dict[str, str]:
    logger.info("nightly retraining placeholder started")
    return {"status": "scheduled", "task": "retrain_models"}


@celery_app.task(name="trading_bot.services.scheduler.weekly_walk_forward")
def weekly_walk_forward() -> dict[str, str]:
    logger.info("weekly walk-forward validation placeholder started")
    return {"status": "scheduled", "task": "weekly_walk_forward"}
