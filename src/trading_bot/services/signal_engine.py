from __future__ import annotations

import asyncio
import logging

from trading_bot.config import load_settings
from trading_bot.data.providers import YFinanceProvider
from trading_bot.selection.scoring import AssetScoringEngine
from trading_bot.strategies import (
    MLDirectionStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    TrendFollowingStrategy,
)
from trading_bot.utils.logging import configure_logging

logger = logging.getLogger(__name__)


async def run_once() -> None:
    settings = load_settings()
    provider = YFinanceProvider()
    history = {}
    for symbol in settings.get("universe.symbols", []):
        try:
            history[symbol] = await provider.history(symbol, days=int(settings.get("data.history_days", 800)))
        except Exception:
            logger.exception("history failed for %s", symbol)
    rankings = AssetScoringEngine().score(history)
    logger.info("top assets: %s", [(row.symbol, row.score) for row in rankings[:5]])
    strategies = [
        TrendFollowingStrategy(),
        MomentumStrategy(),
        MeanReversionStrategy(),
        MLDirectionStrategy(),
    ]
    for ranking in rankings:
        df = history[ranking.symbol]
        for strategy in strategies:
            signal = strategy.generate_signal(ranking.symbol, df)
            if signal.signal != "hold":
                logger.info("signal=%s", signal)


async def main() -> None:
    configure_logging()
    while True:
        await run_once()
        await asyncio.sleep(60 * 15)


if __name__ == "__main__":
    asyncio.run(main())
