from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def deep_get(data: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root_dir: Path

    @property
    def database_url(self) -> str:
        return os.getenv(
            "DATABASE_URL",
            "postgresql://trader:trader_password@localhost:5432/trading",
        )

    @property
    def redis_url(self) -> str:
        return os.getenv("REDIS_URL", "redis://localhost:6379/0")

    @property
    def mongo_url(self) -> str:
        return os.getenv("MONGO_URL", "mongodb://localhost:27017")

    @property
    def trading_mode(self) -> str:
        return os.getenv("TRADING_MODE", deep_get(self.raw, "app.trading_mode", "paper"))

    @property
    def live_trading_enabled(self) -> bool:
        env_value = os.getenv("ENABLE_LIVE_TRADING")
        if env_value is not None:
            return env_value.lower() == "true"
        return bool(deep_get(self.raw, "app.enable_live_trading", False))

    def get(self, path: str, default: Any = None) -> Any:
        return deep_get(self.raw, path, default)


def _default_settings_path() -> Path:
    env_path = os.getenv("SETTINGS_PATH")
    if env_path:
        return Path(env_path)
    return Path.cwd() / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings(path: str | None = None) -> Settings:
    settings_path = Path(path) if path else _default_settings_path()
    with settings_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Settings(raw=raw, root_dir=settings_path.parent.parent)
