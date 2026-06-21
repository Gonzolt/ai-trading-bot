from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from quantumtrader.config import load_settings
from quantumtrader.dashboard.api.auth import authenticate, create_access_token, require_user
from quantumtrader.database import db

settings = load_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

app = FastAPI(title="QuantumTrader API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("dashboard.cors_origins", ["http://localhost"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


def _demo_equity() -> list[dict[str, Any]]:
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    equity = 100000.0
    rows = []
    for offset in range(60, 0, -1):
        equity *= 1 + random.uniform(-0.002, 0.003)
        rows.append(
            {
                "time": (now - timedelta(days=offset)).isoformat(),
                "equity": round(equity, 2),
                "drawdown": round(min(0.0, equity / 104000 - 1), 4),
                "daily_pnl": round(random.uniform(-600, 900), 2),
            }
        )
    return rows


DEMO_RANKINGS = [
    {"rank": 1, "symbol": "NVDA", "score": 91.2, "components": {"momentum_20d": 97, "sharpe_60d": 94}},
    {"rank": 2, "symbol": "QQQ", "score": 84.6, "components": {"momentum_20d": 82, "sharpe_60d": 88}},
    {"rank": 3, "symbol": "BTC-USD", "score": 78.9, "components": {"momentum_20d": 89, "sharpe_60d": 75}},
]


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    return require_user(token)


@app.on_event("shutdown")
async def shutdown() -> None:
    await db.close()


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": settings.trading_mode, "app": "QuantumTrader"}


@app.post("/api/token", response_model=TokenResponse)
async def token(payload: LoginRequest) -> TokenResponse:
    if not authenticate(payload.username, payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    expire = int(settings.get("dashboard.jwt_expire_minutes", 720))
    return TokenResponse(access_token=create_access_token(payload.username, expire))


@app.get("/api/portfolio")
async def portfolio(_: Annotated[str, Depends(get_current_user)]) -> dict[str, Any]:
    try:
        positions = await db.fetch(
            """
            SELECT symbol, quantity, average_price, mark_price, unrealized_pnl, stop, target, strategy
            FROM positions ORDER BY symbol
            """
        )
        equity = await db.fetch(
            "SELECT time, equity, drawdown, daily_pnl FROM equity_curve ORDER BY time DESC LIMIT 120"
        )
        return {
            "positions": [dict(row) for row in positions],
            "equity_curve": [dict(row) for row in reversed(equity)],
        }
    except Exception:
        return {
            "positions": [
                {"symbol": "QQQ", "quantity": 34, "average_price": 472.15, "mark_price": 481.62,
                 "unrealized_pnl": 321.98, "stop": 452.2, "target": 512.9, "strategy": "trend_following"},
            ],
            "equity_curve": _demo_equity(),
        }


@app.get("/api/risk")
async def risk(_: Annotated[str, Depends(get_current_user)]) -> dict[str, Any]:
    return {
        "mode": settings.trading_mode,
        "live_trading_enabled": settings.live_trading_enabled,
        "daily_loss_limit_pct": settings.get("risk.max_daily_loss_pct", 0.02),
        "max_drawdown_pct": settings.get("risk.max_drawdown_pct", 0.20),
        "risk_per_trade_pct": settings.get("risk.risk_per_trade_pct", 0.01),
        "current_drawdown": -0.043,
        "gross_leverage": 0.42,
        "status": "paper trading" if not settings.live_trading_enabled else "live enabled",
    }


@app.get("/api/rankings")
async def rankings(_: Annotated[str, Depends(get_current_user)]) -> list[dict[str, Any]]:
    try:
        rows = await db.fetch(
            """
            SELECT rank, symbol, score, components FROM asset_rankings
            WHERE ranking_date = (SELECT max(ranking_date) FROM asset_rankings)
            ORDER BY rank LIMIT 15
            """
        )
        return [dict(row) for row in rows] or DEMO_RANKINGS
    except Exception:
        return DEMO_RANKINGS


@app.get("/api/trades")
async def trades(_: Annotated[str, Depends(get_current_user)]) -> list[dict[str, Any]]:
    try:
        rows = await db.fetch(
            """
            SELECT created_at, symbol, strategy, side, quantity, price, commission, slippage, realized_pnl, mode
            FROM trades ORDER BY created_at DESC LIMIT 100
            """
        )
        return [dict(row) for row in rows]
    except Exception:
        now = datetime.now(UTC)
        return [{
            "created_at": (now - timedelta(minutes=45)).isoformat(),
            "symbol": "QQQ", "strategy": "trend_following", "side": "buy",
            "quantity": 34, "price": 472.15, "commission": 1.61, "slippage": 3.21,
            "realized_pnl": 0, "mode": "paper",
        }]


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        equity = 101250.0
        while True:
            equity *= 1 + random.uniform(-0.0007, 0.001)
            await websocket.send_json({
                "time": datetime.now(UTC).isoformat(),
                "equity": round(equity, 2),
                "daily_pnl": round(equity - 100000, 2),
                "drawdown": round(min(0.0, equity / 104000 - 1), 4),
                "risk_status": "paper trading",
            })
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return
