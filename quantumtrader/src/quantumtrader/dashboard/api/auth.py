from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from jose import jwt
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def authenticate(username: str, password: str) -> bool:
    expected_user = os.getenv("DASHBOARD_USERNAME", "admin")
    expected_pass = os.getenv("DASHBOARD_PASSWORD", "QuantumTrader2026!")
    return username == expected_user and password == expected_pass


def create_access_token(username: str, expire_minutes: int = 720) -> str:
    secret = os.getenv("JWT_SECRET", "change-me")
    payload = {
        "sub": username,
        "exp": datetime.now(UTC) + timedelta(minutes=expire_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def require_user(token: str) -> str:
    from fastapi import HTTPException

    secret = os.getenv("JWT_SECRET", "change-me")
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return str(payload["sub"])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
