from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from trading_bot.config import load_settings

bearer = HTTPBearer(auto_error=False)
ALGORITHM = "HS256"


def _secret() -> str:
    return os.getenv("JWT_SECRET", "dev-only-change-me")


def create_access_token(subject: str) -> str:
    settings = load_settings()
    expires = datetime.now(UTC) + timedelta(
        minutes=int(settings.get("dashboard.jwt_expire_minutes", 720))
    )
    return jwt.encode({"sub": subject, "exp": expires}, _secret(), algorithm=ALGORITHM)


def authenticate(username: str, password: str) -> bool:
    expected_username = os.getenv("DASHBOARD_USERNAME", "admin")
    expected_password = os.getenv("DASHBOARD_PASSWORD", "change-me")
    return username == expected_username and password == expected_password


async def require_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    try:
        payload = jwt.decode(credentials.credentials, _secret(), algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return str(subject)
