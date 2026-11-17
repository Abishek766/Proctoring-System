"""JWT token utilities."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from jose import jwt

from proctoring_engine.settings import get_settings


def create_access_token(user_id: str) -> str:
    """Create a JWT access token for the given user_id."""
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm="HS256")
