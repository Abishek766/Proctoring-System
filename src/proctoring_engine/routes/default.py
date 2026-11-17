"""Default HTTP endpoints — health, readiness, and root."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from proctoring_engine.services.dependencies import get_session_manager

router = APIRouter(tags=["health"])


class RootResponse(BaseModel):
    name: str
    version: str
    description: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    active_sessions: int


@router.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        name="video-proctoring-engine",
        version="0.1.0",
        description="Real-time exam video proctoring engine API",
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 if the process is running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready() -> ReadyResponse:
    """Readiness probe — returns active session count."""
    manager = get_session_manager()
    return ReadyResponse(status="ok", active_sessions=manager.active_count)
