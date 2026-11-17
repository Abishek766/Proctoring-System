"""Error response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status_code": 401,
                "error_code": "INVALID_CREDENTIALS",
                "message": "Invalid email or password",
                "details": {"email": "john@example.com"}
            }
        }
    )

    status_code: int
    """HTTP status code"""

    error_code: str
    """Application-specific error code"""

    message: str
    """Human-readable error message"""

    details: dict | None = None
    """Additional error details"""
