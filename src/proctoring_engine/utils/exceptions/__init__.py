"""Exception utilities for the proctoring engine."""

from proctoring_engine.utils.exceptions.error_codes import (
    ERR_AUDIO_THREAD_FAILED,
    ERR_CAMERA_INIT_FAILED,
    ERR_MODEL_INIT_FAILED,
)
from proctoring_engine.utils.exceptions.exceptions import (
    AudioThreadError,
    AuthenticationException,
    CameraInitError,
    DatabaseException,
    ModelInitError,
    ProctoringEngineError,
    ValidationException,
)

__all__ = [
    "ProctoringEngineError",
    "ModelInitError",
    "CameraInitError",
    "AudioThreadError",
    "AuthenticationException",
    "DatabaseException",
    "ValidationException",
    "ERR_MODEL_INIT_FAILED",
    "ERR_CAMERA_INIT_FAILED",
    "ERR_AUDIO_THREAD_FAILED",
]
