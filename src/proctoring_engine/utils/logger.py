"""Centralised logging configuration for the Video Proctoring Engine.

Named loggers and their routing:

  video_pipeline  → video_violations.log  + console
  audio_pipeline  → audio_violations.log  + console
  yolo            → yolo_detections.log   + console
  mediapipe       → mediapipe_detections.log + console
  yamnet          → yamnet_detections.log  + console
  benchmark       → video_violations.log + audio_violations.log + console
  api_session     → api_sessions.log + console   ← API flow only
  (unknown)       → video_violations.log + audio_violations.log + console (fallback)
"""

import logging
import logging.handlers
import sys
from pathlib import Path

from proctoring_engine.settings import get_settings

# ─── Logger name → log-file routing ───────────────────────────────────────────
_VIDEO_LOGGERS:     frozenset[str] = frozenset({"video_pipeline"})
_AUDIO_LOGGERS:     frozenset[str] = frozenset({"audio_pipeline"})
_YOLO_LOGGERS:      frozenset[str] = frozenset({"yolo"})
_MEDIAPIPE_LOGGERS: frozenset[str] = frozenset({"mediapipe"})
_YAMNET_LOGGERS:    frozenset[str] = frozenset({"yamnet"})
_BOTH_LOGGERS:      frozenset[str] = frozenset({"benchmark"})
_API_LOGGERS:       frozenset[str] = frozenset({"api_session"})
_FACE_VERIFY_LOGGERS: frozenset[str] = frozenset({"face_verification"})
_ERROR_LOGGERS:     frozenset[str] = frozenset({"error_logger"})
_BACKGROUND_LOGGERS: frozenset[str] = frozenset({"background_validation"})

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_MAX_BYTES    = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5


def _make_file_handler(log_path: str) -> logging.handlers.RotatingFileHandler:
    """Create a rotating file handler for *log_path*."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _make_console_handler() -> logging.StreamHandler:
    """Create a stdout console handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a named logger with appropriate file and console handlers attached.

    Routing table:
    ┌──────────────────┬──────────────────────────┬─────────┐
    │ Logger name      │ Log file                 │ Console │
    ├──────────────────┼──────────────────────────┼─────────┤
    │ video_pipeline   │ video_violations.log     │ ✅      │
    │ audio_pipeline   │ audio_violations.log     │ ✅      │
    │ yolo             │ yolo_detections.log      │ ✅      │
    │ mediapipe        │ mediapipe_detections.log │ ✅      │
    │ yamnet           │ yamnet_detections.log    │ ✅      │
    │ face_verification│ face_verification.log    │ ✅      │
    │ benchmark        │ video + audio logs       │ ✅      │
    │ (unknown)        │ video + audio logs       │ ✅      │
    └──────────────────┴──────────────────────────┴─────────┘
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured — return cached instance

    logger.setLevel(level)
    logger.propagate = False

    # ── File handler routing ───────────────────────────────────────────────────
    if name in _VIDEO_LOGGERS:
        logger.addHandler(_make_file_handler(settings.video_violations_log))

    elif name in _AUDIO_LOGGERS:
        logger.addHandler(_make_file_handler(settings.audio_violations_log))

    elif name in _YOLO_LOGGERS:
        logger.addHandler(_make_file_handler(settings.yolo_detections_log))

    elif name in _MEDIAPIPE_LOGGERS:
        logger.addHandler(_make_file_handler(settings.mediapipe_log))

    elif name in _YAMNET_LOGGERS:
        logger.addHandler(_make_file_handler(settings.yamnet_log))

    elif name in _API_LOGGERS:
        logger.addHandler(_make_file_handler("api_integrate_logs/api_sessions.log"))

    elif name in _FACE_VERIFY_LOGGERS:
        logger.addHandler(_make_file_handler("engine_logs/face_verification.log"))

    elif name in _ERROR_LOGGERS:
        logger.addHandler(_make_file_handler("engine_logs/errors.log"))

    elif name in _BACKGROUND_LOGGERS:
        logger.addHandler(_make_file_handler("engine_logs/background_violations.log"))

    else:
        # benchmark + all unknown loggers → both violation logs
        logger.addHandler(_make_file_handler(settings.video_violations_log))
        logger.addHandler(_make_file_handler(settings.audio_violations_log))

    # ── Console handler (always attached) ─────────────────────────────────────
    logger.addHandler(_make_console_handler())

    return logger


"""Centralized logging configuration."""

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""

    def add_fields(self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["filename"] = record.filename
        log_record["line"] = record.lineno

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stacktrace": self.formatException(record.exc_info) if record.exc_info else None,
            }


def setup_logger(name: str = "registration", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure logger with JSON formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create JSON formatter
    formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(logger)s %(filename)s %(line)d %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncpg").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    return logger


# Create default logger instance
logger = setup_logger()
