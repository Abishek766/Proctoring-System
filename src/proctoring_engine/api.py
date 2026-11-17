"""Entry point for the API server.

Usage:
    python -m proctoring_engine.api

This starts a uvicorn server on 0.0.0.0:8000 with the number of workers
set to 1 (each worker holds its own pipeline instances in memory).

For production with multiple candidates, run multiple workers behind a
load balancer — each worker handles its own set of sessions independently.
"""

from __future__ import annotations

import sys

import uvicorn

from proctoring_engine.utils.logger import get_logger

logger = get_logger("video_pipeline")

# ─── Server configuration ─────────────────────────────────────────────────────
HOST    = "0.0.0.0"
PORT    = 8001
# Single worker — all sessions share the same process and pipeline instances.
# Increase workers only if you have enough RAM for multiple YOLO + MediaPipe sets.
WORKERS = 1
LOG_LEVEL = "info"


def main() -> int:
    logger.info(f"Starting Proctoring Engine API on {HOST}:{PORT} ...")
    try:
        uvicorn.run(
            "proctoring_engine.app:app",
            host=HOST,
            port=PORT,
            workers=WORKERS,
            log_level=LOG_LEVEL,
            # WebSocket keep-alive — disable server-side ping/pong
            # to prevent uvicorn from killing long-running sessions
            # when the client doesn't respond to pings quickly enough.
            ws_ping_interval=None,
            ws_ping_timeout=None,
        )
    except KeyboardInterrupt:
        logger.info("API server stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
