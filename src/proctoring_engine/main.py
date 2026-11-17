"""Entry point for the Video Proctoring Engine Benchmark.

Usage:
    python -m proctoring_engine.main

Press 'q' in the OpenCV window to stop the session.
"""

from __future__ import annotations

import sys

from proctoring_engine.core.audio_pipeline import AudioPipeline
from proctoring_engine.services.proctoring_service import ProctoringService
from proctoring_engine.settings import get_settings
from proctoring_engine.utils.exceptions import CameraInitError, ModelInitError
from proctoring_engine.utils.logger import get_logger

logger = get_logger("benchmark")


def main() -> int:
    """Initialise all pipelines and run the benchmark capture loop.

    Returns:
        0 on clean exit, 1 on fatal error.
    """
    settings = get_settings()
    logger.info("Starting Video Proctoring Engine Benchmark...")

    try:
        proctoring = ProctoringService(settings)
    except ModelInitError as exc:
        logger.error(f"Model initialisation failed [{exc.error_code}]: {exc}")
        return 1

    # Start audio pipeline (runs in background thread)
    try:
        audio = AudioPipeline(
            settings=settings,
            on_violation=proctoring.on_audio_violation,
            get_mar=proctoring.get_mar,
        )
    except ModelInitError as exc:
        logger.error(f"YAMNet initialisation failed [{exc.error_code}]: {exc}")
        return 1

    try:
        proctoring.run()
    except CameraInitError as exc:
        logger.error(f"Camera error [{exc.error_code}]: {exc}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        audio.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
