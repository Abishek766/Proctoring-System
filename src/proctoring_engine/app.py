"""FastAPI application factory for the Video Proctoring Engine API.

Usage:
    python -m proctoring_engine.api          # starts uvicorn
    uvicorn proctoring_engine.app:app        # direct uvicorn

The existing camera-based entry point is unchanged:
    python -m proctoring_engine.main

Startup model pre-loading:
    All models (FaceMesh, MediaPipe Hands, YOLOv8s ONNX, YAMNet) are fully
    loaded and warmed up during the lifespan startup — before any WebSocket
    connection arrives.  This means the first frame goes straight to inference
    with zero model-loading latency.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from proctoring_engine.database.connection import close_db, init_db
from proctoring_engine.routes.auth_routes import router as auth_router
from proctoring_engine.routes.default import router as default_router
from proctoring_engine.routes.proctoring_ws import router as ws_router
from proctoring_engine.routes.verification_routes import router as verification_router
from proctoring_engine.services.dependencies import get_session_manager
from proctoring_engine.utils.logger import get_logger

logger = get_logger("video_pipeline")


async def _preload_models() -> None:
    """Load and warm up every model before the first connection arrives.

    Uses centralized model_loader.py for all model initialization.
    This ensures models are loaded once and reused across all requests.

    What gets pre-loaded:
      - InsightFace (ArcFace)      — face recognition for /register and /verify
      - MediaPipe FaceMesh         — face landmarks for head pose/gaze detection
      - MediaPipe Hands            — hand detection for external hand detection
      - MediaPipe Selfie Segmentation — person segmentation for background validation
      - YOLOv8s                    — object detection for prohibited objects
      - YAMNet (TF Hub)            — audio classification for audio anomalies
      - DINOv2 (ViT-B/14)          — background embeddings for location validation
    """
    import asyncio  # noqa: PLC0415

    from proctoring_engine.core.model_loader import (  # noqa: PLC0415
        preload_dinov2_model,
        preload_facemesh_model,
        preload_hands_model,
        preload_insightface_model,
        preload_selfie_segmentation_model,
        preload_yamnet_model,
        preload_yolo_model,
    )
    from proctoring_engine.settings import get_settings  # noqa: PLC0415

    settings = get_settings()
    loop = asyncio.get_event_loop()

    logger.info("=" * 80)
    logger.info("[PRELOAD] Starting model preloading...")
    logger.info("=" * 80)

    # ── 1. InsightFace — face recognition for registration/verification ───────
    logger.info("[PRELOAD] (1/7) Loading InsightFace (ArcFace)...")
    logger.info("[PRELOAD]        Used for: /register and /verify endpoints")
    logger.info("[PRELOAD]        This may take 5-10 seconds on first run...")

    def _load_insightface():
        preload_insightface_model()

    await loop.run_in_executor(None, _load_insightface)
    logger.info("[PRELOAD] ✓ InsightFace loaded successfully")

    # ── 2. MediaPipe FaceMesh — face landmarks ────────────────────────────────
    logger.info("[PRELOAD] (2/7) Loading MediaPipe FaceMesh...")
    logger.info("[PRELOAD]        Used for: Head pose and eye gaze detection")

    def _load_facemesh():
        preload_facemesh_model()

    await loop.run_in_executor(None, _load_facemesh)
    logger.info("[PRELOAD] ✓ MediaPipe FaceMesh loaded successfully")

    # ── 3. MediaPipe Hands — hand detection ───────────────────────────────────
    logger.info("[PRELOAD] (3/7) Loading MediaPipe Hands...")
    logger.info("[PRELOAD]        Used for: External hand detection")

    def _load_hands():
        preload_hands_model()

    await loop.run_in_executor(None, _load_hands)
    logger.info("[PRELOAD] ✓ MediaPipe Hands loaded successfully")

    # ── 4. MediaPipe Selfie Segmentation — person segmentation ────────────────
    logger.info("[PRELOAD] (4/7) Loading MediaPipe Selfie Segmentation...")
    logger.info("[PRELOAD]        Used for: Background validation (person removal)")

    def _load_selfie_segmentation():
        preload_selfie_segmentation_model()

    await loop.run_in_executor(None, _load_selfie_segmentation)
    logger.info("[PRELOAD] ✓ MediaPipe Selfie Segmentation loaded successfully")

    # ── 5. YOLOv8 — object detection ──────────────────────────────────────────
    logger.info("[PRELOAD] (5/7) Loading YOLOv8...")
    logger.info(f"[PRELOAD]        Model path: {settings.yolo_model}")
    logger.info("[PRELOAD]        Used for: Prohibited object detection")

    def _load_yolo():
        preload_yolo_model(settings.yolo_model)

    await loop.run_in_executor(None, _load_yolo)
    logger.info("[PRELOAD] ✓ YOLOv8 loaded successfully")

    # ── 6. YAMNet — audio classification ──────────────────────────────────────
    logger.info("[PRELOAD] (6/7) Loading YAMNet (TensorFlow Hub)...")
    logger.info(f"[PRELOAD]        Model URL: {settings.yamnet_model_url}")
    logger.info("[PRELOAD]        Used for: Audio anomaly detection")
    logger.info("[PRELOAD]        This may take 10-30 seconds on first run...")

    def _load_yamnet():
        preload_yamnet_model(settings.yamnet_model_url)

    await loop.run_in_executor(None, _load_yamnet)
    logger.info("[PRELOAD] ✓ YAMNet loaded successfully")

    # ── 7. DINOv2 — background embeddings ─────────────────────────────────────
    logger.info("[PRELOAD] (7/7) Loading DINOv2 (ViT-B/14)...")
    logger.info("[PRELOAD]        Used for: Background validation (location detection)")
    logger.info("[PRELOAD]        This may take 10-30 seconds on first run (downloading ~330MB)...")

    def _load_dinov2():
        preload_dinov2_model()

    await loop.run_in_executor(None, _load_dinov2)
    logger.info("[PRELOAD] ✓ DINOv2 loaded successfully")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("[PRELOAD] All models loaded successfully!")
    logger.info("[PRELOAD] Summary:")
    logger.info("[PRELOAD]   ✓ InsightFace (ArcFace)          - Face recognition")
    logger.info("[PRELOAD]   ✓ MediaPipe FaceMesh             - Face landmarks")
    logger.info("[PRELOAD]   ✓ MediaPipe Hands                - Hand detection")
    logger.info("[PRELOAD]   ✓ MediaPipe Selfie Segmentation  - Person segmentation")
    logger.info("[PRELOAD]   ✓ YOLOv8                         - Object detection")
    logger.info("[PRELOAD]   ✓ YAMNet                         - Audio classification")
    logger.info("[PRELOAD]   ✓ DINOv2                         - Background embeddings")
    logger.info("=" * 80)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — pre-load all models and initialize database before accepting connections."""
    logger.info("Proctoring Engine API starting up...")

    # Initialize database
    await init_db()

    # Eagerly initialise the SessionManager singleton
    get_session_manager()

    # Pre-load and warm up all models
    await _preload_models()

    logger.info("Proctoring Engine API ready — all models loaded, accepting connections.")
    yield

    # Shutdown
    manager = get_session_manager()
    logger.info(
        f"Proctoring Engine API shutting down. "
        f"Active sessions: {manager.active_count}"
    )
    
    # Close database connections
    await close_db()
    
    # Clean up shared MediaPipe models
    logger.info("Cleaning up shared models...")
    try:
        from proctoring_engine.core.model_loader import cleanup_models  # noqa: PLC0415
        cleanup_models()
        logger.info("Models cleaned up successfully")
    except Exception as e:
        logger.warning(f"Error cleaning up models: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Video Proctoring Engine",
        description=(
            "Real-time exam video proctoring API. "
            "Connect via WebSocket at /ws/proctor/{session_id} "
            "and stream 30 FPS JPEG frames with optional PCM audio."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS — allow all origins (no auth required per spec) ──────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── ngrok browser-warning bypass ──────────────────────────────────────────
    # ngrok intercepts HTTP requests and shows an interstitial warning page
    # unless this header is present in the RESPONSE.  Adding it here means
    # every response from our server carries it, so ngrok forwards all
    # requests (including the health check) without the interstitial.
    @app.middleware("http")
    async def add_ngrok_header(request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response

    # ── Routes ─────────────────────────────────────────────────────────────────
    # Health checks
    app.include_router(default_router)
    
    # Proctoring WebSocket
    app.include_router(ws_router)
    
    # Registration/Authentication API
    app.include_router(auth_router)
    
    # Face Verification API
    app.include_router(verification_router)

    return app


# Module-level app instance — used by uvicorn and `python -m proctoring_engine.api`
app = create_app()
