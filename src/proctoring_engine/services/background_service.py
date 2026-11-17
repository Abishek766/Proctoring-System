"""Background validation service - handles background registration and validation with database.

This service coordinates background validation operations:
- Registers background embeddings during initial verification
- Validates background during periodic checks
- Stores/retrieves background embeddings from database
"""

from __future__ import annotations

import numpy as np

from proctoring_engine.core.background_pipeline import (
    BackgroundPipeline,
    BackgroundValidationResult,
)
from proctoring_engine.repositories.user_repository import UserRepository
from proctoring_engine.settings import Settings
from proctoring_engine.utils.logger import get_logger

logger = get_logger("background_validation")


class BackgroundService:
    """Service for background validation operations with database integration.
    
    Mirrors FaceVerificationService pattern for consistency.
    """
    
    def __init__(
        self,
        user_repository: UserRepository | None = None,
        settings: Settings | None = None
    ) -> None:
        """Initialize the background service.
        
        Args:
            user_repository: Repository for user database operations (optional, lazy-loaded)
            settings: Application settings (optional, will use default if not provided)
        """
        self.user_repository = user_repository
        
        if settings is None:
            from proctoring_engine.settings import get_settings  # noqa: PLC0415
            settings = get_settings()
        
        self._settings = settings
        self._pipeline = BackgroundPipeline(settings)
        
        logger.info("[INIT] BackgroundService initialized")
    
    async def register_background_from_frame(
        self,
        bgr_frame: np.ndarray,
        user_id: str,
        frame_id: int,
        timestamp_ms: int,
    ) -> BackgroundValidationResult:
        """Register background from a frame and store in database.
        
        This should be called during initial verification to capture the
        candidate's exam environment.
        
        Mirrors FaceVerificationService.verify_face() pattern.
        
        Args:
            bgr_frame: BGR frame from camera
            user_id: User ID to associate background with
            frame_id: Frame ID
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            BackgroundValidationResult with registration status
        """
        logger.info(
            f"[REGISTER_START] user_id={user_id} frame={frame_id} ts={timestamp_ms}"
        )
        
        try:
            # Step 1: Validate user exists (same as face verification)
            user = await self.user_repository.get_user_by_id(user_id)
            if not user:
                logger.warning(f"[USER_NOT_FOUND] user_id={user_id}")
                return BackgroundValidationResult(
                    status="error",
                    similarity_score=0.0,
                    threshold=self._settings.bg_similarity_threshold,
                    consecutive_failures=0,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    latency_ms=0.0,
                    message=f"User with ID '{user_id}' not found"
                )
            
            logger.info(
                f"[USER_FOUND] user_id={user_id} username={user.username}"
            )
            
            # Step 2: Register background using pipeline
            result = self._pipeline.register_background(
                bgr_frame,
                frame_id,
                timestamp_ms
            )
            
            if result.status == "registered":
                # Step 3: Get the registered embedding
                embedding = self._pipeline.get_registered_embedding()
                
                if embedding is not None:
                    # Step 4: Store in database (creates new record, doesn't overwrite)
                    bg_id = await self.user_repository.update_background_embedding(
                        user_id,
                        embedding.tolist()
                    )
                    
                    # Step 5: Store background_id in pipeline for validation
                    self._pipeline.set_background_id(bg_id)
                    
                    logger.info(
                        f"[REGISTER_SUCCESS] user_id={user_id} username={user.username} "
                        f"frame={frame_id} bg_id={bg_id} embedding_stored=True embedding_dim={len(embedding)}"
                    )
                else:
                    logger.warning(
                        f"[REGISTER_WARNING] user_id={user_id} frame={frame_id} "
                        f"embedding_is_none"
                    )
            else:
                logger.error(
                    f"[REGISTER_FAILED] user_id={user_id} frame={frame_id} "
                    f"status={result.status} message={result.message}"
                )
            
            return result
            
        except Exception as exc:
            logger.error(
                f"[REGISTER_ERROR] user_id={user_id} frame={frame_id} "
                f"error={str(exc)}",
                exc_info=True
            )
            
            return BackgroundValidationResult(
                status="error",
                similarity_score=0.0,
                threshold=self._settings.bg_similarity_threshold,
                consecutive_failures=0,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                latency_ms=0.0,
                message=f"Background registration failed: {str(exc)}"
            )
    
    async def load_registered_background(self, user_id: str) -> bool:
        """Load registered background from database.
        
        This should be called when resuming a session to restore the
        registered background.
        
        Fetches the LATEST background embedding from the background_embeddings table
        and stores the background_id for subsequent validations.
        
        Args:
            user_id: User ID to load background for
        
        Returns:
            True if background was loaded successfully, False otherwise
        """
        logger.info(f"[LOAD_START] user_id={user_id}")
        
        try:
            # Step 1: Fetch latest background embedding from database
            result = await self.user_repository.get_latest_background_embedding(user_id)
            
            if result is None:
                logger.warning(
                    f"[LOAD_FAILED] user_id={user_id} reason=no_background_embedding"
                )
                return False
            
            bg_id, embedding_list = result
            
            logger.info(
                f"[LATEST_BG_FOUND] user_id={user_id} bg_id={bg_id} "
                f"embedding_dim={len(embedding_list)}"
            )
            
            # Step 2: Convert to numpy array
            embedding = np.array(embedding_list, dtype=np.float32)
            
            # Step 3: Set in pipeline
            self._pipeline._registered_embedding = embedding
            
            # Step 4: Store background_id in pipeline for validation
            self._pipeline.set_background_id(bg_id)
            
            logger.info(
                f"[LOAD_SUCCESS] user_id={user_id} bg_id={bg_id} "
                f"embedding_dim={len(embedding)}"
            )
            
            return True
            
        except Exception as exc:
            logger.error(
                f"[LOAD_ERROR] user_id={user_id} error={str(exc)}",
                exc_info=True
            )
            return False
    
    async def validate_from_frame(
        self,
        bgr_frame: np.ndarray,
        user_id: str,
        frame_id: int,
        timestamp_ms: int,
    ) -> dict:
        """Validate background from frame - mirrors verify_from_frame pattern.
        
        This is the main validation method that should be called periodically.
        Returns a dictionary similar to face verification for consistency.
        
        Uses repository's calculate_cosine_similarity() method (same as face verification).
        
        NOTE: Timing checks are handled by the caller (WebSocket handler).
        This method performs validation immediately when called.
        
        Args:
            bgr_frame: BGR frame from camera
            user_id: User ID for logging
            frame_id: Frame ID
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            Dictionary with validation result (similar to face verification)
        """
        logger.info(
            f"[VALIDATE_FROM_FRAME] user_id={user_id} frame={frame_id} ts={timestamp_ms}"
        )
        
        try:
            # Check if background is registered
            if not self._pipeline.is_registered():
                logger.warning(
                    f"[VALIDATE_SKIPPED] user_id={user_id} frame={frame_id} "
                    f"reason=not_registered"
                )
                return {
                    "is_match": False,
                    "similarity": 0.0,
                    "confidence": "none",
                    "message": "Background not registered",
                    "user_id": user_id,
                    "status": "error",
                    "consecutive_failures": 0,
                }
            
            # Step 1: Fetch stored embedding using background_id from pipeline
            bg_id = self._pipeline.get_background_id()
            if bg_id is None:
                logger.warning(
                    f"[NO_BG_ID] user_id={user_id} frame={frame_id} "
                    f"reason=background_id_not_set"
                )
                return {
                    "is_match": False,
                    "similarity": 0.0,
                    "confidence": "none",
                    "message": "Background ID not set",
                    "user_id": user_id,
                    "status": "error",
                    "consecutive_failures": 0,
                }
            
            stored_embedding = await self.user_repository.get_background_embedding_by_id(bg_id)
            if stored_embedding is None:
                logger.warning(
                    f"[NO_STORED_EMBEDDING] user_id={user_id} bg_id={bg_id} frame={frame_id}"
                )
                return {
                    "is_match": False,
                    "similarity": 0.0,
                    "confidence": "none",
                    "message": "No stored background embedding found",
                    "user_id": user_id,
                    "status": "error",
                    "consecutive_failures": 0,
                }
            
            logger.info(
                f"[STORED_EMBEDDING_FETCHED] user_id={user_id} bg_id={bg_id} "
                f"embedding_dim={len(stored_embedding)}"
            )
            
            # Step 2: Extract current background embedding
            import time  # noqa: PLC0415
            start_time = time.perf_counter()
            
            current_embedding = self._pipeline._extract_background_embedding(bgr_frame, frame_id, "validation")
            
            logger.info(
                f"[CURRENT_EMBEDDING_EXTRACTED] user_id={user_id} frame={frame_id} "
                f"embedding_dim={len(current_embedding)}"
            )
            
            # Step 3: Calculate cosine similarity using repository (same as face verification)
            similarity = await self.user_repository.calculate_cosine_similarity(
                stored_embedding,
                current_embedding.tolist()
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            logger.info(
                f"[SIMILARITY_CALCULATED] user_id={user_id} frame={frame_id} "
                f"similarity={similarity:.4f} threshold={self._settings.bg_similarity_threshold} "
                f"latency={latency_ms:.1f}ms"
            )
            
            # Step 4: Update timing for next validation
            self._pipeline._last_validation_time = time.perf_counter()
            
            # Step 5: Determine result based on threshold (same logic as face verification)
            is_match = similarity >= self._settings.bg_similarity_threshold
            
            if is_match:
                # Background matched - reset failure counter
                self._pipeline._consecutive_failures = 0
                
                # Exit warning state if we were in it
                if self._pipeline._in_warning_state:
                    self._pipeline._in_warning_state = False
                    logger.info(
                        f"[WARNING_CLEARED] user_id={user_id} frame={frame_id} "
                        f"similarity={similarity:.4f} returning_to_normal_interval"
                    )
                
                # Determine confidence level
                if similarity >= 0.90:
                    confidence = "high"
                elif similarity >= 0.85:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                logger.info(
                    f"[MATCHED] user_id={user_id} frame={frame_id} "
                    f"similarity={similarity:.4f} confidence={confidence} "
                    f"latency={latency_ms:.1f}ms"
                )
                
                return {
                    "is_match": True,
                    "similarity": round(similarity, 4),
                    "confidence": confidence,
                    "message": "Background matched",
                    "user_id": user_id,
                    "status": "matched",
                    "consecutive_failures": 0,
                    "threshold": self._settings.bg_similarity_threshold,
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "latency_ms": round(latency_ms, 2),
                }
            
            else:
                # Background mismatch - increment failure counter
                self._pipeline._consecutive_failures += 1
                consecutive_failures = self._pipeline._consecutive_failures
                
                # Check if we should send warning
                # If already warned (count >= 3): send warning immediately on next mismatch
                # If not warned yet (count < 3): wait for 3 consecutive mismatches
                if consecutive_failures >= self._settings.bg_failure_threshold:
                    logger.warning(
                        f"[WARNING] user_id={user_id} frame={frame_id} "
                        f"similarity={similarity:.4f} threshold={self._settings.bg_similarity_threshold} "
                        f"consecutive_failures={consecutive_failures} latency={latency_ms:.1f}ms"
                    )
                    
                    return {
                        "is_match": False,
                        "similarity": round(similarity, 4),
                        "confidence": "none",
                        "message": "Background changed! Candidate may have changed location",
                        "user_id": user_id,
                        "status": "warning",
                        "consecutive_failures": consecutive_failures,
                        "threshold": self._settings.bg_similarity_threshold,
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "latency_ms": round(latency_ms, 2),
                    }
                
                else:
                    # Mismatch but not yet warning threshold
                    logger.warning(
                        f"[MISMATCH] user_id={user_id} frame={frame_id} "
                        f"similarity={similarity:.4f} threshold={self._settings.bg_similarity_threshold} "
                        f"consecutive_failures={consecutive_failures}/{self._settings.bg_failure_threshold} "
                        f"latency={latency_ms:.1f}ms"
                    )
                    
                    return {
                        "is_match": False,
                        "similarity": round(similarity, 4),
                        "confidence": "none",
                        "message": f"Background mismatch ({consecutive_failures}/{self._settings.bg_failure_threshold})",
                        "user_id": user_id,
                        "status": "mismatch",
                        "consecutive_failures": consecutive_failures,
                        "threshold": self._settings.bg_similarity_threshold,
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "latency_ms": round(latency_ms, 2),
                    }
            
        except Exception as exc:
            logger.error(
                f"[VALIDATE_ERROR] user_id={user_id} frame={frame_id} "
                f"error={str(exc)}",
                exc_info=True
            )
            return {
                "is_match": False,
                "similarity": 0.0,
                "confidence": "error",
                "message": f"Validation error: {str(exc)}",
                "user_id": user_id,
                "status": "error",
                "consecutive_failures": self._pipeline.get_consecutive_failures(),
            }
    
    def should_validate(self) -> bool:
        """Check if background validation should be performed based on timing.
        
        Returns:
            True if validation should be performed, False otherwise
        """
        return self._pipeline.should_validate()
    
    def is_registered(self) -> bool:
        """Check if background is registered.
        
        Returns:
            True if background is registered, False otherwise
        """
        return self._pipeline.is_registered()
    
    def get_consecutive_failures(self) -> int:
        """Get the number of consecutive validation failures.
        
        Returns:
            Number of consecutive failures
        """
        return self._pipeline.get_consecutive_failures()
