"""Background validation pipeline using DINOv2 embeddings and MediaPipe Selfie Segmentation.

This module implements the background validation system that detects if a candidate
has changed their exam location during a proctoring session.

Key Features:
- Person removal using MediaPipe Selfie Segmentation
- Background preprocessing (inpainting and CLAHE normalization)
- Background embedding extraction using DINOv2 Vision Transformer
- Cosine similarity comparison with configurable threshold
- Position-invariant embeddings using global pooling
- Lighting-robust preprocessing with CLAHE

Architecture:
- Initial registration: Capture background on first frame
- Periodic validation: Check background every 60 seconds (normal) or 5 seconds (warning)
- Consecutive failure tracking: 3 failures → warning state
- Warning state: Continuous checking until background matches again

Preprocessing Pipeline (Simplified for Generic Use):
1. Person Removal: MediaPipe Selfie Segmentation + Telea Inpainting
2. Lighting Normalization: CLAHE on L channel (LAB color space)
3. Embedding Extraction: DINOv2 with global pooling (768D)

Technical Specifications:
- DINOv2-base (ViT-B/14): 86M parameters, 768D embeddings
- MediaPipe Selfie Segmentation: Model 1 (landscape)
- Similarity threshold: 0.80 (configurable)
- Inpainting: Telea algorithm with 7px radius
- CLAHE: Clip limit 2.0, tile grid 8×8
- Global Pooling: Averages 256 patch tokens for position invariance
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from torchvision import transforms

from proctoring_engine.core.model_loader import (
    get_dinov2_model,
    get_selfie_segmentation_model,
)
from proctoring_engine.settings import Settings
from proctoring_engine.utils.logger import get_logger

logger = get_logger("background_validation")

# DINOv2 image transform (standard ImageNet normalization)
_DINOV2_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@dataclass
class BackgroundValidationResult:
    """Result of a background validation check."""
    
    status: str
    """Status: 'registered', 'matched', 'mismatch', 'warning'"""
    
    similarity_score: float
    """Cosine similarity score (0.0 to 1.0)"""
    
    threshold: float
    """Similarity threshold used for comparison"""
    
    consecutive_failures: int
    """Number of consecutive validation failures"""
    
    frame_id: int
    """Frame ID where validation was performed"""
    
    timestamp_ms: int
    """Timestamp when validation was performed"""
    
    latency_ms: float
    """Processing latency in milliseconds"""
    
    message: str
    """Human-readable message"""


class BackgroundPipeline:
    """Background validation pipeline using DINOv2 and MediaPipe Selfie Segmentation.
    
    This pipeline:
    1. Removes the person from the frame using MediaPipe Selfie Segmentation
    2. Preprocesses the background (morphological ops, bilateral filter, inpainting, CLAHE)
    3. Extracts 768D embedding using DINOv2 Vision Transformer
    4. Compares with registered background using cosine similarity
    5. Tracks consecutive failures and triggers warning state
    """
    
    def __init__(self, settings: Settings) -> None:
        """Initialize the background validation pipeline.
        
        Args:
            settings: Application settings with background validation configuration
        """
        self._settings = settings
        
        # Load models from centralized loader
        self._segmentation = get_selfie_segmentation_model()
        self._dinov2 = get_dinov2_model()
        
        # Registered background embedding (set during initial verification)
        self._registered_embedding: np.ndarray | None = None
        
        # Background ID from database (to fetch the same background during validation)
        self._background_id: str | None = None
        
        # Consecutive failure tracking
        self._consecutive_failures = 0
        self._in_warning_state = False
        
        # Timing for periodic validation
        self._last_validation_time = 0.0
        
        logger.info(
            f"BackgroundPipeline initialized: "
            f"threshold={settings.bg_similarity_threshold}, "
            f"failure_threshold={settings.bg_failure_threshold}, "
            f"check_interval={settings.bg_check_interval_seconds}s"
        )
    
    def register_background(
        self,
        bgr_frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
    ) -> BackgroundValidationResult:
        """Register the initial background from the first frame.
        
        This should be called during initial verification to capture the
        candidate's exam environment.
        
        Args:
            bgr_frame: BGR frame from camera
            frame_id: Frame ID
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            BackgroundValidationResult with registration status
        """
        start_time = time.perf_counter()
        
        try:
            # Extract background embedding
            embedding = self._extract_background_embedding(bgr_frame, frame_id, "registration")
            
            # Store as registered background
            self._registered_embedding = embedding
            
            # Reset state
            self._consecutive_failures = 0
            self._in_warning_state = False
            self._last_validation_time = time.perf_counter()
            
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            result = BackgroundValidationResult(
                status="registered",
                similarity_score=1.0,  # Perfect match with itself
                threshold=self._settings.bg_similarity_threshold,
                consecutive_failures=0,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                latency_ms=round(latency_ms, 2),
                message="Background registered successfully"
            )
            
            logger.info(
                f"[REGISTERED] frame={frame_id} ts={timestamp_ms} "
                f"latency={latency_ms:.1f}ms embedding_shape={embedding.shape}"
            )
            
            return result
            
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"[REGISTRATION_ERROR] frame={frame_id} error={str(exc)}",
                exc_info=True
            )
            
            return BackgroundValidationResult(
                status="error",
                similarity_score=0.0,
                threshold=self._settings.bg_similarity_threshold,
                consecutive_failures=0,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                latency_ms=round(latency_ms, 2),
                message=f"Background registration failed: {str(exc)}"
            )
    
    def should_validate(self) -> bool:
        """Check if background validation should be performed based on timing.
        
        Always uses the standard check interval (60 seconds), regardless of warning state.
        
        Returns:
            True if validation should be performed, False otherwise
        """
        if self._registered_embedding is None:
            return False  # Not registered yet
        
        current_time = time.perf_counter()
        elapsed = current_time - self._last_validation_time
        
        # Always use standard check interval (60 seconds)
        interval = self._settings.bg_check_interval_seconds
        
        return elapsed >= interval
    
    def validate_background(
        self,
        bgr_frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
    ) -> BackgroundValidationResult | None:
        """Validate the current background against the registered background.
        
        This should be called periodically (every 60s normal, 5s warning) to check
        if the candidate has changed their location.
        
        Args:
            bgr_frame: BGR frame from camera
            frame_id: Frame ID
            timestamp_ms: Timestamp in milliseconds
        
        Returns:
            BackgroundValidationResult if validation was performed, None if skipped
        """
        # Check if validation should be performed
        if not self.should_validate():
            return None
        
        # Check if background is registered
        if self._registered_embedding is None:
            logger.warning(
                f"[VALIDATION_SKIPPED] frame={frame_id} reason=not_registered"
            )
            return None
        
        start_time = time.perf_counter()
        
        try:
            # Extract current background embedding
            current_embedding = self._extract_background_embedding(bgr_frame, frame_id, "validation")
            
            # Compute cosine similarity
            similarity = self._compute_cosine_similarity(
                self._registered_embedding,
                current_embedding
            )
            
            # Update last validation time
            self._last_validation_time = time.perf_counter()
            
            latency_ms = (self._last_validation_time - start_time) * 1000.0
            
            # Check if background matches
            if similarity >= self._settings.bg_similarity_threshold:
                # Background matched - reset failure counter
                self._consecutive_failures = 0
                
                # Exit warning state if we were in it
                if self._in_warning_state:
                    self._in_warning_state = False
                    logger.info(
                        f"[WARNING_CLEARED] frame={frame_id} "
                        f"similarity={similarity:.4f} "
                        f"returning_to_normal_interval"
                    )
                
                result = BackgroundValidationResult(
                    status="matched",
                    similarity_score=round(similarity, 4),
                    threshold=self._settings.bg_similarity_threshold,
                    consecutive_failures=self._consecutive_failures,
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    latency_ms=round(latency_ms, 2),
                    message="Background matched"
                )
                
                logger.info(
                    f"[MATCHED] frame={frame_id} ts={timestamp_ms} "
                    f"similarity={similarity:.4f} threshold={self._settings.bg_similarity_threshold} "
                    f"latency={latency_ms:.1f}ms"
                )
                
                return result
            
            else:
                # Background mismatch - increment failure counter
                self._consecutive_failures += 1
                
                # Check if we should enter warning state
                if self._consecutive_failures >= self._settings.bg_failure_threshold:
                    if not self._in_warning_state:
                        self._in_warning_state = True
                        logger.warning(
                            f"[WARNING_ACTIVATED] frame={frame_id} "
                            f"consecutive_failures={self._consecutive_failures} "
                            f"threshold_reached"
                        )
                    
                    result = BackgroundValidationResult(
                        status="warning",
                        similarity_score=round(similarity, 4),
                        threshold=self._settings.bg_similarity_threshold,
                        consecutive_failures=self._consecutive_failures,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        latency_ms=round(latency_ms, 2),
                        message="Background changed! Candidate may have changed location"
                    )
                    
                    logger.warning(
                        f"[WARNING] frame={frame_id} ts={timestamp_ms} "
                        f"similarity={similarity:.4f} threshold={self._settings.bg_similarity_threshold} "
                        f"consecutive_failures={self._consecutive_failures} "
                        f"latency={latency_ms:.1f}ms"
                    )
                    
                    return result
                
                else:
                    # Mismatch but not yet warning state
                    result = BackgroundValidationResult(
                        status="mismatch",
                        similarity_score=round(similarity, 4),
                        threshold=self._settings.bg_similarity_threshold,
                        consecutive_failures=self._consecutive_failures,
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        latency_ms=round(latency_ms, 2),
                        message=f"Background mismatch ({self._consecutive_failures}/{self._settings.bg_failure_threshold})"
                    )
                    
                    logger.warning(
                        f"[MISMATCH] frame={frame_id} ts={timestamp_ms} "
                        f"similarity={similarity:.4f} threshold={self._settings.bg_similarity_threshold} "
                        f"consecutive_failures={self._consecutive_failures}/{self._settings.bg_failure_threshold} "
                        f"latency={latency_ms:.1f}ms"
                    )
                    
                    return result
        
        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"[VALIDATION_ERROR] frame={frame_id} error={str(exc)}",
                exc_info=True
            )
            
            return BackgroundValidationResult(
                status="error",
                similarity_score=0.0,
                threshold=self._settings.bg_similarity_threshold,
                consecutive_failures=self._consecutive_failures,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                latency_ms=round(latency_ms, 2),
                message=f"Background validation failed: {str(exc)}"
            )
    
    def _extract_background_embedding(self, bgr_frame: np.ndarray, frame_id: int = 0, stage: str = "unknown") -> np.ndarray:
        """Extract 768D background embedding from regions OUTSIDE the person.
        
        Pipeline:
        1. Use Selfie Segmentation to identify person region
        2. Apply morphological refinement to clean mask
        3. Apply Telea inpainting to fill person region
        4. Apply CLAHE normalization for lighting robustness
        5. Extract DINOv2 embedding from full frame
        
        Args:
            bgr_frame: BGR frame from camera
            frame_id: Frame ID for debug logging
            stage: Stage name (registration/validation) for debug logging
        
        Returns:
            768D normalized embedding vector
        """
        # Step 1: Get background-only frame (person removed via inpainting)
        background_frame = self._extract_background_regions(bgr_frame)
        
        # DEBUG: Save background frame for inspection
        import uuid
        from pathlib import Path
        debug_id = str(uuid.uuid4())[:8]
        src_dir = Path(__file__).resolve().parent.parent.parent
        debug_path = src_dir / f"debug_background_{stage}_frame{frame_id}_{debug_id}.jpg"
        cv2.imwrite(str(debug_path), background_frame)
        logger.info(
            f"[DEBUG_BG_SAVED] stage={stage} frame={frame_id} "
            f"path={debug_path.name} shape={background_frame.shape}"
        )
        
        # Step 2: Preprocess background frame
        preprocessed = self._preprocess_background(background_frame)
        
        # Step 3: Extract DINOv2 embedding
        embedding = self._extract_dinov2_embedding(preprocessed)
        
        return embedding
    
    def _extract_background_regions(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Extract background by removing person using Selfie Segmentation and inpainting.
        
        Uses MediaPipe Selfie Segmentation to identify person region,
        then fills the person region using Telea inpainting algorithm.
        This allows DINOv2 to analyze the full frame background.
        
        Args:
            bgr_frame: BGR frame from camera
        
        Returns:
            Full frame with person region inpainted
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # Run segmentation to get person mask
        results = self._segmentation.process(rgb_frame)
        
        # Get segmentation mask (0.0 = background, 1.0 = person)
        mask = results.segmentation_mask
        
        # Create binary mask (255 = person, 0 = background)
        person_mask = (mask > 0.5).astype(np.uint8) * 255
        
        # Apply morphological refinement to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Closing: fill small holes in person region
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Opening: remove small noise outside person
        person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Dilate slightly to ensure person edges are covered
        person_mask = cv2.dilate(person_mask, kernel, iterations=1)
        
        # Blur mask for smoother inpainting
        person_mask = cv2.GaussianBlur(person_mask, (5, 5), 0)
        
        # Threshold again after blur
        _, person_mask = cv2.threshold(person_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Inpaint person region using Telea algorithm
        inpaint_radius = self._settings.bg_inpaint_radius
        background_frame = cv2.inpaint(bgr_frame, person_mask, inpaint_radius, cv2.INPAINT_TELEA)
        
        return background_frame
    
    def _preprocess_background(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Preprocess background for robust embedding extraction.
        
        Applies CLAHE normalization for lighting robustness (ESSENTIAL).
        
        Args:
            bgr_frame: Background regions image
        
        Returns:
            Preprocessed BGR frame
        """
        # Apply CLAHE normalization (ESSENTIAL - always enabled)
        if self._settings.bg_clahe_normalization:
            # Convert to LAB color space
            lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            lab = cv2.merge([l, a, b])
            bgr_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return bgr_frame
    
    def _extract_dinov2_embedding(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Extract 768D embedding using DINOv2 Vision Transformer with ONNX Runtime.
        
        Uses ONNX Runtime for optimized CPU inference (3-5x faster than PyTorch).
        Applies standard DINOv2 transform and global pooling for position-invariant embeddings.
        
        Args:
            bgr_frame: Preprocessed BGR frame
        
        Returns:
            768D L2-normalized embedding vector
        """
        # Convert BGR to RGB (required for DINOv2 transform)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        # Apply DINOv2 transform (resize, crop, normalize)
        # Output shape: [3, 224, 224]
        input_tensor = _DINOV2_TRANSFORM(rgb_frame)
        # Shape: [3, 224, 224] - [channels, height, width]
        
        # Add batch dimension and convert to numpy
        input_array = input_tensor.unsqueeze(0).numpy()
        # Shape: [1, 3, 224, 224] - [batch_size, channels, height, width]
        # Type: numpy.float32
        
        # Run ONNX inference
        # Get input name from ONNX model
        input_name = self._dinov2.get_inputs()[0].name
        
        # Run inference
        outputs = self._dinov2.run(None, {input_name: input_array})
        
        # Extract embedding based on pooling method
        if self._settings.bg_global_pooling:
            # Method 1: Global Pooling (Position-Invariant) ✅ Recommended
            # ONNX model output contains all tokens (CLS + patches)
            features = outputs[0]
            # Shape: [1, 257, 768]
            # 257 tokens = 1 CLS token + 256 patch tokens (16×16 patches)
            
            # Remove CLS token (index 0) and keep only patch tokens
            patch_tokens = features[:, 1:, :]
            # Shape: [1, 256, 768]
            
            # Average all patch tokens for position invariance
            embedding_np = np.mean(patch_tokens, axis=1).squeeze()
            # Shape: (768,)
        else:
            # Method 2: CLS Token (Position-Dependent)
            # Use only the CLS (classification) token (first token)
            features = outputs[0]
            # Shape: [1, 257, 768]
            
            embedding_np = features[:, 0, :].squeeze()
            # Shape: (768,)
        
        # Type: numpy.float32
        
        # L2 normalize to unit length
        embedding_np = embedding_np / (np.linalg.norm(embedding_np) + 1e-8)
        # Shape: (768,)
        # Values: Range approximately [-1.0, 1.0]
        
        return embedding_np
    
    @staticmethod
    def _compute_cosine_similarity(
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(embedding1, embedding2)
        
        # Clip to [0, 1] range (should already be in this range for normalized vectors)
        similarity = np.clip(similarity, 0.0, 1.0)
        
        return float(similarity)
    
    def get_registered_embedding(self) -> np.ndarray | None:
        """Get the registered background embedding.
        
        Returns:
            768D embedding vector or None if not registered
        """
        return self._registered_embedding
    
    def set_background_id(self, background_id: str) -> None:
        """Set the background ID from database.
        
        Args:
            background_id: Background embedding ID from database
        """
        self._background_id = background_id
    
    def get_background_id(self) -> str | None:
        """Get the background ID from database.
        
        Returns:
            Background embedding ID or None if not set
        """
        return self._background_id
    
    def is_registered(self) -> bool:
        """Check if background is registered.
        
        Returns:
            True if background is registered, False otherwise
        """
        return self._registered_embedding is not None
    
    def is_in_warning_state(self) -> bool:
        """Check if pipeline is in warning state.
        
        Returns:
            True if in warning state (3+ consecutive failures), False otherwise
        """
        return self._in_warning_state
    
    def get_consecutive_failures(self) -> int:
        """Get the number of consecutive validation failures.
        
        Returns:
            Number of consecutive failures
        """
        return self._consecutive_failures
