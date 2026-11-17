"""Face recognition using InsightFace ArcFace."""

import numpy as np
from insightface.app import FaceAnalysis

from proctoring_engine.core.model_loader import get_insightface_model
from proctoring_engine.settings import get_settings
from proctoring_engine.utils.exceptions import ValidationException
from proctoring_engine.utils.exceptions.error_codes import EMBEDDING_GENERATION_FAILED
from proctoring_engine.utils.logger import get_logger

logger = get_logger("face_recognition")
error_logger = get_logger("error_logger")

# Get settings instance
_settings = get_settings()


def get_face_analysis_model() -> FaceAnalysis:
    """Get the global InsightFace model instance.
    
    Uses the centralized model loader to ensure the model is shared
    across all requests and preloaded at startup.
    
    Returns:
        FaceAnalysis: Initialized InsightFace model
    """
    return get_insightface_model()


def generate_face_embedding(image: np.ndarray) -> np.ndarray:
    """Generate 512D face embedding using InsightFace ArcFace.

    Args:
        image: Input face image (BGR format)

    Returns:
        512D numpy array of face embeddings

    Raises:
        ValidationException: If embedding generation fails
    """
    # Log detailed image properties for debugging
    image_info = {
        "shape": image.shape,
        "dtype": str(image.dtype),
        "min_val": float(np.min(image)),
        "max_val": float(np.max(image)),
        "mean_val": float(np.mean(image)),
        "is_empty": image.size == 0,
    }
    
    logger.info(f"[STEP_1_START] Generating face embeddings with InsightFace, image_info={image_info}")

    try:
        # Step 1: Get InsightFace model
        logger.info("[STEP_2_MODEL] Loading InsightFace model instance")
        app = get_face_analysis_model()
        logger.info("[STEP_2_MODEL] InsightFace model loaded successfully")

        # Step 2: Detect faces and get embeddings
        logger.info("[STEP_3_DETECT] Calling InsightFace app.get() to detect faces")
        faces = app.get(image)
        logger.info(f"[STEP_3_DETECT] InsightFace detected {len(faces)} face(s)")

        if not faces:
            # Log detailed error to error log file
            error_msg = (
                f"[STEP_3_DETECT_FAILED] InsightFace failed to detect face in image. "
                f"Image properties: shape={image.shape}, dtype={image.dtype}, "
                f"min={np.min(image)}, max={np.max(image)}, mean={np.mean(image):.2f}, "
                f"size={image.size} pixels. "
                f"Possible reasons: (1) Face too small, (2) Poor lighting, (3) Face angle too extreme, "
                f"(4) Image quality too low, (5) Face partially occluded"
            )
            logger.error(error_msg)
            error_logger.error(
                f"[INSIGHTFACE_NO_FACE] {error_msg}",
                extra={"image_properties": image_info, "failure_stage": "face_detection"}
            )
            
            raise ValidationException(
                message="Failed to generate face embeddings. Please ensure your face is clearly visible.",
                error_code=EMBEDDING_GENERATION_FAILED,
                details=image_info,
            )

        if len(faces) > 1:
            logger.warning(f"[STEP_3_DETECT] Multiple faces detected by InsightFace: {len(faces)}, using first face")

        # Step 3: Extract embedding from first face
        logger.info("[STEP_4_EXTRACT] Extracting embedding from first detected face")
        embedding = faces[0].embedding
        logger.info(f"[STEP_4_EXTRACT] Raw embedding extracted: shape={embedding.shape}, dtype={embedding.dtype}")

        # Step 4: Normalize embedding
        logger.info("[STEP_5_NORMALIZE] Normalizing embedding vector")
        embedding_norm = np.linalg.norm(embedding)
        logger.info(f"[STEP_5_NORMALIZE] Embedding norm before normalization: {embedding_norm:.4f}")
        
        if embedding_norm == 0:
            error_msg = "[STEP_5_NORMALIZE_FAILED] Embedding norm is zero, cannot normalize"
            logger.error(error_msg)
            error_logger.error(
                f"[INSIGHTFACE_ZERO_NORM] {error_msg}",
                extra={"image_properties": image_info, "failure_stage": "normalization"}
            )
            raise ValidationException(
                message="Invalid face embedding generated (zero norm).",
                error_code=EMBEDDING_GENERATION_FAILED,
                details={"embedding_norm": 0.0},
            )
        
        embedding = embedding / embedding_norm
        logger.info(f"[STEP_5_NORMALIZE] Embedding normalized: new_norm={np.linalg.norm(embedding):.4f}")

        # Step 5: Validate embedding size
        logger.info(f"[STEP_6_VALIDATE] Validating embedding size: {embedding.shape[0]} == {_settings.embedding_size}")
        if embedding.shape[0] != _settings.embedding_size:
            error_msg = f"[STEP_6_VALIDATE_FAILED] Invalid embedding size: {embedding.shape[0]} != {_settings.embedding_size}"
            logger.error(error_msg)
            error_logger.error(
                f"[INSIGHTFACE_INVALID_SIZE] {error_msg}",
                extra={
                    "embedding_size": embedding.shape[0], 
                    "expected": _settings.embedding_size,
                    "failure_stage": "size_validation"
                }
            )
            
            raise ValidationException(
                message="Invalid face embedding generated.",
                error_code=EMBEDDING_GENERATION_FAILED,
                details={"embedding_size": embedding.shape[0], "expected": _settings.embedding_size},
            )

        # Step 6: Convert to float32
        logger.info("[STEP_7_CONVERT] Converting embedding to float32")
        embedding = embedding.astype(np.float32)
        logger.info(f"[STEP_7_CONVERT] Embedding conversion complete: dtype={embedding.dtype}")
        
        logger.info(f"[SUCCESS] Face embedding generated successfully: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
        return embedding

    except ValidationException:
        raise
    except Exception as e:
        error_msg = f"[EXCEPTION] Unexpected error during embedding generation: {str(e)}"
        logger.exception(error_msg)
        error_logger.exception(
            f"[INSIGHTFACE_EXCEPTION] {error_msg}",
            extra={
                "image_properties": image_info, 
                "exception_type": type(e).__name__,
                "failure_stage": "unknown_exception"
            }
        )
        
        raise ValidationException(
            message="Failed to generate face embeddings. Please try again.",
            error_code=EMBEDDING_GENERATION_FAILED,
            details={"error": str(e), "image_properties": image_info},
        )
