"""Face processing service for orchestrating face detection and recognition."""

import cv2
import numpy as np

from proctoring_engine.core.face_detection import (
    create_convex_hull_mask,
    detect_face_with_mediapipe,
    get_face_landmarks,
)
from proctoring_engine.core.face_recognition import generate_face_embedding
from proctoring_engine.core.image_processing import bytes_to_image, crop_face_region
from proctoring_engine.core.image_validation import validate_brightness, validate_image_quality
from proctoring_engine.settings import get_settings
from proctoring_engine.utils.exceptions import ValidationException
from proctoring_engine.utils.exceptions.error_codes import (
    EMBEDDING_GENERATION_FAILED,
    FACE_NOT_DETECTED,
    INTERNAL_SERVER_ERROR,
    MULTIPLE_FACES_DETECTED,
)
from proctoring_engine.utils.logger import get_logger

logger = get_logger("face_service")


class FaceService:
    """Service for face processing operations."""

    def __init__(self) -> None:
        """Initialize face service with settings."""
        self._settings = get_settings()

    async def process_face_for_registration(self, image_bytes: bytes) -> dict:
        """Process face image for registration.

        Complete flow:
        1. Convert bytes to image
        2. Validate image quality (brightness, sharpness, resolution)
        3. Detect face using MediaPipe Face Detection
        4. Get facial landmarks using MediaPipe Face Mesh
        5. Create convex hull mask for face segmentation
        6. Apply mask to extract only face region
        7. Crop face region with padding
        8. Generate face embeddings using InsightFace

        Args:
            image_bytes: Raw image bytes from frontend

        Returns:
            dict with processed_photo (bytes) and face_embedding (list)

        Raises:
            ValidationException: If any validation step fails
        """
        logger.info("Starting face processing for registration")

        try:
            # Step 1: Convert bytes to image
            logger.info("Converting bytes to image")
            image = bytes_to_image(image_bytes)
            logger.info(f"Image loaded: shape={image.shape}, dtype={image.dtype}")

            # Step 2: Validate image quality
            logger.info("Validating image quality")
            validate_image_quality(image)
            logger.info("Image quality validation passed")

            # Step 3: Validate brightness
            logger.info("Validating brightness")
            brightness_info = validate_brightness(image)
            logger.info(f"Brightness validation passed: {brightness_info}")

            # Step 4: Detect face with MediaPipe Face Detection
            logger.info("Detecting face with MediaPipe Face Detection")
            face_detection_result = detect_face_with_mediapipe(image)
            bbox = face_detection_result["bbox"]
            logger.info(f"Face detected: bbox={bbox}, confidence={face_detection_result['confidence']:.3f}")

            # Step 5: Get facial landmarks with MediaPipe Face Mesh
            logger.info("Extracting facial landmarks with MediaPipe Face Mesh")
            landmarks = get_face_landmarks(image)
            
            if landmarks is None:
                logger.warning("Failed to extract facial landmarks")
                raise ValidationException(
                    message="Failed to extract facial features. Please ensure your face is clearly visible.",
                    error_code=FACE_NOT_DETECTED,
                    details=None,
                )
            
            logger.info(f"Facial landmarks extracted: {len(landmarks)} points")

            # Step 6: Create convex hull mask for face segmentation
            logger.info("Creating convex hull mask for face segmentation")
            face_mask = create_convex_hull_mask(image, landmarks)
            logger.info("Convex hull mask created")

            # Step 7: Apply mask to extract only face region (segmentation)
            logger.info("Applying face segmentation mask using bitwise AND operation")
            segmented_face = cv2.bitwise_and(image, image, mask=face_mask)
            
            # Calculate segmentation statistics
            non_zero_pixels = np.count_nonzero(segmented_face)
            total_pixels = segmented_face.shape[0] * segmented_face.shape[1] * segmented_face.shape[2]
            segmentation_ratio = non_zero_pixels / total_pixels
            logger.info(f"Face segmentation completed: {non_zero_pixels}/{total_pixels} pixels retained ({segmentation_ratio:.2%})")
            logger.info(f"Segmented face shape: {segmented_face.shape}")

            # Step 8: Crop face region with padding
            logger.info("Cropping segmented face region")
            cropped_face = crop_face_region(segmented_face, bbox)
            logger.info(f"Face cropped: shape={cropped_face.shape}")

            # Step 9: Generate face embeddings with InsightFace
            logger.info("Generating face embeddings with InsightFace")
            face_embedding = generate_face_embedding(cropped_face)

            if face_embedding is None or len(face_embedding) != self._settings.embedding_size:
                logger.error(
                    f"Invalid embedding generated: size={len(face_embedding) if face_embedding is not None else 0}"
                )
                raise ValidationException(
                    message="Failed to generate face embeddings. Please try again with a clearer photo.",
                    error_code=EMBEDDING_GENERATION_FAILED,
                    details={
                        "expected_size": self._settings.embedding_size,
                        "actual_size": len(face_embedding) if face_embedding is not None else 0,
                    },
                )

            logger.info(f"Face embedding generated successfully: size={len(face_embedding)}")

            # Step 10: Convert cropped segmented face to bytes for storage
            logger.info("Converting segmented face to bytes")
            _, buffer = cv2.imencode(".jpg", cropped_face)
            processed_photo_bytes = buffer.tobytes()
            logger.info(f"Processed photo size: {len(processed_photo_bytes)} bytes")

            logger.info("Face processing completed successfully")
            return {
                "processed_photo": processed_photo_bytes,
                "face_embedding": face_embedding.tolist(),  # Convert numpy array to list
            }

        except ValidationException:
            # Re-raise validation exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error during face processing: {str(e)}", exc_info=True)
            raise ValidationException(
                message=f"Face processing failed: {str(e)}",
                error_code=INTERNAL_SERVER_ERROR,
                details={"error": str(e)},
            ) from e

    async def process_face_for_verification(
        self, image_bytes: bytes, validate_quality: bool = True
    ) -> dict:
        """Process face image for verification.

        Two flows based on validate_quality parameter:
        1. validate_quality=True (verify flow):
           - Full image quality validation
           - Strict face detection (exactly 1 face)
        
        2. validate_quality=False (violation_verify flow):
           - Skip image quality validation
           - Lenient face detection (takes first face if multiple)

        Args:
            image_bytes: Raw image bytes from frontend
            validate_quality: Whether to validate image quality

        Returns:
            dict with face_embedding (list)

        Raises:
            ValidationException: If face processing fails
        """
        logger.info(
            f"Starting face processing for verification: validate_quality={validate_quality}"
        )

        try:
            # Step 1: Convert bytes to image
            logger.info("Converting bytes to image")
            image = bytes_to_image(image_bytes)
            logger.info(f"Image loaded: shape={image.shape}, dtype={image.dtype}")

            # Step 2: Validate image quality (only for "verify" flow)
            if validate_quality:
                logger.info("Validating image quality")
                validate_image_quality(image)
                logger.info("Image quality validation passed")

                logger.info("Validating brightness")
                brightness_info = validate_brightness(image)
                logger.info(f"Brightness validation passed: {brightness_info}")

            # Step 3: Detect face with MediaPipe Face Detection
            logger.info("Detecting face with MediaPipe Face Detection")
            try:
                face_detection_result = detect_face_with_mediapipe(image)
                bbox = face_detection_result["bbox"]
                logger.info(
                    f"Face detected: bbox={bbox}, confidence={face_detection_result['confidence']:.3f}"
                )
            except ValidationException as e:
                # For violation_verify flow, if multiple faces detected, log and continue
                if not validate_quality and e.error_code == MULTIPLE_FACES_DETECTED:
                    logger.warning(
                        f"Multiple faces detected in violation_verify flow: {e.details}"
                    )
                    # Re-raise to let caller handle
                raise

            # Step 4: Get facial landmarks with MediaPipe Face Mesh
            logger.info("Extracting facial landmarks with MediaPipe Face Mesh")
            landmarks = get_face_landmarks(image)
            
            if landmarks is None:
                logger.warning("Failed to extract facial landmarks")
                raise ValidationException(
                    message="Failed to extract facial features. Please ensure your face is clearly visible.",
                    error_code=FACE_NOT_DETECTED,
                    details=None,
                )
            
            logger.info(f"Facial landmarks extracted: {len(landmarks)} points")

            # Step 5: Create convex hull mask for face segmentation
            logger.info("Creating convex hull mask for face segmentation")
            face_mask = create_convex_hull_mask(image, landmarks)
            logger.info("Convex hull mask created")

            # Step 6: Apply mask to extract only face region (segmentation)
            logger.info("Applying face segmentation mask")
            segmented_face = cv2.bitwise_and(image, image, mask=face_mask)
            
            non_zero_pixels = np.count_nonzero(segmented_face)
            total_pixels = segmented_face.shape[0] * segmented_face.shape[1] * segmented_face.shape[2]
            segmentation_ratio = non_zero_pixels / total_pixels
            logger.info(
                f"Face segmentation completed: {non_zero_pixels}/{total_pixels} pixels retained "
                f"({segmentation_ratio:.2%})"
            )

            # Step 7: Crop face region with padding
            logger.info("Cropping segmented face region")
            cropped_face = crop_face_region(segmented_face, bbox)
            logger.info(f"Face cropped: shape={cropped_face.shape}")

            # Step 8: Generate face embeddings with InsightFace
            logger.info("Generating face embeddings with InsightFace")
            face_embedding = generate_face_embedding(cropped_face)

            if face_embedding is None or len(face_embedding) != self._settings.embedding_size:
                logger.error(
                    f"Invalid embedding generated: size={len(face_embedding) if face_embedding is not None else 0}"
                )
                raise ValidationException(
                    message="Failed to generate face embeddings. Please try again with a clearer photo.",
                    error_code=EMBEDDING_GENERATION_FAILED,
                    details={
                        "expected_size": self._settings.embedding_size,
                        "actual_size": len(face_embedding) if face_embedding is not None else 0,
                    },
                )

            logger.info(f"Face embedding generated successfully: size={len(face_embedding)}")
            logger.info("Face processing for verification completed successfully")

            return {
                "face_embedding": face_embedding.tolist(),  # Convert numpy array to list
            }

        except ValidationException:
            # Re-raise validation exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error during face processing: {str(e)}", exc_info=True)
            raise ValidationException(
                message=f"Face processing failed: {str(e)}",
                error_code=INTERNAL_SERVER_ERROR,
                details={"error": str(e)},
            ) from e
