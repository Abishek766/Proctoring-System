"""Image quality validation."""

import cv2
import numpy as np

from proctoring_engine.settings import get_settings
from proctoring_engine.utils.exceptions import ValidationException
from proctoring_engine.utils.exceptions.error_codes import BLURRY_IMAGE, IMAGE_TOO_SMALL, LOW_LIGHT_CONDITION
from proctoring_engine.utils.logger import get_logger

logger = get_logger("image_validation")

# Get settings instance
_settings = get_settings()


def validate_brightness(image: np.ndarray) -> dict:
    """Validate image brightness.

    Args:
        image: Input image (BGR format)

    Returns:
        dict with brightness score and status

    Raises:
        ValidationException: If brightness is too low
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate mean brightness
    brightness = np.mean(gray)

    logger.info(f"Image brightness: {brightness:.2f}")

    if brightness < _settings.min_brightness:
        logger.warning(f"Low light detected: {brightness:.2f} < {_settings.min_brightness}")
        raise ValidationException(
            message="Image is too dark. Please move to a brighter area.",
            error_code=LOW_LIGHT_CONDITION,
            details={"brightness_score": float(brightness), "min_required": _settings.min_brightness},
        )

    if brightness > _settings.max_brightness:
        logger.warning(f"Too bright: {brightness:.2f} > {_settings.max_brightness}")
        raise ValidationException(
            message="Image is too bright. Please reduce lighting or move away from direct light.",
            error_code=LOW_LIGHT_CONDITION,
            details={"brightness_score": float(brightness), "max_allowed": _settings.max_brightness},
        )

    is_optimal = _settings.optimal_brightness_min <= brightness <= _settings.optimal_brightness_max

    return {
        "brightness": float(brightness),
        "is_optimal": is_optimal,
        "status": "optimal" if is_optimal else "acceptable",
    }


def validate_sharpness(image: np.ndarray) -> dict:
    """Validate image sharpness using Laplacian variance.

    Args:
        image: Input image (BGR format)

    Returns:
        dict with sharpness score and status

    Raises:
        ValidationException: If image is too blurry
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    logger.info(f"Image sharpness (Laplacian variance): {laplacian_var:.2f}")

    if laplacian_var < _settings.min_sharpness:
        logger.warning(f"Blurry image detected: {laplacian_var:.2f} < {_settings.min_sharpness}")
        raise ValidationException(
            message="Image is too blurry. Please hold the camera steady and ensure good focus.",
            error_code=BLURRY_IMAGE,
            details={"sharpness_score": float(laplacian_var), "min_required": _settings.min_sharpness},
        )

    return {"sharpness": float(laplacian_var), "status": "sharp"}


def validate_resolution(image: np.ndarray) -> dict:
    """Validate image resolution.

    Args:
        image: Input image

    Returns:
        dict with resolution info

    Raises:
        ValidationException: If resolution is too low
    """
    height, width = image.shape[:2]

    logger.info(f"Image resolution: {width}x{height}")

    if width < _settings.min_resolution_width or height < _settings.min_resolution_height:
        logger.warning(f"Low resolution: {width}x{height}")
        raise ValidationException(
            message=f"Image resolution too low. Minimum required: {_settings.min_resolution_width}x{_settings.min_resolution_height}",
            error_code=IMAGE_TOO_SMALL,
            details={
                "current_resolution": f"{width}x{height}",
                "min_required": f"{_settings.min_resolution_width}x{_settings.min_resolution_height}",
            },
        )

    return {"width": width, "height": height, "status": "valid"}


def validate_image_quality(image: np.ndarray) -> dict:
    """Run all image quality validations.

    Args:
        image: Input image (BGR format)

    Returns:
        dict with all validation results

    Raises:
        ValidationException: If any validation fails
    """
    logger.info("Starting image quality validation")

    results = {}

    # Validate resolution
    results["resolution"] = validate_resolution(image)

    # Validate brightness
    results["brightness"] = validate_brightness(image)

    # Validate sharpness
    results["sharpness"] = validate_sharpness(image)

    logger.info("Image quality validation passed")

    return results
