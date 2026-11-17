"""Image processing utilities."""

import cv2
import numpy as np

from proctoring_engine.utils.logger import get_logger

logger = get_logger("image_processing")


def bytes_to_image(image_bytes: bytes) -> np.ndarray:
    """Convert bytes to OpenCV image.

    Args:
        image_bytes: Raw image bytes

    Returns:
        OpenCV image (BGR format)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image from bytes")

    logger.info(f"Image decoded: shape={image.shape}")
    return image


def image_to_bytes(image: np.ndarray, format: str = ".jpg") -> bytes:
    """Convert OpenCV image to bytes.

    Args:
        image: OpenCV image
        format: Image format (.jpg, .png)

    Returns:
        Image as bytes
    """
    success, buffer = cv2.imencode(format, image)

    if not success:
        raise ValueError("Failed to encode image to bytes")

    return buffer.tobytes()


def crop_face_region(image: np.ndarray, bbox: dict, padding: float = 0.5) -> np.ndarray:
    """Crop face region from image with padding.

    Args:
        image: Input image
        bbox: Bounding box dict with x, y, width, height
        padding: Padding ratio (0.5 = 50% padding for better context)

    Returns:
        Cropped face image
    """
    height, width = image.shape[:2]

    x = bbox["x"]
    y = bbox["y"]
    w = bbox["width"]
    h = bbox["height"]

    # Add padding (increased to 50% for better context)
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(width, x + w + pad_w)
    y2 = min(height, y + h + pad_h)

    cropped = image[y1:y2, x1:x2]

    logger.info(f"Face cropped: original_bbox=({x},{y},{w},{h}), padded_bbox=({x1},{y1},{x2-x1},{y2-y1}), cropped_shape={cropped.shape}")

    return cropped


def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Resize image while maintaining aspect ratio.

    Args:
        image: Input image
        max_size: Maximum dimension size

    Returns:
        Resized image
    """
    height, width = image.shape[:2]

    if max(height, width) <= max_size:
        return image

    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    logger.info(f"Image resized: {width}x{height} -> {new_width}x{new_height}")

    return resized
