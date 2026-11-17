"""Constants for face recognition system."""

from pathlib import Path

# Model paths
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
INSIGHTFACE_MODEL = "buffalo_l"

# Brightness thresholds
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 220
OPTIMAL_BRIGHTNESS_MIN = 90
OPTIMAL_BRIGHTNESS_MAX = 180

# Face detection thresholds
DETECTION_CONFIDENCE = 0.6
MIN_FACE_SIZE = 60  # pixels (reduced from 80 for more flexibility)
MIN_FACE_RATIO = 0.04  # 4% of image (reduced from 8% for comfortable distance)
MAX_FACES_ALLOWED = 1

# Image quality thresholds
MIN_SHARPNESS = 50  # Laplacian variance for production quality
MIN_RESOLUTION_WIDTH = 640
MIN_RESOLUTION_HEIGHT = 480

# Face embedding
EMBEDDING_SIZE = 512

# Face verification
SIMILARITY_THRESHOLD = 0.70  # 70% similarity required for verification

# Error codes (imported from error_codes.py for convenience)
