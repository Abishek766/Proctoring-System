"""Centralised error code constants."""

# ─── Proctoring engine errors ─────────────────────────────────────────────────
ERR_MODEL_INIT_FAILED = "MODEL_INIT_FAILED"
ERR_CAMERA_INIT_FAILED = "CAMERA_INIT_FAILED"
ERR_AUDIO_THREAD_FAILED = "AUDIO_THREAD_FAILED"
ERR_FRAME_PROCESSING_FAILED = "FRAME_PROCESSING_FAILED"
ERR_AUDIO_PROCESSING_FAILED = "AUDIO_PROCESSING_FAILED"

# ─── Authentication errors ────────────────────────────────────────────────────
INVALID_CREDENTIALS = "INVALID_CREDENTIALS"
USER_NOT_FOUND = "USER_NOT_FOUND"
USER_ALREADY_EXISTS = "USER_ALREADY_EXISTS"

# ─── Database errors ──────────────────────────────────────────────────────────
DATABASE_ERROR = "DATABASE_ERROR"

# ─── Validation errors ────────────────────────────────────────────────────────
FACE_NOT_DETECTED = "FACE_NOT_DETECTED"
FACE_TOO_SMALL = "FACE_TOO_SMALL"
MULTIPLE_FACES_DETECTED = "MULTIPLE_FACES_DETECTED"
EMBEDDING_GENERATION_FAILED = "EMBEDDING_GENERATION_FAILED"
INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
IMAGE_TOO_DARK = "IMAGE_TOO_DARK"
IMAGE_TOO_BRIGHT = "IMAGE_TOO_BRIGHT"

# ─── Image quality validation errors ──────────────────────────────────────────
LOW_LIGHT_CONDITION = "LOW_LIGHT_CONDITION"
BLURRY_IMAGE = "BLURRY_IMAGE"
IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"

# ─── General errors ───────────────────────────────────────────────────────────
INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
