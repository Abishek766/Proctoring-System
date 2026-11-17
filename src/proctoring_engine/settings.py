"""Centralised application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Resolve src/ directory (parent of proctoring_engine/) ─────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent

# ─── Log file names ────────────────────────────────────────────────────────────
VIDEO_VIOLATIONS_LOG  = "engine_logs/video_violations.log"
AUDIO_VIOLATIONS_LOG  = "engine_logs/audio_violations.log"
YOLO_DETECTIONS_LOG   = "engine_logs/yolo_detections.log"
MEDIAPIPE_LOG         = "engine_logs/mediapipe_detections.log"
YAMNET_LOG            = "engine_logs/yamnet_detections.log"
BACKGROUND_VIOLATIONS_LOG = "engine_logs/background_violations.log"

# ─── YAMNet model URL ──────────────────────────────────────────────────────────
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# ─── Audio segment length (seconds) ───────────────────────────────────────────
AUDIO_BUFFER_SECONDS = 0.975

# ─── YAMNet speech-related class labels ───────────────────────────────────────
# These trigger the lip-sync check:
#   if candidate's lips are moving → no violation (candidate speaking)
#   if lips are still              → external voice violation
SPEECH_CLASSES: frozenset[str] = frozenset({
    # speech / conversation
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Whispering",
    # narration / monologue
    "Narration, monologue",
    "Male singing",
    "Female singing",
})

# ─── YAMNet device-sound / background-noise class labels ──────────────────────
# These always trigger a device_sound violation regardless of lip movement.
DEVICE_SOUND_CLASSES: frozenset[str] = frozenset({
    # music
    "Music",
    "Musical instrument",
    "Electronic music",
    "Pop music",
    "Hip hop music",
    "Rock music",
    "Rhythm and blues music",
    "Soul music",
    "Reggae",
    "Country",
    "Funk",
    "Folk music",
    "Middle Eastern music",
    "Classical music",
    "Opera",
    "Electronic dance music",
    "House music",
    "Techno",
    "Disco",
    "Ambient music",
    "Theme music",
    "Jingle, tinkle",
    "Background music",
    # ringtone / alarm / notification
    "Ringtone",
    "Telephone",
    "Alarm",
    "Alarm clock",
    "Siren",
    "Civil defense siren",
    "Buzzer",
    "Smoke detector, smoke alarm",
    "Fire alarm",
    "Notification",
    # background noise (indicates phone/speaker playing audio)
    "Background noise",
    "White noise",
    "Pink noise",
    "Noise",
})

# ─── Benchmark flush intervals ────────────────────────────────────────────────
VIDEO_LOG_FLUSH_INTERVAL = 100
AUDIO_LOG_FLUSH_INTERVAL = 10


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Database (for registration module)
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/video_proctoring"
    debug: bool = False

    # JWT
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 30

    # Camera
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720

    # Head pose (REQ-001)
    head_pose_yaw_threshold: float = 20.0
    head_pose_pitch_threshold: float = 20.0
    head_pose_roll_threshold: float = 15.0
    head_pose_consecutive_frames: int = 3

    # Eye gaze (REQ-002)
    eye_gaze_left_threshold: float = 0.35
    eye_gaze_right_threshold: float = 0.65
    eye_gaze_up_threshold: float = 0.35
    eye_gaze_down_threshold: float = 0.65
    eye_gaze_consecutive_frames: int = 3
    eye_gaze_min_confidence: float = 0.7

    # Hand detection (REQ-003)
    hand_proximity_threshold_px: float = 80.0

    # YOLO (REQ-004)
    yolo_model: str = str(_SRC_DIR / "yolov8s.pt")
    yolo_confidence_threshold: float = 0.35
    yolo_person_confidence_threshold: float = 0.60   # stricter threshold for person class
    yolo_input_size: int = 640
    prohibited_objects: str = "cell phone,phone,earphones,laptop,tablet,book,remote,keyboard,person"
    
    # YOLO frame skip — only submit every Nth frame to YOLO to prevent queue backlog.
    # At 30 FPS with skip=6, YOLO receives ~5 FPS — enough to detect objects that
    # are visible for seconds (phones, laptops, persons) while keeping the queue
    # from growing faster than YOLO can process on CPU.
    yolo_frame_skip: int = 6

    # Audio / YAMNet (REQ-005)
    audio_sample_rate: int = 16000
    audio_buffer_seconds: float = AUDIO_BUFFER_SECONDS
    lip_sync_mar_threshold: float = 0.02   # used by AudioPipeline for lip-sync suppression
    mouth_open_mar_threshold: float = 0.25  # used for MOUTH_OPEN detection
    # 3-point MAR typical values:
    #   Mouth closed:        0.04 – 0.18
    #   Mouth slightly open: 0.20 – 0.30
    #   Mouth wide open:     0.35 – 0.60
    # 0.25 sits clearly above closed and catches all open states
    yamnet_model_url: str = YAMNET_MODEL_URL

    # Benchmark reporting (REQ-006)
    video_log_flush_interval: int = VIDEO_LOG_FLUSH_INTERVAL
    audio_log_flush_interval: int = AUDIO_LOG_FLUSH_INTERVAL
    video_violations_log: str = VIDEO_VIOLATIONS_LOG
    audio_violations_log: str = AUDIO_VIOLATIONS_LOG
    yolo_detections_log: str = YOLO_DETECTIONS_LOG
    mediapipe_log: str = MEDIAPIPE_LOG
    yamnet_log: str = YAMNET_LOG

    # Face verification (REQ-FV)
    face_similarity_threshold: float = 0.70
    """Cosine similarity threshold for face verification (0.60 = 60% similar, handles angle variations)"""
    
    # Periodic face verification configuration
    verification_interval_seconds: int = 60
    """Trigger periodic face verification every N seconds"""
    
    max_verification_attempts: int = 3
    """Maximum verification attempts per cycle"""
    
    frames_between_attempts: int = 6
    """Wait N frames (~0.2s at 30 FPS) between verification attempts"""
    
    # Face detection and quality thresholds
    detection_confidence: float = 0.6
    """InsightFace detection confidence threshold"""
    
    min_face_size: int = 60
    """Minimum face size in pixels (reduced from 80 for more flexibility)"""
    
    min_face_ratio: float = 0.04
    """Minimum face ratio (4% of image, reduced from 8% for comfortable distance)"""
    
    max_faces_allowed: int = 1
    """Maximum number of faces allowed in frame"""
    
    min_brightness: int = 60
    """Minimum brightness threshold"""
    
    max_brightness: int = 220
    """Maximum brightness threshold"""
    
    optimal_brightness_min: int = 90
    """Optimal brightness minimum"""
    
    optimal_brightness_max: int = 180
    """Optimal brightness maximum"""
    
    min_sharpness: int = 50
    """Minimum sharpness (Laplacian variance) for production quality"""
    
    min_resolution_width: int = 640
    """Minimum frame width resolution"""
    
    min_resolution_height: int = 480
    """Minimum frame height resolution"""
    
    embedding_size: int = 512
    """Face embedding dimension size (InsightFace ArcFace)"""
    
    background_embedding_size: int = 768
    """Background embedding dimension size (DINOv2 ViT-B/14)"""
    
    insightface_model: str = "buffalo_l"
    """InsightFace model name"""
    
    @property
    def model_dir(self) -> Path:
        """Return the models directory path (video-proctoring-engine/models)"""
        return Path(__file__).resolve().parent.parent.parent / "models"

    # Background validation (REQ-BG)
    bg_similarity_threshold: float = 0.85
    """Cosine similarity threshold for background matching (0.70 = 70% similar)"""
    
    bg_failure_threshold: int = 3
    """Number of consecutive failures before triggering warning"""
    
    bg_check_interval_seconds: float = 60.0
    """Background validation interval (seconds) - always 60s"""
    
    bg_inpaint_radius: int = 10
    """Telea inpainting radius (pixels) for filling person region (range: 3-15, default: 10)"""
    
    bg_inpainting: bool = True
    """Enable Telea inpainting to fill person region (ESSENTIAL)"""
    
    bg_clahe_normalization: bool = True
    """Enable CLAHE normalization for lighting robustness (ESSENTIAL)"""
    
    bg_global_pooling: bool = True
    """Enable global pooling for position-invariant embeddings (ESSENTIAL)"""

    @property
    def prohibited_objects_list(self) -> list[str]:
        """Return prohibited objects as a list of lowercase strings."""
        return [obj.strip().lower() for obj in self.prohibited_objects.split(",")]

    @property
    def audio_buffer_samples(self) -> int:
        """Number of audio samples per YAMNet buffer."""
        return int(self.audio_sample_rate * self.audio_buffer_seconds)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
