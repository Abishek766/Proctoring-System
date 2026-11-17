"""Pydantic models for structured violation events."""

from enum import Enum
from typing import Union

from pydantic import BaseModel, Field


class HeadPoseDirection(str, Enum):
    """Possible head pose violation directions."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class GazeDirection(str, Enum):
    """Possible eye gaze violation directions."""

    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class AudioSubtype(str, Enum):
    """Audio anomaly subtypes."""

    EXTERNAL_VOICE = "external_voice"
    DEVICE_SOUND = "device_sound"


class HeadPoseEvent(BaseModel):
    """Violation event for abnormal head pose (REQ-001)."""

    type: str = "head_pose"
    direction: HeadPoseDirection
    angle_deg: float
    frame_id: int
    timestamp_ms: int
    latency_ms: float = 0.0


class EyeGazeEvent(BaseModel):
    """Violation event for suspicious eye gaze (REQ-002)."""

    type: str = "eye_gaze"
    direction: GazeDirection
    ratio: float
    frame_id: int
    timestamp_ms: int
    latency_ms: float = 0.0


class ExternalHandEvent(BaseModel):
    """Violation event for an external hand detected in frame (REQ-003)."""

    type: str = "external_hand"
    hand_id: int
    distance_px: float
    frame_id: int
    timestamp_ms: int
    latency_ms: float = 0.0


class ProhibitedObjectEvent(BaseModel):
    """Violation event for a prohibited object detected by YOLO (REQ-004)."""

    type: str = "prohibited_object"
    object_class: str
    confidence: float
    bbox: list[int] = Field(default_factory=list)
    """Bounding box as [x1, y1, x2, y2]."""
    frame_id: int
    timestamp_ms: int
    latency_ms: float = 0.0


class AudioAnomalyEvent(BaseModel):
    """Violation event for background audio anomaly (REQ-005)."""

    type: str = "audio_anomaly"
    subtype: AudioSubtype
    yamnet_class: str
    confidence: float
    timestamp_ms: int
    segment_id: int
    latency_ms: float = 0.0


class MouthOpenEvent(BaseModel):
    """Candidate's mouth is open — detected via Mouth Aspect Ratio (MAR).

    Emitted every frame where MAR >= lip_sync_mar_threshold.
    The backend uses this to determine if the candidate is speaking.
    """

    type: str = "MOUTH_OPEN"
    mar: float
    """Mouth Aspect Ratio: ratio of mouth opening height to width (0.0–1.0).
    Higher values = more open. Threshold is lip_sync_mar_threshold in settings.
    """
    frame_id: int
    timestamp_ms: int
    latency_ms: float = 0.0


class MultiplePersonsEvent(BaseModel):
    """More than one person detected in the frame by YOLO.

    Emitted at the window level — fires if ANY frame within the 30-frame
    window contained a YOLO person detection.  The `person` class is stripped
    from prohibited_objects in the API response and replaced by this event.
    """

    type: str = "MULTIPLE_PERSONS"
    keyword: str = "multiple_person"
    message: str = "Other person detected on the screen, only candidate should be present in the room"
    frame_id: int
    """Frame ID of the first frame where a person was detected."""
    timestamp_ms: int
    """Timestamp of the first frame where a person was detected."""
    confidence: float
    """YOLO detection confidence for the person."""


class NoPersonEvent(BaseModel):
    """No person detected in the frame by YOLO.

    Emitted when YOLO fails to detect any person in the frame, meaning
    the candidate is not visible or has moved out of frame.
    """

    type: str = "NO_PERSON"
    keyword: str = "no_person"
    message: str = "Candidate face should be visible"
    frame_id: int
    """Frame ID where no person was detected."""
    timestamp_ms: int
    """Timestamp when no person was detected."""
    latency_ms: float = 0.0


class IdentityVerificationEvent(BaseModel):
    """Identity verification result event.

    Emitted when face verification is triggered (on no-person or multiple-person detection)
    to verify that the person in the frame matches the registered user.
    """

    type: str = "IDENTITY_VERIFICATION"
    is_match: bool
    """Whether the face matches the registered user."""
    similarity: float
    """Cosine similarity score (0.0 to 1.0)."""
    confidence: str
    """Confidence level: 'high', 'medium', 'low', 'very_low', 'none', 'error'."""
    message: str
    """Human-readable verification message."""
    user_id: str | None = None
    """User ID that was verified against."""
    username: str | None = None
    """Username of the verified user."""
    frame_id: int
    """Frame ID where verification was performed."""
    timestamp_ms: int
    """Timestamp when verification was performed."""
    latency_ms: float = 0.0
    """Verification latency in milliseconds."""
    is_initial_verification: bool = False
    """Whether this is an initial verification (session start) vs periodic/violation verification."""


class BackgroundValidationEvent(BaseModel):
    """Background validation result event.
    
    Emitted when background validation is performed to check if the candidate
    has changed their exam location.
    """
    
    type: str = "BACKGROUND_VALIDATION"
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
    latency_ms: float = 0.0
    """Processing latency in milliseconds"""
    message: str
    """Human-readable message"""


# Union type for all video violation events
ViolationEvent = Union[
    HeadPoseEvent,
    EyeGazeEvent,
    ExternalHandEvent,
    ProhibitedObjectEvent,
    AudioAnomalyEvent,
    MouthOpenEvent,
    MultiplePersonsEvent,
    NoPersonEvent,
    IdentityVerificationEvent,
    BackgroundValidationEvent,
]
