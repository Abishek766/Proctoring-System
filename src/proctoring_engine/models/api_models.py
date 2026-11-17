"""Pydantic models for the WebSocket API — request messages and response payloads.

Wire format (client → server):
    {
        "session_id": "abc123",
        "frame_id": 42,
        "timestamp_ms": 1700000000000,
        "frame_b64": "<base64-encoded JPEG>",
        "audio_b64": "<base64-encoded float32 PCM, optional>"
    }

Wire format (server → client, sent every 6 frames):
    {
        "session_id": "abc123",
        "window_start_ms": 1700000000000,
        "window_end_ms":   1700000001000,
        "frame_count": 6,
        "active_violations": {
            "head_pose": {
                "message": "candidate turning to the left/right/up/down",
                "severity": 1.0
            },
            "gaze_position": {
                "message": "candidate looking to the left/right/up/down",
                "severity": 0.8
            },
            "no_person": {
                "message": "candidate face should be visible",
                "severity": 1.0
            }
        },
        "mean_face_latency_ms": 45.2,
        "total_latency_ms": 167.8,
        "has_violations": true
    }

Severity levels:
    1.0 = violation just detected (full intensity)
    0.7 = violation 3 frames ago (fading)
    0.3 = violation 7 frames ago (almost gone)
    0.0 = violation cleared (removed from active_violations)

Frontend keywords:
    - head_pose: Candidate turning head (consolidates left/right/up/down)
    - gaze_position: Candidate looking away (consolidates left/right/up/down)
    - no_person: Candidate face not visible
    - multiple_person: Other person detected
    - candidate_speaking: Candidate is speaking
    - environment_speech: External voice detected
    - environment_noise: Background noise detected
    - object_*: Prohibited objects (cell_phone, laptop, etc.)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from proctoring_engine.models.violations import (
    AudioAnomalyEvent,
    ExternalHandEvent,
    EyeGazeEvent,
    HeadPoseEvent,
    IdentityVerificationEvent,
    MouthOpenEvent,
    MultiplePersonsEvent,
    NoPersonEvent,
    ProhibitedObjectEvent,
)

# ─── Client → Server ──────────────────────────────────────────────────────────

class FrameMessage(BaseModel):
    """Single WebSocket message sent by the client for each video frame.

    The client sends this at 30 FPS.  Audio is optional — if the client
    sends audio chunks at a different rate it can omit audio_b64 on frames
    where no new audio is available.
    """

    session_id: str
    """Unique identifier for this proctoring session."""

    frame_id: int
    """Monotonically increasing frame counter (starts at 1)."""

    timestamp_ms: int
    """Wall-clock timestamp in milliseconds when the frame was captured."""

    frame_b64: str
    """Base64-encoded JPEG image (320×240 recommended)."""

    audio_b64: str | None = None
    """Base64-encoded float32 PCM audio chunk (4096 samples at 16 kHz, mono).
    Little-endian IEEE 754 float32 bytes, then base64-encoded.
    Omit or set to null if no audio is available for this frame.
    """


# ─── Server → Client ──────────────────────────────────────────────────────────

class ViolationState(BaseModel):
    """Current state of a violation with fade-out severity.

    Used to implement smooth fade-out animations in the frontend.
    """

    message: str
    """User-friendly warning message to display in the frontend."""

    severity: float = Field(ge=0.0, le=1.0)
    """Severity level from 0.0 (fully faded) to 1.0 (just detected).
    Frontend should use this for opacity/color intensity.
    """


class ViolationSummary(BaseModel):
    """All violations detected within a one-second window."""

    head_pose: list[HeadPoseEvent] = Field(default_factory=list)
    eye_gaze: list[EyeGazeEvent] = Field(default_factory=list)
    external_hands: list[ExternalHandEvent] = Field(default_factory=list)
    prohibited_objects: list[ProhibitedObjectEvent] = Field(default_factory=list)
    """Non-person prohibited objects only (phones, laptops, etc.).
    Person detections are reported separately in multiple_persons and no_person.
    """
    multiple_persons: list[MultiplePersonsEvent] = Field(default_factory=list)
    """Populated if any frame in the window had ≥2 YOLO person detections."""
    no_person: list[NoPersonEvent] = Field(default_factory=list)
    """Populated if any frame in the window had 0 YOLO person detections."""
    mouth_open: list[MouthOpenEvent] = Field(default_factory=list)
    audio_anomalies: list[AudioAnomalyEvent] = Field(default_factory=list)
    identity_verification: list[IdentityVerificationEvent] = Field(default_factory=list)
    """Identity verification results when triggered by no_person or multiple_persons."""

    @property
    def has_violations(self) -> bool:
        return bool(
            self.head_pose
            or self.eye_gaze
            or self.external_hands
            or self.prohibited_objects
            or self.multiple_persons
            or self.no_person
            or self.mouth_open
            or self.audio_anomalies
            or self.identity_verification
        )

    def total_count(self) -> int:
        return (
            len(self.head_pose)
            + len(self.eye_gaze)
            + len(self.external_hands)
            + len(self.prohibited_objects)
            + len(self.multiple_persons)
            + len(self.no_person)
            + len(self.mouth_open)
            + len(self.audio_anomalies)
            + len(self.identity_verification)
        )


class WindowResponse(BaseModel):
    """Response sent to the client once per second (after ~30 frames).

    Sent regardless of whether violations were found — the client uses
    this as a heartbeat to confirm the engine is processing.
    """

    session_id: str
    window_start_ms: int
    """Timestamp of the first frame in this window."""

    window_end_ms: int
    """Timestamp of the last frame in this window."""

    frame_count: int
    """Number of frames processed in this window."""

    active_violations: dict[str, ViolationState] = Field(default_factory=dict)
    """Currently active or fading violations with severity levels.

    Keys are frontend violation keywords:
    - "head_pose" - Candidate turning head (any direction)
    - "gaze_position" - Candidate looking away (any direction)
    - "no_person" - Candidate face not visible
    - "multiple_person" - Other person detected
    - "candidate_speaking" - Candidate is speaking
    - "environment_speech" - External voice detected
    - "environment_noise" - Background noise detected
    - "object_*" - Prohibited objects (cell_phone, laptop, etc.)

    Values contain message and severity (0.0-1.0) for fade-out animations.

    Example:
        {
            "head_pose": {
                "message": "candidate turning to the left/right/up/down",
                "severity": 1.0
            },
            "gaze_position": {
                "message": "candidate looking to the left/right/up/down",
                "severity": 0.8
            },
            "no_person": {
                "message": "candidate face should be visible",
                "severity": 1.0
            }
        }

    Note: Multiple internal violations (e.g., head_pose_left, head_pose_right)
    are consolidated into a single frontend keyword (head_pose) with the
    highest severity among them.
    """

    mean_face_latency_ms: float
    """Mean FaceMesh + hand pipeline latency across frames in this window."""

    total_latency_ms: float
    """Wall-clock time from first frame received to response sent (ms)."""

    has_violations: bool
    """Convenience flag — True if any violation was detected."""
    
    trigger_face_verification: bool = False
    """Internal flag — True if face verification should be triggered for this window.
    Used by WebSocket handler to trigger verification when person violations reach severity 0.0.
    Not sent to frontend."""


class SessionStartResponse(BaseModel):
    """Sent immediately after the WebSocket connection is accepted."""

    session_id: str
    status: str = "connected"
    message: str = "Session started. Send FrameMessage at 30 FPS."


# InitialVerificationResponse removed — reuse WindowResponse for all frontend responses

    session_id: str


class ErrorResponse(BaseModel):
    """Sent when a message cannot be processed."""

    session_id: str | None = None
    error: str
    detail: str | None = None

