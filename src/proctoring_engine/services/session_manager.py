"""Session manager — creates, tracks, and destroys per-candidate InferenceEngine instances.

Each WebSocket connection maps to one Session.  The session owns:
  - One InferenceEngine (all pipelines: FaceMesh, Hands, YOLO, YAMNet)
  - A one-second frame window accumulator
  - The WebSocket connection reference

Thread-safety:
  - _sessions dict is protected by an asyncio.Lock (all access from the
    async event loop).
  - InferenceEngine internals use their own threading locks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from proctoring_engine.models.api_models import ViolationState, WindowResponse
from proctoring_engine.models.violations import MultiplePersonsEvent, NoPersonEvent
from proctoring_engine.services.inference_engine import FrameViolations, InferenceEngine
from proctoring_engine.settings import Settings
from proctoring_engine.utils.logger import get_logger

logger = get_logger("video_pipeline")

# How many frames to accumulate before sending a response.
# At 30 FPS this is ~1 second.
WINDOW_SIZE_FRAMES = 6

# Fade-out configuration (matches camera flow's STICKY_CLEAR_FRAMES)
FADE_OUT_FRAMES = 10  # Violation fades over 10 frames after it stops

# Violation keyword and message mapping for frontend display
# Maps internal violation keys to (frontend_keyword, message) tuples
VIOLATION_MAPPING: dict[str, tuple[str, str]] = {
    # Head pose violations (all directions map to single "head_pose" keyword)
    "head_pose_left": ("head_pose", "candidate turning away from the screen"),
    "head_pose_right": ("head_pose", "candidate turning away from the screen"),
    "head_pose_up": ("head_pose", "candidate turning away from the screen"),
    "head_pose_down": ("head_pose", "candidate turning away from the screen"),
    # Eye gaze violations (all directions map to single "gaze_position" keyword)
    "eye_gaze_left": ("gaze_position", "candidate looking away from the screen"),
    "eye_gaze_right": ("gaze_position", "candidate looking away from the screen"),
    "eye_gaze_up": ("gaze_position", "candidate looking away from the screen"),
    "eye_gaze_down": ("gaze_position", "candidate looking away from the screen"),
    # Person violations
    "multiple_persons": ("multiple_person", "other person detected on the screen, only candidate should be present in the room"),
    "no_person": ("no_person", "candidate face should be visible"),
    # Identity verification violations
    "face_mismatch": ("face_mismatch", "face does not match registered candidate, possible impersonation detected"),
    # Background validation violations
    "background_mismatch": ("background_mismatch", "Background changed! Candidate may have changed location"),
    # Mouth/speaking violations
    "mouth_open": ("candidate_speaking", "candidate is speaking, maintain silence"),
    # Audio violations
    "audio_external_voice": ("environment_speech", "speech detected in your environment, make sure that no other person present in the room"),
    "audio_device_sound": ("environment_noise", "please make sure that your environment is silent"),
}


def get_violation_keyword_and_message(internal_key: str) -> tuple[str, str]:
    """Get frontend keyword and message for an internal violation key.

    Args:
        internal_key: Internal violation key (e.g., "head_pose_left", "eye_gaze_right")

    Returns:
        Tuple of (frontend_keyword, message)
        Example: ("head_pose", "candidate turning to the left/right/up/down")
    """
    if internal_key in VIOLATION_MAPPING:
        return VIOLATION_MAPPING[internal_key]

    # Handle dynamic object violations (all map to single "object_detected" keyword)
    if internal_key.startswith("object_"):
        object_name = internal_key.replace("object_", "").replace("_", " ")
        return ("object_detected", f"{object_name} detected in frame")

    # Fallback for unknown violations
    return (internal_key, "Violation detected")


@dataclass
class Session:
    """State for one active proctoring session."""

    session_id: str
    engine: InferenceEngine
    user_id: str | None = None  # User ID for identity verification

    # Accumulated frame results for the current window
    window_frames: list[FrameViolations] = field(default_factory=list)
    window_start_ms: int = 0

    # Wall-clock time when the first frame of the current window arrived
    window_wall_start: float = field(default_factory=time.perf_counter)

    # Track last known person count from YOLO (to avoid false NO_PERSON during queue gaps)
    last_person_count: int = 1  # Assume 1 person initially (candidate present)

    # Active violations with fade-out tracking
    # Maps violation_key → (last_seen_frame_id, frames_since_last, sent_zero, triggered_verification)
    # sent_zero tracks whether we've already sent severity 0.0 for this violation
    # triggered_verification tracks whether we've triggered face verification for this violation
    active_violations: dict[str, tuple[int, int, bool, bool]] = field(default_factory=dict)

    def reset_window(self) -> None:
        self.window_frames = []
        self.window_start_ms = 0
        self.window_wall_start = time.perf_counter()

    def add_frame(self, result: FrameViolations) -> None:
        if not self.window_frames:
            self.window_start_ms = result.timestamp_ms
            self.window_wall_start = time.perf_counter()
        self.window_frames.append(result)

    def is_window_ready(self) -> bool:
        return len(self.window_frames) >= WINDOW_SIZE_FRAMES

    def build_response(self) -> WindowResponse:
        """Aggregate all frame results into a WindowResponse and reset the window."""
        frames = self.window_frames
        if not frames:
            raise ValueError("No frames in window")

        # Collect all violation keys detected in this window
        current_violations: set[str] = set()

        # ── Separate person detections from other prohibited objects ──────────
        # YOLO emits one ProhibitedObjectEvent per bounding box.
        # Special case: confidence=0.0 is a marker meaning "no person detected"
        # Simple logic:
        #   - 0 persons detected → NO_PERSON violation
        #   - 1 person detected  → Normal (candidate present)
        #   - ≥2 persons detected → MULTIPLE_PERSONS violation
        all_prohibited = [ev for f in frames for ev in f.prohibited_objects]
        person_events  = [ev for ev in all_prohibited if ev.object_class.lower() == "person" and ev.confidence > 0]
        no_person_markers = [ev for ev in all_prohibited if ev.object_class.lower() == "person" and ev.confidence == 0.0]
        other_objects  = [ev for ev in all_prohibited if ev.object_class.lower() != "person"]

        # Update last_person_count only when we have YOLO results
        if all_prohibited:
            if no_person_markers:
                # YOLO confirmed 0 persons in at least one frame
                self.last_person_count = 0
            elif person_events:
                # Count unique persons across all frames in this window
                from collections import defaultdict  # noqa: PLC0415
                persons_per_frame: dict[int, list] = defaultdict(list)
                for ev in person_events:
                    persons_per_frame[ev.frame_id].append(ev)

                # Use the maximum person count seen in any frame
                self.last_person_count = max(len(detections) for detections in persons_per_frame.values())
            else:
                # No person events and no markers - keep last known count
                pass

        # Debug logging
        from proctoring_engine.utils.logger import get_logger  # noqa: PLC0415
        logger = get_logger("api_session")
        logger.info(
            f"YOLO_PERSONS_RESPONSE window_frames={len(frames)} "
            f"all_prohibited_count={len(all_prohibited)} "
            f"person_events_count={len(person_events)} "
            f"last_person_count={self.last_person_count} "
            f"other_objects_count={len(other_objects)} "
            f"all_classes=[{', '.join(ev.object_class for ev in all_prohibited)}]"
        )

        # Group person detections by frame_id for detailed analysis
        from collections import defaultdict  # noqa: PLC0415
        persons_per_frame: dict[int, list] = defaultdict(list)
        for ev in person_events:
            persons_per_frame[ev.frame_id].append(ev)

        # ── Check for MULTIPLE_PERSONS (≥2 persons in any frame) ──────────────
        multiple_persons: list[MultiplePersonsEvent] = []
        if self.last_person_count >= 2:
            for frame_id_key in sorted(persons_per_frame):
                detections = persons_per_frame[frame_id_key]
                if len(detections) >= 2:
                    # Use the highest-confidence detection for reporting
                    best = max(detections, key=lambda e: e.confidence)
                    multiple_persons.append(MultiplePersonsEvent(
                        frame_id=best.frame_id,
                        timestamp_ms=best.timestamp_ms,
                        confidence=best.confidence,
                    ))
                    break   # one event per window is enough

        # ── Check for NO_PERSON (0 persons detected) ──────────────────────────
        no_person: list[NoPersonEvent] = []

        # Only report NO_PERSON when YOLO confirmed 0 persons
        if self.last_person_count == 0 and frames:
            # Report using the first frame in the window
            first_frame = frames[0]
            no_person.append(NoPersonEvent(
                frame_id=first_frame.frame_id,
                timestamp_ms=first_frame.timestamp_ms,
                latency_ms=0.0,
            ))

        # ── Merge all other violations across frames ───────────────────────────
        # Track violation keys for fade-out
        for f in frames:
            for ev in f.head_pose:
                current_violations.add(f"head_pose_{ev.direction.value}")
            for ev in f.eye_gaze:
                current_violations.add(f"eye_gaze_{ev.direction.value}")
            for ev in f.external_hands:
                current_violations.add("external_hand")
            for ev in f.mouth_open:
                current_violations.add("mouth_open")
            for ev in f.audio_anomalies:
                current_violations.add(f"audio_{ev.subtype.value}")
            # Process identity verification events
            for ev in f.identity_verification:
                # Special handling for initial verification
                if ev.is_initial_verification:
                    # Initial verification: ALWAYS add face_mismatch if not matched
                    # Don't skip based on confidence - we need to show the result immediately
                    if not ev.is_match:
                        current_violations.add("face_mismatch")
                        logger.warning(
                            f"[INITIAL_VERIFICATION_FAILED] frame={f.frame_id} "
                            f"similarity={ev.similarity:.4f} confidence={ev.confidence} "
                            f"message=\"{ev.message}\" - adding to violations"
                        )
                    elif ev.is_match:
                        logger.info(
                            f"[INITIAL_VERIFICATION_SUCCESS] frame={f.frame_id} "
                            f"similarity={ev.similarity:.4f} confidence={ev.confidence}"
                        )
                # Regular verification (periodic/violation-triggered)
                else:
                    # Only trigger face_mismatch if it's an actual identity mismatch
                    # Skip if confidence is 'none' or 'error' (quality check failed or error occurred)
                    if not ev.is_match and ev.confidence not in ('none', 'error'):
                        # Face mismatch detected - add to violations
                        current_violations.add("face_mismatch")
                        logger.info(
                            f"[FACE_MISMATCH_DETECTED] frame={f.frame_id} "
                            f"similarity={ev.similarity:.4f} confidence={ev.confidence}"
                        )
                    elif ev.is_match:
                        # Face matched - remove face_mismatch if it exists
                        # This will allow it to fade out naturally
                        logger.info(
                            f"[FACE_MATCH_SUCCESS] frame={f.frame_id} "
                            f"similarity={ev.similarity:.4f} confidence={ev.confidence} "
                            f"- face_mismatch will fade out"
                        )
                    elif ev.confidence in ('none', 'error'):
                        # Quality check failed or error - skip (will retry automatically)
                        logger.debug(
                            f"[FACE_VERIFY_SKIP] frame={f.frame_id} "
                            f"confidence={ev.confidence} message=\"{ev.message}\" - will retry"
                        )

        # Add person-related violations
        if self.last_person_count == 0:
            current_violations.add("no_person")
        elif self.last_person_count >= 2:
            current_violations.add("multiple_persons")

        # Add prohibited objects
        for ev in other_objects:
            current_violations.add(f"object_{ev.object_class.replace(' ', '_')}")

        # ── Build active_violations dict with fade-out severity ───────────────
        last_frame_id = frames[-1].frame_id

        # Update active violations: reset counter for active ones, increment for inactive
        for key in current_violations:
            self.active_violations[key] = (last_frame_id, 0, False, False)  # just seen, no fade, not sent zero, not triggered verification

        # Age out inactive violations and remove after sending severity 0.0
        # Also trigger face verification when severity reaches 0.0 for person-related violations
        to_remove = []
        violations_reaching_zero = []  # Track violations that just reached severity 0.0
        
        for key, (last_seen_frame, frames_since, sent_zero, triggered_verification) in self.active_violations.items():
            if key not in current_violations:
                new_frames_since = frames_since + len(frames)  # increment by window size
                
                # Check if we should remove (already sent 0.0 in previous window)
                if new_frames_since >= FADE_OUT_FRAMES and sent_zero:
                    # Already sent severity 0.0, now remove
                    to_remove.append(key)
                else:
                    # Check if this violation just reached severity 0.0
                    just_reached_zero = (new_frames_since >= FADE_OUT_FRAMES and not sent_zero)
                    
                    # Mark as sent_zero if we've reached the fade-out threshold
                    new_sent_zero = sent_zero or just_reached_zero
                    
                    # Track if we need to trigger verification for this violation
                    if just_reached_zero and not triggered_verification:
                        # Only trigger for person-related violations
                        if key in ("no_person", "multiple_persons"):
                            violations_reaching_zero.append(key)
                            triggered_verification = True
                    
                    self.active_violations[key] = (last_seen_frame, new_frames_since, new_sent_zero, triggered_verification)

        for key in to_remove:
            del self.active_violations[key]
        
        # Check if face_mismatch is currently active (needs continuous verification)
        # Verification continues until face matches AND severity reaches 0.0 (fully removed)
        face_mismatch_active = "face_mismatch" in self.active_violations
        
        # Set flag if face verification should be triggered
        # Trigger if: (1) violations reaching zero OR (2) face_mismatch is active (even while fading)
        trigger_verification = len(violations_reaching_zero) > 0 or face_mismatch_active
        
        if trigger_verification:
            if face_mismatch_active:
                mismatch_data = self.active_violations['face_mismatch']
                current_severity = max(0.0, 1.0 - (mismatch_data[1] / FADE_OUT_FRAMES))
                logger.info(
                    f"[VERIFICATION_TRIGGER] reason=face_mismatch_active "
                    f"frame_id={frames[-1].frame_id} "
                    f"face_mismatch_severity={current_severity:.2f} "
                    f"frames_since={mismatch_data[1]} "
                    f"status={'fading' if mismatch_data[1] > 0 else 'active'}"
                )
            if violations_reaching_zero:
                logger.info(
                    f"[VERIFICATION_TRIGGER] reason=violations_reaching_zero "
                    f"violations={violations_reaching_zero} frame_id={frames[-1].frame_id}"
                )

        # Build ViolationState dict for response
        # Group by frontend keyword (multiple internal keys may map to same frontend keyword)
        active_violations_response: dict[str, ViolationState] = {}
        
        for internal_key, (last_seen_frame, frames_since, sent_zero, triggered_verification) in self.active_violations.items():
            # Get frontend keyword and message
            frontend_keyword, message = get_violation_keyword_and_message(internal_key)
            
            # Calculate severity: 1.0 = just detected, 0.0 = fully faded
            severity = max(0.0, 1.0 - (frames_since / FADE_OUT_FRAMES))
            
            # If this frontend keyword already exists, keep the highest severity
            if frontend_keyword in active_violations_response:
                existing_severity = active_violations_response[frontend_keyword].severity
                if severity > existing_severity:
                    active_violations_response[frontend_keyword] = ViolationState(
                        message=message,
                        severity=round(severity, 2),
                    )
            else:
                active_violations_response[frontend_keyword] = ViolationState(
                    message=message,
                    severity=round(severity, 2),
                )

        mean_face_lat = (
            sum(f.face_latency_ms for f in frames) / len(frames)
        )
        total_latency = (time.perf_counter() - self.window_wall_start) * 1000.0

        # Calculate has_violations from active_violations
        has_violations = len(active_violations_response) > 0

        response = WindowResponse(
            session_id=self.session_id,
            window_start_ms=frames[0].timestamp_ms,
            window_end_ms=frames[-1].timestamp_ms,
            frame_count=len(frames),
            active_violations=active_violations_response,
            mean_face_latency_ms=round(mean_face_lat, 2),
            total_latency_ms=round(total_latency, 2),
            has_violations=has_violations,
            trigger_face_verification=trigger_verification,
        )
        
        self.reset_window()
        return response


class SessionManager:
    """Creates and manages InferenceEngine instances per WebSocket session."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, session_id: str, user_id: str | None = None, db_session=None) -> Session:
        """Create a new session with a fresh InferenceEngine.

        Args:
            session_id: Unique session identifier
            user_id: Optional user ID for identity verification
            db_session: Optional database session for verification queries

        Raises ValueError if session_id already exists.
        """
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session '{session_id}' already exists.")
            engine = InferenceEngine(self._settings, user_id=user_id, db_session=db_session)
            session = Session(session_id=session_id, engine=engine, user_id=user_id)
            self._sessions[session_id] = session
            logger.info(f"Session created: {session_id}, user_id={user_id}  (active={len(self._sessions)})")
            return session

    async def get_session(self, session_id: str) -> Session | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """Shut down the engine and remove the session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            # Engine close is blocking (joins YOLO thread) — run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, session.engine.close)
            logger.info(f"Session closed: {session_id}  (active={len(self._sessions)})")

    @property
    def active_count(self) -> int:
        return len(self._sessions)
