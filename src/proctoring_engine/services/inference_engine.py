"""Stateless per-frame inference engine — shared by both the camera loop and the API.

This module extracts the pure detection logic from ProctoringService so it can
be reused by API sessions without any camera, overlay, or benchmark coupling.

The camera-based ProctoringService is NOT modified — it continues to work
exactly as before via `python -m proctoring_engine.main`.

Usage (API session):
    engine = InferenceEngine(settings)
    result = engine.process_frame(bgr_frame, frame_id, timestamp_ms)
    engine.process_audio(pcm_float32_array)   # call whenever audio arrives
    engine.close()
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

# ── TensorFlow and TF Hub imported at module level ────────────────────────────
# Lazy imports inside threaded functions cause partial-module errors:
#   "module 'tensorflow' has no attribute '__version__'"
#   "module 'tensorflow' has no attribute 'cast'"
# Importing here ensures TF is fully initialised in the main thread before
# any thread pool executor calls _run_yamnet().
try:
    import tensorflow as _tf
    import tensorflow_hub as _hub
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

from proctoring_engine.core.face_pipeline import FacePipeline
from proctoring_engine.core.pose_pipeline import HandPipeline
from proctoring_engine.core.yolo_pipeline import YoloPipeline
from proctoring_engine.models.violations import (
    AudioAnomalyEvent,
    EyeGazeEvent,
    ExternalHandEvent,
    HeadPoseEvent,
    IdentityVerificationEvent,
    MouthOpenEvent,
    ProhibitedObjectEvent,
)
from proctoring_engine.settings import Settings
from proctoring_engine.utils.logger import get_logger

logger = get_logger("video_pipeline")
verify_logger = get_logger("face_verification")


@dataclass
class FrameViolations:
    """All violations detected for a single video frame."""

    frame_id: int
    timestamp_ms: int
    head_pose: list[HeadPoseEvent] = field(default_factory=list)
    eye_gaze: list[EyeGazeEvent] = field(default_factory=list)
    external_hands: list[ExternalHandEvent] = field(default_factory=list)
    prohibited_objects: list[ProhibitedObjectEvent] = field(default_factory=list)
    mouth_open: list[MouthOpenEvent] = field(default_factory=list)
    # Audio violations accumulated since the last frame result
    audio_anomalies: list[AudioAnomalyEvent] = field(default_factory=list)
    # Identity verification results (triggered on no-person/multiple-person)
    identity_verification: list[IdentityVerificationEvent] = field(default_factory=list)
    # Per-pipeline latencies (ms)
    face_latency_ms: float = 0.0
    hand_latency_ms: float = 0.0


class InferenceEngine:
    """Runs all detection pipelines on frames/audio supplied by the caller.

    Thread-safety:
      - process_frame() must be called from a single thread (MediaPipe is not
        thread-safe across instances).
      - process_audio() is called from a separate audio thread and is safe
        because it only appends to a thread-safe list.
      - YOLO runs in its own background thread (same as the camera mode).
    """

    def __init__(self, settings: Settings, user_id: str | None = None, db_session=None) -> None:
        self._settings = settings
        self._user_id = user_id
        self._db_session = db_session
        
        # Store event loop reference for scheduling async tasks from threads
        import asyncio  # noqa: PLC0415
        try:
            self._event_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._event_loop = None
        
        # ── Verification coordination state (set by proctoring_ws.py) ─────────
        self.initial_verification_in_progress = False
        self.periodic_verification_in_progress = False

        # ── Detection pipelines (same instances as camera mode) ───────────────
        self._face = FacePipeline(settings)
        self._pose = HandPipeline(settings)
        self._yolo = YoloPipeline(settings, on_violation=self._on_yolo_violation)

        # ── Latest frame for MAR (mouth aspect ratio) used by audio pipeline ──
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_lock = threading.Lock()
        
        # ── Cached MAR value — updated by process_frame, read by YAMNet thread ──
        # This avoids calling MediaPipe from multiple threads (not thread-safe).
        self._cached_mar: float = 0.0
        self._mar_lock = threading.Lock()

        # ── Pending YOLO events from background thread ─────────────────────────
        self._yolo_pending: list[ProhibitedObjectEvent] = []
        self._yolo_lock = threading.Lock()

        # ── Pending audio events from audio processing ─────────────────────────
        self._audio_pending: list[AudioAnomalyEvent] = []
        self._audio_lock = threading.Lock()

        # ── Audio buffer — accumulates PCM chunks until we have enough for YAMNet
        self._audio_buffer: list[np.ndarray] = []
        self._audio_buffer_lock = threading.Lock()
        self._audio_samples_needed = settings.audio_buffer_samples  # ~15,600 samples

        # ── YAMNet thread pool — runs inference off the frame-processing thread
        #    so a 500 ms YAMNet call never inflates total_latency_ms.
        self._yamnet_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yamnet")

        # ── YAMNet — use centralized model loader instead of loading here
        #    The model is already preloaded at startup, so this is instant.
        self._yamnet_model = None
        if _TF_AVAILABLE:
            try:
                from proctoring_engine.core.model_loader import get_yamnet_model  # noqa: PLC0415
                self._yamnet_model = get_yamnet_model()
                logger.info("YAMNet model retrieved from centralized loader.")
            except Exception as exc:
                logger.warning(f"YAMNet failed to load: {exc} — audio classification disabled.")

        # ── Face Verification — for identity validation on no-person/multiple-person
        self._face_verification_service = None  # Lazy-loaded when needed
        self._verification_pending: list[IdentityVerificationEvent] = []
        self._verification_lock = threading.Lock()
        self._last_verification_frame_id: int = -1  # Initialize to -1 so frame 0 can trigger
        self._verification_cooldown_frames: int = 30  # Wait 30 frames (~1 sec) between verifications
        self._last_verification_matched: bool = False  # Track if last verification matched
        
        # ── Person count tracking — to detect person returning after absence
        self._last_person_count: int | None = None  # Track previous frame's person count (None = not initialized yet)

        logger.info(f"InferenceEngine initialised (user_id={user_id}).")

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_mar(self) -> float:
        """Return the current mouth aspect ratio for lip-sync suppression.
        
        Returns the cached MAR value computed by the last process_frame() call.
        This is thread-safe and avoids calling MediaPipe from multiple threads.
        """
        with self._mar_lock:
            return self._cached_mar

    def trigger_face_verification(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        triggered_by_violation: bool = False,
        is_initial_verification: bool = False,
    ) -> None:
        """Trigger face verification as async task (runs in parallel with main event loop).

        This method is called when no-person or multiple-person is detected.
        It runs face verification asynchronously in parallel and stores the result in
        _verification_pending for the next process_frame() call to drain.

        Args:
            frame: BGR frame to verify
            frame_id: Frame ID
            timestamp_ms: Timestamp in milliseconds
            triggered_by_violation: If True, bypass the "already matched" check
            is_initial_verification: If True, skip quality checks (for initial verification)
        """
        verify_logger.info(
            f"[TRIGGER] frame={frame_id} ts={timestamp_ms} user_id={self._user_id} "
            f"triggered_by_violation={triggered_by_violation} is_initial={is_initial_verification}"
        )
        
        # Check cooldown to prevent duplicate verifications
        # Continue verifying every 30 frames if last verification failed (mismatch)
        # Stop verifying once match is found (UNLESS triggered by violation)
        frames_since_last = frame_id - self._last_verification_frame_id
        
        if self._last_verification_matched and not triggered_by_violation:
            # Last verification matched - stop continuous verification
            # But allow violation-triggered verification to proceed
            verify_logger.debug(
                f"[VERIFICATION_COMPLETE] frame={frame_id} "
                f"last_verification={self._last_verification_frame_id} "
                f"reason=face_matched_stopping_continuous_verification"
            )
            return
        
        # Last verification failed or first verification - check cooldown
        # Skip cooldown check for violation-triggered verification
        if not triggered_by_violation and self._last_verification_frame_id >= 0 and frames_since_last < self._verification_cooldown_frames:
            verify_logger.debug(
                f"[COOLDOWN_WAIT] frame={frame_id} "
                f"last_verification={self._last_verification_frame_id} "
                f"frames_since={frames_since_last} "
                f"cooldown_frames={self._verification_cooldown_frames} "
                f"reason=waiting_for_cooldown_to_retry"
            )
            return

        # Check if user_id and db_session are available
        if not self._user_id or not self._db_session:
            verify_logger.warning(
                f"[MISSING_DATA] frame={frame_id} user_id={self._user_id} "
                f"has_db_session={self._db_session is not None}"
            )
            return

        # Update last verification frame
        self._last_verification_frame_id = frame_id

        # Lazy-load face verification service with repositories
        if self._face_verification_service is None:
            from proctoring_engine.services.face_verification_service import FaceVerificationService  # noqa: PLC0415
            from proctoring_engine.repositories.user_repository import UserRepository  # noqa: PLC0415
            from proctoring_engine.services.face_service import FaceService  # noqa: PLC0415
            
            user_repo = UserRepository(self._db_session)
            face_service = FaceService()
            self._face_verification_service = FaceVerificationService(user_repo, face_service)
            verify_logger.info("[SERVICE_INIT] FaceVerificationService initialized with repositories")

        # Run verification as async task in parallel (schedule on event loop from thread)
        import asyncio  # noqa: PLC0415
        if self._event_loop and self._event_loop.is_running():
            # Schedule task on the event loop from this thread
            asyncio.run_coroutine_threadsafe(
                self._run_face_verification_async(
                    frame.copy(),  # Copy frame to avoid race conditions
                    frame_id,
                    timestamp_ms,
                    is_initial_verification,
                ),
                self._event_loop
            )
            verify_logger.info(
                f"[SUBMITTED] frame={frame_id} ts={timestamp_ms} "
                f"user_id={self._user_id} is_initial={is_initial_verification} status=queued_for_async_processing"
            )
        else:
            verify_logger.warning(
                f"[NO_EVENT_LOOP] frame={frame_id} cannot schedule async task"
            )

    async def _run_face_verification_async(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        is_initial_verification: bool = False,
    ) -> None:
        """Run face verification as async task (runs in parallel with main event loop).

        Args:
            frame: BGR frame to verify
            frame_id: Frame ID
            timestamp_ms: Timestamp in milliseconds
            is_initial_verification: If True, skip quality checks
        """
        verify_logger.info(
            f"[VERIFY_START] frame={frame_id} ts={timestamp_ms} "
            f"user_id={self._user_id} frame_shape={frame.shape} is_initial={is_initial_verification}"
        )
        
        try:
            start_time = time.time()
            
            # Call async verify_from_frame (runs in parallel with main event loop)
            # Skip quality check for initial verification
            result = await self._face_verification_service.verify_from_frame(
                frame, 
                self._user_id,
                skip_quality_check=is_initial_verification
            )
            
            latency_ms = (time.time() - start_time) * 1000.0

            verify_logger.info(
                f"[VERIFY_RESULT] frame={frame_id} "
                f"is_match={result['is_match']} "
                f"similarity={result['similarity']:.4f} "
                f"confidence={result['confidence']} "
                f"user_id={result.get('user_id')} "
                f"username={result.get('username')} "
                f"latency_ms={latency_ms:.2f}"
            )

            # Create verification event
            event = IdentityVerificationEvent(
                is_match=result["is_match"],
                similarity=result["similarity"],
                confidence=result["confidence"],
                message=result["message"],
                user_id=result.get("user_id"),
                username=result.get("username"),
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                latency_ms=round(latency_ms, 2),
                is_initial_verification=is_initial_verification,
            )

            # Add to pending queue
            with self._verification_lock:
                self._verification_pending.append(event)
                
            # Update match status for cooldown logic
            self._last_verification_matched = result["is_match"]

            verify_logger.info(
                f"[VERIFY_COMPLETE] frame={frame_id} "
                f"is_match={event.is_match} similarity={event.similarity:.4f} "
                f"confidence={event.confidence} latency={latency_ms:.1f}ms "
                f"message=\"{event.message}\" "
                f"will_continue_verification={not result['is_match']}"
            )

        except Exception as exc:
            verify_logger.error(
                f"[VERIFY_ERROR] frame={frame_id} "
                f"user_id={self._user_id} error={str(exc)}",
                exc_info=True
            )
            
            # Create error event
            error_event = IdentityVerificationEvent(
                is_match=False,
                similarity=0.0,
                confidence="error",
                message=f"Verification failed: {str(exc)}",
                user_id=self._user_id,
                username=None,
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                latency_ms=0.0,
                is_initial_verification=is_initial_verification,
            )
            
            with self._verification_lock:
                self._verification_pending.append(error_event)
            
            verify_logger.info(
                f"[VERIFY_ERROR_EVENT] frame={frame_id} "
                f"error_event_created message=\"{error_event.message}\""
            )

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
    ) -> FrameViolations:
        """Run all synchronous pipelines on one BGR frame.

        Args:
            bgr_frame:    320×240 (or any size) BGR numpy array.
                          The frame must already be horizontally flipped
                          (mirrored) by the caller — same convention as the
                          camera mode.
            frame_id:     monotonically increasing frame counter for this session.
            timestamp_ms: wall-clock timestamp in milliseconds.

        Returns:
            FrameViolations with all events detected in this frame plus any
            pending audio and YOLO events accumulated since the last call.
        """
        result = FrameViolations(frame_id=frame_id, timestamp_ms=timestamp_ms)
        h, w = bgr_frame.shape[:2]

        # Store for MAR computation
        with self._latest_frame_lock:
            self._latest_frame = bgr_frame

        # ── FaceMesh: head pose + eye gaze ────────────────────────────────────
        head_events, gaze_events, face_lat, face_landmarks = self._face.process_frame(
            bgr_frame, frame_id, timestamp_ms
        )
        result.face_latency_ms = round(face_lat, 2)
        result.head_pose = head_events
        result.eye_gaze = gaze_events

        # ── Mouth open detection — uses landmarks already computed above ───────
        # compute_mar_from_landmarks() reuses face_landmarks from process_frame
        # directly — no second FaceMesh inference call.
        if face_landmarks is not None:
            mar = self._face.compute_mar_from_landmarks(face_landmarks, bgr_frame.shape)
            
            # Cache MAR for YAMNet thread (lip-sync suppression)
            with self._mar_lock:
                self._cached_mar = mar
            
            if mar >= self._settings.mouth_open_mar_threshold:
                result.mouth_open.append(MouthOpenEvent(
                    mar=round(mar, 4),
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    latency_ms=round(face_lat, 2),
                ))
        else:
            # No face detected — reset cached MAR
            with self._mar_lock:
                self._cached_mar = 0.0

        # ── Extract face geometry for hand ownership ──────────────────────────
        face_cx: float | None = None
        face_width: float | None = None
        if face_landmarks is not None:
            lm = face_landmarks.landmark
            xs = [lm[i].x * w for i in range(len(lm))]
            face_cx    = float(np.mean(xs))
            face_width = float(max(xs) - min(xs))

        # ── Hands: external hand detection ────────────────────────────────────
        # Skip entirely when no face is detected — face_cx/face_width are None
        # so all three ownership signals return False anyway, meaning no external
        # hand can ever be flagged. Skipping saves 40–80 ms per frame.
        if face_landmarks is not None:
            hand_events, hand_lat, _ = self._pose.process_frame(
                bgr_frame,
                frame_id,
                timestamp_ms,
                face_cx=face_cx,
                face_width=face_width,
            )
            result.hand_latency_ms = round(hand_lat, 2)
            result.external_hands = hand_events

        # ── YOLO: submit frame to background thread, drain pending results ─────
        self._yolo.submit_frame(bgr_frame, frame_id, timestamp_ms)
        with self._yolo_lock:
            result.prohibited_objects = self._yolo_pending[:]
            self._yolo_pending.clear()

        # ── Audio: drain pending audio violations ─────────────────────────────
        with self._audio_lock:
            result.audio_anomalies = self._audio_pending[:]
            self._audio_pending.clear()

        # ── Identity Verification: drain pending verification events ──────────
        with self._verification_lock:
            result.identity_verification = self._verification_pending[:]
            self._verification_pending.clear()
            
        if result.identity_verification:
            verify_logger.info(
                f"[DRAIN_EVENTS] frame={frame_id} "
                f"verification_events_count={len(result.identity_verification)} "
                f"events={[{'is_match': e.is_match, 'similarity': e.similarity, 'confidence': e.confidence} for e in result.identity_verification]}"
            )

        # ── Trigger face verification on no-person/multiple-person detection ──
        # ONLY trigger when YOLO results are actually present for this frame
        # YOLO runs every 5-6 frames, so we only check when we have fresh results
        # COORDINATION: Skip if initial or periodic verification is running
        if self._user_id and self._db_session and result.prohibited_objects:
            # Check if other verifications are blocking
            if self.initial_verification_in_progress or self.periodic_verification_in_progress:
                verify_logger.info(
                    f"[VIOLATION_SKIPPED] frame={frame_id} user_id={self._user_id} "
                    f"reason=other_verification_in_progress "
                    f"initial_in_progress={self.initial_verification_in_progress} "
                    f"periodic_in_progress={self.periodic_verification_in_progress} "
                    f"note=violation_detection_still_sent_to_frontend"
                )
                # Note: Violation DETECTION (no_person/multiple_persons) is still sent to frontend
                # Only violation-triggered VERIFICATION is skipped
            else:
                # Count person detections in current frame's YOLO results
                person_detections = [
                    ev for ev in result.prohibited_objects
                    if ev.object_class.lower() == "person" and ev.confidence > 0
                ]
                no_person_markers = [
                    ev for ev in result.prohibited_objects
                    if ev.object_class.lower() == "person" and ev.confidence == 0.0
                ]

                # Current person count
                current_person_count = len(person_detections) if not no_person_markers else 0
                
                verify_logger.debug(
                    f"[PERSON_COUNT_CHECK] frame={frame_id} "
                    f"person_detections={len(person_detections)} "
                    f"no_person_markers={len(no_person_markers)} "
                    f"current_count={current_person_count} "
                    f"last_count={self._last_person_count} "
                    f"total_prohibited_objects={len(result.prohibited_objects)}"
                )

                # Trigger verification if:
                # 1. No person detected (confidence=0.0 marker present) - person left
                # 2. Multiple persons detected (≥2 person detections) - extra person appeared
                # 3. Person returned after absence (last_count=0, current_count=1) - verify it's same person
                #    BUT only if we've seen a previous state (last_count is not None)
                should_verify = False
                trigger_reason = ""
                
                if no_person_markers:
                    trigger_reason = "no_person_detected"
                    should_verify = True
                    verify_logger.info(
                        f"[ANOMALY_DETECTED] frame={frame_id} "
                        f"type=no_person markers_count={len(no_person_markers)}"
                    )
                elif len(person_detections) >= 2:
                    trigger_reason = "multiple_persons_detected"
                    should_verify = True
                    verify_logger.info(
                        f"[ANOMALY_DETECTED] frame={frame_id} "
                        f"type=multiple_persons count={len(person_detections)}"
                    )
                elif self._last_person_count is not None and self._last_person_count == 0 and current_person_count == 1:
                    # Person returned after being absent - verify it's the same person
                    # Only trigger if we have previous state (not first frame)
                    trigger_reason = "person_returned_after_absence"
                    should_verify = True
                    verify_logger.info(
                        f"[ANOMALY_DETECTED] frame={frame_id} "
                        f"type=person_returned last_count={self._last_person_count} current_count={current_person_count}"
                    )

                if should_verify:
                    verify_logger.info(
                        f"[TRIGGER_DECISION] frame={frame_id} "
                        f"reason={trigger_reason} should_verify=True"
                    )
                    # Pass triggered_by_violation=True to bypass "already matched" check
                    self.trigger_face_verification(
                        bgr_frame, frame_id, timestamp_ms, triggered_by_violation=True
                    )
                else:
                    verify_logger.debug(
                        f"[TRIGGER_DECISION] frame={frame_id} "
                        f"reason=normal_person_count should_verify=False"
                    )
                
                # Update last person count for next frame
                self._last_person_count = current_person_count

        return result

    def process_audio(self, pcm_chunk: np.ndarray) -> None:
        """Accumulate a PCM audio chunk and run YAMNet when buffer is full.

        Args:
            pcm_chunk: float32 numpy array at 16 kHz mono.
                       Expected size: 4096 samples (~0.256 s) per call.
                       Smaller or larger chunks are handled correctly.

        This method is safe to call from a separate thread.
        """
        with self._audio_buffer_lock:
            self._audio_buffer.append(pcm_chunk)
            total = sum(len(c) for c in self._audio_buffer)

            if total < self._audio_samples_needed:
                return   # not enough audio yet

            # Concatenate and slice exactly what YAMNet needs
            combined = np.concatenate(self._audio_buffer)
            segment  = combined[: self._audio_samples_needed]
            leftover = combined[self._audio_samples_needed :]
            self._audio_buffer = [leftover] if len(leftover) > 0 else []

        # Run YAMNet in a dedicated background thread so it never blocks the
        # frame-processing executor thread.  Results arrive via _audio_pending
        # and are drained on the next process_frame() call — same as before,
        # just without blocking the window latency clock.
        self._yamnet_executor.submit(self._run_yamnet, segment)

    def close(self) -> None:
        """Release all pipeline resources."""
        self._yolo.stop()
        self._yamnet_executor.shutdown(wait=False)
        self._face.close()
        self._pose.close()
        logger.info("InferenceEngine closed.")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _on_yolo_violation(self, events: list[ProhibitedObjectEvent]) -> None:
        """Callback from YOLO background thread."""
        with self._yolo_lock:
            self._yolo_pending.extend(events)

    def _run_yamnet(self, audio: np.ndarray) -> None:
        """Run YAMNet on one audio segment and buffer any violations."""
        if not _TF_AVAILABLE:
            logger.warning("TensorFlow not available — skipping YAMNet inference.")
            return
        if self._yamnet_model is None:
            logger.warning("YAMNet model not loaded — skipping audio inference.")
            return
        try:
            from proctoring_engine.settings import DEVICE_SOUND_CLASSES, SPEECH_CLASSES  # noqa: PLC0415

            # Sanitise — replace NaN/inf before peak normalisation
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

            # Peak normalisation
            amplified = audio / (np.max(np.abs(audio)) + 1e-9)

            # Convert to float32 numpy — YAMNet accepts numpy directly.
            # Do NOT use tf.cast here: it fails in thread-pool contexts.
            waveform = amplified.astype(np.float32)

            scores, _, _ = self._yamnet_model(waveform)
            scores_np  = scores.numpy()
            top_idx    = int(np.argmax(np.mean(scores_np, axis=0)))
            top_score  = float(np.mean(scores_np, axis=0)[top_idx])
            top_class  = self._get_yamnet_class(top_idx)

            timestamp_ms = int(time.time() * 1000)
            mar          = self.get_mar()
            event        = self._classify_audio(top_class, top_score, timestamp_ms, mar)

            if event:
                with self._audio_lock:
                    self._audio_pending.append(event)
                logger.info(f"Audio anomaly (API): {event.model_dump_json()}")

        except Exception as exc:
            logger.error(f"YAMNet inference error: {exc}", exc_info=True)

    def _classify_audio(
        self,
        top_class: str,
        confidence: float,
        timestamp_ms: int,
        mar: float,
    ) -> AudioAnomalyEvent | None:
        """Mirror the classification logic from AudioPipeline."""
        from proctoring_engine.models.violations import AudioSubtype  # noqa: PLC0415
        from proctoring_engine.settings import DEVICE_SOUND_CLASSES, SPEECH_CLASSES  # noqa: PLC0415

        if top_class in SPEECH_CLASSES:
            if mar >= self._settings.lip_sync_mar_threshold:
                return None   # candidate speaking — suppress
            return AudioAnomalyEvent(
                subtype=AudioSubtype.EXTERNAL_VOICE,
                yamnet_class=top_class,
                confidence=round(confidence, 4),
                timestamp_ms=timestamp_ms,
                segment_id=0,
                latency_ms=0.0,
            )

        if top_class in DEVICE_SOUND_CLASSES:
            return AudioAnomalyEvent(
                subtype=AudioSubtype.DEVICE_SOUND,
                yamnet_class=top_class,
                confidence=round(confidence, 4),
                timestamp_ms=timestamp_ms,
                segment_id=0,
                latency_ms=0.0,
            )

        return None

    @staticmethod
    def _get_yamnet_class(class_index: int) -> str:
        """Return the YAMNet class name for *class_index*."""
        try:
            import csv  # noqa: PLC0415
            # Use module-level _hub (already initialised in main thread)
            class_map_path = (
                _hub.resolve("https://tfhub.dev/google/yamnet/1")
                + "/assets/yamnet_class_map.csv"
            )
            with open(class_map_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i == class_index:
                        return row.get("display_name", f"class_{class_index}")
        except Exception:
            pass
        return f"class_{class_index}"
