"""Main proctoring orchestration service.

Coordinates all detection pipelines, the benchmark service, and the overlay
renderer within the main video capture loop.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from proctoring_engine.core.face_pipeline import FacePipeline
from proctoring_engine.core.pose_pipeline import HandPipeline
from proctoring_engine.core.yolo_pipeline import YoloPipeline
from proctoring_engine.models.violations import AudioAnomalyEvent, ProhibitedObjectEvent
from proctoring_engine.services.benchmark_service import BenchmarkService
from proctoring_engine.services.diagnostics_window import DiagnosticsWindow, FrameDiagData
from proctoring_engine.services.overlay_service import OverlayService
from proctoring_engine.settings import Settings
from proctoring_engine.utils.exceptions import ERR_CAMERA_INIT_FAILED, CameraInitError
from proctoring_engine.utils.logger import get_logger

logger = get_logger("video_pipeline")


class ProctoringService:
    """Orchestrates the full video proctoring pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._benchmark = BenchmarkService(settings)
        self._overlay = OverlayService()
        self._diagnostics = DiagnosticsWindow(panel_w=480, panel_h=320)

        # Detection pipelines
        self._face = FacePipeline(settings)
        self._pose = HandPipeline(settings)
        self._yolo = YoloPipeline(
            settings,
            on_violation=self._on_yolo_violation,
        )

        # Latest audio anomaly for overlay (set by AudioPipeline callback)
        self._latest_audio_event: AudioAnomalyEvent | None = None

        # Latest frame for MAR computation (shared with AudioPipeline)
        self._latest_frame: np.ndarray | None = None

        # Pending YOLO events from background thread — drained each frame
        self._yolo_pending: list[ProhibitedObjectEvent] = []
        self._yolo_lock = threading.Lock()

        # Latest YOLO latency from worker thread (for diagnostics panel)
        self._latest_yolo_latency: float = 0.0

        # Track last known person count from YOLO (to avoid false NO_PERSON during queue gaps)
        self._last_person_count: int = 1  # Assume 1 person initially (candidate present)

        # Latest YAMNet latency from audio thread (for diagnostics panel)
        self._latest_yamnet_latency: float = 0.0

        self._cap: cv2.VideoCapture | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_mar(self) -> float:
        """Return the current mouth aspect ratio (used by AudioPipeline)."""
        if self._latest_frame is None:
            return 0.0
        return self._face.get_mouth_aspect_ratio(self._latest_frame)

    def on_audio_violation(self, event: AudioAnomalyEvent) -> None:
        """Callback invoked by AudioPipeline when an audio anomaly is detected."""
        self._latest_audio_event = event
        self._latest_yamnet_latency = event.latency_ms
        self._benchmark.log_audio_anomaly(event)
        logger.info(f"Audio anomaly: {event.model_dump_json()}")

    def run(self) -> None:
        """Open the camera and run the main capture loop until user presses 'q'."""
        self._cap = self._open_camera()
        frame_id = 0

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera.")
                    break

                # Flip horizontally so left/right directions match the candidate's
                # actual movement (mirror mode — same as a selfie camera).
                frame = cv2.flip(frame, 1)
                self._latest_frame = frame
                timestamp_ms = int(time.time() * 1000)
                frame_id += 1
                self._benchmark.increment_video_frame()

                frame = self._process_frame(frame, frame_id, timestamp_ms)

                cv2.imshow("Video Proctoring Engine", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested stop.")
                    break
        finally:
            self._shutdown()

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _open_camera(self) -> cv2.VideoCapture:
        """Open the camera device. Raises CameraInitError on failure."""
        cap = cv2.VideoCapture(self._settings.camera_index)
        if not cap.isOpened():
            raise CameraInitError(
                f"Cannot open camera at index {self._settings.camera_index}",
                error_code=ERR_CAMERA_INIT_FAILED,
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.camera_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info(
            f"Camera opened: index={self._settings.camera_index}, "
            f"{self._settings.camera_width}x{self._settings.camera_height} @ 30 FPS"
        )
        return cap

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
    ) -> np.ndarray:
        """Run all synchronous pipelines on *frame* and render overlays."""
        h, w = frame.shape[:2]
        active_keys: list[str] = []

        # ── FaceMesh (REQ-001 + REQ-002) ──────────────────────────────────────
        head_events, gaze_events, face_lat, face_landmarks = self._face.process_frame(
            frame, frame_id, timestamp_ms
        )
        self._benchmark.record_face_latency(face_lat)

        # Keep a clean copy of the frame for hand inference.
        # draw_face_mesh draws the green tessellation onto `frame` in-place;
        # passing that annotated frame to MediaPipe Hands degrades detection.
        clean_frame = frame.copy()

        # Extract face centroid and width for hand ownership detection
        face_cx: float | None = None
        face_width: float | None = None
        if face_landmarks is not None:
            lm = face_landmarks.landmark
            xs = [lm[i].x * w for i in range(len(lm))]
            face_cx    = float(np.mean(xs))
            face_width = float(max(xs) - min(xs))
            frame = self._overlay.draw_face_mesh(frame, face_landmarks, w, h)

            # ── Mouth open detection — reuses landmarks, no extra inference ───
            mar = self._face.compute_mar_from_landmarks(face_landmarks, frame.shape)
            if mar >= self._settings.mouth_open_mar_threshold:
                active_keys.append("MOUTH_OPEN")
                # Draw MAR value on frame for debugging
                cv2.putText(
                    frame,
                    f"MOUTH_OPEN  MAR:{mar:.2f}",
                    (10, frame.shape[0] - 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA,
                )

        for ev in head_events:
            ev.latency_ms = round(face_lat, 2)
            self._benchmark.log_head_pose(ev)
            frame = self._overlay.draw_head_pose(frame, ev)
            active_keys.append(f"HEAD:{ev.direction.value.upper()}")
            logger.info(f"Head pose violation: {ev.model_dump_json()}")

        for ev in gaze_events:
            ev.latency_ms = round(face_lat, 2)
            self._benchmark.log_eye_gaze(ev)
            frame = self._overlay.draw_eye_gaze(frame, ev)
            active_keys.append(f"GAZE:{ev.direction.value.upper()}")
            logger.info(f"Eye gaze violation: {ev.model_dump_json()}")

        # ── Hands (REQ-003) — arm-origin + handedness detection ───────────────
        # Use clean_frame (no overlay drawn) so MediaPipe Hands gets an
        # unmodified image — the tessellation overlay degrades hand detection.
        hand_events, pose_lat, hand_landmarks_list = self._pose.process_frame(
            clean_frame, frame_id, timestamp_ms,
            face_cx=face_cx,
            face_width=face_width,
        )
        self._benchmark.record_pose_latency(pose_lat)

        if hand_landmarks_list:
            external_ids = {ev.hand_id for ev in hand_events}
            frame = self._overlay.draw_hands(frame, hand_landmarks_list, external_ids)

        for ev in hand_events:
            ev.latency_ms = round(pose_lat, 2)
            self._benchmark.log_external_hand(ev)
            frame = self._overlay.draw_external_hand(frame, ev)
            active_keys.append("EXT_HAND")
            logger.info(f"External hand violation: {ev.model_dump_json()}")

        # ── YOLO (REQ-004) — drain pending results from background thread ────────
        self._yolo.submit_frame(frame, frame_id, timestamp_ms)
        with self._yolo_lock:
            pending_yolo = self._yolo_pending[:]
            self._yolo_pending.clear()

        # ── Person detection logic (multiple_person / no_person) ──────────────
        # Separate person detections from other prohibited objects
        # Special case: confidence=0.0 is a marker from YOLO meaning "no person detected"
        person_detections = [ev for ev in pending_yolo if ev.object_class.lower() == "person" and ev.confidence > 0]
        no_person_marker = any(ev.object_class.lower() == "person" and ev.confidence == 0.0 for ev in pending_yolo)
        other_objects = [ev for ev in pending_yolo if ev.object_class.lower() != "person"]

        person_count = len(person_detections)

        # Only update last_person_count when YOLO actually returns results
        # (pending_yolo not empty means YOLO finished processing a frame)
        if pending_yolo:
            if no_person_marker:
                self._last_person_count = 0  # YOLO confirmed 0 persons
            else:
                self._last_person_count = person_count

        # Debug logging
        logger.info(
            f"YOLO_PERSONS_RESPONSE frame={frame_id} "
            f"pending_yolo_count={len(pending_yolo)} "
            f"person_count={person_count} "
            f"last_person_count={self._last_person_count} "
            f"other_objects_count={len(other_objects)} "
            f"all_classes=[{', '.join(ev.object_class for ev in pending_yolo)}]"
        )

        # Decision based on last known person count (not current frame's empty queue)
        if self._last_person_count == 0:
            # NO_PERSON violation - YOLO confirmed 0 persons
            active_keys.append("NO_PERSON")
            frame = self._overlay.draw_no_person_violation(frame)
        elif self._last_person_count >= 2:
            # MULTIPLE_PERSONS violation - YOLO confirmed 2+ persons
            active_keys.append("MULTIPLE_PERSON")
            frame = self._overlay.draw_multiple_persons_violation(frame, self._last_person_count)

        # Draw bounding boxes only when we have fresh results this frame
        if pending_yolo:
            if person_count >= 2:
                # Draw all person bboxes for multiple persons
                for ev in person_detections:
                    frame = self._overlay.draw_prohibited_object(frame, ev)
            elif person_count == 1:
                # Draw single person bbox
                for ev in person_detections:
                    frame = self._overlay.draw_prohibited_object(frame, ev)

        # Draw other prohibited objects (not persons)
        for ev in other_objects:
            frame = self._overlay.draw_prohibited_object(frame, ev)
            active_keys.append(f"OBJECT:{ev.object_class.upper()}")

        # ── Audio overlay (REQ-005) ────────────────────────────────────────────
        audio_event_for_diag = self._latest_audio_event
        if self._latest_audio_event:
            frame = self._overlay.draw_audio_anomaly(frame, self._latest_audio_event)
            active_keys.append(f"AUDIO:{self._latest_audio_event.subtype.value.upper()}")
            self._latest_audio_event = None

        # ── Overlay HUD + sticky violation labels ──────────────────────────────
        self._overlay.update_latencies({"FaceMesh": face_lat, "Hands": pose_lat})
        self._overlay.update_violations(active_keys)
        # Advance incident cooldown counters with this frame's active keys
        self._benchmark.tick_frame(set(active_keys))
        frame = self._overlay.render(frame)

        # ── Update diagnostics window (second window with 4 panels) ────────────
        diag_data = FrameDiagData(
            raw_frame=clean_frame,
            face_landmarks=face_landmarks,
            head_events=head_events,
            gaze_events=gaze_events,
            face_latency_ms=face_lat,
            hand_landmarks_list=hand_landmarks_list,
            hand_events=hand_events,
            hand_latency_ms=pose_lat,
            yolo_events=pending_yolo,
            yolo_latency_ms=self._latest_yolo_latency,
            audio_event=audio_event_for_diag,
            yamnet_latency_ms=self._latest_yamnet_latency,
        )
        self._diagnostics.update(diag_data)

        return frame

    def _on_yolo_violation(self, events: list[ProhibitedObjectEvent]) -> None:
        """Callback from YOLO background thread — buffer for main thread drawing."""
        with self._yolo_lock:
            self._yolo_pending.extend(events)
        # Update diagnostics with latest YOLO latency
        if events:
            self._latest_yolo_latency = events[0].latency_ms if hasattr(events[0], 'latency_ms') else 0.0
            self._diagnostics.update_yolo_latency(self._latest_yolo_latency)
        for ev in events:
            self._benchmark.log_prohibited_object(ev)
            logger.info(f"Prohibited object: {ev.model_dump_json()}")

    def _shutdown(self) -> None:
        """Release all resources and print the session summary."""
        logger.info("Shutting down proctoring service...")
        self._yolo.stop()
        self._face.close()
        self._pose.close()
        self._diagnostics.close()
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self._benchmark.print_summary()
        self._benchmark.close()
