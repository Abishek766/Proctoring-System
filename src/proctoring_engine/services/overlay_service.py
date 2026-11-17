"""OpenCV overlay rendering service.

Draws real-time violation labels, FPS, per-pipeline latency, MediaPipe
landmarks, and bounding boxes on the live video frame.

Sticky violation state:
  Once a violation fires it stays on screen until the head/gaze returns to
  neutral for STICKY_CLEAR_FRAMES consecutive frames, preventing blinking.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from proctoring_engine.models.violations import (
    AudioAnomalyEvent,
    ExternalHandEvent,
    EyeGazeEvent,
    HeadPoseEvent,
    ProhibitedObjectEvent,
)

# ─── Colours (BGR) ─────────────────────────────────────────────────────────────
_RED     = (0, 0, 255)
_GREEN   = (0, 255, 0)
_YELLOW  = (0, 255, 255)
_WHITE   = (255, 255, 255)
_BLACK   = (0, 0, 0)
_ORANGE  = (0, 165, 255)
_CYAN    = (255, 255, 0)
_PINK    = (180, 105, 255)
_LIME    = (0, 255, 128)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FPS_WINDOW = 30   # frames for rolling FPS average

# How many consecutive neutral frames before a sticky label is cleared
STICKY_CLEAR_FRAMES = 10


class OverlayService:
    """Renders live benchmark overlays onto OpenCV frames."""

    def __init__(self) -> None:
        self._frame_times: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._latencies: dict[str, float] = {}

        # ── Sticky violation state ─────────────────────────────────────────────
        # Maps violation key → (display label, neutral-frame counter)
        # The label stays visible until neutral_count reaches STICKY_CLEAR_FRAMES.
        self._sticky: dict[str, tuple[str, int]] = {}

        # Keys reported as active this frame (set by update_violations)
        self._current_keys: set[str] = set()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update_latencies(self, latencies: dict[str, float]) -> None:
        """Update per-pipeline latency values for display."""
        self._latencies.update(latencies)

    def update_violations(self, active_keys: list[str]) -> None:
        """Called every frame with the list of currently active violation keys.

        Keys that are present → reset their neutral counter (violation ongoing).
        Keys that are absent  → increment neutral counter; clear after N frames.
        New keys              → add to sticky state immediately.
        """
        self._current_keys = set(active_keys)

        # Add new violations
        for key in active_keys:
            if key not in self._sticky:
                self._sticky[key] = (key, 0)
            else:
                label, _ = self._sticky[key]
                self._sticky[key] = (label, 0)  # reset neutral counter

        # Age out violations that are no longer active
        to_remove = []
        for key, (label, neutral_count) in self._sticky.items():
            if key not in self._current_keys:
                new_count = neutral_count + 1
                if new_count >= STICKY_CLEAR_FRAMES:
                    to_remove.append(key)
                else:
                    self._sticky[key] = (label, new_count)
        for key in to_remove:
            del self._sticky[key]

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw HUD (FPS, latency, sticky violation labels) onto *frame*."""
        self._frame_times.append(time.perf_counter())
        fps = self._compute_fps()

        # Semi-transparent HUD background (top-left)
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (320, 30 + 20 * (1 + len(self._latencies))), _BLACK, -1)
        cv2.addWeighted(hud, 0.45, frame, 0.55, 0, frame)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22), _FONT, 0.65, _GREEN, 2)

        # Per-pipeline latencies
        y = 44
        for name, lat in self._latencies.items():
            cv2.putText(frame, f"{name}: {lat:.1f}ms", (10, y), _FONT, 0.48, _WHITE, 1)
            y += 20

        # Sticky violation labels (top-right)
        y_viol = 30
        for key, (label, neutral_count) in self._sticky.items():
            # Fade colour as it ages toward clearing
            alpha = 1.0 - (neutral_count / STICKY_CLEAR_FRAMES)
            colour = (
                int(_RED[0] * alpha),
                int(_RED[1] * alpha),
                int(255 * alpha),
            )
            cv2.putText(
                frame, label,
                (frame.shape[1] - 360, y_viol),
                _FONT, 0.65, colour, 2,
            )
            y_viol += 30

        return frame

    # ── Per-pipeline annotation methods ───────────────────────────────────────

    def draw_face_mesh(self, frame: np.ndarray, face_landmarks, w: int, h: int) -> np.ndarray:
        """Draw MediaPipe FaceMesh tessellation + contours + landmark dots."""
        mp_drawing  = _get_mp_drawing()
        mp_face_mesh = _get_mp_face_mesh()

        # Tessellation (grey mesh)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(80, 80, 80), thickness=1, circle_radius=0
            ),
        )
        # Contours (cyan outline)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=_CYAN, thickness=1, circle_radius=1
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=_CYAN, thickness=1, circle_radius=0
            ),
        )
        # Iris circles (green)
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=_GREEN, thickness=1, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=_GREEN, thickness=1, circle_radius=0
            ),
        )
        return frame

    def draw_pose(self, frame: np.ndarray, pose_landmarks) -> np.ndarray:
        """Draw MediaPipe Pose skeleton on *frame*."""
        mp_drawing = _get_mp_drawing()
        mp_pose = _get_mp_pose()

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=_LIME, thickness=2, circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=_WHITE, thickness=2, circle_radius=1
            ),
        )
        return frame

    def draw_hands(
        self,
        frame: np.ndarray,
        hand_landmarks_list: list,
        external_hand_ids: set[int] | None = None,
    ) -> np.ndarray:
        """Draw MediaPipe Hands landmarks on *frame*.

        Args:
            hand_landmarks_list: raw MediaPipe hand landmark objects.
            external_hand_ids:   set of hand indices flagged as external.
                                 Those hands get a red skeleton + label above
                                 the wrist. Candidate hands stay orange/pink.
        """
        mp_drawing = _get_mp_drawing()
        mp_hands   = _get_mp_hands()
        ext_ids    = external_hand_ids or set()
        h, w       = frame.shape[:2]

        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            is_external = idx in ext_ids

            # Skeleton colour: red for external, orange/pink for candidate
            dot_colour  = _RED    if is_external else _PINK
            line_colour = _RED    if is_external else _ORANGE

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=dot_colour, thickness=2, circle_radius=4
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=line_colour, thickness=2, circle_radius=2
                ),
            )

            # Draw "EXTERNAL HAND" label above the wrist for flagged hands
            if is_external:
                wrist = hand_landmarks.landmark[0]
                wx = int(wrist.x * w)
                wy = int(wrist.y * h)

                label = "EXTERNAL HAND"
                (tw, th), _ = cv2.getTextSize(label, _FONT, 0.65, 2)

                # Background pill so the label is readable on any background
                pad = 4
                bx1 = wx - pad
                by1 = wy - th - 18
                bx2 = wx + tw + pad
                by2 = wy - 8
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), _RED, -1)
                cv2.putText(
                    frame, label,
                    (wx, wy - 12),
                    _FONT, 0.65, _WHITE, 2, cv2.LINE_AA,
                )

        return frame

    def draw_head_pose(self, frame: np.ndarray, event: HeadPoseEvent) -> np.ndarray:
        """Draw head pose violation label at bottom of frame."""
        label = f"HEAD: {event.direction.value.upper()}  {event.angle_deg:.1f}deg"
        cv2.putText(frame, label, (10, frame.shape[0] - 60), _FONT, 0.75, _RED, 2)
        return frame

    def draw_eye_gaze(self, frame: np.ndarray, event: EyeGazeEvent) -> np.ndarray:
        """Draw eye gaze violation label at bottom of frame."""
        label = f"GAZE: {event.direction.value.upper()}  ({event.ratio:.2f})"
        cv2.putText(frame, label, (10, frame.shape[0] - 30), _FONT, 0.75, _ORANGE, 2)
        return frame

    def draw_external_hand(self, frame: np.ndarray, event: ExternalHandEvent) -> np.ndarray:
        """Draw external hand violation label."""
        label = f"EXTERNAL HAND #{event.hand_id}  ({event.distance_px:.0f}px)"
        cv2.putText(frame, label, (10, frame.shape[0] - 90), _FONT, 0.7, _RED, 2)
        return frame

    def draw_prohibited_object(self, frame: np.ndarray, event: ProhibitedObjectEvent) -> np.ndarray:
        """Draw YOLO bounding box + class label + confidence."""
        if len(event.bbox) == 4:
            x1, y1, x2, y2 = event.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), _RED, 2)
            label = f"{event.object_class}  {event.confidence:.2f}"
            # Background pill behind text
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), _RED, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 6), _FONT, 0.6, _WHITE, 2)
        return frame

    def draw_audio_anomaly(self, frame: np.ndarray, event: AudioAnomalyEvent) -> np.ndarray:
        """Draw audio anomaly notification."""
        label = f"AUDIO: {event.subtype.value.upper()}  ({event.yamnet_class})"
        cv2.putText(frame, label, (10, frame.shape[0] - 120), _FONT, 0.65, _YELLOW, 2)
        return frame

    def draw_multiple_persons_violation(self, frame: np.ndarray, person_count: int) -> np.ndarray:
        """Draw multiple persons violation warning.
        
        Args:
            frame: The video frame to draw on
            person_count: Number of persons detected
        """
        # Main violation message
        label = f"MULTIPLE_PERSON: {person_count} persons detected"
        message = "Other person detected on the screen, only candidate should be present in the room"
        
        # Draw main label at bottom
        cv2.putText(frame, label, (10, frame.shape[0] - 180), _FONT, 0.75, _RED, 2)
        
        # Draw detailed message below
        # Split message into multiple lines if too long
        max_width = frame.shape[1] - 20
        words = message.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = " ".join(current_line + [word])
            (tw, _), _ = cv2.getTextSize(test_line, _FONT, 0.55, 1)
            if tw > max_width and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(" ".join(current_line))
        
        # Draw each line
        y_offset = frame.shape[0] - 210
        for line in lines:
            cv2.putText(frame, line, (10, y_offset), _FONT, 0.55, _ORANGE, 1)
            y_offset -= 20
        
        return frame

    def draw_no_person_violation(self, frame: np.ndarray) -> np.ndarray:
        """Draw no person violation warning."""
        label = "NO_PERSON: Candidate not visible"
        message = "Candidate face should be visible"
        
        # Draw main label at bottom
        cv2.putText(frame, label, (10, frame.shape[0] - 180), _FONT, 0.75, _RED, 2)
        
        # Draw detailed message below
        cv2.putText(frame, message, (10, frame.shape[0] - 210), _FONT, 0.55, _ORANGE, 1)
        
        return frame

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0


# ─── Lazy MediaPipe accessors (avoid re-importing every frame) ─────────────────

def _get_mp_drawing():
    import mediapipe as mp  # noqa: PLC0415
    return mp.solutions.drawing_utils

def _get_mp_face_mesh():
    import mediapipe as mp  # noqa: PLC0415
    return mp.solutions.face_mesh

def _get_mp_pose():
    import mediapipe as mp  # noqa: PLC0415
    return mp.solutions.pose

def _get_mp_hands():
    import mediapipe as mp  # noqa: PLC0415
    return mp.solutions.hands
