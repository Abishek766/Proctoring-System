"""Diagnostics window — 2×2 panel view for the main camera loop only.

Shows each model in its own panel with:
  - Live annotated frame (landmarks / bounding boxes)
  - Per-model latency (ms)
  - Per-panel FPS counter
  - Fading violation labels (same sticky logic as OverlayService)

Layout:
  ┌─────────────────┬─────────────────┐
  │   FaceMesh      │  Hand Landmarks │
  │  (head + gaze)  │  (ext. hands)   │
  ├─────────────────┼─────────────────┤
  │     YOLO        │    YAMNet       │
  │ (objects/person)│  (audio class)  │
  └─────────────────┴─────────────────┘

Usage (called from ProctoringService every frame):
    diag = DiagnosticsWindow(panel_w=480, panel_h=360)
    diag.update(frame_data)   # FrameDiagData dataclass
    diag.show()               # cv2.imshow — non-blocking
    diag.close()              # cv2.destroyWindow
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

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
_RED    = (0,   0,   255)
_GREEN  = (0,   255, 0  )
_YELLOW = (0,   255, 255)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0  )
_ORANGE = (0,   165, 255)
_CYAN   = (255, 255, 0  )
_PINK   = (180, 105, 255)
_LIME   = (0,   255, 128)
_BLUE   = (255, 100, 0  )
_PURPLE = (200, 0,   200)

_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FPS_WINDOW = 30   # rolling window for FPS
_STICKY_FRAMES = 15  # frames before a faded label disappears

_WINDOW_NAME = "Proctoring Engine — Diagnostics"


@dataclass
class FrameDiagData:
    """All data needed to render one diagnostics frame.  Populated by ProctoringService."""

    # Raw BGR frame (already flipped, before any overlay)
    raw_frame: np.ndarray

    # FaceMesh
    face_landmarks: object | None          # mediapipe NormalizedLandmarkList
    head_events:    list[HeadPoseEvent]    = field(default_factory=list)
    gaze_events:    list[EyeGazeEvent]     = field(default_factory=list)
    face_latency_ms: float = 0.0

    # Hand Landmarks
    hand_landmarks_list: list = field(default_factory=list)
    hand_events:         list[ExternalHandEvent] = field(default_factory=list)
    hand_latency_ms:     float = 0.0

    # YOLO
    yolo_events:      list[ProhibitedObjectEvent] = field(default_factory=list)
    yolo_latency_ms:  float = 0.0   # latest result latency from the worker

    # YAMNet
    audio_event:      AudioAnomalyEvent | None = None
    yamnet_latency_ms: float = 0.0


class _PanelFPS:
    """Rolling FPS counter for a single panel."""

    def __init__(self) -> None:
        self._times: deque[float] = deque(maxlen=_FPS_WINDOW)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class _StickyLabels:
    """Fading sticky violation labels — same logic as OverlayService."""

    def __init__(self) -> None:
        # key → (label_text, neutral_frame_count)
        self._sticky: dict[str, tuple[str, int]] = {}

    def update(self, active_labels: list[str]) -> None:
        active = set(active_labels)
        for lbl in active_labels:
            self._sticky[lbl] = (lbl, 0)

        to_remove = []
        for key, (lbl, count) in self._sticky.items():
            if key not in active:
                new_count = count + 1
                if new_count >= _STICKY_FRAMES:
                    to_remove.append(key)
                else:
                    self._sticky[key] = (lbl, new_count)
        for key in to_remove:
            del self._sticky[key]

    def draw(self, panel: np.ndarray, start_y: int = 30) -> None:
        y = start_y
        for key, (lbl, count) in self._sticky.items():
            alpha = 1.0 - (count / _STICKY_FRAMES)
            colour = (
                int(_RED[0] * alpha),
                int(_RED[1] * alpha),
                int(255 * alpha),
            )
            cv2.putText(
                panel, lbl,
                (panel.shape[1] - 260, y),
                _FONT, 0.52, colour, 1, cv2.LINE_AA,
            )
            y += 22


class DiagnosticsWindow:
    """Builds and displays the 2×2 diagnostics panel window."""

    def __init__(self, panel_w: int = 480, panel_h: int = 360) -> None:
        self._pw = panel_w
        self._ph = panel_h

        # Per-panel FPS counters
        self._fps_face  = _PanelFPS()
        self._fps_hand  = _PanelFPS()
        self._fps_yolo  = _PanelFPS()
        self._fps_audio = _PanelFPS()

        # Per-panel sticky labels
        self._sticky_face  = _StickyLabels()
        self._sticky_hand  = _StickyLabels()
        self._sticky_yolo  = _StickyLabels()
        self._sticky_audio = _StickyLabels()

        # Latest YOLO latency — updated via update_yolo_latency() from the
        # YOLO worker callback (different thread) so we store it separately.
        self._latest_yolo_latency: float = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update_yolo_latency(self, latency_ms: float) -> None:
        """Called from the YOLO worker callback to record the latest latency."""
        self._latest_yolo_latency = latency_ms

    def update(self, data: FrameDiagData) -> None:
        """Build and display the 2×2 grid from *data*."""
        pw, ph = self._pw, self._ph

        panel_face  = self._build_face_panel(data, pw, ph)
        panel_hand  = self._build_hand_panel(data, pw, ph)
        panel_yolo  = self._build_yolo_panel(data, pw, ph)
        panel_audio = self._build_audio_panel(data, pw, ph)

        # Assemble 2×2 grid
        top    = np.hstack([panel_face, panel_hand])
        bottom = np.hstack([panel_yolo, panel_audio])
        grid   = np.vstack([top, bottom])

        cv2.imshow(_WINDOW_NAME, grid)

    def close(self) -> None:
        """Close the diagnostics window if it exists."""
        try:
            cv2.destroyWindow(_WINDOW_NAME)
        except cv2.error:
            pass  # Window was never created or already destroyed

    # ──────────────────────────────────────────────────────────────────────────
    # Panel builders
    # ──────────────────────────────────────────────────────────────────────────

    def _build_face_panel(self, data: FrameDiagData, pw: int, ph: int) -> np.ndarray:
        panel = self._base_panel(data.raw_frame, pw, ph)
        fps   = self._fps_face.tick()

        # Draw FaceMesh landmarks
        if data.face_landmarks is not None:
            self._draw_face_mesh(panel, data.face_landmarks)

        # Violation labels
        active: list[str] = []
        for ev in data.head_events:
            label = f"HEAD {ev.direction.value.upper()} {ev.angle_deg:.1f}°"
            active.append(label)
            cv2.putText(panel, label, (8, ph - 48), _FONT, 0.55, _RED, 1, cv2.LINE_AA)
        for ev in data.gaze_events:
            label = f"GAZE {ev.direction.value.upper()} ({ev.ratio:.2f})"
            active.append(label)
            cv2.putText(panel, label, (8, ph - 28), _FONT, 0.55, _ORANGE, 1, cv2.LINE_AA)

        self._sticky_face.update(active)
        self._sticky_face.draw(panel)
        self._draw_hud(panel, "FaceMesh", data.face_latency_ms, fps, _CYAN)
        return panel

    def _build_hand_panel(self, data: FrameDiagData, pw: int, ph: int) -> np.ndarray:
        panel = self._base_panel(data.raw_frame, pw, ph)
        fps   = self._fps_hand.tick()

        # Draw hand landmarks
        if data.hand_landmarks_list:
            ext_ids = {ev.hand_id for ev in data.hand_events}
            self._draw_hands(panel, data.hand_landmarks_list, ext_ids)

        active: list[str] = []
        for ev in data.hand_events:
            label = f"EXT HAND #{ev.hand_id} ({ev.distance_px:.0f}px)"
            active.append(label)
            cv2.putText(panel, label, (8, ph - 28), _FONT, 0.55, _RED, 1, cv2.LINE_AA)

        self._sticky_hand.update(active)
        self._sticky_hand.draw(panel)
        self._draw_hud(panel, "Hand Landmarks", data.hand_latency_ms, fps, _LIME)
        return panel

    def _build_yolo_panel(self, data: FrameDiagData, pw: int, ph: int) -> np.ndarray:
        panel = self._base_panel(data.raw_frame, pw, ph)
        fps   = self._fps_yolo.tick()

        # Scale bounding boxes to panel size
        orig_h, orig_w = data.raw_frame.shape[:2]
        sx = pw / orig_w
        sy = ph / orig_h

        active: list[str] = []
        for ev in data.yolo_events:
            if len(ev.bbox) == 4:
                x1 = int(ev.bbox[0] * sx)
                y1 = int(ev.bbox[1] * sy)
                x2 = int(ev.bbox[2] * sx)
                y2 = int(ev.bbox[3] * sy)
                cv2.rectangle(panel, (x1, y1), (x2, y2), _RED, 2)
                lbl = f"{ev.object_class} {ev.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, _FONT, 0.5, 1)
                cv2.rectangle(panel, (x1, y1 - th - 8), (x1 + tw + 4, y1), _RED, -1)
                cv2.putText(panel, lbl, (x1 + 2, y1 - 4), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)
            label = f"OBJECT: {ev.object_class.upper()} {ev.confidence:.2f}"
            active.append(label)

        self._sticky_yolo.update(active)
        self._sticky_yolo.draw(panel)
        lat = self._latest_yolo_latency if self._latest_yolo_latency > 0 else data.yolo_latency_ms
        self._draw_hud(panel, "YOLO (YOLOv8s)", lat, fps, _YELLOW)
        return panel

    def _build_audio_panel(self, data: FrameDiagData, pw: int, ph: int) -> np.ndarray:
        panel = self._base_panel(data.raw_frame, pw, ph)
        fps   = self._fps_audio.tick()

        # Audio panel: dark background with waveform-style decoration
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (pw, ph), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, panel, 0.45, 0, panel)

        active: list[str] = []
        if data.audio_event:
            ev = data.audio_event
            label_type  = f"TYPE:  {ev.subtype.value.upper()}"
            label_class = f"CLASS: {ev.yamnet_class}"
            label_conf  = f"CONF:  {ev.confidence:.3f}"
            for i, lbl in enumerate([label_type, label_class, label_conf]):
                cv2.putText(
                    panel, lbl,
                    (pw // 2 - 100, ph // 2 - 20 + i * 28),
                    _FONT, 0.6, _YELLOW, 1, cv2.LINE_AA,
                )
            active.append(f"AUDIO: {ev.subtype.value.upper()}")
        else:
            cv2.putText(
                panel, "No audio event",
                (pw // 2 - 80, ph // 2),
                _FONT, 0.55, (100, 100, 100), 1, cv2.LINE_AA,
            )

        self._sticky_audio.update(active)
        self._sticky_audio.draw(panel)
        self._draw_hud(panel, "YAMNet (Audio)", data.yamnet_latency_ms, fps, _PURPLE)
        return panel

    # ──────────────────────────────────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _base_panel(raw_frame: np.ndarray, pw: int, ph: int) -> np.ndarray:
        """Resize raw frame to panel dimensions."""
        return cv2.resize(raw_frame.copy(), (pw, ph), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _draw_hud(
        panel: np.ndarray,
        model_name: str,
        latency_ms: float,
        fps: float,
        colour: tuple[int, int, int],
    ) -> None:
        """Draw semi-transparent HUD bar at the top of a panel."""
        h, w = panel.shape[:2]
        hud = panel.copy()
        cv2.rectangle(hud, (0, 0), (w, 42), _BLACK, -1)
        cv2.addWeighted(hud, 0.5, panel, 0.5, 0, panel)

        # Model name
        cv2.putText(panel, model_name, (8, 16), _FONT, 0.52, colour, 1, cv2.LINE_AA)
        # Latency + FPS on the same line, right-aligned
        stats = f"{latency_ms:.1f}ms  |  {fps:.1f} FPS"
        (tw, _), _ = cv2.getTextSize(stats, _FONT, 0.48, 1)
        cv2.putText(panel, stats, (w - tw - 8, 16), _FONT, 0.48, _WHITE, 1, cv2.LINE_AA)

        # Thin colour bar under the HUD
        cv2.line(panel, (0, 20), (w, 20), colour, 1)

        # Panel border
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), colour, 1)

    @staticmethod
    def _draw_face_mesh(panel: np.ndarray, face_landmarks) -> None:
        """Draw FaceMesh tessellation + contours + iris onto *panel*."""
        import mediapipe as mp  # noqa: PLC0415
        drawing  = mp.solutions.drawing_utils
        face_mesh_sol = mp.solutions.face_mesh

        drawing.draw_landmarks(
            image=panel,
            landmark_list=face_landmarks,
            connections=face_mesh_sol.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing.DrawingSpec(
                color=(60, 60, 60), thickness=1, circle_radius=0
            ),
        )
        drawing.draw_landmarks(
            image=panel,
            landmark_list=face_landmarks,
            connections=face_mesh_sol.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing.DrawingSpec(
                color=_CYAN, thickness=1, circle_radius=1
            ),
            connection_drawing_spec=drawing.DrawingSpec(
                color=_CYAN, thickness=1, circle_radius=0
            ),
        )
        drawing.draw_landmarks(
            image=panel,
            landmark_list=face_landmarks,
            connections=face_mesh_sol.FACEMESH_IRISES,
            landmark_drawing_spec=drawing.DrawingSpec(
                color=_GREEN, thickness=1, circle_radius=2
            ),
            connection_drawing_spec=drawing.DrawingSpec(
                color=_GREEN, thickness=1, circle_radius=0
            ),
        )

    @staticmethod
    def _draw_hands(
        panel: np.ndarray,
        hand_landmarks_list: list,
        external_ids: set[int],
    ) -> None:
        """Draw hand skeletons — red for external, pink/orange for candidate."""
        import mediapipe as mp  # noqa: PLC0415
        drawing = mp.solutions.drawing_utils
        hands   = mp.solutions.hands
        h, w    = panel.shape[:2]

        for idx, hand_lm in enumerate(hand_landmarks_list):
            is_ext     = idx in external_ids
            dot_colour = _RED  if is_ext else _PINK
            ln_colour  = _RED  if is_ext else _ORANGE

            drawing.draw_landmarks(
                image=panel,
                landmark_list=hand_lm,
                connections=hands.HAND_CONNECTIONS,
                landmark_drawing_spec=drawing.DrawingSpec(
                    color=dot_colour, thickness=2, circle_radius=3
                ),
                connection_drawing_spec=drawing.DrawingSpec(
                    color=ln_colour, thickness=2, circle_radius=1
                ),
            )
            if is_ext:
                wrist = hand_lm.landmark[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(
                    panel, "EXT",
                    (wx, wy - 10),
                    _FONT, 0.5, _RED, 1, cv2.LINE_AA,
                )
