"""Face landmark pipeline — head pose (REQ-001) and eye gaze (REQ-002) detection.

Uses MediaPipe FaceMesh with refine_landmarks=True for iris tracking.
Both detectors share a single FaceMesh inference call per frame.

Head pose approach (after cv2.flip horizontal mirror):
  We use the nose-tip x-position relative to the face bounding-box centre as a
  reliable left/right indicator — this is always correct regardless of the
  rotation-matrix sign convention, which varies by OpenCV version and 3D model.

  Yaw magnitude is still computed via PnP for the angle value in the event.

Gaze approach (after cv2.flip horizontal mirror):
  The frame is flipped so the right eye in the image IS the candidate's right eye.
  We use the LEFT eye (landmark indices 474-477) for gaze because after flipping
  it is the eye that is most visible when the candidate looks left/right.
  h_ratio < 0.35 → iris near inner corner → candidate looking LEFT
  h_ratio > 0.65 → iris near outer corner → candidate looking RIGHT
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

from proctoring_engine.models.violations import (
    EyeGazeEvent,
    GazeDirection,
    HeadPoseDirection,
    HeadPoseEvent,
)
from proctoring_engine.settings import Settings
from proctoring_engine.utils.logger import get_logger

logger = get_logger("video_pipeline")
mp_logger = get_logger("mediapipe")  # → mediapipe_detections.log

# ─── 3-D face model (standard 6-point, units in mm) ───────────────────────────
_FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0  ],   # Nose tip          (1)
    [0.0,   -63.6, -12.5  ],   # Chin              (152)
    [-43.3,  32.7, -26.0  ],   # Left eye corner   (263)
    [43.3,   32.7, -26.0  ],   # Right eye corner  (33)
    [-28.9, -28.9, -24.1  ],   # Left mouth corner (287)
    [28.9,  -28.9, -24.1  ],   # Right mouth corner(57)
], dtype=np.float64)

_FACE_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]

# ─── Iris indices (refined FaceMesh 468-477) ──────────────────────────────────
# After horizontal flip: index 469-472 = right iris in image = candidate's right eye
_RIGHT_IRIS_IDX = [469, 470, 471, 472]

# Nose tip and face-width landmarks for head direction check
_NOSE_TIP    = 1
_LEFT_CHEEK  = 234   # leftmost face point
_RIGHT_CHEEK = 454   # rightmost face point


class FacePipeline:
    """Runs FaceMesh once per frame and produces head-pose and eye-gaze events."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        
        # Use centralized model loader instead of creating new instance
        from proctoring_engine.core.model_loader import get_facemesh_model  # noqa: PLC0415
        self._face_mesh = get_facemesh_model()
        
        self._head_pose_buffer: deque[HeadPoseDirection | None] = deque(
            maxlen=settings.head_pose_consecutive_frames
        )
        self._gaze_buffer: deque[GazeDirection | None] = deque(
            maxlen=settings.eye_gaze_consecutive_frames
        )
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
    ) -> tuple[list[HeadPoseEvent], list[EyeGazeEvent], float, object | None]:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        head_events: list[HeadPoseEvent] = []
        gaze_events: list[EyeGazeEvent] = []
        face_landmarks = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            if self._camera_matrix is None:
                self._camera_matrix = self._build_camera_matrix(w, h)

            head_event = self._detect_head_pose(face_landmarks, w, h, frame_id, timestamp_ms)
            if head_event:
                head_events.append(head_event)

            gaze_event = self._detect_eye_gaze(face_landmarks, w, h, frame_id, timestamp_ms)
            if gaze_event:
                gaze_events.append(gaze_event)

            # ── MediaPipe per-frame log ────────────────────────────────────────
            mp_logger.debug(
                f"frame={frame_id} ts={timestamp_ms} "
                f"face=detected "
                f"head_violation={head_event.direction.value if head_event else 'none'} "
                f"head_angle={head_event.angle_deg if head_event else 0.0} "
                f"gaze_violation={gaze_event.direction.value if gaze_event else 'none'} "
                f"gaze_ratio={gaze_event.ratio if gaze_event else 0.0} "
                f"latency_ms={round((time.perf_counter() - t0) * 1000.0, 2)}"
            )
        else:
            self._head_pose_buffer.clear()
            self._gaze_buffer.clear()
            mp_logger.debug(f"frame={frame_id} ts={timestamp_ms} face=not_detected")

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return head_events, gaze_events, latency_ms, face_landmarks

    def get_mouth_aspect_ratio(self, frame: np.ndarray) -> float:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return 0.0
        return self._compute_mar(
            results.multi_face_landmarks[0].landmark, frame.shape
        )

    def compute_mar_from_landmarks(self, face_landmarks, frame_shape: tuple) -> float:
        """Compute MAR directly from already-available landmarks — no extra inference.

        Call this inside process_frame to avoid a second FaceMesh call.
        """
        return self._compute_mar(face_landmarks.landmark, frame_shape)

    @staticmethod
    def _compute_mar(lm, frame_shape: tuple) -> float:
        """Compute Mouth Aspect Ratio using a 3-point vertical average.

        Uses three vertical measurement pairs across the mouth opening,
        normalised by mouth width — same approach as the Eye Aspect Ratio (EAR).

        Landmark pairs (MediaPipe FaceMesh 468-point model):
          Vertical pair 1: 82  (upper lip left-mid)  ↔ 87  (lower lip left-mid)
          Vertical pair 2: 13  (upper inner lip)     ↔ 14  (lower inner lip)
          Vertical pair 3: 312 (upper lip right-mid) ↔ 317 (lower lip right-mid)
          Horizontal:      61  (left corner)         ↔ 291 (right corner)

        MAR = mean(vertical_1, vertical_2, vertical_3) / horizontal_width

        Typical values:
          Mouth closed:        0.04 – 0.18
          Mouth slightly open: 0.20 – 0.30
          Mouth wide open:     0.35 – 0.60

        Previous implementation used only landmarks 0/17 (outer lip surfaces)
        which include natural lip thickness and give 0.30–0.38 even when
        the mouth is fully closed, making threshold-based detection unreliable.
        """
        h, w = frame_shape[:2]

        def _dist(a: int, b: int) -> float:
            return float(np.linalg.norm(
                np.array([lm[a].x * w, lm[a].y * h]) -
                np.array([lm[b].x * w, lm[b].y * h])
            ))

        v1 = _dist(82, 87)    # left-mid vertical
        v2 = _dist(13, 14)    # centre inner vertical
        v3 = _dist(312, 317)  # right-mid vertical
        horizontal = _dist(61, 291) + 1e-9

        return (v1 + v2 + v3) / (3.0 * horizontal)


    def close(self) -> None:
        """Close the pipeline.
        
        Note: We don't close the FaceMesh model because it's shared across
        all sessions via the centralized model loader. Closing it would break
        other active sessions. The model will be cleaned up on app shutdown.
        """
        # Don't close shared model - it's managed by model_loader
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_camera_matrix(width: int, height: int) -> np.ndarray:
        f = width
        return np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1]], dtype=np.float64)

    def _detect_head_pose(
        self, face_landmarks, width: int, height: int, frame_id: int, timestamp_ms: int
    ) -> HeadPoseEvent | None:
        lm = face_landmarks.landmark

        # ── Direction via nose-tip offset from face centre ─────────────────────
        # This is sign-convention-free and always correct after the horizontal flip.
        nose_x    = lm[_NOSE_TIP].x    * width
        l_cheek_x = lm[_LEFT_CHEEK].x  * width
        r_cheek_x = lm[_RIGHT_CHEEK].x * width
        face_cx   = (l_cheek_x + r_cheek_x) / 2.0
        face_w    = abs(r_cheek_x - l_cheek_x) + 1e-9

        # Normalised offset: positive = nose right of centre = head turned RIGHT
        nose_offset = (nose_x - face_cx) / face_w

        # ── Pitch via nose-tip y relative to face height ──────────────────────
        # Nose naturally sits at ~58% of face height from forehead to chin.
        # We compare actual nose position to this neutral reference.
        # In image coords: smaller y = higher on screen.
        # nose_y < neutral_y → nose is ABOVE neutral → head tilted UP
        # nose_y > neutral_y → nose is BELOW neutral → head tilted DOWN
        nose_y     = lm[_NOSE_TIP].y * height
        chin_y     = lm[152].y       * height
        forehead_y = lm[10].y        * height
        face_h     = abs(chin_y - forehead_y) + 1e-9
        neutral_nose_y = forehead_y + face_h * 0.58   # natural nose position
        # Normalised: positive = nose above neutral = UP, negative = DOWN
        pitch_offset = (neutral_nose_y - nose_y) / face_h

        # ── Yaw magnitude via PnP (for the angle value only) ──────────────────
        image_pts = np.array(
            [[lm[i].x * width, lm[i].y * height] for i in _FACE_LANDMARK_INDICES],
            dtype=np.float64,
        )
        yaw_deg = 0.0
        pitch_deg = 0.0
        success, rvec, _ = cv2.solvePnP(
            _FACE_3D_MODEL, image_pts,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            rmat, _ = cv2.Rodrigues(rvec)
            pitch_deg = float(np.degrees(
                np.arctan2(-rmat[2, 0], np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2))
            ))
            yaw_deg = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))

        s = self._settings
        direction: HeadPoseDirection | None = None

        # Use nose offset for left/right (reliable, sign-convention-free)
        # Use pitch offset for up/down — threshold 0.12 catches clear head tilts
        # without triggering on minor natural seated posture variation.
        if abs(nose_offset) > 0.12:
            direction = HeadPoseDirection.RIGHT if nose_offset > 0 else HeadPoseDirection.LEFT
        elif abs(pitch_offset) > 0.12:
            direction = HeadPoseDirection.UP if pitch_offset > 0 else HeadPoseDirection.DOWN

        self._head_pose_buffer.append(direction)

        if direction is None:
            return None

        if (
            len(self._head_pose_buffer) == s.head_pose_consecutive_frames
            and all(d == direction for d in self._head_pose_buffer)
        ):
            angle = min(abs(yaw_deg), 89.9) if direction in (
                HeadPoseDirection.LEFT, HeadPoseDirection.RIGHT
            ) else min(abs(pitch_deg), 89.9)
            return HeadPoseEvent(
                direction=direction,
                angle_deg=round(angle, 1),
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )
        return None

    def _detect_eye_gaze(
        self, face_landmarks, width: int, height: int, frame_id: int, timestamp_ms: int
    ) -> EyeGazeEvent | None:
        lm = face_landmarks.landmark

        # Use the RIGHT iris (469-472) in the flipped frame.
        # After cv2.flip(frame, 1), landmark indices stay the same but x-coords
        # are mirrored. The right iris in the flipped image = candidate's right eye.
        #
        # Eye corners for the right eye in MediaPipe:
        #   33  = right eye LEFT  corner (in image coords = nose side after flip)
        #   133 = right eye RIGHT corner (in image coords = ear side after flip)
        #
        # h_ratio = (iris_x - left_corner_x) / (right_corner_x - left_corner_x)
        #   → 0.0 = iris at left corner of image  = candidate looking LEFT
        #   → 1.0 = iris at right corner of image = candidate looking RIGHT
        #   → 0.5 = iris centred = looking straight
        #
        # No inversion needed — after flip, image-left = candidate-left.

        iris_x = float(np.mean([lm[i].x for i in _RIGHT_IRIS_IDX])) * width
        iris_y = float(np.mean([lm[i].y for i in _RIGHT_IRIS_IDX])) * height

        # Right eye corners (image coordinates after flip)
        eye_left_x  = lm[33].x  * width    # left  side of right eye in image
        eye_right_x = lm[133].x * width    # right side of right eye in image
        eye_top_y   = lm[159].y * height
        eye_bot_y   = lm[145].y * height

        h_ratio = (iris_x - eye_left_x)  / (eye_right_x - eye_left_x + 1e-9)
        v_ratio = (iris_y - eye_top_y)   / (eye_bot_y   - eye_top_y  + 1e-9)

        direction = self._classify_gaze_direction(h_ratio, v_ratio)
        self._gaze_buffer.append(direction)

        if direction is None:
            return None

        s = self._settings
        if (
            len(self._gaze_buffer) == s.eye_gaze_consecutive_frames
            and all(d == direction for d in self._gaze_buffer)
        ):
            ratio = h_ratio if direction in (GazeDirection.LEFT, GazeDirection.RIGHT) else v_ratio
            return EyeGazeEvent(
                direction=direction,
                ratio=round(ratio, 4),
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
            )
        return None

    def _classify_gaze_direction(self, h_ratio: float, v_ratio: float) -> GazeDirection | None:
        """Classify gaze direction from normalised iris ratios.

        h_ratio: 0 = iris at left edge of eye, 1 = iris at right edge
          < 0.35 → GAZE LEFT   (iris near left corner)
          > 0.65 → GAZE RIGHT  (iris near right corner)

        v_ratio: 0 = iris at top of eye, 1 = iris at bottom
          In image coords, when you look UP the iris moves toward the TOP of the
          eye opening → v_ratio decreases → small v_ratio = looking DOWN
          (the eyelid covers the top of the iris when looking up, but the iris
          centre moves toward the bottom of the visible eye opening).

          Empirically with MediaPipe:
            Look DOWN → iris moves to bottom of eye → v_ratio > 0.65 → DOWN
            Look UP   → iris moves to top of eye   → v_ratio < 0.35 → UP
        """
        s = self._settings
        if h_ratio < s.eye_gaze_left_threshold:
            return GazeDirection.LEFT
        if h_ratio > s.eye_gaze_right_threshold:
            return GazeDirection.RIGHT
        if v_ratio > s.eye_gaze_down_threshold:
            return GazeDirection.DOWN
        if v_ratio < s.eye_gaze_up_threshold:
            return GazeDirection.UP
        return None
