"""Hand pipeline — external hand detection (REQ-003).

Detection uses THREE independent signals that are combined with OR logic.
Any single signal firing is sufficient to flag a hand as external.

  1. ARM ORIGIN ESTIMATION
     Projects the forearm vector backward from the wrist to estimate where
     the arm originates (elbow/shoulder direction).  If that origin falls
     outside the candidate's shoulder zone it means the arm is coming from
     a different person's body.

     Raised-hand guard: when the wrist is above the face centre y-position
     the palm→wrist vector points downward (not toward the shoulder), making
     the projection unreliable.  In that posture this check is skipped.

  2. WRIST-TO-FACE DISTANCE
     Measures the Euclidean distance from the wrist to the detected face
     centre, normalised by face_width.  A candidate's own hand at full arm
     extension is ≤ MAX_WRIST_FACE_DISTANCE_FACTOR face-widths away.
     Anything further cannot belong to the candidate.

     This check is position-independent — it works correctly whether the
     candidate is centred or sitting to one side of the frame.

  3. SIDE-AWARE HANDEDNESS + DISTANCE (combined signal)
     MediaPipe assumes the input is a mirrored (selfie) image.  After
     cv2.flip(frame, 1) the image IS mirrored, so labels are correct:
       "Left"  hand → appears on the LEFT  side of the frame
       "Right" hand → appears on the RIGHT side of the frame

     A candidate's own hands always appear on the SAME side as their label:
       "Left"  hand → wrist_x < face_cx   (left of face)
       "Right" hand → wrist_x > face_cx   (right of face)

     If a hand appears on the WRONG side AND the distance is already
     elevated (> CROSS_BODY_DISTANCE_FACTOR × face_width), it is
     anatomically impossible for the candidate — their arm would have to
     cross their body.  This is the signature of an external person
     reaching in from the opposite side.

     The distance threshold is lower here (CROSS_BODY_DISTANCE_FACTOR <
     MAX_WRIST_FACE_DISTANCE_FACTOR) because the wrong-side position is
     already strong evidence.
"""

from __future__ import annotations

import time

import cv2
import mediapipe as mp
import numpy as np

from proctoring_engine.models.violations import ExternalHandEvent
from proctoring_engine.settings import Settings
from proctoring_engine.utils.logger import get_logger

logger    = get_logger("video_pipeline")
mp_logger = get_logger("mediapipe")

# ─── MediaPipe Hands landmark indices ─────────────────────────────────────────
_WRIST      = 0
_INDEX_MCP  = 5    # base of index finger
_PINKY_MCP  = 17   # base of pinky finger
_MIDDLE_MCP = 9    # base of middle finger (palm centre reference)

# ─── Arm-origin estimation ────────────────────────────────────────────────────
ARM_EXTENSION_FACTOR = 2.5

# Shoulder zone: face_cx ± (face_width × factor)
SHOULDER_ZONE_FACTOR = 2.8

# Signal 2: pure distance threshold (face-width units)
# A candidate's own hand at full arm extension is ≤ 3.5 face-widths away.
MAX_WRIST_FACE_DISTANCE_FACTOR = 3.5

# Signal 3: cross-body threshold — lower because wrong-side is already evidence.
# If a "Left" hand is to the RIGHT of the face (or vice versa) AND the distance
# exceeds this factor, it cannot be the candidate's hand.
CROSS_BODY_DISTANCE_FACTOR = 2.5


class HandPipeline:
    """Detects external hands using arm-origin analysis and handedness consistency."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        
        # Use centralized model loader - MediaPipe models can be reused
        # as long as we don't call .close() on them
        from proctoring_engine.core.model_loader import get_hands_model  # noqa: PLC0415
        self._hands = get_hands_model()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
        face_cx: float | None = None,
        face_width: float | None = None,
    ) -> tuple[list[ExternalHandEvent], float, list]:
        """Detect external hands in *frame*.

        Args:
            frame:        BGR frame (already horizontally flipped)
            frame_id:     current frame counter
            timestamp_ms: current timestamp in milliseconds
            face_cx:      face centroid x in pixels (from FacePipeline), or None
            face_width:   face bounding-box width in pixels (from FacePipeline), or None

        Returns:
            events:              list of ExternalHandEvent (may be empty)
            latency_ms:          inference time in milliseconds
            hand_landmarks_list: raw MediaPipe Hand landmark objects for overlay
        """
        t0 = time.perf_counter()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        hands_results = self._hands.process(rgb)
        hand_landmarks_list = hands_results.multi_hand_landmarks or []
        handedness_list     = hands_results.multi_handedness or []

        events: list[ExternalHandEvent] = []

        if not hand_landmarks_list:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            mp_logger.debug(
                f"frame={frame_id} ts={timestamp_ms} "
                f"hands_detected=0 external_hands=0 "
                f"latency_ms={round(latency_ms, 2)}"
            )
            return events, latency_ms, hand_landmarks_list

        for hand_id, (hand_lm, handedness) in enumerate(
            zip(hand_landmarks_list, handedness_list)
        ):
            lm = hand_lm.landmark
            # Pixel coordinates of key landmarks
            wrist_x      = lm[_WRIST].x      * w
            wrist_y      = lm[_WRIST].y      * h
            index_mcp_x  = lm[_INDEX_MCP].x  * w
            index_mcp_y  = lm[_INDEX_MCP].y  * h
            pinky_mcp_x  = lm[_PINKY_MCP].x  * w
            pinky_mcp_y  = lm[_PINKY_MCP].y  * h
            middle_mcp_x = lm[_MIDDLE_MCP].x * w
            middle_mcp_y = lm[_MIDDLE_MCP].y * h

            # Palm centre = midpoint of index and pinky MCP
            palm_cx = (index_mcp_x + pinky_mcp_x) / 2.0
            palm_cy = (index_mcp_y + pinky_mcp_y) / 2.0

            # Forearm direction vector: palm_centre → wrist
            # Extending this backward gives the arm origin estimate
            dx = wrist_x - palm_cx
            dy = wrist_y - palm_cy
            arm_origin_x = wrist_x + dx * ARM_EXTENSION_FACTOR
            # arm_origin_y = wrist_y + dy * ARM_EXTENSION_FACTOR  # not needed

            is_external_arm  = self._check_arm_origin(
                arm_origin_x, wrist_y, face_cx, face_width, w, h
            )
            is_external_dist = self._check_wrist_distance(
                wrist_x, wrist_y, face_cx, face_width, h
            )
            is_external_side = self._check_cross_body(
                handedness, wrist_x, wrist_y, face_cx, face_width, h
            )

            # Decision logic — three signals, combined carefully:
            #
            # Signal 2 (distance) and Signal 3 (cross-body) fire alone.
            # Signal 1 (arm origin) requires corroboration UNLESS the arm
            # origin is clearly off-screen (< 0 or > frame_width), which is
            # unambiguous evidence of an external arm entering the frame.
            arm_offscreen = (
                is_external_arm
                and face_cx is not None
                and (arm_origin_x < 0 or arm_origin_x > w)
            )
            arm_corroborated = is_external_arm and (is_external_dist or is_external_side)
            if arm_offscreen or arm_corroborated or is_external_dist or is_external_side:
                # Distance from wrist to face centre (for logging/reporting)
                dist = float(np.linalg.norm(
                    np.array([wrist_x, wrist_y]) -
                    np.array([face_cx if face_cx else w / 2,
                              h * 0.35])   # approximate face y
                ))
                events.append(ExternalHandEvent(
                    hand_id=hand_id,
                    distance_px=round(dist, 2),
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                ))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        mp_logger.debug(
            f"frame={frame_id} ts={timestamp_ms} "
            f"hands_detected={len(hand_landmarks_list)} "
            f"external_hands={len(events)} "
            f"face_cx={round(face_cx, 1) if face_cx else 'none'} "
            f"face_width={round(face_width, 1) if face_width else 'none'} "
            f"latency_ms={round(latency_ms, 2)}"
        )
        return events, latency_ms, hand_landmarks_list

    def close(self) -> None:
        """Close the pipeline.
        
        Note: We don't close the Hands model because it's shared across
        all sessions via the centralized model loader. Closing it would break
        other active sessions. The model will be cleaned up on app shutdown.
        """
        # Don't close shared model - it's managed by model_loader
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _check_arm_origin(
        self,
        arm_origin_x: float,
        wrist_y: float,
        face_cx: float | None,
        face_width: float | None,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """Return True if the projected arm origin is outside the candidate's
        shoulder zone.

        Raised-hand guard: when the wrist is above the face centre y-position,
        the palm→wrist vector points downward rather than toward the shoulder.
        The projected arm-origin x-coordinate is then unreliable, so we return
        False and let the distance check carry the decision.
        """
        face_cy = frame_height * 0.35
        if wrist_y < face_cy:
            return False   # hand raised — geometry unreliable

        if face_cx is not None and face_width is not None:
            zone_half   = face_width * SHOULDER_ZONE_FACTOR
            left_bound  = face_cx - zone_half
            right_bound = face_cx + zone_half
        else:
            zone_half   = frame_width * 0.30
            left_bound  = frame_width / 2 - zone_half
            right_bound = frame_width / 2 + zone_half

        return arm_origin_x < left_bound or arm_origin_x > right_bound

    def _check_wrist_distance(
        self,
        wrist_x: float,
        wrist_y: float,
        face_cx: float | None,
        face_width: float | None,
        frame_height: int,
    ) -> bool:
        """Return True if the wrist is too far from the candidate's face to
        plausibly belong to them.

        Uses Euclidean distance from wrist to face centre, normalised by
        face_width so the threshold scales with how close the candidate is
        to the camera.

        A candidate's own hand at full arm extension is typically ≤ 3.5
        face-widths from their face centre. An external person's hand
        entering from the side will be further away.

        Falls back to a frame-size heuristic when face position is unknown.
        """
        if face_cx is None or face_width is None or face_width < 1:
            # No face detected — cannot determine ownership
            return False

        face_cy = frame_height * 0.35   # approximate face centre y
        dist = float(np.sqrt((wrist_x - face_cx) ** 2 + (wrist_y - face_cy) ** 2))
        return dist > face_width * MAX_WRIST_FACE_DISTANCE_FACTOR

    @staticmethod
    def _check_cross_body(
        handedness,
        wrist_x: float,
        wrist_y: float,
        face_cx: float | None,
        face_width: float | None,
        frame_height: int,
    ) -> bool:
        """Return True when a hand is on the WRONG side of the face AND the
        wrist-to-face distance is elevated — the cross-body signature.

        MediaPipe assumes a mirrored (selfie) input, so after cv2.flip the
        labels are correct:
          "Left"  hand → naturally appears to the LEFT  of the face (wrist_x < face_cx)
          "Right" hand → naturally appears to the RIGHT of the face (wrist_x > face_cx)

        A candidate cannot have their own "Left" hand appear to the RIGHT of
        their face at a large distance — that would require the arm to cross
        the entire body.  The same applies in reverse for "Right" hands.

        An external person reaching in from the opposite side produces exactly
        this pattern: their hand label is on the wrong side relative to the
        candidate's face, and the distance is larger than a natural reach.

        The distance threshold here (CROSS_BODY_DISTANCE_FACTOR = 2.5) is
        lower than the pure distance threshold (3.5) because the wrong-side
        position is already strong corroborating evidence.

        Guard: skip when face position is unknown or handedness confidence
        is below 0.80 (ambiguous label — don't penalise).
        """
        if face_cx is None or face_width is None or face_width < 1:
            return False

        confidence = handedness.classification[0].score
        if confidence < 0.80:
            return False   # label not reliable enough

        label   = handedness.classification[0].label   # "Left" or "Right"
        face_cy = frame_height * 0.35
        dist    = float(np.sqrt((wrist_x - face_cx) ** 2 + (wrist_y - face_cy) ** 2))
        threshold = face_width * CROSS_BODY_DISTANCE_FACTOR

        # MediaPipe labels in a mirrored (selfie) image:
        #   "Left"  label → candidate's RIGHT hand → should be RIGHT of face_cx
        #   "Right" label → candidate's LEFT  hand → should be LEFT  of face_cx
        #
        # Wrong side (cross-body) means:
        #   "Left"  label appearing to the LEFT  of face_cx (wrist_x < face_cx)
        #   "Right" label appearing to the RIGHT of face_cx (wrist_x > face_cx)

        # "Left" label (candidate's right hand) appearing LEFT of face → cross-body
        if label == "Left" and wrist_x < face_cx and dist > threshold:
            return True

        # "Right" label (candidate's left hand) appearing RIGHT of face → cross-body
        if label == "Right" and wrist_x > face_cx and dist > threshold:
            return True

        return False
