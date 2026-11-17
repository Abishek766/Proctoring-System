"""YOLO object detection pipeline — prohibited object detection (REQ-004).

Runs YOLOv8 nano in a separate thread with a frame queue to avoid blocking
the main capture loop.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from proctoring_engine.models.violations import ProhibitedObjectEvent
from proctoring_engine.settings import Settings
from proctoring_engine.utils.exceptions import ERR_MODEL_INIT_FAILED, ModelInitError
from proctoring_engine.utils.logger import get_logger

logger = get_logger("video_pipeline")
yolo_logger = get_logger("yolo")  # → yolo_detections.log

_FRAME_QUEUE_MAX = 2


class YoloPipeline:
    """Runs YOLOv8 inference in a background thread.

    Frames are submitted via :meth:`submit_frame`. Only every Nth frame is
    actually submitted based on ``settings.yolo_frame_skip`` — this keeps the
    YOLO queue from building up a backlog on CPU-only hardware while still
    providing adequate coverage (objects like phones and laptops are visible
    for seconds, not single frames).

    All results that YOLO produces are always delivered via the
    *on_violation* callback — nothing is ever discarded.
    """

    def __init__(
        self,
        settings: Settings,
        on_violation: Callable[[list[ProhibitedObjectEvent]], None],
    ) -> None:
        self._settings = settings
        self._on_violation = on_violation
        self._frame_queue: queue.Queue[tuple[np.ndarray, int, int] | None] = queue.Queue(
            maxsize=_FRAME_QUEUE_MAX
        )
        # Counts how many frames have been offered to submit_frame so we can
        # apply the skip interval without relying on the caller's frame_id
        # (which may not start at 0 or be contiguous across sessions).
        self._submit_counter: int = 0
        
        # Use centralized model loader instead of loading model here
        from proctoring_engine.core.model_loader import get_yolo_model  # noqa: PLC0415
        self._model = get_yolo_model()
        
        self._thread = threading.Thread(target=self._worker, daemon=True, name="yolo-worker")
        self._thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def submit_frame(self, frame: np.ndarray, frame_id: int, timestamp_ms: int) -> None:
        """Submit a frame for YOLO inference (non-blocking; drops if queue is full).

        Only every ``yolo_frame_skip``-th frame is forwarded to the queue.
        At 30 FPS with skip=6 this gives YOLO ~5 FPS — enough to reliably
        detect objects that are visible for seconds, while keeping the queue
        from growing faster than YOLO can drain it on CPU.

        When the queue is full the frame is dropped silently.  This is safe
        because the next eligible frame will be submitted on the next cycle.
        """
        self._submit_counter += 1
        skip = max(1, self._settings.yolo_frame_skip)
        if self._submit_counter % skip != 0:
            return  # not this frame's turn — skip without queuing

        try:
            self._frame_queue.put_nowait((frame.copy(), frame_id, timestamp_ms))
        except queue.Full:
            pass  # queue still busy — drop this frame, next eligible will retry

    def stop(self) -> None:
        """Signal the worker thread to stop and wait for it to finish."""
        self._frame_queue.put(None)
        self._thread.join(timeout=5.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model(self):
        """Load YOLOv8 .pt model, auto-downloading if missing."""
        try:
            from ultralytics import YOLO  # noqa: PLC0415

            model_path = Path(self._settings.yolo_model)

            if not model_path.exists():
                model_name = model_path.stem  # e.g. "yolov8s"
                pt_name = f"{model_name}.pt"
                logger.info(f"Model '{model_path}' not found. Downloading '{pt_name}' from Ultralytics hub ...")
                import shutil  # noqa: PLC0415
                YOLO(pt_name)  # triggers auto-download to cwd
                if Path(pt_name).exists():
                    shutil.move(pt_name, str(model_path))
                    logger.info(f"Downloaded to {model_path}")

            model = YOLO(str(model_path))
            logger.info(f"YOLOv8 model loaded: {model_path}")
            prohibited = self._settings.prohibited_objects_list
            matched = [v for v in model.names.values() if v.lower() in prohibited]
            logger.info(f"YOLO will detect: {matched}")
            print(f"[YOLO] Prohibited classes matched in COCO: {matched}")
            return model
        except Exception as exc:
            raise ModelInitError(
                f"Failed to load YOLO model '{self._settings.yolo_model}': {exc}",
                error_code=ERR_MODEL_INIT_FAILED,
            ) from exc

    def _worker(self) -> None:
        """Background thread: dequeue frames, run inference, fire callbacks."""
        while True:
            item = self._frame_queue.get()
            if item is None:
                break

            frame, frame_id, timestamp_ms = item
            try:
                events, latency = self._infer(frame, frame_id, timestamp_ms)
                if events:
                    self._on_violation(events)
            except Exception as exc:
                import traceback  # noqa: PLC0415
                logger.error(f"YOLO inference error on frame {frame_id}: {exc}")
                traceback.print_exc()  # also print to console so it's visible

    def _infer(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_ms: int,
    ) -> tuple[list[ProhibitedObjectEvent], float]:
        """Run YOLOv8 inference and return violation events + latency."""
        t0 = time.perf_counter()

        results = self._model(
            frame,
            imgsz=self._settings.yolo_input_size,
            conf=self._settings.yolo_confidence_threshold,
            verbose=False,
        )

        events: list[ProhibitedObjectEvent] = []
        prohibited = self._settings.prohibited_objects_list
        default_conf = self._settings.yolo_confidence_threshold
        person_conf  = self._settings.yolo_person_confidence_threshold

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                cls_name   = result.names[cls_id].lower()
                confidence = float(box.conf[0])

                if cls_name not in prohibited:
                    continue

                # Apply a stricter confidence threshold for the person class
                # to reduce false positives from bags, chairs, and background people.
                required_conf = person_conf if cls_name == "person" else default_conf
                if confidence < required_conf:
                    continue

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                events.append(
                    ProhibitedObjectEvent(
                        object_class=cls_name,
                        confidence=round(confidence, 4),
                        bbox=[x1, y1, x2, y2],
                        frame_id=frame_id,
                        timestamp_ms=timestamp_ms,
                        latency_ms=0.0,  # will be set below after inference completes
                    )
                )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Populate latency_ms for all events
        for ev in events:
            ev.latency_ms = round(latency_ms, 2)

        # ── Check if YOLO detected any persons (for NO_PERSON tracking) ────────
        # Count all person detections (including low-confidence ones filtered out)
        person_count_total = 0
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    cls_name = result.names[int(box.cls[0])].lower()
                    if cls_name == "person":
                        person_count_total += 1

        # If YOLO ran but found 0 persons, add a marker event with confidence=0
        # This tells the main thread that YOLO actively checked and found nothing
        if person_count_total == 0:
            events.append(
                ProhibitedObjectEvent(
                    object_class="person",
                    confidence=0.0,  # marker: 0 confidence = no person detected
                    bbox=[],
                    frame_id=frame_id,
                    timestamp_ms=timestamp_ms,
                    latency_ms=round(latency_ms, 2),
                )
            )

        # ── YOLO per-inference log ─────────────────────────────────────────────
        all_detections = []
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    cls_name = result.names[int(box.cls[0])].lower()
                    conf = round(float(box.conf[0]), 4)
                    all_detections.append(f"{cls_name}:{conf}")

        # Log non-person prohibited objects at INFO level for visibility
        non_person_events = [e for e in events if e.object_class != "person"]
        if non_person_events:
            for e in non_person_events:
                yolo_logger.info(
                    f"[PROHIBITED OBJECT] frame={frame_id} ts={timestamp_ms} "
                    f"class={e.object_class} confidence={e.confidence} "
                    f"bbox={e.bbox}"
                )
                logger.info(
                    f"[YOLO] Prohibited object detected: {e.object_class} "
                    f"conf={e.confidence} bbox={e.bbox} frame={frame_id}"
                )

        yolo_logger.debug(
            f"frame={frame_id} ts={timestamp_ms} "
            f"all_detections=[{', '.join(all_detections) if all_detections else 'none'}] "
            f"prohibited_violations={len(events)} "
            f"prohibited=[{', '.join(e.object_class for e in events)}] "
            f"latency_ms={round(latency_ms, 2)}"
        )

        return events, latency_ms
