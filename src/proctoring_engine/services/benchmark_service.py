"""Benchmark metrics collection and reporting service (REQ-006).

Violation counting uses INCIDENT tracking:
  - An incident starts when a violation type transitions from inactive → active.
  - Subsequent frames where the same violation is still active do NOT increment
    the count — they are part of the same incident.
  - The incident ends when the violation type is absent for INCIDENT_COOLDOWN_FRAMES
    consecutive frames. Only then can a new incident begin.

This means 302 head-pose frames = 1 incident (or a small number if the candidate
repeatedly turns their head), not 302 counts.
"""

from __future__ import annotations

import json
import time

from proctoring_engine.models.metrics import BenchmarkReport, PipelineMetrics
from proctoring_engine.models.violations import (
    AudioAnomalyEvent,
    ExternalHandEvent,
    EyeGazeEvent,
    HeadPoseEvent,
    ProhibitedObjectEvent,
)
from proctoring_engine.settings import Settings
from proctoring_engine.utils.logger import get_logger

logger = get_logger("benchmark")

_PIPELINE_NAMES = ["face_mesh", "pose_hands", "yolo", "audio"]

# Frames a violation must be absent before a new incident can be counted.
# At ~5 FPS this is ~2 seconds of neutral before the next turn counts as new.
INCIDENT_COOLDOWN_FRAMES = 10


class BenchmarkService:
    """Accumulates per-frame metrics and writes structured JSON-lines logs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._start_time = time.perf_counter()
        self._video_frame_count = 0
        self._audio_segment_count = 0

        # Per-pipeline latency metrics
        self._metrics: dict[str, PipelineMetrics] = {
            name: PipelineMetrics(pipeline_name=name) for name in _PIPELINE_NAMES
        }

        # Incident counts — incremented only on transition inactive → active
        self._violation_counts: dict[str, int] = {}

        # Incident state tracking per violation type:
        #   True  = violation is currently active (incident in progress)
        #   int   = cooldown counter (frames since last active)
        # State machine per key:
        #   absent → "inactive" (not in dict)
        #   active → _active_violations contains the key
        #   cooling → _cooldown[key] counts up to INCIDENT_COOLDOWN_FRAMES
        self._active_violations: set[str] = set()
        self._cooldown: dict[str, int] = {}

        # Open log files
        self._video_log = open(settings.video_violations_log, "a", encoding="utf-8")  # noqa: SIM115
        self._audio_log = open(settings.audio_violations_log, "a", encoding="utf-8")  # noqa: SIM115

        logger.info(
            f"BenchmarkService started. "
            f"Video log: {settings.video_violations_log}, "
            f"Audio log: {settings.audio_violations_log}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Latency recording
    # ──────────────────────────────────────────────────────────────────────────

    def record_face_latency(self, latency_ms: float) -> None:
        self._metrics["face_mesh"].record_latency(latency_ms)

    def record_pose_latency(self, latency_ms: float) -> None:
        self._metrics["pose_hands"].record_latency(latency_ms)

    def record_yolo_latency(self, latency_ms: float) -> None:
        self._metrics["yolo"].record_latency(latency_ms)

    def record_audio_latency(self, latency_ms: float) -> None:
        self._metrics["audio"].record_latency(latency_ms)

    def increment_video_frame(self) -> None:
        self._video_frame_count += 1

    def increment_audio_segment(self) -> None:
        self._audio_segment_count += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Violation logging — called every frame a violation fires
    # ──────────────────────────────────────────────────────────────────────────

    def log_head_pose(self, event: HeadPoseEvent) -> None:
        self._write_video_log(event.model_dump())
        self._record_incident(f"head_pose_{event.direction.value}")

    def log_eye_gaze(self, event: EyeGazeEvent) -> None:
        self._write_video_log(event.model_dump())
        self._record_incident(f"eye_gaze_{event.direction.value}")

    def log_external_hand(self, event: ExternalHandEvent) -> None:
        self._write_video_log(event.model_dump())
        self._record_incident("external_hand")

    def log_prohibited_object(self, event: ProhibitedObjectEvent) -> None:
        self._write_video_log(event.model_dump())
        self._record_incident(f"prohibited_{event.object_class}")

    def log_audio_anomaly(self, event: AudioAnomalyEvent) -> None:
        self._write_audio_log(event.model_dump())
        self._record_incident(f"audio_{event.subtype.value}")

    def tick_frame(self, active_violation_keys: set[str]) -> None:
        """Call once per frame with the set of currently active violation keys.

        Advances the cooldown counters for violations that are no longer active.
        """
        # Age out violations that are no longer firing
        cooled_out = []
        for key in list(self._cooldown.keys()):
            if key in active_violation_keys:
                # Still active — reset cooldown
                self._cooldown[key] = 0
            else:
                self._cooldown[key] += 1
                if self._cooldown[key] >= INCIDENT_COOLDOWN_FRAMES:
                    # Fully cooled — now truly inactive
                    cooled_out.append(key)
                    self._active_violations.discard(key)

        for key in cooled_out:
            del self._cooldown[key]

    # ──────────────────────────────────────────────────────────────────────────
    # Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def build_report(self) -> BenchmarkReport:
        duration = time.perf_counter() - self._start_time
        return BenchmarkReport(
            total_video_frames=self._video_frame_count,
            total_audio_segments=self._audio_segment_count,
            pipeline_metrics=self._metrics,
            violation_counts=self._violation_counts,
            session_duration_s=round(duration, 2),
        )

    def print_summary(self) -> None:
        report = self.build_report()
        print("\n" + "=" * 70)
        print("  VIDEO PROCTORING ENGINE — SESSION SUMMARY")
        print("=" * 70)
        print(f"  Duration          : {report.session_duration_s:.1f} s")
        print(f"  Total frames      : {report.total_video_frames}")
        print(f"  Total audio segs  : {report.total_audio_segments}")
        print(f"  Overall FPS       : {report.overall_fps:.1f}")
        print()
        print("  Pipeline Latencies (mean / p95 / p99 ms):")
        for name, m in report.pipeline_metrics.items():
            print(
                f"    {name:<12} : {m.mean_latency_ms:6.1f} / "
                f"{m.p95_latency_ms:6.1f} / {m.p99_latency_ms:6.1f}"
            )
        print()
        print("  Violation Incidents (each = one distinct event, not frame count):")
        if not report.violation_counts:
            print("    None detected.")
        else:
            for vtype, count in sorted(report.violation_counts.items()):
                print(f"    {vtype:<35} : {count}")
        print("=" * 70 + "\n")

    def close(self) -> None:
        self._video_log.flush()
        self._video_log.close()
        self._audio_log.flush()
        self._audio_log.close()
        logger.info("BenchmarkService closed.")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _record_incident(self, violation_key: str) -> None:
        """Increment count only when transitioning from fully-inactive → active.

        State machine:
          not in _active_violations AND not in _cooldown
            → NEW incident: increment count, add to active + start cooldown
          in _active_violations (incident ongoing, whether firing or cooling)
            → do nothing (same incident)
        """
        if violation_key not in self._active_violations:
            # Genuinely new incident — key is fully inactive
            self._violation_counts[violation_key] = (
                self._violation_counts.get(violation_key, 0) + 1
            )
            self._active_violations.add(violation_key)
        # Reset cooldown counter since it's active this frame
        self._cooldown[violation_key] = 0

    def _write_video_log(self, data: dict) -> None:
        self._video_log.write(json.dumps(data) + "\n")
        if self._video_frame_count % self._settings.video_log_flush_interval == 0:
            self._video_log.flush()

    def _write_audio_log(self, data: dict) -> None:
        self._audio_log.write(json.dumps(data) + "\n")
        if self._audio_segment_count % self._settings.audio_log_flush_interval == 0:
            self._audio_log.flush()
