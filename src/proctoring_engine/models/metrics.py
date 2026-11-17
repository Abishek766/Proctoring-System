"""Pydantic models for benchmark metrics collection and reporting (REQ-006)."""

from pydantic import BaseModel, ConfigDict


class PipelineMetrics(BaseModel):
    """Per-pipeline latency and throughput metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pipeline_name: str
    total_frames: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = []

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def mean_latency_ms(self) -> float:
        """Mean inference latency in milliseconds."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def p95_latency_ms(self) -> float:
        """95th-percentile inference latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        """99th-percentile inference latency in milliseconds."""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * precision * recall / (precision + recall)."""
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom > 0 else 0.0

    def record_latency(self, latency_ms: float) -> None:
        """Append a latency sample and increment frame counter."""
        self.latencies.append(latency_ms)
        self.total_frames += 1
        self.total_latency_ms += latency_ms


class BenchmarkReport(BaseModel):
    """End-of-session aggregate benchmark report."""

    total_video_frames: int = 0
    total_audio_segments: int = 0
    pipeline_metrics: dict[str, PipelineMetrics] = {}
    violation_counts: dict[str, int] = {}
    session_duration_s: float = 0.0

    @property
    def overall_fps(self) -> float:
        """Approximate FPS based on session duration."""
        if self.session_duration_s <= 0:
            return 0.0
        return self.total_video_frames / self.session_duration_s
