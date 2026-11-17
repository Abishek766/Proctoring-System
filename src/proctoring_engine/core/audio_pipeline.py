"""Audio classification pipeline — background audio anomaly detection (REQ-005).

Captures microphone audio at 16 kHz in a background thread, peak-normalises
the signal, and classifies it with YAMNet. Lip-sync correlation suppresses
false positives for the candidate's own speech.

Audio chunk saving (always on for debugging):
  Every audio segment is saved as a WAV file in audio_chunks/ at the project
  root. File name format:
    seg_{id:04d}_{timestamp_ms}_{top_class_sanitised}.wav
  Listen to these files to verify what the microphone is actually capturing.

Thread-safety fix:
  get_mar() calls face_mesh.process() which is NOT thread-safe. After the
  main loop closes FaceMesh, the audio thread crashes with '_graph is None'.
  Fix: wrap get_mar() in a try/except — if it fails, assume MAR=0.0 (lips
  still) which is the conservative choice (may flag candidate speech as
  external, but won't crash).
"""

from __future__ import annotations

import queue
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from proctoring_engine.models.violations import AudioAnomalyEvent, AudioSubtype
from proctoring_engine.settings import DEVICE_SOUND_CLASSES, SPEECH_CLASSES, Settings
from proctoring_engine.utils.exceptions import ERR_AUDIO_THREAD_FAILED, AudioThreadError
from proctoring_engine.utils.logger import get_logger

logger        = get_logger("audio_pipeline")
yamnet_logger = get_logger("yamnet")

_MAX_RETRIES      = 3
_AUDIO_CHUNKS_DIR = Path(__file__).resolve().parents[3] / "audio_chunks"

# ─── YAMNet class map — loaded once at startup ────────────────────────────────
_CLASS_NAMES: dict[int, str] = {}


def _load_class_map() -> dict[int, str]:
    """Load YAMNet class names from the TF Hub cached CSV."""
    import csv  # noqa: PLC0415

    candidates: list[Path] = []

    try:
        import tensorflow_hub as hub  # noqa: PLC0415
        resolved = hub.resolve("https://tfhub.dev/google/yamnet/1")
        candidates.append(Path(resolved) / "assets" / "yamnet_class_map.csv")
    except Exception:
        pass

    # Bundled copy alongside this file
    candidates.append(Path(__file__).parent / "yamnet_class_map.csv")

    for path in candidates:
        if path.exists():
            try:
                names: dict[int, str] = {}
                with open(path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        names[i] = row.get("display_name", f"class_{i}")
                logger.info(f"YAMNet class map loaded: {len(names)} classes from {path}")
                return names
            except Exception as exc:
                logger.warning(f"Failed to read class map from {path}: {exc}")

    logger.warning(
        "YAMNet class map not found — class names will show as 'class_N'. "
        "Music/ringtone detection will NOT work until the class map is loaded."
    )
    return {}


class AudioPipeline:
    """Captures audio and classifies it with YAMNet in a background thread."""

    def __init__(
        self,
        settings: Settings,
        on_violation: Callable[[AudioAnomalyEvent], None],
        get_mar: Callable[[], float],
    ) -> None:
        self._settings    = settings
        self._on_violation = on_violation
        self._get_mar     = get_mar
        self._stop_event  = threading.Event()
        self._segment_id  = 0

        # Load class map once
        global _CLASS_NAMES
        if not _CLASS_NAMES:
            _CLASS_NAMES = _load_class_map()

        # Create audio_chunks directory
        _AUDIO_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Audio chunks will be saved to: {_AUDIO_CHUNKS_DIR}")

        # Use centralized model loader instead of loading model here
        from proctoring_engine.core.model_loader import get_yamnet_model  # noqa: PLC0415
        self._yamnet = get_yamnet_model()
        
        self._thread = threading.Thread(
            target=self._run_with_retry, daemon=True, name="audio-worker"
        )
        self._thread.start()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _safe_get_mar(self) -> float:
        """Get mouth aspect ratio safely.

        FaceMesh is NOT thread-safe. After the main loop closes it,
        calling process() raises '_graph is None'. We catch that here
        and return 0.0 (conservative: treat as lips-still = external voice).
        """
        try:
            return self._get_mar()
        except Exception:
            return 0.0

    def _run_with_retry(self) -> None:
        retries = 0
        while retries < _MAX_RETRIES and not self._stop_event.is_set():
            try:
                self._capture_loop()
            except AudioThreadError:
                raise
            except Exception as exc:
                retries += 1
                logger.error(
                    f"Audio thread error (attempt {retries}/{_MAX_RETRIES}): {exc}",
                    exc_info=True,
                )
                if retries >= _MAX_RETRIES:
                    logger.error("Audio thread exceeded max retries. Stopping.")
                    raise AudioThreadError(
                        f"Audio thread failed after {_MAX_RETRIES} retries: {exc}",
                        error_code=ERR_AUDIO_THREAD_FAILED,
                    ) from exc
                time.sleep(1.0)

    def _capture_loop(self) -> None:
        import sounddevice as sd  # noqa: PLC0415

        sample_rate    = self._settings.audio_sample_rate
        target_samples = sample_rate  # exactly 1 second = 16000 samples
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Accumulation buffer — collects raw chunks until we have 1 full second
        accumulator: list[np.ndarray] = []
        accumulated_samples = 0

        # Use a small blocksize for low-latency capture (512 samples = 32ms)
        # We accumulate these small blocks until we reach 1 second.
        capture_blocksize = 512

        def _callback(indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ARG001
            if status:
                logger.warning(f"Audio stream status: {status}")
            audio_queue.put(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=capture_blocksize,
            callback=_callback,
        ):
            logger.info(
                f"Audio capture started. "
                f"Accumulating {target_samples} samples ({target_samples / sample_rate:.1f}s) "
                f"per YAMNet segment."
            )
            while not self._stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=2.0)
                except queue.Empty:
                    continue

                accumulator.append(chunk)
                accumulated_samples += len(chunk)

                # Once we have at least 1 second of audio, send to YAMNet
                if accumulated_samples >= target_samples:
                    # Concatenate and take exactly target_samples
                    full_audio = np.concatenate(accumulator)
                    segment    = full_audio[:target_samples]

                    # Keep any overflow for the next segment
                    overflow = full_audio[target_samples:]
                    if len(overflow) > 0:
                        accumulator = [overflow]
                        accumulated_samples = len(overflow)
                    else:
                        accumulator = []
                        accumulated_samples = 0

                    self._process_audio_chunk(segment)

    def _process_audio_chunk(self, audio: np.ndarray) -> None:
        """Amplify, classify, save WAV, log, and emit violation events."""
        t0 = time.perf_counter()
        self._segment_id += 1
        timestamp_ms = int(time.time() * 1000)

        # ── Peak normalisation (amplify to maximum gain) ───────────────────────
        peak = float(np.max(np.abs(audio)))
        amplified = audio / (peak + 1e-9)

        # ── YAMNet inference ───────────────────────────────────────────────────
        import tensorflow as tf  # noqa: PLC0415

        waveform = tf.cast(amplified, tf.float32)
        scores, _embeddings, _spectrogram = self._yamnet(waveform)
        scores_np   = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)

        top_idx   = int(np.argmax(mean_scores))
        top_score = float(mean_scores[top_idx])
        top_class = _CLASS_NAMES.get(top_idx, f"class_{top_idx}")

        # Top-5 classes for richer logging
        top5_idx   = np.argsort(mean_scores)[::-1][:5]
        top5       = [(
            _CLASS_NAMES.get(int(i), f"class_{i}"),
            round(float(mean_scores[i]), 4)
        ) for i in top5_idx]

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # ── Save WAV chunk for debugging ───────────────────────────────────────
        safe_class = top_class.replace(" ", "_").replace(",", "").replace("/", "_")[:30]
        wav_path   = _AUDIO_CHUNKS_DIR / (
            f"seg_{self._segment_id:04d}_{timestamp_ms}_{safe_class}.wav"
        )
        self._save_wav(amplified, wav_path, self._settings.audio_sample_rate)

        # ── Lip-sync MAR (thread-safe) ─────────────────────────────────────────
        mar   = self._safe_get_mar()
        event = self._classify_audio_event(top_class, top_score, timestamp_ms, latency_ms)

        # ── YAMNet per-segment log ─────────────────────────────────────────────
        is_speech = top_class in SPEECH_CLASSES
        is_device = top_class in DEVICE_SOUND_CLASSES
        decision  = event.subtype.value if event else "no_violation"

        yamnet_logger.debug(
            f"segment={self._segment_id} ts={timestamp_ms} "
            f"peak_amplitude={round(peak, 4)} "
            f"top_class='{top_class}' confidence={round(top_score, 4)} "
            f"top5={top5} "
            f"is_speech={is_speech} is_device_sound={is_device} "
            f"mar={round(mar, 4)} lip_sync_threshold={self._settings.lip_sync_mar_threshold} "
            f"decision={decision} latency_ms={round(latency_ms, 2)} "
            f"wav={wav_path.name}"
        )

        if event:
            self._on_violation(event)

    def _classify_audio_event(
        self,
        top_class: str,
        confidence: float,
        timestamp_ms: int,
        latency_ms: float,
    ) -> AudioAnomalyEvent | None:
        if top_class in SPEECH_CLASSES:
            mar = self._safe_get_mar()
            if mar >= self._settings.lip_sync_mar_threshold:
                return None  # candidate speaking — no violation
            return AudioAnomalyEvent(
                subtype=AudioSubtype.EXTERNAL_VOICE,
                yamnet_class=top_class,
                confidence=round(confidence, 4),
                timestamp_ms=timestamp_ms,
                segment_id=self._segment_id,
                latency_ms=round(latency_ms, 2),
            )

        if top_class in DEVICE_SOUND_CLASSES:
            return AudioAnomalyEvent(
                subtype=AudioSubtype.DEVICE_SOUND,
                yamnet_class=top_class,
                confidence=round(confidence, 4),
                timestamp_ms=timestamp_ms,
                segment_id=self._segment_id,
                latency_ms=round(latency_ms, 2),
            )

        return None

    @staticmethod
    def _save_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
        """Save a float32 audio array as a 16-bit PCM WAV file."""
        try:
            # Convert float32 [-1, 1] → int16
            pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)   # 16-bit = 2 bytes
                wf.setframerate(sample_rate)
                wf.writeframes(pcm.tobytes())
        except Exception as exc:
            logger.warning(f"Failed to save audio chunk {path.name}: {exc}")
