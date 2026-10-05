import logging
import queue
import threading
from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import sounddevice as sd

from realtime_qa.config import Settings
from realtime_qa.transcription import StreamingTranscriber, TranscriptView

logger = logging.getLogger(__name__)
_MICROPHONE_LOCK = threading.Lock()


def input_devices() -> list[dict[str, Any]]:
    hosts = sd.query_hostapis()
    return [
        {
            "id": index,
            "name": device["name"],
            "host": hosts[device["hostapi"]]["name"],
            "sample_rate": int(device["default_samplerate"]),
            "default": index == sd.default.device[0],
        }
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]


@dataclass(frozen=True)
class WaveformBin:
    start: float
    end: float
    minimum: float
    maximum: float


class WaveformHistory:
    def __init__(self) -> None:
        self.bins: deque[WaveformBin] = deque(maxlen=400)
        self.elapsed = 0.0

    def append(self, samples: npt.NDArray[np.float32], rate: int) -> None:
        if rate <= 0:
            raise ValueError("Sample rate must be positive")
        if not len(samples):
            return
        clean = np.clip(np.nan_to_num(samples, nan=0, posinf=1, neginf=-1), -1, 1)
        end = self.elapsed + len(samples) / rate
        self.bins.append(WaveformBin(self.elapsed, end, float(clean.min()), float(clean.max())))
        self.elapsed = end
        while self.bins and self.bins[0].end < end - 20:
            self.bins.popleft()

    def snapshot(self) -> tuple[WaveformBin, ...]:
        return tuple(self.bins)


@dataclass(frozen=True)
class SilenceInterval:
    start: float
    duration: float = 0
    quiet_seconds: float = 0


@dataclass(frozen=True)
class AudioStats:
    captured_seconds: float = 0
    quiet_seconds: float = 0
    intervals: tuple[SilenceInterval, ...] = ()


class AudioMetrics:
    def __init__(self, threshold_dbfs: float) -> None:
        self.threshold = 10 ** (threshold_dbfs / 20)
        self.captured_seconds = 0.0
        self.quiet_seconds = 0.0
        self.intervals: deque[SilenceInterval] = deque(maxlen=120)

    def append(self, samples: npt.NDArray[np.float32], rate: int) -> None:
        if rate <= 0:
            raise ValueError("Sample rate must be positive")
        if not len(samples):
            return
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if not np.isfinite(rms):
            raise ValueError("Nonfinite audio cannot be measured")
        quiet = rms < self.threshold
        duration = len(samples) / rate
        self.captured_seconds += duration
        self.quiet_seconds += duration if quiet else 0
        remaining = duration
        while remaining > 1e-9:
            if not self.intervals or self.intervals[-1].duration >= 5 - 1e-9:
                self.intervals.append(SilenceInterval(self.captured_seconds - remaining))
            current = self.intervals[-1]
            amount = min(5 - current.duration, remaining)
            self.intervals[-1] = replace(
                current,
                duration=current.duration + amount,
                quiet_seconds=current.quiet_seconds + (amount if quiet else 0),
            )
            remaining -= amount

    def snapshot(self) -> AudioStats:
        return AudioStats(self.captured_seconds, self.quiet_seconds, tuple(self.intervals))


@dataclass(frozen=True)
class AudioView:
    transcript: TranscriptView = TranscriptView()
    level: float = 0
    clipping: bool = False
    lag_ms: float = 0
    dropped_blocks: int = 0
    error: str = ""
    waveform: tuple[WaveformBin, ...] = ()
    stats: AudioStats = AudioStats()


class Microphone:
    def __init__(self, settings: Settings, device: int | None) -> None:
        self.settings = settings
        self.device = device
        self.blocks: queue.Queue[tuple[npt.NDArray[np.float32], float]] = queue.Queue(
            maxsize=settings.audio_queue_blocks
        )
        self.stopping = threading.Event()
        self.lock = threading.Lock()
        self.view = AudioView()
        self.dropped = 0
        self.stream: sd.InputStream | None = None
        self.worker: threading.Thread | None = None
        self.owns_mic = False

    def start(self) -> None:
        if not _MICROPHONE_LOCK.acquire(blocking=False):
            raise RuntimeError("Another session already owns the microphone")
        self.owns_mic = True
        try:
            transcriber = StreamingTranscriber(self.settings)
            info = sd.query_devices(self.device, "input")
            rate = int(info["default_samplerate"])
            sd.check_input_settings(
                device=self.device, channels=1, dtype="float32", samplerate=rate
            )
            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                dtype="float32",
                samplerate=rate,
                blocksize=int(rate * 0.05),
                callback=self._callback,
            )
            self.worker = threading.Thread(
                target=self._work, args=(transcriber, rate), name="realtime-qa-asr", daemon=True
            )
            self.worker.start()
            self.stream.start()
        except Exception:
            self.stop()
            raise

    def _callback(
        self, samples: npt.NDArray[np.float32], frames: int, time: Any, status: Any
    ) -> None:
        if status.input_overflow:
            self.dropped += 1
        try:
            self.blocks.put_nowait((samples[:, 0].copy(), perf_counter()))
        except queue.Full:
            self.dropped += 1

    def _work(self, transcriber: StreamingTranscriber, rate: int) -> None:
        waveform = WaveformHistory()
        metrics = AudioMetrics(self.settings.silence_threshold_dbfs)
        try:
            while not self.stopping.is_set() or not self.blocks.empty():
                try:
                    samples, captured_at = self.blocks.get(timeout=0.1)
                except queue.Empty:
                    continue
                waveform.append(samples, rate)
                metrics.append(samples, rate)
                transcript = transcriber.feed(samples, rate)
                with self.lock:
                    self.view = AudioView(
                        transcript=transcript,
                        level=min(1.0, float(np.sqrt(np.mean(samples**2))) * 5),
                        clipping=bool(np.max(np.abs(samples)) >= 0.99),
                        lag_ms=(perf_counter() - captured_at) * 1000,
                        dropped_blocks=self.dropped,
                        waveform=waveform.snapshot(),
                        stats=metrics.snapshot(),
                    )
            with self.lock:
                self.view = replace(
                    self.view, transcript=transcriber.finish(), dropped_blocks=self.dropped, level=0
                )
        except Exception:
            logger.exception("Streaming recognition worker failed")
            with self.lock:
                self.view = replace(
                    self.view, error="Speech recognizer failed. Restart and run doctor."
                )

    def snapshot(self) -> AudioView:
        with self.lock:
            return replace(self.view, dropped_blocks=self.dropped)

    def stop(self) -> AudioView:
        try:
            if self.stream is not None:
                try:
                    self.stream.stop()
                finally:
                    self.stream.close()
                    self.stream = None
        finally:
            self.stopping.set()
            if self.worker is not None:
                self.worker.join(timeout=10)
            if self.owns_mic:
                self.owns_mic = False
                _MICROPHONE_LOCK.release()
        if self.worker is not None and self.worker.is_alive():
            with self.lock:
                self.view = replace(
                    self.view, error="Recognizer did not finish; session is incomplete."
                )
        return self.snapshot()
