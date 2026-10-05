import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from realtime_qa.audio import AudioMetrics, Microphone, WaveformHistory
from realtime_qa.config import Settings
from realtime_qa.setup_models import model_paths, verify_artifact


def test_audio_queue_is_bounded_and_overflow_visible() -> None:
    microphone = Microphone(Settings(audio_queue_blocks=5), None)
    samples = np.zeros((100, 1), dtype=np.float32)
    for _ in range(8):
        microphone._callback(samples, 100, None, SimpleNamespace(input_overflow=False))
    assert microphone.blocks.qsize() == 5
    assert microphone.snapshot().dropped_blocks == 3
    microphone._callback(samples, 100, None, SimpleNamespace(input_overflow=True))
    assert microphone.snapshot().dropped_blocks == 5


def test_waveform_uses_real_extrema_and_sample_time() -> None:
    history = WaveformHistory()
    history.append(np.array([-0.2, 0.7, 0.1, -0.5], dtype=np.float32), 16000)
    point = history.snapshot()[0]
    assert point.start == 0
    assert point.end == 4 / 16000
    assert point.minimum == pytest.approx(-0.5)
    assert point.maximum == pytest.approx(0.7)
    history.append(np.zeros(480, dtype=np.float32), 48000)
    assert history.snapshot()[-1].end == pytest.approx(4 / 16000 + 0.01)
    assert history.snapshot()[-1].minimum == history.snapshot()[-1].maximum == 0


def test_waveform_is_bounded_and_sanitizes_nonfinite_values() -> None:
    history = WaveformHistory()
    for _ in range(500):
        history.append(np.zeros(800, dtype=np.float32), 16000)
    assert len(history.snapshot()) <= 400
    assert history.snapshot()[0].start >= 4.9
    history.append(np.array([np.nan, np.inf, -np.inf], dtype=np.float32), 16000)
    assert history.snapshot()[-1].minimum == -1
    assert history.snapshot()[-1].maximum == 1
    size = len(history.snapshot())
    history.append(np.array([], dtype=np.float32), 16000)
    assert len(history.snapshot()) == size


def test_worker_publishes_waveform_without_opening_a_microphone() -> None:
    from time import perf_counter

    from realtime_qa.transcription import TranscriptView

    class FakeTranscriber:
        def feed(self, samples, rate) -> TranscriptView:
            return TranscriptView(elapsed=len(samples) / rate)

        def finish(self) -> TranscriptView:
            return TranscriptView()

    microphone = Microphone(Settings(), None)
    microphone.blocks.put((np.array([-0.4, 0.8], dtype=np.float32), perf_counter()))
    microphone.stopping.set()
    microphone._work(FakeTranscriber(), 16000)
    view = microphone.snapshot()
    assert not view.error
    assert len(view.waveform) == 1
    assert view.waveform[0].minimum == pytest.approx(-0.4)
    assert view.waveform[0].maximum == pytest.approx(0.8)


def test_silence_counts_duration_and_splits_interval_boundaries() -> None:
    metrics = AudioMetrics(-40)
    metrics.append(np.zeros(4 * 16000, dtype=np.float32), 16000)
    metrics.append(np.full(2 * 48000, 0.1, dtype=np.float32), 48000)
    stats = metrics.snapshot()
    assert stats.captured_seconds == pytest.approx(6)
    assert stats.quiet_seconds == pytest.approx(4)
    assert len(stats.intervals) == 2
    assert stats.intervals[0].duration == 5
    assert stats.intervals[0].quiet_seconds == 4
    assert stats.intervals[1].duration == 1
    assert stats.intervals[1].quiet_seconds == 0
    assert AudioMetrics(-40).snapshot().captured_seconds == 0


def test_silence_threshold_and_history_bounds() -> None:
    metrics = AudioMetrics(-40)
    metrics.append(np.full(16000, 0.005, dtype=np.float32), 16000)
    assert metrics.snapshot().quiet_seconds == 1
    for _ in range(130):
        metrics.append(np.full(5 * 100, 0.1, dtype=np.float32), 100)
    assert len(metrics.snapshot().intervals) <= 120
    assert metrics.snapshot().captured_seconds == pytest.approx(651)
    with pytest.raises(ValueError):
        metrics.append(np.array([np.nan], dtype=np.float32), 16000)


def test_hash_validation_and_missing_models(tmp_path) -> None:
    path = tmp_path / "model"
    data = b"test model"
    path.write_bytes(data)
    assert verify_artifact(path, len(data), hashlib.sha256(data).hexdigest())
    assert not verify_artifact(path, len(data) + 1, hashlib.sha256(data).hexdigest())
    assert not verify_artifact(path, len(data), "0" * 64)
    with pytest.raises(ValueError, match="download-asr"):
        model_paths(tmp_path)


@pytest.mark.parametrize(
    "url", ["http://example.com", "https://127.0.0.1", "http://user@localhost"]
)
def test_ollama_is_loopback_only(url: str) -> None:
    with pytest.raises(ValueError):
        Settings(ollama_url=url)
