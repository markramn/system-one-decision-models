from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from realtime_qa.config import Settings
from realtime_qa.schemas import Utterance


@dataclass(frozen=True)
class TranscriptView:
    utterances: tuple[Utterance, ...] = ()
    partial: str = ""
    partial_start: float = 0
    elapsed: float = 0

    @property
    def text(self) -> str:
        return "\n".join(
            [u.text for u in self.utterances] + ([self.partial] if self.partial else [])
        )


class StreamingTranscriber:
    def __init__(self, settings: Settings, *, recognizer: Any = None) -> None:
        if recognizer is None:
            import sherpa_onnx

            from realtime_qa.setup_models import model_paths

            paths = model_paths(settings.model_dir)
            recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                **{key: str(value) for key, value in paths.items()},
                num_threads=settings.asr_threads,
                provider="cpu",
                sample_rate=16000,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=settings.endpoint_silence,
                rule3_min_utterance_length=20,
                decoding_method="greedy_search",
            )
        self.recognizer = recognizer
        self.stream = recognizer.create_stream()
        self.utterances: list[Utterance] = []
        self.elapsed = 0.0
        self.segment_start = 0.0
        self.rate = 16000
        self.finished = False
        self.view = TranscriptView()

    def _decode(self) -> str:
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return self.recognizer.get_result(self.stream).strip()

    def _commit(self, text: str) -> None:
        if text:
            self.utterances.append(
                Utterance(
                    id=len(self.utterances) + 1,
                    start=self.segment_start,
                    end=self.elapsed,
                    text=text,
                )
            )
        self.segment_start = self.elapsed

    def feed(self, samples: npt.NDArray[np.float32], rate: int) -> TranscriptView:
        if self.finished:
            raise RuntimeError("Cannot feed a finalized transcriber")
        self.rate = rate
        self.elapsed += len(samples) / rate
        self.stream.accept_waveform(rate, samples)
        text = self._decode()
        if self.recognizer.is_endpoint(self.stream):
            self._commit(text)
            self.recognizer.reset(self.stream)
            text = ""
        self.view = TranscriptView(tuple(self.utterances), text, self.segment_start, self.elapsed)
        return self.view

    def finish(self) -> TranscriptView:
        if not self.finished:
            self.stream.accept_waveform(self.rate, np.zeros(int(self.rate * 0.6), dtype=np.float32))
            self.stream.input_finished()
            self._commit(self._decode())
            self.finished = True
            self.view = TranscriptView(tuple(self.utterances), elapsed=self.elapsed)
        return self.view
