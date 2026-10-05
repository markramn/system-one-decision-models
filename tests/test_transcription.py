import numpy as np

from realtime_qa.config import Settings
from realtime_qa.transcription import StreamingTranscriber


class FakeStream:
    def __init__(self) -> None:
        self.rates = []
        self.finished = False

    def accept_waveform(self, rate, samples) -> None:
        self.rates.append(rate)

    def input_finished(self) -> None:
        self.finished = True


class FakeRecognizer:
    def __init__(self) -> None:
        self.text = "hello"
        self.endpoint = False
        self.stream = FakeStream()

    def create_stream(self):
        return self.stream

    def is_ready(self, stream) -> bool:
        return False

    def get_result(self, stream) -> str:
        return self.text

    def is_endpoint(self, stream) -> bool:
        return self.endpoint

    def reset(self, stream) -> None:
        self.text = ""
        self.endpoint = False


def test_partial_replacement_endpoint_and_final_flush() -> None:
    engine = FakeRecognizer()
    asr = StreamingTranscriber(Settings(), recognizer=engine)
    first = asr.feed(np.zeros(4800, dtype=np.float32), 48000)
    assert first.partial == "hello"
    engine.text = "hello there"
    second = asr.feed(np.zeros(4800, dtype=np.float32), 48000)
    assert second.partial == "hello there"
    assert not second.utterances
    engine.endpoint = True
    third = asr.feed(np.zeros(4800, dtype=np.float32), 48000)
    assert [u.text for u in third.utterances] == ["hello there"]
    assert third.partial == ""
    engine.text = "goodbye"
    flushed = asr.finish()
    assert [u.text for u in flushed.utterances] == ["hello there", "goodbye"]
    assert engine.stream.finished
    assert engine.stream.rates[0] == 48000
    assert asr.finish() == flushed
