import asyncio

import pytest

from realtime_qa.config import Settings
from realtime_qa.ollama import DecisionBatch, DecisionError
from realtime_qa.schemas import Decision, SentimentDecision
from realtime_qa.scoring import load_scorecard
from realtime_qa.session import Session


class FakeClient:
    digest = "fixture-model"

    def __init__(self) -> None:
        self.calls = []
        self.delay = 0.02
        self.fail = False
        self.active = 0
        self.peak = 0

    async def prepare(self, *, warmup: bool = False) -> str:
        return self.digest

    async def close(self) -> None:
        pass

    async def decide(self, card, text, *, final):
        self.calls.append((text, final))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            if self.fail:
                raise DecisionError("unavailable")
            answers = {
                q.id: Decision(
                    type="choice",
                    choice="not_observed",
                    confidence=0.9,
                    probabilities={
                        k: 0.98 if k == "not_observed" else 0.02 / (len(q.criteria) - 1)
                        for k in q.criteria
                    },
                )
                for q in card.questions
            }
            return DecisionBatch(
                answers,
                20,
                {},
                SentimentDecision(
                    type="choice",
                    choice="positive",
                    confidence=0.8,
                    probabilities={
                        "positive": 0.94,
                        "negative": 0.01,
                        "neutral": 0.03,
                        "unclear": 0.02,
                    },
                ),
            )
        finally:
            self.active -= 1


def make_session() -> tuple[Session, FakeClient]:
    settings = Settings(score_interval=0.1, partial_debounce=0)
    client = FakeClient()
    return Session(settings, load_scorecard(settings.scorecard_path), client=client), client


async def test_newest_wins_single_request_and_final_pass() -> None:
    session, client = make_session()
    await session.start("text")
    session.add_text("Hello")
    session.tick()
    await asyncio.sleep(0.005)
    session.add_text("I am Alex")
    session.add_text("Thank you")
    await session.end()
    assert client.peak == 1
    assert client.calls[-1] == ("Hello\nI am Alex\nThank you", True)
    assert session.snapshot.final
    assert session.snapshot.revision == session.revision
    assert session.state == "ended"
    assert session.started_wall_at > 0
    assert '"started_at": "' in session.export()
    assert session.snapshot.sentiment.status == "positive"
    assert session.sentiment_history[-1].value == 93
    assert session.transcript_metrics.word_count == 6
    assert session.transcript_metrics.cadence_wpm is None
    assert '"visual_analytics"' in session.export()
    await session.close()
    assert session.started_wall_at == 0
    assert not session.sentiment_history
    assert not session.transcript_metrics.words


async def test_audio_metrics_are_unavailable_for_text_or_incomplete_capture() -> None:
    from realtime_qa.audio import AudioStats, AudioView
    from realtime_qa.schemas import Utterance
    from realtime_qa.transcription import TranscriptView

    session, _ = make_session()
    session.view = TranscriptView(
        utterances=(Utterance(id=1, start=0, end=5, text="Refund account payment reference"),)
    )
    session.audio_view = AudioView(stats=AudioStats(captured_seconds=10, quiet_seconds=4))
    assert session.silence_percent == 40
    assert session.transcript_metrics.cadence_wpm == 24
    session.incomplete = True
    assert session.silence_percent is None
    assert session.transcript_metrics.cadence_wpm is None
    session.incomplete = False
    session.mode = "text"
    assert session.silence_percent is None
    assert session.transcript_metrics.cadence_wpm is None
    assert session.transcript_metrics.word_count == 4
    await session.close()


async def test_empty_session_not_authoritative_and_start_guard() -> None:
    session, _ = make_session()
    await session.start("text")
    with pytest.raises(ValueError, match="active"):
        await session.start("text")
    await session.end()
    assert session.snapshot is None
    assert "empty" in session.message.lower()
    await session.close()


async def test_failed_final_pass_does_not_produce_final_score() -> None:
    session, client = make_session()
    await session.start("text")
    session.add_text("Hello")
    client.fail = True
    await session.end()
    assert session.snapshot is None
    assert session.scoring_error
    assert not session.authoritative
    await session.close()


async def test_duration_limit_finalizes_without_losing_text() -> None:
    from time import perf_counter

    session, _ = make_session()
    await session.start("text")
    session.add_text("Hello")
    session.started_at = perf_counter() - 301
    session.tick()
    await session.end_task
    assert session.state == "ended"
    assert session.snapshot.final
    assert session.snapshot.transcript == "Hello"
    await session.close()


async def test_reset_invalidates_old_results() -> None:
    session, client = make_session()
    await session.start("text")
    session.add_text("Old session")
    session.tick()
    await asyncio.sleep(0.005)
    old_id = session.session_id
    await session.reset()
    await session.start("text")
    assert session.session_id != old_id
    await asyncio.sleep(0.05)
    assert session.snapshot is None
    assert not session.sentiment_history
    assert session.transcript == ""
    await session.close()


async def test_prompt_budget_keeps_previous_valid_transcript() -> None:
    session, _ = make_session()
    await session.start("text")
    session.add_text("Hello")
    with pytest.raises(ValueError, match="budget"):
        session.add_text("long " * 10000)
    assert session.transcript == "Hello"
    await session.close()


async def test_continuous_partials_cannot_starve_scoring() -> None:
    from time import perf_counter

    from realtime_qa.transcription import TranscriptView

    session, client = make_session()
    session.settings.partial_debounce = 0.4
    await session.start("text")
    session._accept(TranscriptView(partial="Hello, I am still speaking"))
    session.last_request = perf_counter() - 2.1
    session.tick()
    await asyncio.sleep(0.05)
    assert len(client.calls) == 1
    assert session.snapshot.transcript == "Hello, I am still speaking"
    await session.close()


async def test_old_session_response_cannot_update_current_session() -> None:
    session, _ = make_session()
    await session.start("text")
    await session._score("different-session", 100, "Old text", final=False)
    assert session.snapshot is None
    await session.close()


async def test_retries_are_bounded_and_capture_errors_are_exported() -> None:
    import json

    from realtime_qa.audio import AudioView

    session, client = make_session()
    await session.start("text")
    session.add_text("Hello")
    client.fail = True
    for _ in range(6):
        session.retry_after = 0
        session.last_request = 0
        session.tick()
        await asyncio.sleep(0.03)
    assert len(client.calls) == 3
    session._accept_audio(AudioView(transcript=session.view, dropped_blocks=1))
    await session.end()
    exported = json.loads(session.export())
    assert exported["incomplete_capture"]
    assert exported["requires_review"]
    assert exported["scoring_error"]
    await session.close()


async def test_capture_close_failure_still_ends_session() -> None:
    from realtime_qa.schemas import Utterance
    from realtime_qa.transcription import TranscriptView

    class BrokenMicrophone:
        def stop(self):
            raise RuntimeError("device lost")

    session, client = make_session()
    await session.start("text")
    session._accept(
        TranscriptView(
            utterances=(Utterance(id=1, start=0, end=1, text="Finalized words"),),
            partial="Unfinished words",
        )
    )
    session.microphone = BrokenMicrophone()
    await session.end()
    assert session.state == "ended"
    assert session.microphone is None
    assert session.incomplete
    assert client.calls[-1] == ("Finalized words", True)
    assert not session.authoritative
    await session.close()
