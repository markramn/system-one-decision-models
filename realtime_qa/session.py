import asyncio
import json
import logging
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter, time
from typing import Any, Literal
from uuid import uuid4

from realtime_qa.audio import AudioView, Microphone
from realtime_qa.config import Settings
from realtime_qa.ollama import DecisionError, OllamaClient
from realtime_qa.schemas import MetricPoint, Scorecard, ScoreSnapshot, TranscriptMetrics, Utterance
from realtime_qa.scoring import build_request, evaluate, evaluate_sentiment, transcript_metrics
from realtime_qa.transcription import TranscriptView

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, settings: Settings, card: Scorecard, *, client: Any = None) -> None:
        self.settings = settings
        self.card = card
        self.client = client or OllamaClient(settings)
        self.microphone: Microphone | None = None
        self.request_task: asyncio.Task | None = None
        self.end_task: asyncio.Task | None = None
        self.lifecycle = asyncio.Lock()
        self.owner_id: str | None = None
        self._clear()

    def _clear(self) -> None:
        self.session_id = str(uuid4())
        self.state = "idle"
        self.mode = "microphone"
        self.message = "Ready. Audio is captured only after you press Start session."
        self.scoring_error = ""
        self.incomplete = False
        self.view = TranscriptView()
        self.audio_view = AudioView()
        self.revision = 0
        self.snapshot: ScoreSnapshot | None = None
        self.started_at = 0.0
        self.started_wall_at = 0.0
        self.ended_at = 0.0
        self.changed_at = 0.0
        self.last_request = 0.0
        self.last_attempt = -1
        self.failures = 0
        self.retry_after = 0.0
        self.updates = 0
        self.history: deque[dict] = deque(maxlen=100)
        self.sentiment_history: deque[MetricPoint] = deque(maxlen=256)

    @property
    def transcript(self) -> str:
        return self.view.text

    @property
    def transcript_metrics(self) -> TranscriptMetrics:
        seconds = (
            self.audio_view.stats.captured_seconds
            if self.mode == "microphone" and not self.incomplete
            else 0
        )
        return transcript_metrics(self.view.utterances, seconds)

    @property
    def silence_percent(self) -> float | None:
        stats = self.audio_view.stats
        if self.mode != "microphone" or self.incomplete or stats.captured_seconds <= 0:
            return None
        return 100 * stats.quiet_seconds / stats.captured_seconds

    @property
    def active(self) -> bool:
        return self.state in {"preparing", "listening", "finalizing"}

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0
        return (self.ended_at or perf_counter()) - self.started_at

    @property
    def authoritative(self) -> bool:
        return bool(
            self.snapshot
            and self.snapshot.final
            and self.snapshot.revision == self.revision
            and not self.incomplete
            and not self.scoring_error
            and not self.snapshot.unresolved
        )

    @property
    def prompt_fraction(self) -> float:
        try:
            body = build_request(self.settings, self.card, self.transcript, final=True)
            return (
                len(json.dumps(body, ensure_ascii=False).encode()) / self.settings.max_prompt_bytes
            )
        except ValueError:
            return 1.0

    async def start(self, mode: Literal["text", "microphone"], device: int | None = None) -> None:
        async with self.lifecycle:
            if self.active:
                raise ValueError("A session is already active")
            if self.state != "idle":
                raise ValueError("Choose New session before starting again")
            if mode not in {"text", "microphone"}:
                raise ValueError("Unknown input mode")
            build_request(self.settings, self.card, "", final=True)
            self.state = "preparing"
            self.mode = mode
            self.message = "Checking local Ollama and loading the speech recognizer..."
            try:
                await self.client.prepare(warmup=True)
                if mode == "microphone":
                    self.microphone = Microphone(self.settings, device)
                    await asyncio.to_thread(self.microphone.start)
            except Exception:
                logger.exception("Session preparation failed")
                await self._stop_microphone()
                self.state = "error"
                self.message = (
                    "Could not start. Check Ollama, the ASR download, and microphone access."
                )
                raise
            self.started_at = perf_counter()
            self.started_wall_at = time()
            self.state = "listening"
            self.message = (
                "Listening to this PC's microphone. First decision may include model load time."
                if mode == "microphone"
                else "Text demo: microphone and transcription are bypassed."
            )

    def _accept(self, view: TranscriptView) -> None:
        if self.view == view:
            return
        if self.view.text != view.text or bool(self.view.partial) != bool(view.partial):
            self.revision += 1
            self.changed_at = perf_counter()
        self.view = view

    def add_text(self, text: str) -> None:
        if self.state != "listening" or self.mode != "text":
            raise ValueError("Start a text demo before adding speech")
        text = text.strip()
        if not text:
            return
        view = TranscriptView(
            utterances=(
                *self.view.utterances,
                Utterance(
                    id=len(self.view.utterances) + 1,
                    text=text,
                    start=self.elapsed,
                    end=self.elapsed,
                ),
            ),
            elapsed=self.elapsed,
        )
        build_request(self.settings, self.card, view.text, final=True)
        self._accept(view)

    def _accept_audio(self, view: AudioView) -> None:
        self.audio_view = view
        if view.error or view.dropped_blocks:
            self.incomplete = True
            self.message = view.error or "Audio overflow detected. This session is incomplete."
        self._accept(view.transcript)

    def tick(self) -> None:
        if self.state != "listening":
            return
        if self.microphone is not None:
            self._accept_audio(self.microphone.snapshot())
        if (
            self.elapsed >= self.settings.max_session_seconds
            or self.prompt_fraction >= 0.95
            or self.audio_view.error
        ):
            self.message = "Finalizing: session/context limit or an audio error was reached."
            if self.end_task is None or self.end_task.done():
                self.end_task = asyncio.create_task(self.end())
            return
        now = perf_counter()
        if not self.transcript.strip() or (self.request_task and not self.request_task.done()):
            return
        if self.last_attempt == self.revision and not (self.scoring_error and self.failures < 3):
            return
        if now < self.retry_after or now - self.last_request < self.settings.score_interval:
            return
        if (
            self.view.partial
            and now - self.changed_at < self.settings.partial_debounce
            and now - self.last_request < max(2.0, self.settings.score_interval)
        ):
            return
        self.last_request = now
        self.last_attempt = self.revision
        self.request_task = asyncio.create_task(
            self._score(self.session_id, self.revision, self.transcript, final=False)
        )

    async def _score(self, session_id: str, revision: int, text: str, *, final: bool) -> None:
        at_seconds = (
            self.audio_view.stats.captured_seconds if self.mode == "microphone" else self.elapsed
        )
        try:
            batch = await self.client.decide(self.card, text, final=final)
        except (DecisionError, ValueError) as exc:
            if session_id == self.session_id:
                self.scoring_error = str(exc)
                self.failures += 1
                self.retry_after = perf_counter() + min(15, 2**self.failures)
            return
        except Exception:
            logger.exception("Decision worker failed")
            if session_id == self.session_id:
                self.scoring_error = "Decision worker failed; scores are unavailable."
                self.failures = 3
            return
        if session_id != self.session_id or (self.snapshot and revision < self.snapshot.revision):
            return
        results = evaluate(
            self.card,
            batch.answers,
            final=final,
            probability_threshold=self.settings.probability_threshold,
            margin_threshold=self.settings.margin_threshold,
        )
        previous = {r.question_id: r.status for r in self.snapshot.results} if self.snapshot else {}
        for result in results:
            if previous.get(result.question_id) != result.status:
                self.history.append(
                    {
                        "at_seconds": round(self.elapsed, 1),
                        "revision": revision,
                        "question": result.question_id,
                        "status": result.status,
                        "final": final,
                    }
                )
        sentiment = evaluate_sentiment(batch.sentiment, self.settings)
        self.sentiment_history.append(MetricPoint(seconds=at_seconds, value=sentiment.index))
        self.snapshot = ScoreSnapshot(
            session_id=session_id,
            revision=revision,
            final=final,
            transcript=text,
            results=results,
            latency_ms=batch.latency_ms,
            completed_at=time(),
            usage=batch.usage,
            sentiment=sentiment,
            at_seconds=at_seconds,
        )
        self.updates += 1
        self.scoring_error = ""
        self.failures = 0
        self.retry_after = 0

    async def _stop_microphone(self) -> None:
        microphone, self.microphone = self.microphone, None
        if microphone is not None:
            try:
                self._accept_audio(await asyncio.to_thread(microphone.stop))
            except Exception:
                logger.exception("Microphone shutdown failed; session is incomplete")
                self.incomplete = True
                self.message = "Microphone shutdown failed. Only finalized speech can be assessed."

    async def end(self) -> None:
        async with self.lifecycle:
            if self.state != "listening":
                return
            self.state = "finalizing"
            self.message = "Finishing the last words and evaluating the final transcript..."
            await self._stop_microphone()
            if self.request_task is not None:
                await self.request_task
                self.request_task = None
            final_text = "\n".join(u.text for u in self.view.utterances)
            if final_text.strip():
                await self._score(self.session_id, self.revision, final_text, final=True)
                self.message = (
                    "Final evaluation complete. This is not a certified compliance result."
                    if self.authoritative
                    else "Session ended. Review uncertain, unavailable, or incomplete results."
                )
            else:
                self.message = "Session was empty; there is no final score."
            self.ended_at = perf_counter()
            self.state = "ended"

    async def reset(self) -> None:
        async with self.lifecycle:
            self.session_id = str(uuid4())
            if self.request_task is not None:
                self.request_task.cancel()
                try:
                    await self.request_task
                except asyncio.CancelledError:
                    pass
                self.request_task = None
            await self._stop_microphone()
            self._clear()

    def export(self) -> str:
        return json.dumps(
            {
                "exported_at": datetime.now(UTC).isoformat(),
                "started_at": (
                    datetime.fromtimestamp(self.started_wall_at, UTC).isoformat()
                    if self.started_wall_at
                    else None
                ),
                "session_id": self.session_id,
                "state": self.state,
                "input_mode": self.mode,
                "model": self.settings.model,
                "model_digest": self.client.digest,
                "scorecard": self.card.model_dump(),
                "transcript_revision": self.revision,
                "utterances": [u.model_dump() for u in self.view.utterances],
                "partial_text": self.view.partial,
                "score": self.snapshot.model_dump() if self.snapshot else None,
                "history": list(self.history),
                "incomplete_capture": self.incomplete,
                "visual_analytics": {
                    **self.transcript_metrics.model_dump(),
                    "sentiment_history": [point.model_dump() for point in self.sentiment_history],
                    "audio": asdict(self.audio_view.stats) if self.mode == "microphone" else None,
                    "silence_threshold_dbfs": self.settings.silence_threshold_dbfs,
                    "silence_percent": self.silence_percent,
                    "sentiment_stale": bool(self.scoring_error)
                    or self.snapshot is None
                    or self.snapshot.revision != self.revision,
                    "sentiment_requires_review": self.snapshot is None
                    or self.snapshot.sentiment.status in {"unclear", "review", "unavailable"},
                    "capture_incomplete": self.incomplete,
                    "definitions": {
                        "sentiment": "Agent text only; index = 100 * (P(positive) - P(negative)).",
                        "cadence": "Finalized words per captured minute, including pauses. "
                        "Five-second bins use uniform allocation within utterances.",
                        "silence": "Captured duration below the RMS threshold; "
                        "not semantic speech detection.",
                        "word_cloud": "Top 18 finalized terms; stop words and numbers omitted.",
                    },
                },
                "requires_review": not self.authoritative,
                "scoring_error": self.scoring_error,
                "elapsed_seconds": round(self.elapsed, 2),
                "notice": "Agent-only demonstration. Probabilities are not calibrated accuracy.",
            },
            indent=2,
            ensure_ascii=False,
        )

    async def close(self) -> None:
        await self.reset()
        await self.client.close()
