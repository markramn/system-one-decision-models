from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx
from pydantic import ValidationError

from realtime_qa.config import Settings
from realtime_qa.schemas import Decision, Scorecard, SentimentDecision
from realtime_qa.scoring import build_request


class DecisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionBatch:
    answers: dict[str, Decision]
    latency_ms: float
    usage: dict[str, int]
    sentiment: SentimentDecision | None = None


class OllamaClient:
    def __init__(
        self, settings: Settings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self.http = httpx.AsyncClient(
            base_url=settings.ollama_url,
            timeout=httpx.Timeout(settings.request_timeout, connect=3),
            transport=transport,
            trust_env=False,
        )
        self.digest: str | None = None

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self.http.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self.http.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise DecisionError(
                "Ollama timed out; scores are unavailable. Try a smaller model."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise DecisionError(
                f"Ollama returned HTTP {exc.response.status_code}. Check the model and server."
            ) from exc
        except httpx.RequestError as exc:
            raise DecisionError("Cannot reach local Ollama. Check that Ollama is running.") from exc
        except ValueError as exc:
            raise DecisionError("Ollama returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise DecisionError("Ollama returned an invalid response shape.")
        return payload

    async def prepare(self, *, warmup: bool = False) -> str:
        tags = await self._request("GET", "/api/tags")
        name = self.settings.model
        canonical = name if ":" in name else f"{name}:latest"
        model = next((m for m in tags.get("models", []) if m.get("name") == canonical), None)
        if model is None:
            raise DecisionError(f"Model {canonical} is not installed in Ollama.")
        info = await self._request("POST", "/api/show", json={"model": name})
        if "decision" not in info.get("capabilities", []):
            raise DecisionError("The selected model does not support System One decisions.")
        self.digest = model.get("digest", "unknown")
        if warmup:
            await self._request(
                "POST",
                "/v1/systemone",
                json={
                    "model": name,
                    "keep_alive": "30m",
                    "state": "Local QA warmup.",
                    "questions": {
                        "ready": {
                            "type": "choice",
                            "instructions": "Does the text mention warmup?",
                            "criteria": {
                                "yes": "Warmup is mentioned",
                                "no": "Warmup is not mentioned",
                            },
                        }
                    },
                },
            )
        return self.digest

    async def decide(self, card: Scorecard, transcript: str, *, final: bool) -> DecisionBatch:
        body = build_request(self.settings, card, transcript, final=final)
        start = perf_counter()
        payload = await self._request("POST", "/v1/systemone", json=body)
        elapsed = (perf_counter() - start) * 1000
        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, dict):
            raise DecisionError("Ollama response did not contain decision answers.")
        answers = {}
        for question in card.questions:
            try:
                answer = Decision.model_validate(raw_answers.get(question.id))
            except ValidationError:
                continue
            if answer.probabilities.keys() == question.criteria.keys():
                answers[question.id] = answer
        try:
            sentiment = SentimentDecision.model_validate(raw_answers.get("_sentiment"))
        except ValidationError:
            sentiment = None
        raw_usage = payload.get("usage", {})
        usage = {
            key: value
            for key, value in (raw_usage.items() if isinstance(raw_usage, dict) else [])
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        }
        return DecisionBatch(answers=answers, latency_ms=elapsed, usage=usage, sentiment=sentiment)
