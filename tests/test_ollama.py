import json

import httpx
import pytest

from realtime_qa.config import Settings
from realtime_qa.ollama import DecisionError, OllamaClient
from realtime_qa.scoring import load_scorecard


def response_answers(card) -> dict:
    return {
        q.id: {
            "type": "choice",
            "choice": "met",
            "confidence": 0.8,
            "probabilities": {
                k: 0.97 if k == "met" else 0.03 / (len(q.criteria) - 1) for k in q.criteria
            },
        }
        for q in card.questions
    }


async def test_client_uses_systemone_and_drops_invalid_answers() -> None:
    settings = Settings()
    card = load_scorecard(settings.scorecard_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone"
        body = json.loads(request.content)
        assert body["state"]["transcript"] == "Hello"
        answers = response_answers(card)
        answers["greeting"]["probabilities"]["met"] = 12
        del answers["closing"]
        return httpx.Response(200, json={"answers": answers, "usage": {"input_tokens": 100}})

    async with OllamaClient(settings, transport=httpx.MockTransport(handler)) as client:
        result = await client.decide(card, "Hello", final=False)
    assert "greeting" not in result.answers
    assert "closing" not in result.answers
    assert len(result.answers) == 6
    assert result.usage["input_tokens"] == 100
    assert result.sentiment is None


@pytest.mark.parametrize("valid", [True, False])
async def test_sentiment_is_an_independent_answer_in_the_same_batch(valid: bool) -> None:
    settings = Settings()
    card = load_scorecard(settings.scorecard_path)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert len(body["questions"]) == 9
        answers = response_answers(card)
        answers["_sentiment"] = {
            "type": "choice",
            "choice": "positive",
            "confidence": 0.8,
            "probabilities": {
                "positive": 0.94 if valid else 9.4,
                "negative": 0.01,
                "neutral": 0.03,
                "unclear": 0.02,
            },
        }
        return httpx.Response(200, json={"answers": answers})

    async with OllamaClient(settings, transport=httpx.MockTransport(handler)) as client:
        result = await client.decide(card, "I am happy to help you", final=False)
    assert len(calls) == 1
    assert len(result.answers) == 8
    assert (result.sentiment is not None) == valid


@pytest.mark.parametrize("status", [404, 500, 503])
async def test_server_failure_is_not_a_qa_result(status: int) -> None:
    settings = Settings()
    card = load_scorecard(settings.scorecard_path)
    async with OllamaClient(
        settings, transport=httpx.MockTransport(lambda _: httpx.Response(status))
    ) as client:
        with pytest.raises(DecisionError):
            await client.decide(card, "Hello", final=False)


async def test_timeout_and_malformed_response() -> None:
    settings = Settings()
    card = load_scorecard(settings.scorecard_path)

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    for handler in [timeout, lambda _: httpx.Response(200, json={"wrong": "shape"})]:
        async with OllamaClient(settings, transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(DecisionError):
                await client.decide(card, "Hello", final=False)


async def test_prepare_checks_capability_and_warms_before_capture() -> None:
    settings = Settings()
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": settings.model, "digest": "test-digest"}]}
            )
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["decision"]})
        assert request.url.path == "/v1/systemone"
        assert json.loads(request.content)["state"] == "Local QA warmup."
        return httpx.Response(200, json={"answers": {}})

    async with OllamaClient(settings, transport=httpx.MockTransport(handler)) as client:
        assert await client.prepare(warmup=True) == "test-digest"
    assert paths == ["/api/tags", "/api/show", "/v1/systemone"]
