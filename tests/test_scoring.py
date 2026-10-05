import json

import pytest
from pydantic import ValidationError

from realtime_qa.config import Settings
from realtime_qa.schemas import Decision, Scorecard, SentimentDecision, Utterance
from realtime_qa.scoring import (
    build_request,
    evaluate,
    evaluate_sentiment,
    load_scorecard,
    transcript_metrics,
)


def test_sample_weights_and_pending() -> None:
    card = load_scorecard(Settings().scorecard_path)
    assert sum(q.weight for q in card.questions) == 100
    answers = {q.id: decision(q, "not_observed") for q in card.questions}
    live = evaluate(card, answers, final=False)
    assert {r.status for r in live} == {"pending"}
    ended = evaluate(card, answers, final=True)
    assert {r.status for r in ended} == {"missed"}


def decision(question, choice: str, probability: float = 0.97) -> Decision:
    count = len(question.criteria)
    return Decision(
        type="choice",
        choice=choice,
        probabilities={
            key: probability if key == choice else (1 - probability) / (count - 1)
            for key in question.criteria
        },
        confidence=0.8,
    )


def test_partial_points_uncertainty_and_missing() -> None:
    card = load_scorecard(Settings().scorecard_path)
    answers = {q.id: decision(q, "met") for q in card.questions}
    answers["identification"] = decision(card.questions[1], "partial")
    results = evaluate(card, answers, final=False)
    assert sum(r.points for r in results) == 95
    answers["greeting"] = decision(card.questions[0], "met", 0.55)
    del answers["closing"]
    results = {r.question_id: r for r in evaluate(card, answers, final=True)}
    assert results["greeting"].status == "review"
    assert results["greeting"].points == 0
    assert results["closing"].status == "unavailable"


def test_corrections_can_remove_points() -> None:
    card = load_scorecard(Settings().scorecard_path)
    question = card.questions[2]
    assert evaluate(card, {question.id: decision(question, "met")}, final=False)[2].points == 15
    result = evaluate(card, {question.id: decision(question, "contradicted")}, final=False)[2]
    assert result.status == "contradicted"
    assert result.points == 0


def test_scorecard_rejects_duplicate_ids_and_bad_weights() -> None:
    raw = json.loads(Settings().scorecard_path.read_text())
    raw["questions"][1]["id"] = raw["questions"][0]["id"]
    with pytest.raises(ValidationError):
        Scorecard.model_validate(raw)
    raw["questions"] = raw["questions"][:1]
    raw["questions"][0]["weight"] = 0
    with pytest.raises(ValidationError):
        Scorecard.model_validate(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability(value: float) -> None:
    with pytest.raises(ValidationError):
        Decision(type="choice", choice="met", probabilities={"met": value}, confidence=0.5)


def test_request_uses_decision_api_and_limits_context() -> None:
    settings = Settings()
    card = load_scorecard(settings.scorecard_path)
    body = build_request(settings, card, "Hello, my name is Alex.", final=False)
    assert body["model"] == settings.model
    assert len(body["questions"]) == 9
    assert body["questions"]["_sentiment"]["type"] == "choice"
    assert body["state"]["phase"] == "live"
    assert all(q["type"] == "choice" for q in body["questions"].values())
    with pytest.raises(ValueError, match="budget"):
        build_request(settings, card, "hello " * 10000, final=False)


def test_sentiment_index_and_uncertainty() -> None:
    answer = SentimentDecision(
        type="choice",
        choice="positive",
        confidence=0.8,
        probabilities={"positive": 0.94, "negative": 0.01, "neutral": 0.03, "unclear": 0.02},
    )
    result = evaluate_sentiment(answer, Settings())
    assert result.status == "positive"
    assert result.index == pytest.approx(93)
    answer = answer.model_copy(
        update={
            "probabilities": {"positive": 0.45, "negative": 0.3, "neutral": 0.2, "unclear": 0.05}
        }
    )
    assert evaluate_sentiment(answer, Settings()).status == "review"
    assert evaluate_sentiment(answer, Settings()).index is None
    assert evaluate_sentiment(None, Settings()).status == "unavailable"
    with pytest.raises(ValidationError):
        SentimentDecision(
            type="choice",
            choice="positive",
            confidence=0.8,
            probabilities={"positive": 0.9, "negative": 0.1},
        )


def test_cadence_and_cloud_are_recomputed_from_finalized_words() -> None:
    utterances = (
        Utterance(id=1, start=0, end=10, text="Refund refund account and THE account 123"),
    )
    result = transcript_metrics(utterances, 20)
    assert result.word_count == 7
    assert result.cadence_wpm == 21
    assert result.words[:2] == [("account", 2), ("refund", 2)]
    assert "123" not in dict(result.words)
    assert [p.value for p in result.cadence] == [42, 42, 0, 0]
    assert transcript_metrics(utterances, 20) == result
    assert transcript_metrics(utterances, 0).cadence_wpm is None
    assert not transcript_metrics((), 20).words


def test_sentiment_reserves_one_question_without_entering_qa_points() -> None:
    card = load_scorecard(Settings().scorecard_path)
    results = evaluate(card, {}, final=False)
    assert len(results) == 8
    assert sum(r.maximum for r in results) == 100
    raw = card.model_dump()
    raw["questions"] = [{**raw["questions"][0], "id": f"question_{i}"} for i in range(64)]
    with pytest.raises(ValidationError):
        Scorecard.model_validate(raw)
