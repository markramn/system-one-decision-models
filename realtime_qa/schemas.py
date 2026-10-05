import math
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Choice = Literal["met", "partial", "not_observed", "contradicted"]
Status = Literal["met", "partial", "pending", "missed", "contradicted", "review", "unavailable"]


def validate_distribution(choice: str, probabilities: Mapping[str, float]) -> None:
    values = list(probabilities.values())
    if (
        choice not in probabilities
        or len(values) < 2
        or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
        or abs(sum(values) - 1) > 0.02
        or probabilities[choice] + 0.001 < max(values)
    ):
        raise ValueError("Invalid choice probability distribution")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    section: str = Field(min_length=1, max_length=60)
    text: str = Field(min_length=5, max_length=300)
    weight: float = Field(gt=0, le=100, allow_inf_nan=False)
    criteria: dict[Choice, str]

    @model_validator(mode="after")
    def valid_criteria(self) -> "Question":
        if not {"met", "not_observed"} <= self.criteria.keys():
            raise ValueError("Every question needs met and not_observed criteria")
        if any(not value.strip() or len(value) > 400 for value in self.criteria.values()):
            raise ValueError("Criteria must be nonblank and at most 400 characters")
        return self


class Scorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    questions: list[Question] = Field(min_length=1, max_length=63)

    @model_validator(mode="after")
    def unique_questions(self) -> "Scorecard":
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("Question IDs must be unique")
        return self

    @property
    def maximum(self) -> float:
        return sum(q.weight for q in self.questions)


class Decision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["choice"]
    choice: Choice
    probabilities: dict[Choice, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_distribution(self) -> "Decision":
        validate_distribution(self.choice, self.probabilities)
        return self


SentimentChoice = Literal["negative", "neutral", "positive", "unclear"]


class SentimentDecision(BaseModel):
    type: Literal["choice"]
    choice: SentimentChoice
    probabilities: dict[SentimentChoice, float]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_distribution(self) -> "SentimentDecision":
        validate_distribution(self.choice, self.probabilities)
        if set(self.probabilities) != {"negative", "neutral", "positive", "unclear"}:
            raise ValueError("Sentiment requires all four probabilities")
        return self


class SentimentResult(BaseModel):
    status: Literal["negative", "neutral", "positive", "unclear", "review", "unavailable"] = (
        "unavailable"
    )
    index: float | None = Field(default=None, ge=-100, le=100, allow_inf_nan=False)
    decision: SentimentDecision | None = None


class MetricPoint(BaseModel):
    seconds: float = Field(ge=0, allow_inf_nan=False)
    value: float | None = Field(default=None, allow_inf_nan=False)


class TranscriptMetrics(BaseModel):
    word_count: int = 0
    words: list[tuple[str, int]] = Field(default_factory=list)
    cadence_wpm: float | None = None
    cadence: list[MetricPoint] = Field(default_factory=list)


class QuestionResult(BaseModel):
    question_id: str
    status: Status
    points: float = 0
    maximum: float
    decision: Decision | None = None


class Utterance(BaseModel):
    id: int
    start: float
    end: float
    text: str


class ScoreSnapshot(BaseModel):
    session_id: str
    revision: int
    final: bool
    transcript: str
    results: list[QuestionResult]
    latency_ms: float
    completed_at: float
    usage: dict[str, int] = Field(default_factory=dict)
    sentiment: SentimentResult = Field(default_factory=SentimentResult)
    at_seconds: float = 0

    @property
    def earned(self) -> float:
        return sum(result.points for result in self.results)

    @property
    def unresolved(self) -> int:
        return sum(r.status in {"pending", "review", "unavailable"} for r in self.results)
