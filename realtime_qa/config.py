import os
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    model: str = Field(default="nimble:9b-q4_K_M", min_length=1, max_length=120)
    ollama_url: str = "http://127.0.0.1:11434"
    scorecard_path: Path = ROOT / "scorecards" / "support_call.json"
    model_dir: Path = ROOT / "models" / "streaming-en"
    request_timeout: float = Field(default=90, ge=1, le=300)
    score_interval: float = Field(default=1.0, ge=0.1, le=30)
    partial_debounce: float = Field(default=0.4, ge=0, le=5)
    probability_threshold: float = Field(default=0.65, ge=0.5, le=1)
    margin_threshold: float = Field(default=0.15, ge=0, le=1)
    max_session_seconds: float = Field(default=300, ge=10, le=600)
    max_prompt_bytes: int = Field(default=7000, ge=4000, le=7000)
    asr_threads: int = Field(default=4, ge=1, le=12)
    endpoint_silence: float = Field(default=1.0, ge=0.5, le=2.4)
    silence_threshold_dbfs: float = Field(default=-40, ge=-80, le=-10, allow_inf_nan=False)
    audio_queue_blocks: int = Field(default=50, ge=5, le=100)
    port: int = Field(default=8088, ge=1024, le=65535)

    @field_validator("ollama_url")
    @classmethod
    def loopback_only(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("Ollama must use a plain HTTP loopback address")
        return value.rstrip("/")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.model_validate(
            {
                name: os.environ[f"REALTIME_QA_{name.upper()}"]
                for name in cls.model_fields
                if f"REALTIME_QA_{name.upper()}" in os.environ
            }
        )
