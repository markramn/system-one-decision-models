import argparse
import asyncio
import json
import math
import statistics
from pathlib import Path

from realtime_qa.config import ROOT, Settings
from realtime_qa.ollama import DecisionError, OllamaClient
from realtime_qa.scoring import build_request, load_scorecard


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


async def benchmark(settings: Settings, repeats: int = 5, fixtures_only: bool = False) -> dict:
    card = load_scorecard(settings.scorecard_path)
    fixtures = json.loads((ROOT / "tests" / "fixtures" / "decisions.json").read_text())
    report: dict = {
        "model": settings.model,
        "question_count": len(card.questions) + 1,
        "qa_question_count": len(card.questions),
        "sentiment_question_count": 1,
        "samples": [],
    }
    async with OllamaClient(settings) as client:
        report["digest"] = await client.prepare()
        if not fixtures_only:
            for words in [40, 100, 200]:
                source = " ".join(f["text"] for f in fixtures[:4])
                text = " ".join((source + " " + source).split()[:words])
                try:
                    build_request(settings, card, text, final=False)
                except ValueError:
                    continue
                timings = []
                for attempt in range(repeats + 1):
                    result = await client.decide(card, text, final=False)
                    print(
                        f"{settings.model} words={words} attempt={attempt} "
                        f"request_ms={result.latency_ms:.0f} answers={len(result.answers)}",
                        flush=True,
                    )
                    if attempt == 0:
                        first = result.latency_ms
                    else:
                        timings.append(result.latency_ms)
                report["samples"].append(
                    {
                        "words": words,
                        "first_ms": round(first),
                        "warm_p50_ms": round(statistics.median(timings)),
                        "warm_p95_ms": round(percentile(timings, 0.95)),
                        "warm_samples": len(timings),
                    }
                )
        if not fixtures_only:
            incremental_timings = []
            incremental_text = ""
            for fixture in fixtures[:4]:
                incremental_text += fixture["text"] + "\n"
                try:
                    build_request(settings, card, incremental_text, final=False)
                except ValueError:
                    report["incremental_stop"] = (
                        "Remaining prefixes exceed the configured prompt budget"
                    )
                    break
                result = await client.decide(card, incremental_text, final=False)
                incremental_timings.append(round(result.latency_ms))
            report["incremental_transcript_request_ms"] = incremental_timings
            report["timing_note"] = (
                "warm_p50/p95 repeat identical prompts and may benefit from caching; "
                "incremental timings change the transcript each time. First request includes "
                "cold loading only when the model was not already resident."
            )
        checks = []
        for fixture in fixtures:
            result = await client.decide(card, fixture["text"], final=True)
            for question, expected in fixture["expected"].items():
                answer = result.answers.get(question)
                actual = answer.choice if answer else "unavailable"
                checks.append(
                    {
                        "case": fixture["name"],
                        "question": question,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        report["fixture_checks"] = checks
        report["fixture_matches"] = sum(c["actual"] == c["expected"] for c in checks)
        report["fixture_total"] = len(checks)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--fixtures-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.model:
        settings.model = args.model
    try:
        report = asyncio.run(benchmark(settings, args.repeats, args.fixtures_only))
    except (DecisionError, ValueError) as exc:
        parser.exit(1, f"Benchmark stopped: {exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
