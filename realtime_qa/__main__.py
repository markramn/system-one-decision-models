import argparse
import asyncio
import json
import logging

import httpx

from realtime_qa.config import Settings
from realtime_qa.ollama import DecisionError, OllamaClient


async def doctor(settings: Settings) -> bool:
    from realtime_qa.audio import input_devices
    from realtime_qa.setup_models import model_paths
    from realtime_qa.transcription import StreamingTranscriber

    healthy = True
    try:
        async with OllamaClient(settings) as client:
            digest = await client.prepare()
            print(f"Ollama decision model ready: {settings.model} ({digest[:12]})")
    except DecisionError as exc:
        print(str(exc))
        healthy = False
    try:
        await asyncio.to_thread(model_paths, settings.model_dir, verify=True)
        await asyncio.to_thread(StreamingTranscriber, settings)
        print("ASR: artifacts verified and native CPU recognizer initialized successfully")
    except (ValueError, RuntimeError, ImportError) as exc:
        print(str(exc))
        healthy = False
    try:
        devices = await asyncio.to_thread(input_devices)
        print(f"Input devices: {len(devices)}; no microphone has been opened")
        for device in devices:
            print(f"  {device['id']}: {device['name']} / {device['host']}")
        if not devices:
            healthy = False
    except Exception:
        logging.exception("Microphone enumeration failed")
        healthy = False
    print(f"UI: http://127.0.0.1:{settings.port}")
    return healthy


def main() -> None:
    parser = argparse.ArgumentParser(description="RealtimeQA: local System One decision showcase")
    parser.add_argument(
        "command",
        choices=["ui", "doctor", "download-asr", "asr-smoke", "benchmark"],
        nargs="?",
        default="ui",
    )
    parser.add_argument("--model", help="Installed Ollama decision model tag")
    parser.add_argument("--repeats", type=int, choices=range(1, 21), default=5)
    parser.add_argument("--fixtures-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    settings = Settings.from_env()
    if args.model:
        settings.model = args.model
    try:
        if args.command == "doctor":
            raise SystemExit(0 if asyncio.run(doctor(settings)) else 1)
        if args.command == "download-asr":
            from realtime_qa.setup_models import download_models

            download_models(settings, progress=lambda text: print(text, flush=True))
        elif args.command == "asr-smoke":
            from realtime_qa.setup_models import smoke_test

            print(json.dumps(smoke_test(settings), indent=2))
        elif args.command == "benchmark":
            from realtime_qa.benchmark import benchmark

            asyncio.run(benchmark(settings, args.repeats, args.fixtures_only))
        else:
            from realtime_qa.ui import run_dashboard

            run_dashboard(settings)
    except httpx.HTTPError:
        parser.exit(1, "Model download failed. Check your connection and retry.\n")
    except (DecisionError, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
