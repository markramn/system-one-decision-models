# RealtimeQA

Standalone Windows-first, all-Python local microphone-to-scorecard showcase. No AnalyticsBackend imports, cloud inference, database, custom JavaScript, or automatic microphone capture.

## Setup and launch

From this directory in PowerShell:

```powershell
uv sync --locked
ollama pull nimble:9b-q4_K_M
uv run python -m realtime_qa download-asr
uv run python -m realtime_qa doctor
uv run python -m realtime_qa
```

Open `http://127.0.0.1:8088`. Select an input device, press Start session, wait for Preparing to finish, and speak in English as the agent. End session flushes the final words and performs a final decision pass. New session clears in-memory data. The text demo bypasses microphone/ASR but uses real Ollama decisions; Next scripted line supplies fictional agent utterances.

All models and dependencies are already installed on the original development PC. Ordinary startup requires no downloads. The first inference after a model unload can be much slower than warm requests; the app warms the model before opening the microphone. Ollama 0.35+ with a decision-capable Nimble model is required. The original Q8 model is preserved: `uv run python -m realtime_qa --model nimble:latest` selects it.

## Architecture and behavior

- `audio.py`: sounddevice capture callback, bounded raw-audio queue, one CPU recognizer worker, microphone ownership and cleanup. The worker also publishes real sample min/max envelopes in a bounded 400-bin / 20-second waveform history; no waveform data is fabricated for text mode.
- `transcription.py`: sherpa-onnx streaming partials, endpoint commits and final flush. Device-rate audio is resampled by the recognizer.
- `ollama.py`: reusable async HTTP client for `/v1/systemone`; validates each named decision. No chat/generation API.
- `scoring.py` and `schemas.py`: bounded requests, explicit rubrics, probability validation and deterministic points.
- `session.py`: versioned transcript, single in-flight decision, newest pending state, bounded retries, lifecycle and export.
- `ui.py`: NiceGUI call-details dashboard and single controlling browser client. Layout order: call metadata, live oscillogram, evaluation-form/status panels, visual metric cards, then transcript on the left and live QA on the right. Updates stay on the UI event loop. Transcript auto-scroll and text export are available. The waveform is a recent live envelope, not a recording player. Phone number and extension remain explicitly unconnected in this standalone POC.
- `scorecards/support_call.json`: eight positive agent-speech checks, 100 points total. The settings dialog edits/imports JSON in memory; download it or edit this file to persist changes.

Met earns full points; supported Partial earns half. Not observed stays Pending during a live call and becomes Missed at final evaluation. Explicit contradictions receive zero. Low-probability or low-margin decisions require review; missing/invalid answers are unavailable. The displayed subtotal is a lower bound while questions remain unresolved. No decisions are permanently latched: later speech can reverse earlier results. Model probabilities and concentration are not calibrated accuracy. Evaluated context is shown, not fabricated evidence or explanations.

All speech is treated as the agent. There is no diarization, customer audio/satisfaction verification, acoustic emotion scoring, or real account-action verification. Short demo sessions are bounded by 300 seconds and a conservative 7,000-byte serialized prompt budget, whichever comes first. The budget includes question instructions and may end a demo earlier than five minutes; prior evidence is never silently discarded.

The app binds only to loopback and uses local assets. Session data stays in memory unless Export JSON is clicked. No audio is written to disk. Closing the controlling browser releases capture and clears the session. Exports contain transcript data: use fictional examples and keep exports out of source control.

## Visual analytics

A four-card Visual Analytics row sits between the evaluation-form section and the transcript/QA panes.

- Agent sentiment is a separate `_sentiment` choice in the same Nimble `/v1/systemone` batch as QA. It assesses cumulative agent text, not customer feelings or vocal tone. The displayed index is `100 * (P(positive) - P(negative))`, from -100 to +100; it is not a percentage or calibrated accuracy. Unclear/low-certainty/missing answers have no numeric index. History uses transcript-snapshot time, preserves uncertain gaps, and is bounded to 256 points. Sentiment never contributes QA points; one model question is reserved for it, so scorecards support at most 63 QA questions.
- Average cadence is finalized recognized words per captured audio minute, including pauses. The five-second bars distribute each utterance's words uniformly over its ASR start/end times; those timings are estimates, not word-level alignment. Partial transcript revisions are excluded. Text-demo mode and incomplete capture have no cadence estimate.
- The word cloud is rebuilt from finalized text, excluding common English words and numeric tokens; the top 18 terms are weighted by frequency. Revisions do not accumulate duplicate counts. Word-cloud data is also available in text mode.
- Call silence is the fraction of captured audio-block duration with RMS below the configurable threshold (default -40 dBFS). It is an acoustic estimate, not VAD or semantic speech detection. Quiet speech/noise can affect it. The chart shows quiet seconds in five-second bins, with the last bin potentially partial. History is bounded to 120 bins; the total remains cumulative. Silence is unavailable in text mode or after capture loss.
- The threshold can be changed in idle-only Settings or with `REALTIME_QA_SILENCE_THRESHOLD_DBFS`. No extra model, GPU allocation, or third-party dependency is required for these calculations.
- JSON exports include `visual_analytics`, raw sentiment decisions in the score snapshot, metric definitions, the threshold, and completeness/staleness flags. `requires_review` still describes QA outcomes; sentiment review flags are separate. Reset clears all metric state. Old latency measurements below cover eight QA questions only; new benchmarks include the ninth sentiment question.

## Brand styling

The dashboard follows a supplied Smarsh visual identity reference and web guide (not included in this repository), refined to match the supplied call-details screenshots: compact light chrome, Deep Blue text, Bright Blue primary actions, white panels with pale neutral headers, extended-neutral borders, and restrained semantic accents. Red Alert is reserved for errors/negative outcomes. Small text uses high-contrast Deep Blue rather than cyan, green, or the lighter neutrals. Manrope is preferred (installed on the development PC), with local Segoe UI/Arial fallbacks and no font CDN. The S1 mark is the app's existing generic monogram, not an official Smarsh logo. Layout measurements are app-specific, not official brand tokens. Palette and browser-rendered theme checks live in the existing UI tests.

## Configuration

Typed defaults and validation live in `realtime_qa/config.py`. Every setting can be overridden by a `REALTIME_QA_` environment variable with its uppercase field name; no dotenv file is loaded automatically.

| Variable | Default / purpose |
|---|---|
| `REALTIME_QA_MODEL` | `nimble:9b-q4_K_M` |
| `REALTIME_QA_OLLAMA_URL` | `http://127.0.0.1:11434`; HTTP loopback only |
| `REALTIME_QA_PORT` | `8088`; localhost UI port |
| `REALTIME_QA_SCORECARD_PATH` | `scorecards/support_call.json`, resolved from project root |
| `REALTIME_QA_MODEL_DIR` | `models/streaming-en`, resolved from project root |
| `REALTIME_QA_REQUEST_TIMEOUT` | `90` seconds, permits cold model load |
| `REALTIME_QA_SCORE_INTERVAL` | `1.0` second minimum between requests |
| `REALTIME_QA_PARTIAL_DEBOUNCE` | `0.4` seconds, with starvation protection |
| `REALTIME_QA_PROBABILITY_THRESHOLD` | `0.65`, heuristic review gate |
| `REALTIME_QA_MARGIN_THRESHOLD` | `0.15`, top-two probability margin gate |
| `REALTIME_QA_MAX_SESSION_SECONDS` | `300`, short demo limit |
| `REALTIME_QA_MAX_PROMPT_BYTES` | `7000`, hard conservative serialized request budget |
| `REALTIME_QA_ASR_THREADS` | `4`, CPU inference threads |
| `REALTIME_QA_ENDPOINT_SILENCE` | `1.0` second trailing silence |
| `REALTIME_QA_SILENCE_THRESHOLD_DBFS` | `-40`; acoustic RMS silence estimate, configurable from -80 to -10 |
| `REALTIME_QA_AUDIO_QUEUE_BLOCKS` | `50` blocks of approximately 50 ms |

Model and scorecard changes are idle-only. The app does not silently pull or replace Ollama models. On capture overflow or ASR errors, results are marked incomplete. On inference errors, earlier scores are visibly stale and retries are bounded.

## Verification

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv pip check
uv run python -m realtime_qa doctor
uv run python -m realtime_qa asr-smoke
uv run python -m realtime_qa benchmark --repeats 5
```

Ordinary pytest uses fake capture/recognition and HTTP mocks, plus NiceGUI's Python user fixture. It does not open a microphone or download models. `asr-smoke` explicitly downloads one licensed upstream WAV into memory, verifies the model artifacts, and streams it through real CPU ASR. `doctor` loads the native recognizer and enumerates devices without recording.

An opt-in real Edge/Playwright test exercises the text demo, real decisions, export and reset, and checks for browser errors and non-local requests. Start the UI separately on port 8088, close other demo tabs, then run:

```powershell
$env:REALTIME_QA_BROWSER_TEST = '1'
uv run pytest tests/test_browser.py -q
Remove-Item Env:REALTIME_QA_BROWSER_TEST
```

This uses installed Microsoft Edge and never opens the microphone. It writes a screenshot to ignored `exports/browser-smoke-final.png`. Set `REALTIME_QA_BROWSER_URL` to a separate loopback UI instance (for example port 8090) when the main demo is already in use. Do not run model benchmarks concurrently with live demonstrations or browser performance checks.

For machine-readable benchmark reports:

```powershell
uv run python -m realtime_qa.benchmark --model nimble:9b-q4_K_M --repeats 5 --output exports/benchmark-q4.json
```

Repeated warm request timings can benefit from identical-prompt caching. The report separately measures changing transcript prefixes. First-request timings include loading only if the model was not already resident. The small semantic fixtures are illustrative regression examples, not production accuracy validation.

## Verified local performance and limitations

On an RTX 5070 Laptop GPU (8 GB), Core Ultra 9 275HX and approximately 32 GB RAM:
- Q8: full eight-question repeated warm requests about 3.4–3.6 seconds; 20/21 fixture comparisons matched.
- Q4: repeated warm requests about 1.3–1.7 seconds; changing transcript prefixes about 2.04–2.16 seconds; 19/21 fixture comparisons matched. GPU usage was approximately 5.8 GB during the Q4 checks.
- Initial cold model requests were approximately 31 seconds for Q8 and 42 seconds for Q4 in those runs; not representative of subsequent live updates.
- Streaming English CPU ASR processed 6.62 seconds of upstream sample audio in 0.55 seconds, excluding model loading/download.
- Both models falsely marked one adversarial instruction-like utterance as Partial closing. Q4 also missed one minimal greeting. Raw decisions and uncertainty are deliberately visible.
- Actual accent, noise and microphone quality require a user-spoken check. This is a demonstration, not certified compliance or production QA.

## Model provenance and Windows dependencies

- Nimble: Apache-2.0; official Ollama decision model. API: https://ollama.com/blog/ollama-now-supports-jev-style-decision-models
- ASR model: Apache-2.0, https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26 at revision `672fbf1b30579d6585301139bb363f42a0ad4a24`. Four INT8 chunk-16/left-64 artifacts total approximately 73 MB. Downloads are pinned and hash-verified; existing invalid files are not overwritten automatically.
- Pin both `sherpa-onnx==1.13.8` and `sherpa-onnx-core==1.13.8`. The main wheel's metadata did not cause uv to install its separate runtime automatically on the development PC. Without core, Windows resolved `C:\Windows\System32\onnxruntime.dll` (1.17.1), causing an API version error. With core, the project uses its own `sherpa_onnx/lib/onnxruntime.dll`. Do not replace Windows DLLs or change global PATH.
- Python 3.12 has verified Windows wheels. uv resolves dependencies with the recorded release cutoff in `pyproject.toml`; do not weaken this constraint to work around installation failures.
- If editor buffers differ from files changed by terminal formatters, Save All before running tests; ensure the tests are executing the latest code. Do not overwrite unrelated user edits.
