# System One Decision Models

**RealtimeQA** is a local, all-Python showcase of fast, typed decisions using **Nimble through Ollama**. Speak into your microphone as a call-centre agent and watch a quality-assurance scorecard update alongside a live transcript and visual analytics.

Rather than asking a chat model to generate a review, the app sends named questions and explicit answer choices to Ollama's `/v1/systemone` endpoint. Decisions include probabilities, while the application calculates the scorecard points.

> This is an English, agent-only proof of concept—not a production call-recording platform or certified compliance evaluator.

## Features

- **Streaming transcription:** open-source sherpa-onnx English ASR on CPU.
- **Live QA:** eight editable support-call checks, weighted to 100 points.
- **Agent sentiment:** a separate Nimble decision in the same request as the scorecard.
- **Visual analytics:** speaking cadence, frequency-weighted word cloud, and estimated acoustic silence.
- **Live oscillogram:** a bounded, recent window of real microphone amplitudes.
- **Call-details UI:** a Python-authored NiceGUI interface with transcript and QA panes, probabilities, and timing diagnostics.
- **Text demo:** try the decision model without speaking, using typed utterances or the built-in script.
- **Exports:** download the transcript or a JSON session report with scorecard results, analytics, and review flags.

## Requirements

- **Python 3.12** and **uv**.
- **Ollama 0.35 or newer**, running locally with a decision-capable model.
- A microphone for audio-based demonstrations.
- Internet access for initial package and model downloads.

The project is Windows-first. It was tested on an RTX 5070 Laptop GPU with 8 GB VRAM, a Core Ultra 9 CPU, and approximately 32 GB RAM. The default `nimble:9b-q4_K_M` model was selected to fit that GPU; transcription stays on CPU to avoid GPU contention. Other hardware may have different latency and memory requirements.

## Quick start

Ensure Ollama is running through its desktop application, or run `ollama serve` in a separate terminal.

```powershell
git clone https://github.com/markramn/system-one-decision-models.git
cd system-one-decision-models

uv sync --locked
ollama pull nimble:9b-q4_K_M
uv run python -m realtime_qa download-asr
uv run python -m realtime_qa doctor
uv run python -m realtime_qa
```

Open **http://127.0.0.1:8088**.

The ASR download is approximately 73 MB; the default Nimble model download is approximately 5.6 GB. They are downloaded separately and are not included in this repository. After setup, transcription and decision inference run locally.

### Microphone mode

1. Choose **Microphone** and select an input device.
2. Click **Start session** and wait for **Listening**. Model warmup happens before capture starts.
3. Speak as the agent. Short pauses allow the recognizer to finalize utterances.
4. Watch the transcript, metric cards, and individual QA questions update.
5. Click **End session** to flush the last words and run a final assessment.
6. Export any results you want to keep, then use **New session** to reset.

Only one browser tab can control the demo at a time. If another tab owns the session, close it and reload. Closing the controlling tab clears the in-memory session.

### Text mode

Select **Text demo**, start a session, then type an agent utterance or click **Next scripted line**. This bypasses microphone capture and ASR but still uses real Nimble decisions.

Sentiment and the word cloud work in text mode. Cadence, silence, and the waveform are unavailable because no audio is captured.

## How scoring works

The default [support-call scorecard](scorecards/support_call.json) checks:

| Check | Points |
|---|---:|
| Professional greeting | 10 |
| Agent and company identification | 10 |
| Recording disclosure | 15 |
| Identity-verification request before account action | 15 |
| Empathy | 10 |
| Ownership or an offer to help | 10 |
| Concrete resolution or next steps | 20 |
| Further-help offer and polite closing | 10 |

- **Met** earns full points. **Partial** earns half where the rubric supports partial credit.
- **Pending** means the behaviour has not been observed yet; it is not a failure during a live call.
- At final evaluation, confidently unobserved required behaviours become **Missed**.
- Explicit contradictions receive zero points. Later speech can change earlier decisions.
- Uncertain decisions are marked **Review needed**; malformed or missing answers are **Unavailable**.

The live subtotal is provisional. Model probabilities are not calibrated accuracy, and an agent saying an action happened does not verify that it actually happened.

Edit the scorecard file, or use **Edit scorecard** while idle. Dialog edits apply in memory; download the JSON or update the file to keep them. Up to 63 QA questions are supported, with one additional question reserved for sentiment.

## Visual analytics

| Metric | Definition |
|---|---|
| **Agent sentiment** | Nimble judges the attitude expressed in the cumulative agent transcript. The index is `100 × (P(positive) − P(negative))`, from −100 to +100—not a percentage. Uncertain or insufficient evidence produces no numeric index. It does not assess the unheard customer or vocal tone. |
| **Average cadence** | Finalized recognized words per captured minute, including pauses. Five-second bars estimate timing by distributing words across each utterance's ASR start/end times. |
| **Word cloud** | The top 18 terms from finalized text, weighted by frequency. Common English words and numeric tokens are excluded. |
| **Call silence** | The fraction of captured audio-block duration below an RMS threshold, defaulting to −40 dBFS. This is an acoustic estimate, not speech detection; background noise and quiet speech can affect it. |

Partial ASR revisions are excluded from word counts to avoid double-counting. Cadence and silence are withheld after incomplete capture. The silence threshold can be adjusted in idle-only **Settings**.

## Configuration

Settings use the `REALTIME_QA_` environment-variable prefix. Common overrides:

| Variable | Default |
|---|---|
| `REALTIME_QA_MODEL` | `nimble:9b-q4_K_M` |
| `REALTIME_QA_OLLAMA_URL` | `http://127.0.0.1:11434`—HTTP loopback only |
| `REALTIME_QA_PORT` | `8088` |
| `REALTIME_QA_SCORECARD_PATH` | Project-root `scorecards/support_call.json` |
| `REALTIME_QA_SILENCE_THRESHOLD_DBFS` | `-40` |
| `REALTIME_QA_MAX_SESSION_SECONDS` | `300` |

For example, to use another **already installed, decision-capable** model:

```powershell
uv run python -m realtime_qa --model nimble:latest
```

Or change the UI port in PowerShell:

```powershell
$env:REALTIME_QA_PORT = '8090'
uv run python -m realtime_qa
```

A `.env` file is not loaded automatically. See [AGENTS.md](AGENTS.md#configuration) for the complete settings reference and [config.py](realtime_qa/config.py) for validation rules.

## Architecture

```text
Local microphone → bounded audio queue
                        ├── amplitude envelope and acoustic silence
                        └── CPU streaming ASR → versioned transcript
                                                   ├── cadence and word cloud
                                                   └── Ollama /v1/systemone
                                                        ├── QA decisions
                                                        └── agent sentiment

                 Transcript, metrics, and decisions → NiceGUI dashboard
```

The scheduler allows one model request in flight and coalesces pending transcript updates. Session IDs and transcript revisions prevent obsolete responses from overwriting newer results. Audio capture and ASR run away from the UI event loop.

## Tests and diagnostics

```powershell
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv pip check
```

The default tests use mocked audio and model responses, plus NiceGUI's UI simulation. They do not open the microphone or require a running model. The real-browser test is opt-in.

```powershell
uv run python -m realtime_qa doctor
uv run python -m realtime_qa asr-smoke
uv run python -m realtime_qa benchmark --repeats 5
```

- **doctor** checks Ollama, initializes the native recognizer, and enumerates devices without recording.
- **asr-smoke** explicitly downloads a licensed upstream sample WAV into memory and tests real CPU transcription.
- **benchmark** measures the combined QA/sentiment request and checks a small fictional fixture set. Repeated prompts may benefit from caching; changing-transcript timings are reported separately.

See [AGENTS.md](AGENTS.md#verification) for the opt-in Edge/Playwright browser test and benchmark export commands.

## Privacy and limitations

- The UI binds to `127.0.0.1`. There is no cloud inference or telephony integration.
- The app does not save microphone audio to disk. Session data stays in memory unless explicitly exported; exports contain transcript data and should be handled accordingly.
- Downloaded models, environments, caches, and exports are excluded from Git.
- All captured speech is treated as the agent. There is no speaker diarization, customer-channel analysis, or acoustic-emotion detection.
- Sessions stop at the configured duration or a conservative 7,000-byte serialized prompt budget, whichever comes first. Long scorecards or transcripts can reach that budget before five minutes.
- Cold model loading can take substantially longer than warm requests. Latency also depends on hardware, transcript length, and question count.
- Transcription errors, background noise, and model misclassification can affect the results. This is a showcase, not validated production QA.

## Models and references

- [Ollama's Jev-style decision-model API](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models)
- [Nimble](https://ollama.com/library/nimble)—Apache-2.0 decision model from Bespoke Labs.
- [Streaming English Zipformer ASR](https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26)—Apache-2.0 model, downloaded from a pinned revision with integrity checks.

The UI follows Smarsh-inspired visual guidance. The S1 mark is a generic app monogram, not an official Smarsh logo or a claim of brand approval. Model licenses are separate from the application code.
