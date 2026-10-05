import hashlib
import io
import wave
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np

from realtime_qa.config import Settings

REPOSITORY = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
REVISION = "672fbf1b30579d6585301139bb363f42a0ad4a24"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"
ARTIFACTS = {
    "encoder": (
        "encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
        71082637,
        "0d072fd4ef956294ba9db9e9a71a541ac70659095ec4934c8453d8b2fe740187",
    ),
    "decoder": (
        "decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
        1307236,
        "98da299f471e38bb4e1a8df579b8cc9122d6039576a77e357b3c60f17dd83b02",
    ),
    "joiner": (
        "joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx",
        259335,
        "d944208d660d67c8d72cd2acaeac971fa5ceb8c80e76c1968148846fedd6e297",
    ),
    "tokens": ("tokens.txt", 5048, "b4d1bf82e68de1354f2a3f5fd51094d61d65c77c"),
}


def verify_artifact(path: Path, size: int, digest: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    if len(digest) == 40:
        data = path.read_bytes()
        return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == digest


def model_paths(directory: Path, *, verify: bool = False) -> dict[str, Path]:
    result = {}
    for key, (name, size, digest) in ARTIFACTS.items():
        path = directory / name
        valid = (
            verify_artifact(path, size, digest)
            if verify
            else (path.is_file() and path.stat().st_size == size)
        )
        if not valid:
            raise ValueError(
                "ASR model is missing or incomplete. Run: uv run python -m realtime_qa download-asr"
            )
        result[key] = path
    return result


def download_models(settings: Settings, progress: Callable[[str], None] = print) -> None:
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name, size, digest in ARTIFACTS.values():
            target = settings.model_dir / name
            if verify_artifact(target, size, digest):
                progress(f"Verified {name}")
                continue
            if target.exists():
                raise ValueError(
                    f"Existing artifact failed validation: {name}. Move it aside and retry."
                )
            partial = target.with_suffix(target.suffix + ".part")
            if partial.exists():
                raise ValueError(
                    f"Partial download already exists: {partial.name}. Move it aside and retry."
                )
            try:
                progress(f"Downloading {name} ({size / 1_000_000:.1f} MB)")
                downloaded = 0
                with client.stream("GET", f"{BASE_URL}/{name}") as response:
                    response.raise_for_status()
                    with partial.open("xb") as handle:
                        for block in response.iter_bytes(1024 * 1024):
                            downloaded += len(block)
                            if downloaded > size:
                                raise ValueError("Download exceeded expected artifact size")
                            handle.write(block)
                if not verify_artifact(partial, size, digest):
                    raise ValueError(f"Integrity verification failed: {name}")
                partial.rename(target)
                progress(f"Verified {name}")
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
    progress("English streaming model ready. Inference now works offline.")


def smoke_test(settings: Settings) -> dict:
    from realtime_qa.transcription import StreamingTranscriber

    model_paths(settings.model_dir, verify=True)
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        response = client.get(f"{BASE_URL}/test_wavs/0.wav")
        response.raise_for_status()
    with wave.open(io.BytesIO(response.content)) as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Expected the upstream mono PCM16 fixture")
        rate = audio.getframerate()
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
            np.float32
        )
        samples /= 32768
    transcriber = StreamingTranscriber(settings)
    start = perf_counter()
    chunk = int(rate * 0.05)
    for offset in range(0, len(samples), chunk):
        transcriber.feed(samples[offset : offset + chunk], rate)
    result = transcriber.finish()
    duration = len(samples) / rate
    report = {
        "source": "Upstream Apache-2.0 model test_wavs/0.wav; not microphone audio",
        "audio_seconds": round(duration, 2),
        "decode_seconds": round(perf_counter() - start, 2),
        "transcript": result.text,
    }
    if not result.text:
        raise RuntimeError("ASR smoke test produced an empty transcript")
    return report
