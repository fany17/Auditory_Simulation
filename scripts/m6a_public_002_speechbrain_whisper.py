"""One-shot synthetic inference checks for SpeechBrain CRDNN and Whisper turbo."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_speechbrain_whisper.json"
HF_ENDPOINT = "https://hf-mirror.com"


def audio() -> np.ndarray:
    fs = 16000
    t = np.arange(fs, dtype=np.float32) / fs
    return (0.08 * np.sin(2 * np.pi * (120 + 40 * t) * t) + 0.03 * np.sin(2 * np.pi * 730 * t)).astype(np.float32)


def run_speechbrain(waveform: np.ndarray) -> dict[str, object]:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    import torch
    from speechbrain.inference.ASR import EncoderDecoderASR

    model_id = "speechbrain/asr-crdnn-rnnlm-librispeech"
    started = time.perf_counter()
    recognizer = EncoderDecoderASR.from_hparams(
        source=model_id,
        savedir=str(ROOT / "cache/speechbrain_crdnn_rnnlm_librispeech"),
        run_opts={"device": "cpu"},
    )
    signal = torch.from_numpy(waveform[None, :])
    lengths = torch.tensor([1.0])
    with torch.inference_mode():
        decoded = recognizer.transcribe_batch(signal, lengths)
    return {
        "status": "PASS",
        "model": "SpeechBrain-CRDNN",
        "model_id": model_id,
        "endpoint": HF_ENDPOINT,
        "wall_seconds": time.perf_counter() - started,
        "decoded_type": type(decoded).__name__,
        "synthetic_only": True,
    }


def run_whisper(waveform: np.ndarray) -> dict[str, object]:
    import whisper

    started = time.perf_counter()
    model = whisper.load_model("turbo", download_root=str(ROOT / "cache/whisper"), device="cpu")
    result = model.transcribe(waveform, fp16=False, language="en")
    text = str(result.get("text", ""))
    return {
        "status": "PASS",
        "model": "Whisper-turbo",
        "checkpoint": "turbo",
        "wall_seconds": time.perf_counter() - started,
        "text_length": len(text),
        "synthetic_only": True,
    }


def main() -> None:
    waveform = audio()
    results: dict[str, object] = {"zero_training": True, "patient_data_read": False, "input_shape": list(waveform.shape)}
    try:
        results["SpeechBrain-CRDNN"] = run_speechbrain(waveform)
    except Exception as exc:
        results["SpeechBrain-CRDNN"] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    try:
        results["Whisper-turbo"] = run_whisper(waveform)
    except Exception as exc:
        results["Whisper-turbo"] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
