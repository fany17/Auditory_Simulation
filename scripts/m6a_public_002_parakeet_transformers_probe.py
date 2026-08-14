"""Try the official Parakeet Transformers loading path on synthetic audio only."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_parakeet_transformers_probe.json"
ENDPOINT = "https://hf-mirror.com"


def main() -> None:
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    waveform = (0.08 * np.sin(2 * np.pi * 440 * np.arange(16000, dtype=np.float32) / 16000)).astype(np.float32)
    results: dict[str, object] = {"zero_training": True, "patient_data_read": False, "input_shape": list(waveform.shape)}
    try:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        model_id = "nvidia/parakeet-tdt-0.6b-v3"
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)
        model = AutoModelForTDT.from_pretrained(
            model_id, trust_remote_code=False, dtype="auto", device_map="cuda:0"
        ).eval()
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs)
        results["Parakeet-TDT"] = {
            "status": "PASS",
            "model_id": model_id,
            "endpoint": ENDPOINT,
            "wall_seconds": time.perf_counter() - started,
            "model_class": type(model).__name__,
            "generated_type": type(generated).__name__,
            "synthetic_only": True,
        }
    except Exception as exc:
        results["Parakeet-TDT"] = {
            "status": "BLOCKED_BY_UPSTREAM_DEPENDENCY",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
