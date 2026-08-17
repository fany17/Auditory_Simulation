"""Official Parakeet AutoModel forward probe on six synthetic waveforms."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_parakeet_forward_probe.json"
ENDPOINT = "https://hf-mirror.com"


def make_probes() -> dict[str, np.ndarray]:
    fs = 16000
    t = np.arange(fs, dtype=np.float32) / fs
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    regular = np.zeros(fs, dtype=np.float32)
    regular[np.arange(1000, fs, 2000)] = 0.8
    jitter = np.zeros(fs, dtype=np.float32)
    jitter[[1000, 2930, 5010, 6900, 9030, 11000, 13150, 15020]] = 0.8
    omission = regular.copy()
    omission[5000] = 0.0
    phase = 0.2 * np.sin(2 * np.pi * 440 * t + np.pi / 2)
    speech = 0.08 * np.sin(2 * np.pi * (120 + 40 * t) * t) + 0.03 * np.sin(2 * np.pi * 730 * t)
    return {"tone": tone, "regular_clicks": regular, "jitter_clicks": jitter,
            "omission": omission, "phase_shift": phase.astype(np.float32), "speech": speech.astype(np.float32)}


def main() -> None:
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    results: dict[str, object] = {"zero_training": True, "patient_data_read": False, "probe_count": 6}
    try:
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        model_id = "nvidia/parakeet-tdt-0.6b-v3"
        model_dir = str(ROOT / "cache/parakeet_tdt_0.6b_v3")
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)
        model = AutoModelForTDT.from_pretrained(
            model_dir, trust_remote_code=False, dtype="auto", device_map="cuda:0", local_files_only=True
        ).eval()
        output_shapes: dict[str, object] = {}
        with torch.inference_mode():
            for name, waveform in make_probes().items():
                inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
                inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
                generated = model.generate(**inputs)
                sequences = getattr(generated, "sequences", generated)
                output_shapes[name] = {
                    "shape": list(sequences.shape),
                    "dtype": str(sequences.dtype),
                    "finite": bool(torch.isfinite(sequences).all().item()),
                }
        results["Parakeet-TDT"] = {
            "status": "PASS",
            "model_id": model_id,
            "endpoint": ENDPOINT,
            "model_dir": model_dir,
            "wall_seconds": time.perf_counter() - started,
            "model_class": type(model).__name__,
            "inference_output": output_shapes,
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
