"""Low-cost pretrained probes for M6A-PUBLIC-002; no training or patient data."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_lowcost_probes.json"
HF_ENDPOINT = "https://hf-mirror.com"


def info(value: object) -> dict[str, object]:
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": bool(np.isfinite(array).all())}


def make_probes() -> dict[str, np.ndarray]:
    fs = 16000
    t = np.arange(fs, dtype=np.float32) / fs
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    clicks = np.zeros(fs, dtype=np.float32)
    clicks[np.arange(1000, fs, 2000)] = 0.8
    jitter = np.zeros(fs, dtype=np.float32)
    jitter[[1000, 2930, 5010, 6900, 9030, 11000, 13150, 15020]] = 0.8
    omission = clicks.copy()
    omission[5000] = 0.0
    phase = 0.2 * np.sin(2 * np.pi * 440 * t + np.pi / 2)
    speech = 0.08 * np.sin(2 * np.pi * (120 + 40 * t) * t)
    return {"tone": tone, "regular_clicks": clicks, "jitter_clicks": jitter,
            "omission": omission, "phase_shift": phase.astype(np.float32), "speech": speech.astype(np.float32)}


def run_convtasnet(values: dict[str, np.ndarray]) -> dict[str, object]:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
    import torch
    from asteroid.models import ConvTasNet

    model_id = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
    started = time.perf_counter()
    model = ConvTasNet.from_pretrained(model_id).eval()
    outputs: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, audio in values.items():
            outputs[name] = model(torch.from_numpy(audio[None, None, :])).cpu().numpy()
    return {
        "status": "PASS",
        "model": "ConvTasNet",
        "model_id": model_id,
        "endpoint": HF_ENDPOINT,
        "wall_seconds": time.perf_counter() - started,
        "tone": info(outputs["tone"]),
        "temporal_probe_shapes": {name: info(value) for name, value in outputs.items()},
    }


def main() -> None:
    results: dict[str, object] = {"zero_training": True, "patient_data_read": False, "probe_count": 6}
    try:
        results["ConvTasNet"] = run_convtasnet(make_probes())
    except Exception as exc:
        results["ConvTasNet"] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
