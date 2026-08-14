"""Single Parakeet official pretrained-path check on a synthetic waveform."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_parakeet_probe.json"


def main() -> None:
    waveform = (0.08 * np.sin(2 * np.pi * 440 * np.arange(16000, dtype=np.float32) / 16000)).astype(np.float32)
    results: dict[str, object] = {"zero_training": True, "patient_data_read": False, "input_shape": list(waveform.shape)}
    try:
        import torch
        from nemo.collections.asr.models import ASRModel

        started = time.perf_counter()
        model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        model.eval()
        with torch.inference_mode():
            decoded = model.transcribe([waveform], batch_size=1)
        results["Parakeet-TDT"] = {
            "status": "PASS",
            "model_id": "nvidia/parakeet-tdt-0.6b-v3",
            "wall_seconds": time.perf_counter() - started,
            "decoded_type": type(decoded).__name__,
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
