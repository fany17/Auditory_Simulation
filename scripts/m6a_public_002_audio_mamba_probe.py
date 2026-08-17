"""Official Audio-Mamba SSAM tiny synthetic probe on server2203."""

from __future__ import annotations

import json
import os
import argparse
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
CODE = ROOT / "cache/audio_mamba_ssam/code/audio-mamba-official-master"
WEIGHTS = ROOT / "cache/audio_mamba_ssam/weights"
PROBES = ROOT / "probes"
OUT = ROOT / "reports/m6a_public_002_audio_mamba_probe.json"


def load_probes() -> dict[str, np.ndarray]:
    names = ("tone", "regular_clicks", "jitter_clicks", "omission", "phase_shift", "speech")
    values = {}
    for name in names:
        value = np.load(PROBES / f"{name}.npy", allow_pickle=False)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"invalid synthetic probe: {name}")
        values[name] = value
    return values


def summarize(outputs: dict[str, np.ndarray]) -> dict[str, object]:
    reference = outputs["tone"]
    reference_flat = reference.reshape(-1).astype(np.float64)
    relative_distance = {}
    for name, value in outputs.items():
        if name == "tone":
            continue
        flat = value.reshape(-1).astype(np.float64)
        n = min(reference_flat.size, flat.size)
        relative_distance[name] = float(
            np.linalg.norm(reference_flat[:n] - flat[:n])
            / max(np.linalg.norm(reference_flat[:n]), 1e-12)
        )

    persistence = {}
    event_response = {}
    for name, value in outputs.items():
        trace = np.mean(np.abs(value), axis=(0, 2))
        if trace.size < 2 or np.std(trace[:-1]) == 0 or np.std(trace[1:]) == 0:
            persistence[name] = None
        else:
            persistence[name] = float(np.corrcoef(trace[:-1], trace[1:])[0, 1])
        indexes = [
            max(0, min(trace.size - 1, int(fraction * trace.size)))
            for fraction in (0.25, 0.5, 0.75)
        ]
        width = max(1, trace.size // 20)
        selected = np.concatenate(
            [trace[index : min(trace.size, index + width)] for index in indexes]
        )
        event_response[name] = float(np.mean(selected)) if selected.size else None

    return {
        "probe_names": list(outputs),
        "shapes": {name: list(value.shape) for name, value in outputs.items()},
        "finite": all(bool(np.isfinite(value).all()) for value in outputs.values()),
        "relative_distance_to_tone": relative_distance,
        "persistence_lag1": persistence,
        "event_triggered_mean": event_response,
    }


def main() -> None:
    os.environ.setdefault("PT_MAMBA_MODEL_DIR", str(WEIGHTS))
    sys.path.insert(0, str(CODE))
    import torch
    from ml_collections import ConfigDict
    from hear_api.runtime import RuntimeSSAST
    from configs.ssam_tiny_200_16x4 import get_config

    # PyTorch 2.6 defaults to weights_only=True.  The official checkpoint
    # contains argparse.Namespace metadata; allowlist only that safe type.
    torch.serialization.add_safe_globals([argparse.Namespace, ConfigDict])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official tiny inference")

    values = load_probes()
    config = get_config()
    model = RuntimeSSAST(config=config, weights_dir=str(WEIGHTS), precision="float32").cuda().eval()
    outputs: dict[str, np.ndarray] = {}
    timestamps: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for name, audio in values.items():
            tensor = torch.from_numpy(audio).cuda().unsqueeze(0)
            embedding, timestamp = model.get_timestamp_embeddings(tensor)
            outputs[name] = embedding.detach().float().cpu().numpy()
            timestamps[name] = timestamp.detach().float().cpu().numpy()
    torch.cuda.synchronize()

    summary = summarize(outputs)
    summary.update(
        {
            "status": "PASS",
            "config": "ssam_tiny_200_16x4",
            "checkpoint_path": str(WEIGHTS / "checkpoints/checkpoint-99.pth"),
            "sample_rate": 16000,
            "wall_seconds": time.perf_counter() - started,
            "timestamp_shapes": {name: list(value.shape) for name, value in timestamps.items()},
            "timestamp_finite": all(bool(np.isfinite(value).all()) for value in timestamps.values()),
            "zero_training": True,
            "patient_data_read": False,
            "downstream_probe": False,
        }
    )
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
