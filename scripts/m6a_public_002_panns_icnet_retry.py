"""One-shot PANNs retry and ICNet official synthetic probe for M6A-PUBLIC-002."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_panns_icnet_retry.json"


def shape_info(value: object) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
    }


def probes() -> dict[str, np.ndarray]:
    sample_rate = 16000
    time_axis = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * time_axis)
    clicks = np.zeros(sample_rate, dtype=np.float32)
    clicks[np.arange(1000, sample_rate, 2000)] = 0.8
    jitter = np.zeros(sample_rate, dtype=np.float32)
    jitter[[1000, 2930, 5010, 6900, 9030, 11000, 13150, 15020]] = 0.8
    omission = clicks.copy()
    omission[5000] = 0.0
    phase = 0.2 * np.sin(2 * np.pi * 440 * time_axis + np.pi / 2)
    speech = 0.08 * np.sin(2 * np.pi * (120 + 40 * time_axis) * time_axis)
    return {
        "tone": tone.astype(np.float32),
        "regular_clicks": clicks,
        "jitter_clicks": jitter,
        "omission": omission,
        "phase_shift": phase.astype(np.float32),
        "speech": speech.astype(np.float32),
    }


def distance(base: np.ndarray, other: np.ndarray) -> float:
    x = np.asarray(base, dtype=np.float64).reshape(-1)
    y = np.asarray(other, dtype=np.float64).reshape(-1)
    length = min(x.size, y.size)
    return float(np.linalg.norm(x[:length] - y[:length]) / max(np.linalg.norm(x[:length]), 1e-12))


def run_panns(values: dict[str, np.ndarray]) -> dict[str, object]:
    import torch

    repo = ROOT / "code/audioset_tagging_cnn-master"
    sys.path.insert(0, str(repo / "pytorch"))
    sys.path.insert(0, str(repo / "utils"))
    from models import Cnn14_16k

    metadata_link = repo / "pytorch/metadata"
    if not metadata_link.exists():
        metadata_link.symlink_to("../metadata", target_is_directory=True)
    old_cwd = os.getcwd()
    os.chdir(repo / "pytorch")
    started = time.perf_counter()
    try:
        import config

        model = Cnn14_16k(16000, 512, 160, 64, 50, 8000, config.classes_num).eval()
        checkpoint = torch.load(
            ROOT / "cache/Cnn14_16k_mAP_0.438.pth",
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        outputs: dict[str, np.ndarray] = {}
        with torch.inference_mode():
            for name, audio in values.items():
                outputs[name] = model(torch.from_numpy(audio[None, :]), None)["embedding"].numpy()
    finally:
        os.chdir(old_cwd)
    return {
        "status": "PASS",
        "model": "PANNs-CNN14",
        "checkpoint": {
            "file": "Cnn14_16k_mAP_0.438.pth",
            "bytes": (ROOT / "cache/Cnn14_16k_mAP_0.438.pth").stat().st_size,
        },
        "wall_seconds": time.perf_counter() - started,
        "tone": shape_info(outputs["tone"]),
        "temporal_probe_relative_distance": {
            name: distance(outputs["tone"], value)
            for name, value in outputs.items()
            if name != "tone"
        },
    }


def run_icnet() -> dict[str, object]:
    import tensorflow as tf
    import yaml
    from scipy import io as scipy_io

    root = ROOT / "code/ICNet-1.0"
    sys.path.insert(0, str(root / "src"))
    from ICNet_functions import simulate_model_responses

    params = yaml.safe_load((root / "DNN/config.yaml").read_text(encoding="utf-8"))
    sample_rate = float(params["fs_audio"])
    time_axis = np.arange(int(sample_rate), dtype=np.float32) / sample_rate
    audio = 0.03 * np.sin(2 * np.pi * 440 * time_axis) + 0.01 * np.sin(2 * np.pi * 880 * time_axis)
    context = int(params["context_size"])
    audio = np.pad(audio.astype(np.float32), (context, 0))
    audio = audio[: audio.size - (audio.size % 32)]
    audio_input = audio[None, :, None]
    channel_cfs = scipy_io.loadmat(root / "DNN/channel_CFs.mat")["channel_CFs"][0]
    started = time.perf_counter()
    with tf.device("/CPU:0"):
        bottleneck, _ = simulate_model_responses(
            str(root / "DNN"), audio_input.copy(), "bottleneck", time_input=0.0, **params
        )
        units, cfs = simulate_model_responses(
            str(root / "DNN"),
            audio_input.copy(),
            "units_1000",
            time_input=0.0,
            channel_CFs=channel_cfs,
            **params,
        )
    return {
        "status": "PASS",
        "model": "ICNet",
        "wall_seconds": time.perf_counter() - started,
        "input": shape_info(audio_input),
        "bottleneck": shape_info(bottleneck),
        "units_1000": shape_info(units),
        "cfs_present": bool(np.asarray(cfs).size),
        "temporal_probe": "synthetic tone; no IC recordings read",
    }


def main() -> None:
    results: dict[str, object] = {
        "zero_training": True,
        "patient_data_read": False,
        "probe_count": len(probes()),
    }
    try:
        results["PANNs-CNN14"] = run_panns(probes())
    except Exception as exc:
        results["PANNs-CNN14"] = {
            "status": "BLOCKED_BY_UPSTREAM_DEPENDENCY",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        results["ICNet"] = run_icnet()
    except Exception as exc:
        results["ICNet"] = {
            "status": "BLOCKED_BY_UPSTREAM_DEPENDENCY",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
