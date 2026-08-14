"""Small, no-training M6A-PUBLIC-002 remote smoke/probe runner.

This file is copied to server2203 for execution. It intentionally uses only
synthetic probes and local model caches/code; no datasets or patient data are
read.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports"
PROBES = ROOT / "probes"
OUT.mkdir(parents=True, exist_ok=True)
PROBES.mkdir(parents=True, exist_ok=True)


def finite_shape(x: object) -> dict[str, object]:
    a = np.asarray(x)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "finite": bool(np.isfinite(a).all())}


def make_probes() -> dict[str, np.ndarray]:
    fs = 16000
    n = fs
    t = np.arange(n, dtype=np.float32) / fs
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    clicks = np.zeros(n, dtype=np.float32)
    clicks[np.arange(1000, n, 2000)] = 0.8
    jitter = np.zeros(n, dtype=np.float32)
    jitter[[1000, 2930, 5010, 6900, 9030, 11000, 13150, 15020]] = 0.8
    omission = clicks.copy()
    omission[5000] = 0.0
    phase = 0.2 * np.sin(2 * np.pi * 440 * t + np.pi / 2)
    speech = 0.08 * np.sin(2 * np.pi * (120 + 40 * t) * t) + 0.03 * np.sin(2 * np.pi * 730 * t)
    out = {"tone": tone, "regular_clicks": clicks, "jitter_clicks": jitter,
           "omission": omission, "phase_shift": phase, "speech": speech.astype(np.float32)}
    for name, audio in out.items():
        np.save(PROBES / f"{name}.npy", audio, allow_pickle=False)
    return out


def probe_distance(base: np.ndarray, other: np.ndarray) -> float:
    x = np.asarray(base, dtype=np.float64).reshape(-1)
    y = np.asarray(other, dtype=np.float64).reshape(-1)
    m = min(x.size, y.size)
    if m == 0:
        return float("nan")
    return float(np.linalg.norm(x[:m] - y[:m]) / max(np.linalg.norm(x[:m]), 1e-12))


def run_wav2vec2(probes: dict[str, np.ndarray]) -> dict[str, object]:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    path = "/home/fanyu/auditory_simulation_m6a/cache/huggingface/facebook_wav2vec2_base_main_20260813"
    t0 = time.perf_counter()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(path, local_files_only=True).eval()
    vals: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, audio in probes.items():
            kwargs = {"sampling_rate": 16000, "return_tensors": "pt", "padding": False}
            if getattr(fe, "return_attention_mask", False):
                kwargs["return_attention_mask"] = True
            batch = fe(audio, **kwargs)
            output = model(**batch, output_hidden_states=True)
            hidden = output.hidden_states
            vals[name] = hidden[-1].cpu().numpy()
    return {"status": "PASS", "model": "facebook/wav2vec2-base", "pretrained": True,
            "wall_seconds": time.perf_counter() - t0, "projected_plus_layers": len(hidden),
            "tone": finite_shape(vals["tone"]),
            "temporal_probe_relative_distance": {k: probe_distance(vals["tone"], v) for k, v in vals.items() if k != "tone"}}


def run_panns(probes: dict[str, np.ndarray]) -> dict[str, object]:
    import torch

    repo = "/home/fanyu/auditory_simulation_m6a/m6a_public_002/code/audioset_tagging_cnn-master"
    sys.path.insert(0, repo + "/pytorch")
    sys.path.insert(0, repo + "/utils")
    from models import Cnn14_16k

    t0 = time.perf_counter()
    old_cwd = os.getcwd()
    metadata_link = Path(repo) / "pytorch/metadata"
    if not metadata_link.exists():
        metadata_link.symlink_to("../metadata", target_is_directory=True)
    os.chdir(repo + "/pytorch")
    try:
        import config
        model = Cnn14_16k(16000, 512, 160, 64, 50, 8000, config.classes_num).eval()
        # This is the official PANNs checkpoint format; its state dictionary
        # contains NumPy metadata that the current weights_only unpickler rejects.
        checkpoint = torch.load(ROOT / "cache/Cnn14_16k_mAP_0.438.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        vals: dict[str, np.ndarray] = {}
        with torch.inference_mode():
            for name, audio in probes.items():
                out = model(torch.from_numpy(audio[None, :]), None)
                vals[name] = out["embedding"].numpy()
    finally:
        os.chdir(old_cwd)
    return {"status": "PASS", "model": "PANNs-CNN14", "pretrained": True,
            "wall_seconds": time.perf_counter() - t0, "tone": finite_shape(vals["tone"]),
            "temporal_probe_relative_distance": {k: probe_distance(vals["tone"], v) for k, v in vals.items() if k != "tone"}}


def run_connear() -> dict[str, object]:
    root = ROOT / "code/CoNNear_periphery-1.0"
    sys.path.insert(0, str(root))
    import keras.models
    from keras import Input

    # Keras 2.15 exposes Input at keras.Input, while the unmodified official
    # script imports it from keras.models. This is a runtime compatibility shim.
    keras.models.Input = Input
    keras.models.Model = keras.Model
    import connear_functions
    original_model_from_json = connear_functions.model_from_json
    connear_functions.model_from_json = lambda text, custom_objects=None: original_model_from_json(
        text, custom_objects={**(custom_objects or {}), "Model": keras.Model, "Input": Input}
    )
    t0 = time.perf_counter()
    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        namespace = runpy.run_path(str(root / "connear_example.py"), run_name="__main__")
    finally:
        os.chdir(old_cwd)
    arrays = {k: namespace[k] for k in ("vbm", "vihc", "anf_hsr", "anf_msr", "anf_lsr")}
    return {"status": "PASS", "model": "CoNNear-periphery", "pretrained": True,
            "wall_seconds": time.perf_counter() - t0, "stages": {k: finite_shape(v) for k, v in arrays.items()}}


def run_icnet() -> dict[str, object]:
    import tensorflow as tf
    import yaml
    from scipy import io as scipy_io

    root = ROOT / "code/ICNet-1.0"
    sys.path.insert(0, str(root / "src"))
    from ICNet_functions import simulate_model_responses

    params = yaml.safe_load((root / "DNN/config.yaml").read_text(encoding="utf-8"))
    fs = float(params["fs_audio"])
    duration = 1.0
    n = int(fs * duration)
    t = np.arange(n, dtype=np.float32) / fs
    audio = (0.03 * np.sin(2 * np.pi * 440 * t) + 0.01 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    context = int(params["context_size"])
    audio = np.pad(audio, (context, 0))
    audio = audio[: audio.size - (audio.size % 32)]
    audio_input = audio[None, :, None]
    unit_cfs = scipy_io.loadmat(root / "DNN/channel_CFs.mat")["channel_CFs"][0]
    t0 = time.perf_counter()
    with tf.device("/CPU:0"):
        bottleneck, _ = simulate_model_responses(
            str(root / "DNN"), audio_input.copy(), "bottleneck", time_input=0.0, **params
        )
        units, cfs = simulate_model_responses(
            str(root / "DNN"), audio_input.copy(), "units_1000", time_input=0.0,
            channel_CFs=unit_cfs, **params
        )
    return {"status": "PASS", "model": "ICNet", "pretrained": True,
            "wall_seconds": time.perf_counter() - t0,
            "input": finite_shape(audio_input), "bottleneck": finite_shape(bottleneck),
            "units_1000": finite_shape(units), "cfs_present": bool(np.asarray(cfs).size),
            "temporal_probe": "synthetic tone; no IC recordings read"}


def main() -> None:
    probes = make_probes()
    results: dict[str, object] = {"probe_count": len(probes), "zero_training": True, "patient_data_read": False}
    for key, fn in (("wav2vec2", run_wav2vec2), ("PANNs-CNN14", run_panns)):
        try:
            results[key] = fn(probes)
        except Exception as exc:  # retain failure evidence and continue
            results[key] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    try:
        results["CoNNear-periphery"] = run_connear()
    except Exception as exc:
        results["CoNNear-periphery"] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    try:
        results["ICNet"] = run_icnet()
    except Exception as exc:
        results["ICNet"] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    (OUT / "m6a_public_002_stage_bc_smoke.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
