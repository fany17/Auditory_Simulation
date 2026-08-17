"""Unified no-fit temporal probes for CoNNear and ICNet on server2203."""

from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_auditory_physio_probes.json"


def probes(fs: float, duration: float) -> dict[str, np.ndarray]:
    n = int(fs * duration)
    t = np.arange(n, dtype=np.float32) / fs
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    regular = np.zeros(n, dtype=np.float32)
    regular[[int(0.025 * fs), int(0.050 * fs), int(0.075 * fs)]] = 0.8
    jitter = np.zeros(n, dtype=np.float32)
    jitter[[int(0.025 * fs), int(0.048 * fs), int(0.078 * fs)]] = 0.8
    omission = regular.copy()
    omission[int(0.050 * fs)] = 0.0
    phase = 0.2 * np.sin(2 * np.pi * 440 * t + np.pi / 2)
    speech = 0.08 * np.sin(2 * np.pi * (120 + 40 * t) * t) + 0.03 * np.sin(2 * np.pi * 730 * t)
    return {"tone": tone.astype(np.float32), "regular_clicks": regular,
            "jitter_clicks": jitter, "omission": omission,
            "phase_shift": phase.astype(np.float32), "speech": speech.astype(np.float32)}


def array_info(value: object) -> dict[str, object]:
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": bool(np.isfinite(array).all())}


def distance(reference: np.ndarray, value: np.ndarray) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(-1)
    y = np.asarray(value, dtype=np.float64).reshape(-1)
    size = min(x.size, y.size)
    return float(np.linalg.norm(x[:size] - y[:size]) / max(np.linalg.norm(x[:size]), 1e-12))


def persistence(value: np.ndarray) -> float | None:
    signal = np.asarray(value, dtype=np.float64)
    signal = np.mean(np.abs(signal), axis=tuple(range(2, signal.ndim))) if signal.ndim > 2 else signal
    x = signal.reshape(signal.shape[0], -1)
    if x.shape[1] < 2:
        return None
    left = x[:, :-1].reshape(-1)
    right = x[:, 1:].reshape(-1)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def event_response(value: np.ndarray, fs: float, event_times: list[float]) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2:
        array = array[:, :, None]
    trace = np.mean(np.abs(array), axis=(0, 2))
    width = max(1, int(0.010 * fs))
    windows = []
    for event in event_times:
        start = min(trace.size, max(0, int(event * fs)))
        windows.append(trace[start:min(trace.size, start + width)])
    selected = np.concatenate([window for window in windows if window.size]) if windows else np.array([])
    return float(np.mean(selected)) if selected.size else float("nan")


def run_connear() -> dict[str, object]:
    root = ROOT / "code/CoNNear_periphery-1.0"
    sys.path.insert(0, str(root))
    import keras.models
    from keras import Input

    keras.models.Input = Input
    keras.models.Model = keras.Model
    import connear_functions

    original = connear_functions.model_from_json
    connear_functions.model_from_json = lambda text, custom_objects=None: original(
        text, custom_objects={**(custom_objects or {}), "Model": keras.Model, "Input": Input}
    )
    import scipy.signal as signal

    fs = 20000.0
    context_left = 7936
    context_right = 256
    values = probes(fs, 0.1)
    modeldir = str(root / "connear") + "/"
    started = time.perf_counter()
    old_cwd = os.getcwd()
    os.chdir(root)
    try:
        cochlea, ihc, anfh, anfm, anfl = connear_functions.build_connear(modeldir, poles="", cf_flag=1)
        batch = []
        for waveform in values.values():
            signal_wave = np.pad(waveform, (context_left + int(0.005 * fs), context_right))
            if signal_wave.size % 16384:
                signal_wave = np.pad(signal_wave, (0, 16384 - signal_wave.size % 16384))
            batch.append(signal_wave)
        stimulus = np.asarray(batch, dtype=np.float32)[:, :, None]
        vbm = cochlea.predict(stimulus, verbose=0)
        vihc = ihc.predict(vbm, verbose=0)
        anf_hsr = anfh.predict(vihc, verbose=0)
        anf_msr = anfm.predict(vihc, verbose=0)
        anf_lsr = anfl.predict(vihc, verbose=0)
    finally:
        os.chdir(old_cwd)
    outputs = {"BM": vbm, "IHC": vihc, "ANF-H": anf_hsr, "ANF-M": anf_msr, "ANF-L": anf_lsr}
    per_probe = {name: {stage: array_info(value[index]) for stage, value in outputs.items()} for index, name in enumerate(values)}
    distances = {stage: {name: distance(value[0], value[index]) for index, name in enumerate(values) if index} for stage, value in outputs.items()}
    return {"status": "PASS", "model": "CoNNear-periphery", "wall_seconds": time.perf_counter() - started,
            "sample_rate": fs, "outputs": ["BM", "IHC", "ANF-H", "ANF-M", "ANF-L"],
            "per_probe": per_probe, "relative_distance_to_tone": distances,
            "persistence_lag1": {stage: persistence(value) for stage, value in outputs.items()},
            "event_triggered_mean": {stage: event_response(value, fs, [0.025, 0.050, 0.075]) for stage, value in outputs.items()},
            "synthetic_only": True}


def run_icnet() -> dict[str, object]:
    import tensorflow as tf
    import yaml
    from scipy import io as scipy_io

    root = ROOT / "code/ICNet-1.0"
    sys.path.insert(0, str(root / "src"))
    from ICNet_functions import simulate_model_responses
    params = yaml.safe_load((root / "DNN/config.yaml").read_text(encoding="utf-8"))
    fs = float(params["fs_audio"])
    values = probes(fs, 1.0)
    channel_cfs = scipy_io.loadmat(root / "DNN/channel_CFs.mat")["channel_CFs"][0]
    outputs: dict[str, dict[str, np.ndarray]] = {}
    started = time.perf_counter()
    with tf.device("/CPU:0"):
        for name, waveform in values.items():
            context = int(params["context_size"])
            audio = np.pad(waveform, (context, 0))
            audio = audio[: audio.size - (audio.size % 32)][None, :, None]
            bottleneck, _ = simulate_model_responses(str(root / "DNN"), audio.copy(), "bottleneck", time_input=0.0, **params)
            units, _ = simulate_model_responses(str(root / "DNN"), audio.copy(), "units_1000", time_input=0.0, channel_CFs=channel_cfs, **params)
            outputs[name] = {"bottleneck": bottleneck, "units_1000": units}
    reference = outputs["tone"]
    return {"status": "PASS", "model": "ICNet", "wall_seconds": time.perf_counter() - started,
            "per_probe": {name: {stage: array_info(value) for stage, value in stages.items()} for name, stages in outputs.items()},
            "relative_distance_to_tone": {stage: {name: distance(reference[stage], stages[stage]) for name, stages in outputs.items() if name != "tone"} for stage in reference},
            "persistence_lag1": {stage: persistence(np.stack([stages[stage] for stages in outputs.values()])) for stage in reference},
            "synthetic_only": True}


def main() -> None:
    results: dict[str, object] = {"zero_training": True, "patient_data_read": False}
    try:
        results["CoNNear-periphery"] = run_connear()
    except Exception as exc:
        results["CoNNear-periphery"] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    try:
        root = ROOT / "code/ICNet-1.0"
        sys.path.insert(0, str(root / "src"))
        results["ICNet"] = run_icnet()
    except Exception as exc:
        results["ICNet"] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
