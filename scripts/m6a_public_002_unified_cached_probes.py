"""Lightweight unified no-fit probe summaries using only existing 2203 caches."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_unified_cached_probes.json"


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
    return {"tone": tone.astype(np.float32), "regular_clicks": regular, "jitter_clicks": jitter,
            "omission": omission, "phase_shift": phase.astype(np.float32), "speech": speech.astype(np.float32)}


def relative_distance(reference: np.ndarray, value: np.ndarray) -> float:
    x = np.asarray(reference, dtype=np.float64).reshape(-1)
    y = np.asarray(value, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    return float(np.linalg.norm(x[:n] - y[:n]) / max(np.linalg.norm(x[:n]), 1e-12))


def persistence(value: np.ndarray, time_axis: int = -2) -> float | None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[time_axis] < 2:
        return None
    array = np.moveaxis(array, time_axis, -1)
    trace = np.mean(np.abs(array), axis=tuple(range(array.ndim - 1)))
    left, right = trace[:-1], trace[1:]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def event_response(value: np.ndarray, time_axis: int = -1) -> float | None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return None
    array = np.moveaxis(array, time_axis, -1)
    trace = np.mean(np.abs(array), axis=tuple(range(array.ndim - 1)))
    indexes = [max(0, min(trace.size - 1, int(fraction * trace.size))) for fraction in (0.25, 0.5, 0.75)]
    width = max(1, trace.size // 20)
    values = np.concatenate([trace[index:min(trace.size, index + width)] for index in indexes])
    return float(np.mean(values)) if values.size else None


def summarize(outputs: dict[str, np.ndarray], time_axis: int = -2) -> dict[str, object]:
    reference = outputs["tone"]
    return {
        "probe_names": list(outputs),
        "shapes": {name: list(np.asarray(value).shape) for name, value in outputs.items()},
        "finite": all(bool(np.isfinite(value).all()) for value in outputs.values()),
        "relative_distance_to_tone": {name: relative_distance(reference, value) for name, value in outputs.items() if name != "tone"},
        "persistence_lag1": persistence(reference, time_axis=time_axis),
        "event_triggered_mean": event_response(reference, time_axis=-1),
    }


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
    try:
        import config
        model = Cnn14_16k(16000, 512, 160, 64, 50, 8000, config.classes_num).eval()
        checkpoint = torch.load(ROOT / "cache/Cnn14_16k_mAP_0.438.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        outputs = {}
        with torch.inference_mode():
            for name, audio in values.items():
                outputs[name] = model(torch.from_numpy(audio[None, :]), None)["embedding"].numpy()
    finally:
        os.chdir(old_cwd)
    return summarize(outputs, time_axis=-1)


def run_convtasnet(values: dict[str, np.ndarray]) -> dict[str, object]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    import torch
    from asteroid.models import ConvTasNet
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k").eval()
    outputs = {}
    with torch.inference_mode():
        for name, audio in values.items():
            outputs[name] = model(torch.from_numpy(audio[None, None, :])).cpu().numpy()
    return summarize(outputs, time_axis=-1)


def run_wav2vec2(values: dict[str, np.ndarray]) -> dict[str, object]:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    path = "/home/fanyu/auditory_simulation_m6a/cache/huggingface/facebook_wav2vec2_base_main_20260813"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(path, local_files_only=True).eval()
    outputs = {}
    with torch.inference_mode():
        for name, audio in values.items():
            kwargs = {"sampling_rate": 16000, "return_tensors": "pt", "padding": False}
            if getattr(extractor, "return_attention_mask", False):
                kwargs["return_attention_mask"] = True
            batch = extractor(audio, **kwargs)
            outputs[name] = model(**batch, output_hidden_states=True).last_hidden_state.cpu().numpy()
    return summarize(outputs, time_axis=1)


def run_speechbrain(values: dict[str, np.ndarray]) -> dict[str, object]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
    import torch
    from speechbrain.inference.ASR import EncoderDecoderASR
    model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir=str(ROOT / "cache/speechbrain_crdnn_rnnlm_librispeech"),
        run_opts={"device": "cpu"},
    )
    outputs = {}
    with torch.inference_mode():
        for name, audio in values.items():
            outputs[name] = model.encode_batch(torch.from_numpy(audio[None, :]), torch.tensor([1.0])).cpu().numpy()
    return summarize(outputs, time_axis=1)


def main() -> None:
    values = make_probes()
    if os.environ.get("M6A_ONLY_MODEL") == "PANNs-CNN14":
        results = {"zero_training": True, "patient_data_read": False, "probe_count": len(values), "PANNs-CNN14": run_panns(values)}
        output = Path(os.environ.get("M6A_UNIFIED_OUT", str(OUT)))
        output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        return
    results: dict[str, object] = {"zero_training": True, "patient_data_read": False, "probe_count": len(values)}
    for name, runner in (("PANNs-CNN14", run_panns), ("ConvTasNet", run_convtasnet), ("wav2vec2", run_wav2vec2), ("SpeechBrain-CRDNN", run_speechbrain)):
        try:
            started = time.perf_counter()
            result = runner(values)
            result["status"] = "PASS"
            result["wall_seconds"] = time.perf_counter() - started
            results[name] = result
        except Exception as exc:
            results[name] = {"status": "BLOCKED_BY_UPSTREAM_DEPENDENCY", "error_type": type(exc).__name__, "error": str(exc)}
    OUT.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
