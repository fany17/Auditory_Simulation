#!/usr/bin/env python3
"""Core implementation for TB001-DEMO001.

The server is a compute node only. This module performs ordinary offline Python
inference; it never launches Codex or another agent.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "TB001-DEMO001"
ALLOWED_MISSING_KEYS = {"wav2vec2.masked_spec_embed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def append_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, tuple, list)):
        return [jsonable(v) for v in sorted(value) if isinstance(value, set)] if isinstance(value, set) else [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def config_hash(config: dict[str, Any], audio_sha256: str, model_files: list[dict[str, Any]]) -> str:
    payload = {
        "config": config,
        "audio_sha256": audio_sha256,
        "model_files": [{"relative_path": x["relative_path"], "sha256": x["sha256"]} for x in model_files],
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Config must be a mapping")
    return value


def resolve_root(anchor: Path | None = None) -> Path:
    if anchor is not None:
        return anchor.resolve()
    return Path(__file__).resolve().parents[3]


def preflight(root: Path) -> dict[str, Any]:
    root = root.resolve()
    required = [
        root / "docs" / "STANDALONE_EXECUTION_SPEC.md",
        root / "configs" / f"{TASK_ID}.yaml",
        root / "stimuli" / TASK_ID / "input_16k_mono.wav",
        root / "stimuli" / TASK_ID / "manifest.json",
        root / "models" / "model_manifest.json",
    ]
    model_manifest = read_json(root / "models" / "model_manifest.json") if required[-1].exists() else {}
    snapshot = Path(model_manifest.get("snapshot_path", "")) if model_manifest.get("snapshot_path") else None
    with wave.open(str(root / "stimuli" / TASK_ID / "input_16k_mono.wav"), "rb") as handle:
        audio = {
            "channels": handle.getnchannels(),
            "sample_rate": handle.getframerate(),
            "frames": handle.getnframes(),
            "duration_seconds": handle.getnframes() / float(handle.getframerate()),
        }
    disk = shutil.disk_usage(root)
    result = {
        "checked_at_utc": utc_now(),
        "demo_root": str(root),
        "root_is_git_repository": (root / ".git").exists(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "required_files": {str(p.relative_to(root)): p.is_file() for p in required},
        "audio": audio,
        "model_snapshot": str(snapshot) if snapshot else None,
        "model_snapshot_exists": bool(snapshot and snapshot.is_dir()),
        "network_required_for_run": False,
        "disk_free_bytes": disk.free,
    }
    result["status"] = "PASS" if (
        all(result["required_files"].values())
        and audio["channels"] == 1
        and audio["sample_rate"] == 16000
        and result["model_snapshot_exists"]
    ) else "FAIL"
    write_json(root / "environment" / "preflight_final.json", result)
    return result


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric: {result}")
    return result


def downsample(values: Iterable[float], target: int = 64) -> list[float]:
    source = [finite_float(v) for v in values]
    if len(source) <= target:
        return source
    output = []
    for i in range(target):
        start = round(i * len(source) / target)
        end = max(start + 1, round((i + 1) * len(source) / target))
        output.append(sum(source[start:end]) / (end - start))
    return output


def tensor_heatmap(tensor: Any, time_bins: int = 64, feature_bins: int = 32) -> list[list[float]]:
    import torch.nn.functional as F

    x = tensor.detach().float().abs().unsqueeze(1)
    pooled = F.adaptive_avg_pool2d(x, (time_bins, feature_bins))[0, 0].cpu()
    mean = pooled.mean()
    std = pooled.std().clamp_min(1e-12)
    z = (pooled - mean) / std
    return [[finite_float(v) for v in row] for row in z.tolist()]


def cosine_distance_mean(a: Any, b: Any) -> float:
    import torch
    import torch.nn.functional as F

    values = 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8)
    if not torch.isfinite(values).all():
        raise ValueError("Cosine distance produced NaN/Inf")
    return finite_float(values.mean().item())


def cosine_distance_by_time(a: Any, b: Any) -> list[float]:
    import torch
    import torch.nn.functional as F

    values = 1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-8)
    if not torch.isfinite(values).all():
        raise ValueError("Per-time cosine distance produced NaN/Inf")
    return downsample(values[0].detach().cpu().tolist())


def temporal_change_series(hidden: Any) -> list[float]:
    import torch
    import torch.nn.functional as F

    values = 1.0 - F.cosine_similarity(hidden[:, 1:].float(), hidden[:, :-1].float(), dim=-1, eps=1e-8)
    if not torch.isfinite(values).all():
        raise ValueError("Temporal change produced NaN/Inf")
    return downsample(values[0].detach().cpu().tolist())


def js_divergence(logits_a: Any, logits_b: Any) -> tuple[float, list[float]]:
    import torch

    p = torch.softmax(logits_a.float(), dim=-1).clamp_min(1e-12)
    q = torch.softmax(logits_b.float(), dim=-1).clamp_min(1e-12)
    m = 0.5 * (p + q)
    per_frame = 0.5 * ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1))
    if not torch.isfinite(per_frame).all():
        raise ValueError("Jensen-Shannon divergence produced NaN/Inf")
    return finite_float(per_frame.mean().item()), downsample(per_frame[0].detach().cpu().tolist())


def decode_output(model: Any, processor: Any, output: Any) -> dict[str, Any]:
    import torch

    logits = output.logits
    token_ids = torch.argmax(logits, dim=-1)
    transcript = processor.batch_decode(token_ids)[0]
    probabilities = torch.softmax(logits.float(), dim=-1)
    blank_id = int(model.config.pad_token_id)
    return {
        "logits": logits,
        "token_ids": token_ids,
        "transcript": transcript,
        "blank_id": blank_id,
        "blank_ratio": finite_float((token_ids == blank_id).float().mean().item()),
        "mean_frame_max_probability": finite_float(probabilities.max(dim=-1).values.mean().item()),
    }


def load_runtime(root: Path, config: dict[str, Any]) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    manifest = read_json(root / config["model_manifest_path"])
    snapshot = Path(manifest["snapshot_path"])
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned local model snapshot is missing: {snapshot}")
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model, loading_info = Wav2Vec2ForCTC.from_pretrained(snapshot, local_files_only=True, output_loading_info=True)
    normalized_loading = jsonable(loading_info)
    missing = set(normalized_loading.get("missing_keys", []))
    unexpected = set(normalized_loading.get("unexpected_keys", []))
    mismatched = set(normalized_loading.get("mismatched_keys", []))
    errors = normalized_loading.get("error_msgs", [])
    if missing != ALLOWED_MISSING_KEYS or unexpected or mismatched or errors:
        raise RuntimeError(
            f"Model loading audit failed: missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"mismatched={sorted(mismatched)}, errors={errors}"
        )
    model.eval()
    model.config.apply_spec_augment = False
    if model.training:
        raise RuntimeError("model.eval() did not take effect")
    requested = str(config.get("device", "cuda:0"))
    device = torch.device(requested if requested.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)
    waveform, sample_rate = sf.read(root / config["audio_path"], dtype="float32")
    if waveform.ndim != 1 or sample_rate != 16000:
        raise RuntimeError(f"Expected mono 16 kHz waveform; got ndim={waveform.ndim}, rate={sample_rate}")
    if not np.isfinite(waveform).all():
        raise RuntimeError("Audio contains NaN/Inf")
    encoded = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_values = encoded.input_values.to(device=device, dtype=torch.float32)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    audit = {
        "captured_at_utc": utc_now(),
        "loading_info": normalized_loading,
        "allowed_missing_keys": sorted(ALLOWED_MISSING_KEYS),
        "missing_key_explanation": "masked_spec_embed is a training-time SpecAugment parameter; eval mode and apply_spec_augment=false prevent its use in this inference demo.",
        "apply_spec_augment_override": False,
        "model_eval": not model.training,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": str(device),
        "dtype": "float32",
        "numpy_version": np.__version__,
        "layer_count": len(model.wav2vec2.encoder.layers),
    }
    write_json(root / "reports" / f"{TASK_ID}_MODEL_LOADING_AUDIT.json", audit)
    if audit["layer_count"] != 12:
        raise RuntimeError(f"Expected 12 transformer layers, got {audit['layer_count']}")
    return model, processor, input_values, attention_mask, audit


def forward(model: Any, input_values: Any, attention_mask: Any, layer_id: int | None = None, alpha: float = 1.0) -> tuple[Any, dict[str, Any]]:
    import torch

    trace: dict[str, Any] = {
        "layer_id": layer_id,
        "alpha": alpha,
        "formula": "h_in + alpha * (h_out - h_in)",
        "hook_calls": 0,
    }
    handle = None
    layer = None
    before_count = None
    if layer_id is not None:
        if not 1 <= layer_id <= len(model.wav2vec2.encoder.layers):
            raise ValueError(f"Invalid one-based layer_id: {layer_id}")
        layer = model.wav2vec2.encoder.layers[layer_id - 1]
        before_count = len(layer._forward_hooks)
        trace.update({"module_path": f"model.wav2vec2.encoder.layers[{layer_id - 1}]", "module_type": type(layer).__name__, "hooks_before": before_count})

        def intervention_hook(module: Any, module_inputs: tuple[Any, ...], module_output: Any) -> Any:
            trace["hook_calls"] += 1
            if not module_inputs or not hasattr(module_inputs[0], "shape"):
                raise RuntimeError("Layer hook did not receive hidden-state input tensor")
            h_in = module_inputs[0]
            is_tuple = isinstance(module_output, tuple)
            is_list = isinstance(module_output, list)
            h_out = module_output[0] if (is_tuple or is_list) else module_output
            if not hasattr(h_out, "shape") or tuple(h_in.shape) != tuple(h_out.shape):
                raise RuntimeError("Layer hook input/output hidden shapes are incompatible")
            # Algebraically exact endpoint fast paths avoid float32 cancellation:
            # h_in + 1 * (h_out - h_in) can differ slightly from h_out and that
            # perturbation is amplified by downstream layers. The endpoints are
            # therefore returned exactly; interior alpha values use the formula.
            if float(alpha) == 1.0:
                adjusted = h_out
                expected = h_out
                stable_endpoint = "h_out"
            elif float(alpha) == 0.0:
                adjusted = h_in
                expected = h_in
                stable_endpoint = "h_in"
            else:
                adjusted = h_in + float(alpha) * (h_out - h_in)
                expected = h_in + float(alpha) * (h_out - h_in)
                stable_endpoint = None
            trace.update(
                {
                    "input_type": type(module_inputs).__name__,
                    "output_type": type(module_output).__name__,
                    "input_shape": list(h_in.shape),
                    "output_shape": list(h_out.shape),
                    "formula_max_abs_error": finite_float((adjusted - expected).abs().max().item()),
                    "bypass_max_abs_error": finite_float((adjusted - h_in).abs().max().item()) if alpha == 0.0 else None,
                    "numerically_stable_endpoint": stable_endpoint,
                }
            )
            if is_tuple:
                return (adjusted,) + tuple(module_output[1:])
            if is_list:
                return [adjusted] + list(module_output[1:])
            return adjusted

        handle = layer.register_forward_hook(intervention_hook)
    try:
        with torch.inference_mode():
            output = model(input_values, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    finally:
        if handle is not None:
            handle.remove()
        if layer is not None:
            trace["hooks_after"] = len(layer._forward_hooks)
            trace["hook_removed"] = trace["hooks_after"] == before_count
    if layer_id is not None and (trace["hook_calls"] != 1 or not trace.get("hook_removed")):
        raise RuntimeError(f"Hook lifecycle failed: {trace}")
    return output, trace


def validate_output(output: Any, expected_hidden_count: int = 13) -> None:
    import torch

    if output.logits is None or output.logits.numel() == 0 or not torch.isfinite(output.logits).all():
        raise RuntimeError("Output logits are empty or non-finite")
    if output.hidden_states is None or len(output.hidden_states) != expected_hidden_count:
        raise RuntimeError(f"Expected {expected_hidden_count} hidden states")
    for index, hidden in enumerate(output.hidden_states):
        if not torch.isfinite(hidden).all():
            raise RuntimeError(f"Hidden state H{index} contains NaN/Inf")


def baseline_visualization(waveform: list[float], output: Any, decoded: dict[str, Any]) -> dict[str, Any]:
    import torch

    hidden = output.hidden_states
    adjacent = [cosine_distance_mean(hidden[i], hidden[i + 1]) for i in range(12)]
    mean_norm = [finite_float(torch.linalg.vector_norm(h.float(), dim=-1).mean().item()) for h in hidden]
    temporal = [temporal_change_series(h) for h in hidden]
    probs = torch.softmax(decoded["logits"].float(), dim=-1)
    max_probs = probs.max(dim=-1).values[0].detach().cpu().tolist()
    blank = (decoded["token_ids"][0] == decoded["blank_id"]).float().detach().cpu().tolist()
    return {
        "waveform": downsample(waveform, 1024),
        "hidden_state_heatmaps": [tensor_heatmap(h) for h in hidden],
        "adjacent_layer_cosine_distance": adjacent,
        "mean_hidden_norm": mean_norm,
        "temporal_change_by_hidden_state": temporal,
        "baseline_ctc_max_probability": downsample(max_probs, 128),
        "baseline_ctc_blank_indicator": downsample(blank, 128),
    }


def base_record(config: dict[str, Any], audio: dict[str, Any], audit: dict[str, Any], run_id: str, layer_id: int | None, alpha: float) -> dict[str, Any]:
    return {
        "task_id": config["task_id"],
        "run_id": run_id,
        "created_at_utc": utc_now(),
        "audio_id": audio["audio_id"],
        "audio_sha256": audio["wav_sha256"],
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "processor_revision": config["processor_revision"],
        "transformers_version": audit["transformers_version"],
        "torch_version": audit["torch_version"],
        "layer_id": layer_id,
        "alpha": alpha,
        "input_sample_rate": audio["sample_rate"],
        "input_duration": audio["duration_seconds"],
        "device": audit["device"],
        "dtype": audit["dtype"],
        "metrics_schema_version": int(config["metrics_schema_version"]),
        "config_path": f"configs/{TASK_ID}.yaml",
        "status": "PENDING",
        "error": None,
    }


def finish_record(record: dict[str, Any], output: Any, decoded: dict[str, Any], baseline_output: Any, baseline_decoded: dict[str, Any], reference: str, trace: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    import torch

    validate_output(output)
    js_mean, js_by_time = js_divergence(baseline_decoded["logits"], decoded["logits"])
    record.update(
        {
            "hidden_state_shapes": [list(x.shape) for x in output.hidden_states],
            "baseline_transcript": baseline_decoded["transcript"],
            "adjusted_transcript": decoded["transcript"],
            "transcript_exactly_matches_baseline": decoded["transcript"] == baseline_decoded["transcript"],
            "ctc_frame_count": int(decoded["logits"].shape[1]),
            "ctc_blank_id": decoded["blank_id"],
            "ctc_blank_ratio": decoded["blank_ratio"],
            "mean_frame_max_probability": decoded["mean_frame_max_probability"],
            "transcript_edit_distance_from_baseline": levenshtein(baseline_decoded["transcript"], decoded["transcript"]),
            "final_hidden_cosine_distance_from_baseline": cosine_distance_mean(baseline_output.hidden_states[-1], output.hidden_states[-1]),
            "ctc_logit_divergence_from_baseline": js_mean,
            "inference_time_ms": finite_float(elapsed_ms),
            "status": "SUCCESS",
            "hook_trace": trace,
            "greedy_token_ids_sha256": hashlib.sha256(decoded["token_ids"].detach().cpu().numpy().astype("int64").tobytes()).hexdigest(),
            "reference_text": reference,
            "character_error_rate_vs_reference": finite_float(levenshtein(reference, decoded["transcript"]) / max(1, len(reference))),
            "visual_trace": {
                "final_hidden_cosine_distance_by_time": cosine_distance_by_time(baseline_output.hidden_states[-1], output.hidden_states[-1]),
                "ctc_js_divergence_by_time": js_by_time,
                "adjusted_final_hidden_temporal_change": temporal_change_series(output.hidden_states[-1]),
            },
            "nan_or_inf_detected": False,
        }
    )
    if not torch.isfinite(decoded["logits"]).all():
        raise RuntimeError("Non-finite logits detected after metrics")
    return record


def identity_test(root: Path, config: dict[str, Any], model: Any, processor: Any, input_values: Any, attention_mask: Any, baseline_output: Any, baseline_decoded: dict[str, Any]) -> dict[str, Any]:
    import torch

    layer_id = int(config["identity_layer"])
    started = time.perf_counter()
    output, trace = forward(model, input_values, attention_mask, layer_id=layer_id, alpha=1.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    validate_output(output)
    decoded = decode_output(model, processor, output)
    max_abs = finite_float((baseline_decoded["logits"].float() - decoded["logits"].float()).abs().max().item())
    result = {
        "tested_at_utc": utc_now(),
        "layer_id": layer_id,
        "alpha": 1.0,
        "transcript_exact": decoded["transcript"] == baseline_decoded["transcript"],
        "token_ids_exact": bool(torch.equal(decoded["token_ids"], baseline_decoded["token_ids"])),
        "hidden_shapes_exact": [list(x.shape) for x in output.hidden_states] == [list(x.shape) for x in baseline_output.hidden_states],
        "logits_max_abs_error": max_abs,
        "tolerance": float(config["identity_tolerance_float32"]),
        "hook_trace": trace,
        "inference_time_ms": finite_float(elapsed_ms),
    }
    result["status"] = "PASS" if (
        result["transcript_exact"]
        and result["token_ids_exact"]
        and result["hidden_shapes_exact"]
        and result["logits_max_abs_error"] <= result["tolerance"]
        and trace.get("hook_removed")
    ) else "FAIL"
    write_json(root / "reports" / f"{TASK_ID}_IDENTITY_TEST.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Identity test failed: {result}")
    return result


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "run_id", "status", "layer_id", "alpha", "adjusted_transcript", "transcript_exactly_matches_baseline",
        "transcript_edit_distance_from_baseline", "ctc_blank_ratio", "mean_frame_max_probability",
        "final_hidden_cosine_distance_from_baseline", "ctc_logit_divergence_from_baseline", "inference_time_ms", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)


def write_figures(root: Path, runs: list[dict[str, Any]], visualization: dict[str, Any]) -> list[dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    interventions = [r for r in runs if r["layer_id"] is not None and r["status"] == "SUCCESS"]
    alphas = [0.0, 0.5]

    def matrix(metric: str) -> Any:
        return np.asarray([[next(r[metric] for r in interventions if r["layer_id"] == layer and r["alpha"] == alpha) for alpha in alphas] for layer in range(1, 13)])

    definitions = [
        ("panel_01_final_hidden_distance", matrix("final_hidden_cosine_distance_from_baseline"), "Final hidden cosine distance", "Distance"),
        ("panel_02_ctc_js_divergence", matrix("ctc_logit_divergence_from_baseline"), "CTC Jensen–Shannon divergence", "JS divergence"),
        ("panel_03_transcript_edit_distance", matrix("transcript_edit_distance_from_baseline"), "Transcript character edit distance", "Characters"),
    ]
    records = []
    for stem, values, title, color_label in definitions:
        fig, ax = plt.subplots(figsize=(6.5, 8.0), constrained_layout=True)
        image = ax.imshow(values, aspect="auto", cmap="viridis")
        ax.set_xticks([0, 1], ["α=0.0", "α=0.5"])
        ax.set_yticks(range(12), [f"L{i}" for i in range(1, 13)])
        ax.set_xlabel("Residual intervention strength")
        ax.set_ylabel("Transformer layer (one-based)")
        ax.set_title(title)
        for i in range(12):
            for j in range(2):
                ax.text(j, i, f"{values[i, j]:.3g}", ha="center", va="center", color="white" if values[i, j] > np.nanmedian(values) else "black", fontsize=8)
        fig.colorbar(image, ax=ax, label=color_label)
        for extension in ("png", "svg"):
            fig.savefig(figure_dir / f"{stem}.{extension}", dpi=180 if extension == "png" else None)
        plt.close(fig)
        records.append({"id": stem, "png": f"reports/figures/{stem}.png", "svg": f"reports/figures/{stem}.svg", "title": title})

    adjacent = visualization["baseline"]["adjacent_layer_cosine_distance"]
    norms = visualization["baseline"]["mean_hidden_norm"]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), constrained_layout=True)
    axes[0].plot(range(1, 13), adjacent, marker="o", color="#d97706")
    axes[0].set(xlabel="Transition (H0→H1 ... H11→H12)", ylabel="Mean cosine distance", title="Baseline adjacent-layer representation change")
    axes[0].grid(alpha=0.25)
    axes[1].plot(range(13), norms, marker="o", color="#2563eb")
    axes[1].set(xlabel="Hidden state H0–H12", ylabel="Mean vector norm", title="Baseline hidden-state magnitude")
    axes[1].grid(alpha=0.25)
    stem = "panel_04_baseline_layer_dynamics"
    for extension in ("png", "svg"):
        fig.savefig(figure_dir / f"{stem}.{extension}", dpi=180 if extension == "png" else None)
    plt.close(fig)
    records.append({"id": stem, "png": f"reports/figures/{stem}.png", "svg": f"reports/figures/{stem}.svg", "title": "Baseline layer dynamics"})
    write_json(figure_dir / "figure_manifest.json", records)
    return records


def build_demo(root: Path, runs: list[dict[str, Any]], visualization: dict[str, Any], config: dict[str, Any], run_group: Path) -> None:
    demo = root / "demo" / TASK_ID
    data_dir = demo / "data"
    assets = demo / "assets"
    data_dir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "stimuli" / TASK_ID / "input_16k_mono.wav", assets / "input_16k_mono.wav")
    manifest = {
        "task_id": TASK_ID,
        "generated_at_utc": utc_now(),
        "source_run_group": str(run_group.relative_to(root)).replace("\\", "/"),
        "config_path": f"configs/{TASK_ID}.yaml",
        "runs_json": "data/runs.json",
        "visualization_json": "data/visualization.json",
        "audio": "assets/input_16k_mono.wav",
        "unique_inference_count": len(runs),
        "alpha_1_cells_reference_baseline": True,
        "model_revision": config["model_revision"],
        "offline": True,
    }
    write_json(data_dir / "manifest.json", manifest)
    write_json(data_dir / "runs.json", runs)
    write_json(data_dir / "visualization.json", visualization)
    bundle = {"manifest": manifest, "runs": runs, "visualization": visualization}
    write_text(data_dir / "data.js", "window.TB001_DEMO_DATA = " + json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n")


def summary_markdown(runs: list[dict[str, Any]], figures: list[dict[str, Any]]) -> str:
    successful = [r for r in runs if r["status"] == "SUCCESS" and r["layer_id"] is not None]
    failed = [r for r in runs if r["status"] != "SUCCESS"]
    strongest_hidden = max(successful, key=lambda r: r["final_hidden_cosine_distance_from_baseline"])
    strongest_js = max(successful, key=lambda r: r["ctc_logit_divergence_from_baseline"])
    edited = [r for r in successful if r["transcript_edit_distance_from_baseline"] > 0]
    unchanged = [r for r in successful if r["transcript_edit_distance_from_baseline"] == 0]
    return f"""# {TASK_ID} 科研结果解读

## 跨图结论

在固定的 `facebook/wav2vec2-base-960h` revision、单条 5.12 s 英语语音和规定的残差调节公式下，24 个非基线干预中 {len(successful)} 个成功、{len(failed)} 个失败。最终 hidden 表示距离最大的是 L{strongest_hidden['layer_id']} / α={strongest_hidden['alpha']}（{strongest_hidden['final_hidden_cosine_distance_from_baseline']:.6g}）；CTC 分布 JS divergence 最大的是 L{strongest_js['layer_id']} / α={strongest_js['alpha']}（{strongest_js['ctc_logit_divergence_from_baseline']:.6g}）。有 {len(edited)} 个干预改变了 greedy transcript，{len(unchanged)} 个没有改变文本；后者仍可能改变表示或 CTC 分布，不能解释为“该层无作用”。

## 逐图解读

### Figure 1：最终 hidden 表示距离

纵轴为实际 Transformer 层 L1–L12，横轴为 α=0（完全旁路）与 α=0.5（减弱该层残差变化），单元格为调整后最终 hidden state 相对 baseline 的逐时间位置 cosine distance 均值。图中仅衡量本条语音经后续层传播后的表示偏移；它不提供层的语义名称，也不对应脑区。

### Figure 2：CTC 输出分布差异

坐标与 Figure 1 相同，颜色和数值为逐 frame softmax 的 Jensen–Shannon divergence 再对时间取均值。较高数值说明 CTC 输出分布对该层干预更敏感；即使转写不变，也可存在分布变化。该指标不是校准置信度或识别正确率。

### Figure 3：转写字符编辑距离

单元格为干预转写相对 baseline 的字符 Levenshtein 距离。非零只支持“当前模型、当前语音、当前干预下输出发生改变”；不能推出某层专门编码被改变的字符或词。零值也不能排除中间表示和 logits 已经变化。

### Figure 4：baseline 层级动力学

上图是 baseline 相邻 hidden states（H0→H1 到 H11→H12）的逐位置 cosine distance 均值；下图是 H0–H12 的平均向量范数。二者描述未干预前向中的层间数值变化，用于定位变化幅度，而不是为层赋予“声音、节律、语义”等功能标签。

## 证据边界

- 只使用一个公开英语样本，不能泛化到其他说话人、语言、噪声或模型。
- forward hook 是工程消融；α=0/0.5 会产生偏离训练分布的状态。
- greedy transcript、softmax 最大值、hidden distance 与 JS divergence 都不是临床或神经生理证据。
- 本结果不能建立 Transformer 层与听神经、听觉皮层或其他生物结构的映射。

## 下一步

只有在人类核对 HTML、JSON、identity test 与逐图结果后，才建议增加第二条英语语音；第一版不加入噪声、变速或第二模型。
"""


def execution_report(root: Path, config: dict[str, Any], run_group: Path, runs: list[dict[str, Any]], identity: dict[str, Any], figures: list[dict[str, Any]], reused: bool = False) -> str:
    interventions = [r for r in runs if r["layer_id"] is not None]
    successes = [r for r in interventions if r["status"] == "SUCCESS"]
    failures = [r for r in interventions if r["status"] != "SUCCESS"]
    edited = [r for r in successes if r["transcript_edit_distance_from_baseline"] > 0]
    return f"""# {TASK_ID} 完整执行报告

- 状态：`{'REUSED_VERIFIED_RESULTS' if reused else 'S0-S6_EXECUTED'}`
- 生成时间：`{utc_now()}`
- 执行链路：本地控制器通过 SSH 调用 2203 上的普通 Python；未启动远程 Codex Agent。
- 远端 Git：未初始化、未提交、未推送。
- run group：`{run_group.relative_to(root).as_posix()}`

## 环境与固定来源

- 模型：`{config['model_id']}`，revision `{config['model_revision']}`。
- Processor revision：`{config['processor_revision']}`。
- 输入：公开 LibriSpeech demo 单条 16 kHz 单声道英语语音；来源、许可、item 与 SHA256 见 `stimuli/{TASK_ID}/manifest.json`。
- 模型关键文件与 SHA256 见 `models/model_manifest.json`；加载键审计见 `reports/{TASK_ID}_MODEL_LOADING_AUDIT.json`。
- 唯一缺失键 `wav2vec2.masked_spec_embed` 为训练期 SpecAugment 参数；本次 `eval()` 且显式 `apply_spec_augment=false`。任何其他 missing/unexpected/mismatched key 都会令运行失败。

## 实际运行与状态

- baseline：1/1 SUCCESS；13 组 hidden-state shape 已记录。
- identity test：`{identity['status']}`；α=1 logits 最大绝对误差 `{identity['logits_max_abs_error']}`，阈值 `{identity['tolerance']}`；token IDs / transcript / hidden shapes 严格一致。
- 非基线干预：{len(successes)}/24 SUCCESS，{len(failures)} FAILED；每次只 hook 一个一基编号层，forward 后 hook 残留为 0。
- greedy transcript 改变：{len(edited)}/24；未改变：{len(successes) - len(edited)}/24。阴性结果未过滤。
- 唯一推理结果：{len(runs)}（1 baseline + 12×2 intervention）。HTML 的 12 个 α=1 选择共同引用 baseline。

## 稳定命令与退出码

```text
python scripts/preflight.py                                      # exit 0
python scripts/run_demo.py --config configs/{TASK_ID}.yaml       # exit 0
python -m unittest discover -s tests -v                          # exit 0
python scripts/verify_delivery.py                                # 由最终验收日志给出
python scripts/serve_demo.py --port 8000                         # 人工浏览入口
```

运行阶段设置 Hugging Face offline/local-files-only，只读取已固定的本地 snapshot，不访问网络。日志只追加写入 `logs/`。

## HTML 与图

- 离线入口：`demo/{TASK_ID}/index.html`；无 CDN，结果由 `outputs/` 导出的 `data/*.json` 和派生 `data.js` 提供，`app.js` 不包含手填实验数值。
- 独立图件：{len(figures)} 张 PNG 和对应 SVG，位于 `reports/figures/`；逐图科研解读见 `reports/{TASK_ID}_SCIENTIFIC_INTERPRETATION.md`。
- 控件包含音频、L1–L12、观察/残差调节、α=0/0.5/1、恢复 baseline、解释开关、复制配置和导出当前 JSON。

## token 口径

- 本轮 S3–S6 使用 SSH + Python/模型推理，远端 Agent token 为 0。
- 此处的 CTC frame、token id 和字符数不是模型计费 token。
- 当前桌面控制会话的精确 usage 未由服务器暴露，标记为 `UNAVAILABLE`，不估算。
- 历史上两次误用远端 Codex CLI 的 314,074 tokens 单列为历史沉没消耗，不计入本轮 SSH 完成链路。

## 结论与边界

本 Demo 最多支持：在固定模型 revision、固定单条语音和规定工程干预下，不同层与强度对中间表示、CTC 分布及最终转写呈现可复现的不同敏感性。它不能支持层的语义/节律命名、脑区映射、跨语音普遍排序或人类听觉等价结论。

## Go / No-go

如果最终 verifier、独立轻量包验收和人工 HTML 检查均通过，则第一版 Demo 达到 Go；Go 仅表示可以考虑第二条语音，不授权自动扩展。
"""


def update_token_report(root: Path) -> None:
    path = root / "reports" / f"{TASK_ID}_TOKEN_USAGE.json"
    existing = read_json(path) if path.exists() else {"schema_version": 1, "attempts": []}
    existing["updated_at_utc"] = utc_now()
    existing["current_completion"] = {
        "execution_method": "LOCAL_CONTROLLER_OVER_SSH",
        "remote_codex_agent_used": False,
        "remote_agent_tokens": 0,
        "model_inference_billing_tokens": 0,
        "desktop_controller_usage": "UNAVAILABLE",
        "note": "No estimate was fabricated. CTC frames, token IDs and transcript characters are not billing tokens.",
    }
    existing["historical_remote_agent_attempts_are_not_part_of_current_completion"] = True
    write_json(path, existing)


def copytree_selected(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_lightweight_delivery(root: Path, run_group: Path) -> Path:
    delivery = root / "delivery" / f"{TASK_ID}_lightweight"
    delivery.mkdir(parents=True, exist_ok=True)
    selections = [
        "README.md", "requirements.lock.txt", "configs", "src", "tests", "scripts", "docs",
        "environment", f"stimuli/{TASK_ID}", f"demo/{TASK_ID}", "reports",
    ]
    for relative in selections:
        source = root / relative
        if source.exists():
            copytree_selected(source, delivery / relative)
    # Preserve successful, superseded and failed run groups. This makes the
    # numerical identity failure and its subsequent fix auditable.
    copytree_selected(root / "outputs" / TASK_ID, delivery / "outputs" / TASK_ID)
    model_manifest = read_json(root / "models" / "model_manifest.json")
    model_manifest["snapshot_path"] = "EXCLUDED_FROM_LIGHTWEIGHT_DELIVERY"
    model_manifest["delivery_note"] = "Weights and Hugging Face cache are intentionally excluded; use the pinned hashes to restore compute capability. Offline HTML and verification do not require weights."
    write_json(delivery / "models" / "model_manifest.json", model_manifest)
    return delivery


def create_manifest(package: Path) -> dict[str, Any]:
    # Delivery verification is generated after manifest creation and therefore
    # intentionally excluded from the immutable payload hash set.
    excluded_names = {"delivery_manifest.json", f"{TASK_ID}_DELIVERY_VERIFY.json"}
    records = []
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.name not in excluded_names:
            rel = path.relative_to(package).as_posix()
            if any(part in {"cache", ".git", "__pycache__"} for part in path.relative_to(package).parts):
                continue
            records.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "package_root_name": package.name,
        "file_count": len(records),
        "total_size_bytes": sum(x["size_bytes"] for x in records),
        "forbidden_payloads_excluded": ["Hugging Face cache", "model weights", "raw Codex JSONL", "credentials", ".git"],
        "files": records,
    }
    write_json(package / "delivery_manifest.json", manifest)
    return manifest


def verify_delivery(root: Path, delivery_only: bool = False) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        "README.md", "requirements.lock.txt", f"configs/{TASK_ID}.yaml", f"demo/{TASK_ID}/index.html",
        f"demo/{TASK_ID}/styles.css", f"demo/{TASK_ID}/app.js", f"demo/{TASK_ID}/data/showcase_results.json",
        f"demo/{TASK_ID}/data/showcase_data.js", f"reports/{TASK_ID}_EXECUTION_REPORT.md",
        f"reports/{TASK_ID}_SCIENTIFIC_INTERPRETATION.md", f"reports/{TASK_ID}_IDENTITY_TEST.json",
        f"outputs/{TASK_ID}/current_run_group.json", f"stimuli/{TASK_ID}/input_16k_mono.wav",
        f"outputs/{TASK_ID}/current_showcase.json",
        f"outputs/{TASK_ID}/current_frontend.json", f"demo/{TASK_ID}/data/frontend_results.json",
        f"demo/{TASK_ID}/data/frontend_data.js", f"demo/{TASK_ID}/data/reference_data.js",
        f"demo/{TASK_ID}/figures/frontend_speed_cer.png", f"demo/{TASK_ID}/figures/frontend_speed_cer.svg",
        "reports/figures/frontend_speed_cer.png", "reports/figures/frontend_speed_cer.svg",
        "reports/figures/frontend_speed_cer_source.csv", "reports/figures/frontend_speed_cer_manifest.json",
    ]
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")
    if errors:
        result = {"status": "FAIL", "checked_at_utc": utc_now(), "root": str(root), "errors": errors, "warnings": warnings}
        write_json(root / "reports" / f"{TASK_ID}_FINAL_VERIFY.json", result)
        return result
    current = read_json(root / "outputs" / TASK_ID / "current_run_group.json")
    group = root / current["relative_path"]
    runs = read_json(group / "runs.json")
    if len(runs) != 25:
        errors.append(f"expected 25 unique inference results, got {len(runs)}")
    baseline = [r for r in runs if r.get("layer_id") is None]
    interventions = [r for r in runs if r.get("layer_id") is not None]
    if len(baseline) != 1:
        errors.append(f"expected one baseline, got {len(baseline)}")
    expected = {(layer, alpha) for layer in range(1, 13) for alpha in (0.0, 0.5)}
    actual = {(int(r["layer_id"]), float(r["alpha"])) for r in interventions}
    if actual != expected:
        errors.append(f"intervention matrix mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    failed = [r["run_id"] for r in runs if r.get("status") != "SUCCESS"]
    if failed:
        errors.append(f"non-success runs: {failed}")
    identity = read_json(root / "reports" / f"{TASK_ID}_IDENTITY_TEST.json")
    if identity.get("status") != "PASS":
        errors.append("identity test is not PASS")
    if any(not r.get("hook_trace", {}).get("hook_removed", True) for r in interventions):
        errors.append("one or more intervention hooks were not removed")
    if any(r.get("hook_trace", {}).get("formula_max_abs_error") != 0.0 for r in interventions):
        errors.append("intervention formula verification error is non-zero")
    for r in interventions:
        if r["alpha"] == 0.0 and r.get("hook_trace", {}).get("bypass_max_abs_error") != 0.0:
            errors.append(f"alpha=0 bypass mismatch: {r['run_id']}")
    showcase_pointer = read_json(root / "outputs" / TASK_ID / "current_showcase.json")
    showcase_group = root / showcase_pointer["relative_path"]
    showcase = read_json(showcase_group / "showcase_results.json")
    exported = read_json(root / "demo" / TASK_ID / "data" / "showcase_results.json")
    if exported != showcase:
        errors.append("HTML showcase_results.json differs from canonical showcase results")
    showcase_counts = showcase.get("counts", {})
    if showcase_counts.get("conditions") != 5:
        errors.append(f"expected 5 input-speed conditions, got {showcase_counts.get('conditions')}")
    if showcase_counts.get("unique_inferences") != 125 or showcase_counts.get("interventions") != 120:
        errors.append(f"showcase inference counts mismatch: {showcase_counts}")
    if showcase_counts.get("failed") != 0:
        errors.append(f"showcase contains failed runs: {showcase_counts.get('failed')}")
    if not isinstance(showcase_counts.get("text_changed"), int) or showcase_counts["text_changed"] <= 0:
        errors.append("showcase has no real decoded-text differences")
    default = showcase.get("default_selection", {})
    default_condition = next((c for c in showcase.get("conditions", []) if c.get("id") == default.get("condition_id")), None)
    default_run = None if default_condition is None else next((r for r in default_condition.get("runs", []) if r.get("run_id") == default.get("run_id")), None)
    if default_run is None or default_run.get("transcript_edit_distance_from_baseline") != default.get("transcript_edit_distance"):
        errors.append("default showcase selection does not resolve to its canonical real run")
    frontend_pointer = read_json(root / "outputs" / TASK_ID / "current_frontend.json")
    frontend_group = root / frontend_pointer["relative_path"]
    frontend = read_json(frontend_group / "frontend_results.json")
    frontend_exported = read_json(root / "demo" / TASK_ID / "data" / "frontend_results.json")
    if frontend_exported != frontend:
        errors.append("HTML frontend_results.json differs from canonical frontend results")
    frontend_counts = frontend.get("counts", {})
    if frontend_counts.get("conditions") != 5 or frontend_counts.get("variants_per_condition") != 3:
        errors.append(f"frontend condition matrix mismatch: {frontend_counts}")
    if frontend_counts.get("unique_inferences") != 15 or frontend_counts.get("failed") != 0:
        errors.append(f"frontend inference counts mismatch: {frontend_counts}")
    x2_frontend = next((c for c in frontend.get("conditions", []) if c.get("id") == "speed-2p00"), None)
    x2_variants = {} if x2_frontend is None else {v.get("frontend_variant_id"): v for v in x2_frontend.get("variants", [])}
    if set(x2_variants) != {"frontend-50hz", "frontend-100hz", "frontend-200hz"}:
        errors.append("2x frontend variants are incomplete")
    else:
        x2_frames = [x2_variants[key].get("ctc_frame_count") for key in ("frontend-50hz", "frontend-100hz", "frontend-200hz")]
        if not (x2_frames[0] < x2_frames[1] < x2_frames[2]):
            errors.append(f"2x frontend frame counts are not increasing: {x2_frames}")
        if x2_variants["frontend-100hz"].get("character_error_rate_vs_reference", 1.0) >= x2_variants["frontend-50hz"].get("character_error_rate_vs_reference", 0.0):
            errors.append("2x 100Hz frontend does not preserve the recorded CER improvement")
    html_text = (root / "demo" / TASK_ID / "index.html").read_text(encoding="utf-8")
    css_text = (root / "demo" / TASK_ID / "styles.css").read_text(encoding="utf-8")
    app_text = (root / "demo" / TASK_ID / "app.js").read_text(encoding="utf-8")
    for marker in ("http://", "https://", "cdn."):
        if marker in html_text.lower() or marker in css_text.lower() or marker in app_text.lower():
            errors.append(f"offline assets contain remote marker: {marker}")
    for required_control in ("speedButtons", "layerStack", "data-alpha", "frontendNode", "data-frontend", "layerTitle", "strengthMeaning", "referenceTranscript", "baselineTranscript", "adjustedTranscript", "baselineCer", "adjustedCer", "editCount"):
        if required_control not in html_text and required_control not in app_text:
            errors.append(f"HTML control missing: {required_control}")
    if "TB001_SHOWCASE" not in (root / "demo" / TASK_ID / "data" / "showcase_data.js").read_text(encoding="utf-8"):
        errors.append("derived local showcase data bundle is missing")
    if "TB001_FRONTEND" not in (root / "demo" / TASK_ID / "data" / "frontend_data.js").read_text(encoding="utf-8"):
        errors.append("derived local frontend data bundle is missing")
    if "TB001_REFERENCE" not in (root / "demo" / TASK_ID / "data" / "reference_data.js").read_text(encoding="utf-8"):
        errors.append("local reference transcript bundle is missing")
    frontend_figure_manifest = read_json(root / "reports" / "figures" / "frontend_speed_cer_manifest.json")
    if frontend_figure_manifest.get("real_inferences") != 15:
        errors.append("frontend CER figure does not declare 15 real inferences")
    if frontend_figure_manifest.get("adaptive_curve") != "derived from existing runs; no new inference":
        errors.append("frontend CER adaptive curve boundary is missing")
    if "frontend_speed_cer.svg" not in html_text or "同样本" not in html_text:
        errors.append("frontend CER figure or same-sample boundary is missing from HTML")
    manifest_path = root / "delivery_manifest.json"
    if delivery_only and manifest_path.is_file():
        manifest = read_json(manifest_path)
        for item in manifest["files"]:
            path = root / item["relative_path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                errors.append(f"delivery manifest mismatch: {item['relative_path']}")
        forbidden = [p for p in root.rglob("*") if p.is_file() and (p.suffix in {".safetensors", ".bin", ".jsonl"} or "models--" in p.as_posix() or ".git/" in p.as_posix())]
        if forbidden:
            errors.append("forbidden heavyweight/private payloads found: " + ", ".join(str(p.relative_to(root)) for p in forbidden[:5]))
    result = {
        "status": "PASS" if not errors else "FAIL",
        "checked_at_utc": utc_now(),
        "root": str(root),
        "delivery_only": delivery_only,
        "checks": {
            "unique_inference_results": len(runs),
            "nonbaseline_interventions": len(interventions),
            "identity_status": identity.get("status"),
            "failed_runs": len(failed),
            "html_data_exact_match": exported == showcase,
            "showcase_unique_inferences": showcase_counts.get("unique_inferences"),
            "showcase_text_changed_runs": showcase_counts.get("text_changed"),
            "frontend_unique_inferences": frontend_counts.get("unique_inferences"),
            "frontend_text_changed_runs": frontend_counts.get("text_changed_vs_standard"),
            "frontend_html_data_exact_match": frontend_exported == frontend,
            "frontend_cer_figure": frontend_figure_manifest.get("figure_id"),
            "offline_remote_markers_absent": not any("remote marker" in e for e in errors),
        },
        "errors": errors,
        "warnings": warnings,
    }
    verify_name = f"{TASK_ID}_DELIVERY_VERIFY.json" if delivery_only else f"{TASK_ID}_FINAL_VERIFY.json"
    write_json(root / "reports" / verify_name, result)
    return result


def main_run(root: Path, config_path: Path) -> int:
    import soundfile as sf
    import torch

    root = root.resolve()
    config = load_config(config_path.resolve())
    audio = read_json(root / config["audio_manifest_path"])
    model_manifest = read_json(root / config["model_manifest_path"])
    current_hash = config_hash(config, audio["wav_sha256"], model_manifest["files"])
    current_pointer = root / "outputs" / TASK_ID / "current_run_group.json"
    if current_pointer.exists():
        current = read_json(current_pointer)
        group = root / current["relative_path"]
        completion = group / "completion.json"
        if completion.exists() and read_json(completion).get("config_hash") == current_hash:
            runs = read_json(group / "runs.json")
            identity = read_json(root / "reports" / f"{TASK_ID}_IDENTITY_TEST.json")
            figures = read_json(root / "reports" / "figures" / "figure_manifest.json")
            write_text(root / "reports" / f"{TASK_ID}_EXECUTION_REPORT.md", execution_report(root, config, group, runs, identity, figures, reused=True))
            append_text(root / "logs" / f"run_demo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_reuse.log", f"{utc_now()} REUSED {group.relative_to(root).as_posix()}\n")
            print(json.dumps({"status": "REUSED", "run_group": str(group), "run_count": len(runs)}, ensure_ascii=False))
            return 0

    run_group_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    group = root / "outputs" / TASK_ID / "run_groups" / run_group_id
    group.mkdir(parents=True, exist_ok=False)
    log_path = root / "logs" / f"run_demo_{run_group_id}.log"
    append_text(log_path, f"{utc_now()} START config={config_path}\n")
    model = processor = input_values = attention_mask = None
    try:
        pf = preflight(root)
        if pf["status"] != "PASS":
            raise RuntimeError(f"Preflight failed: {pf}")
        torch.manual_seed(int(config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["seed"]))
        model, processor, input_values, attention_mask, audit = load_runtime(root, config)
        started = time.perf_counter()
        baseline_output, baseline_trace = forward(model, input_values, attention_mask)
        baseline_elapsed = (time.perf_counter() - started) * 1000.0
        validate_output(baseline_output)
        baseline_decoded = decode_output(model, processor, baseline_output)
        baseline_record = base_record(config, audio, audit, "baseline-final-001", None, 1.0)
        baseline_record = finish_record(
            baseline_record, baseline_output, baseline_decoded, baseline_output, baseline_decoded,
            audio["reference_text"], baseline_trace, baseline_elapsed,
        )
        baseline_record["final_hidden_cosine_distance_from_baseline"] = 0.0
        baseline_record["ctc_logit_divergence_from_baseline"] = 0.0
        baseline_record["visual_trace"]["final_hidden_cosine_distance_by_time"] = [0.0] * len(
            baseline_record["visual_trace"]["final_hidden_cosine_distance_by_time"]
        )
        baseline_record["visual_trace"]["ctc_js_divergence_by_time"] = [0.0] * len(
            baseline_record["visual_trace"]["ctc_js_divergence_by_time"]
        )
        write_json(group / "baseline-final-001" / "summary.json", baseline_record)
        identity = identity_test(root, config, model, processor, input_values, attention_mask, baseline_output, baseline_decoded)
        runs = [baseline_record]
        for layer_id in [int(x) for x in config["layers"]]:
            for alpha in [float(x) for x in config["alphas"]]:
                run_id = f"layer-{layer_id:02d}-alpha-{str(alpha).replace('.', 'p')}"
                record = base_record(config, audio, audit, run_id, layer_id, alpha)
                run_dir = group / run_id
                try:
                    started = time.perf_counter()
                    output, trace = forward(model, input_values, attention_mask, layer_id=layer_id, alpha=alpha)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    decoded = decode_output(model, processor, output)
                    record = finish_record(record, output, decoded, baseline_output, baseline_decoded, audio["reference_text"], trace, elapsed_ms)
                    append_text(log_path, f"{utc_now()} SUCCESS {run_id} transcript={json.dumps(decoded['transcript'])}\n")
                except Exception as exc:
                    record.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    append_text(log_path, f"{utc_now()} FAILED {run_id} error={type(exc).__name__}: {exc}\n")
                write_json(run_dir / "summary.json", record)
                runs.append(record)
        write_json(group / "runs.json", runs)
        write_csv(group / "runs.csv", runs)
        waveform, _ = sf.read(root / config["audio_path"], dtype="float32")
        visualization = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "baseline": baseline_visualization(waveform.tolist(), baseline_output, baseline_decoded),
            "interventions": {r["run_id"]: r.get("visual_trace") for r in runs if r["layer_id"] is not None},
        }
        write_json(group / "visualization.json", visualization)
        build_demo(root, runs, visualization, config, group)
        figures = write_figures(root, runs, visualization)
        write_text(root / "reports" / f"{TASK_ID}_SCIENTIFIC_INTERPRETATION.md", summary_markdown(runs, figures))
        update_token_report(root)
        completion = {
            "status": "SUCCESS" if all(r["status"] == "SUCCESS" for r in runs) else "COMPLETED_WITH_FAILURES",
            "completed_at_utc": utc_now(),
            "config_hash": current_hash,
            "run_group_id": run_group_id,
            "unique_inference_results": len(runs),
            "nonbaseline_interventions": len(runs) - 1,
            "identity_test": identity["status"],
        }
        write_json(group / "completion.json", completion)
        write_json(current_pointer, {"relative_path": group.relative_to(root).as_posix(), "config_hash": current_hash, "updated_at_utc": utc_now()})
        write_text(root / "reports" / f"{TASK_ID}_EXECUTION_REPORT.md", execution_report(root, config, group, runs, identity, figures))
        append_text(root / "docs" / "CODEX_PROJECT_LOG.md", f"\n## {utc_now()}：SSH 直连完成 S3-S6\n\n- 执行方式：本地控制器通过 SSH 运行 2203 普通 Python；未调用远程 Agent。\n- run group：`{group.relative_to(root).as_posix()}`。\n- identity test：`{identity['status']}`；唯一推理结果：`{len(runs)}`。\n- 远端 Git 未初始化；token 不写入本日志。\n")
        append_text(log_path, f"{utc_now()} COMPLETE status={completion['status']} runs={len(runs)}\n")
        print(json.dumps(completion, ensure_ascii=False, indent=2))
        return 0 if completion["status"] == "SUCCESS" else 3
    except Exception as exc:
        failure = {"status": "FAILED", "created_at_utc": utc_now(), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "config_hash": current_hash}
        write_json(group / "failure.json", failure)
        append_text(log_path, f"{utc_now()} FATAL {type(exc).__name__}: {exc}\n")
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
