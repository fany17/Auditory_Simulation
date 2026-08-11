#!/usr/bin/env python3
"""Measure whether denser wav2vec2 convolutional strides change fast-speech decoding."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from route_b_temporal_models.wav2vec2_layer_demo.core import (
    decode_output,
    downsample,
    forward,
    levenshtein,
    load_config,
    load_runtime,
    read_json,
    write_json,
    write_text,
)


TASK_ID = "TB001-DEMO001"
FRONTEND_VARIANTS = [
    {
        "id": "frontend-50hz",
        "label": "标准 50 Hz",
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "description": "预训练模型原始卷积步幅，总下采样 320 samples。",
    },
    {
        "id": "frontend-100hz",
        "label": "加密到 100 Hz",
        "conv_stride": [5, 2, 2, 2, 2, 2, 1],
        "description": "将最后一个卷积层步幅由 2 改为 1，总下采样减至 160 samples。",
    },
    {
        "id": "frontend-200hz",
        "label": "加密到 200 Hz",
        "conv_stride": [5, 2, 2, 2, 2, 1, 1],
        "description": "将最后两个卷积层步幅由 2 改为 1，总下采样减至 80 samples。",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def set_frontend_strides(model, strides: list[int]) -> None:
    layers = model.wav2vec2.feature_extractor.conv_layers
    if len(layers) != len(strides):
        raise RuntimeError(f"Expected {len(strides)} conv layers, found {len(layers)}")
    for layer, stride in zip(layers, strides):
        layer.conv.stride = (int(stride),)
    model.config.conv_stride = list(strides)
    model.wav2vec2.config.conv_stride = list(strides)


def align_time(tensor, target_frames: int):
    import torch.nn.functional as F

    if tensor.shape[1] == target_frames:
        return tensor.float()
    return F.interpolate(
        tensor.float().transpose(1, 2),
        size=target_frames,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def aligned_js(reference_logits, changed_logits) -> float:
    import torch

    changed = align_time(changed_logits, reference_logits.shape[1])
    p = torch.softmax(reference_logits.float(), dim=-1).clamp_min(1e-12)
    q = torch.softmax(changed, dim=-1).clamp_min(1e-12)
    m = 0.5 * (p + q)
    value = 0.5 * ((p * (p.log() - m.log())).sum(-1) + (q * (q.log() - m.log())).sum(-1))
    if not torch.isfinite(value).all():
        raise RuntimeError("Aligned CTC JS divergence is not finite")
    return float(value.mean().item())


def aligned_hidden_distance(reference_hidden, changed_hidden) -> float:
    import torch
    import torch.nn.functional as F

    changed = align_time(changed_hidden, reference_hidden.shape[1])
    value = 1.0 - F.cosine_similarity(reference_hidden.float(), changed, dim=-1, eps=1e-8)
    if not torch.isfinite(value).all():
        raise RuntimeError("Aligned hidden cosine distance is not finite")
    return float(value.mean().item())


def compact(record: dict) -> dict:
    keys = [
        "run_id", "status", "frontend_variant_id", "frontend_label", "conv_stride",
        "total_stride_samples", "nominal_feature_rate_hz", "actual_ctc_frame_rate_hz",
        "ctc_frame_count", "baseline_transcript", "adjusted_transcript",
        "transcript_edit_distance_from_baseline", "character_error_rate_vs_reference",
        "ctc_logit_divergence_from_baseline", "final_hidden_cosine_distance_from_baseline",
        "ctc_blank_ratio", "mean_frame_max_probability", "inference_time_ms", "error",
    ]
    return {key: record.get(key) for key in keys}


def main() -> int:
    import soundfile as sf
    import torch

    config = load_config(ROOT / "configs" / f"{TASK_ID}.yaml")
    model, processor, _, _, audit = load_runtime(ROOT, config)
    showcase_ptr = read_json(ROOT / "outputs" / TASK_ID / "current_showcase.json")
    showcase = read_json(ROOT / showcase_ptr["relative_path"] / "showcase_results.json")
    source_manifest = read_json(ROOT / "stimuli" / TASK_ID / "manifest.json")
    reference_text = source_manifest["reference_text"]
    group_id = datetime.now(timezone.utc).strftime("frontend-%Y%m%dT%H%M%SZ")
    group = ROOT / "outputs" / TASK_ID / "frontend_variants" / group_id
    group.mkdir(parents=True, exist_ok=False)
    conditions = []
    all_records = []
    log_lines = [f"{utc_now()} START {group_id}"]

    for condition in showcase["conditions"]:
        audio_path = ROOT / "demo" / TASK_ID / condition["audio_source"]
        waveform, sample_rate = sf.read(audio_path, dtype="float32")
        encoded = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        input_values = encoded.input_values.to(device=audit["device"], dtype=torch.float32)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(audit["device"])

        condition_records = []
        baseline_logits = None
        baseline_hidden = None
        baseline_transcript = None
        for variant in FRONTEND_VARIANTS:
            set_frontend_strides(model, variant["conv_stride"])
            run_id = f"{condition['id']}-{variant['id']}"
            record = {
                "run_id": run_id,
                "condition_id": condition["id"],
                "speed_factor": condition["speed_factor"],
                "frontend_variant_id": variant["id"],
                "frontend_label": variant["label"],
                "conv_stride": variant["conv_stride"],
                "total_stride_samples": math.prod(variant["conv_stride"]),
                "nominal_feature_rate_hz": sample_rate / math.prod(variant["conv_stride"]),
                "method_description": variant["description"],
                "status": "RUNNING",
                "error": None,
            }
            try:
                started = time.perf_counter()
                output, trace = forward(model, input_values, attention_mask)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                decoded = decode_output(model, processor, output)
                logits = decoded["logits"]
                hidden = output.hidden_states[-1]
                if baseline_logits is None:
                    baseline_logits = logits.detach()
                    baseline_hidden = hidden.detach()
                    baseline_transcript = decoded["transcript"]
                record.update(
                    {
                        "status": "SUCCESS",
                        "baseline_transcript": baseline_transcript,
                        "adjusted_transcript": decoded["transcript"],
                        "transcript_edit_distance_from_baseline": levenshtein(baseline_transcript, decoded["transcript"]),
                        "character_error_rate_vs_reference": levenshtein(reference_text, decoded["transcript"]) / max(1, len(reference_text)),
                        "ctc_logit_divergence_from_baseline": aligned_js(baseline_logits, logits),
                        "final_hidden_cosine_distance_from_baseline": aligned_hidden_distance(baseline_hidden, hidden),
                        "ctc_blank_ratio": decoded["blank_ratio"],
                        "mean_frame_max_probability": decoded["mean_frame_max_probability"],
                        "ctc_frame_count": int(logits.shape[1]),
                        "actual_ctc_frame_rate_hz": int(logits.shape[1]) / float(condition["duration_seconds"]),
                        "inference_time_ms": elapsed_ms,
                        "reference_text": reference_text,
                        "alignment_for_cross_rate_metrics": "linear interpolation to standard-frontend CTC frame count over normalized utterance time",
                        "hook_trace": trace,
                    }
                )
                log_lines.append(f"{utc_now()} SUCCESS {run_id} frames={record['ctc_frame_count']} transcript={json.dumps(record['adjusted_transcript'])}")
            except Exception as exc:
                record.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                log_lines.append(f"{utc_now()} FAILED {run_id} {type(exc).__name__}: {exc}")
            write_json(group / condition["id"] / run_id / "summary.json", record)
            condition_records.append(record)
            all_records.append(record)
        conditions.append(
            {
                "id": condition["id"],
                "label": condition["label"],
                "speed_factor": condition["speed_factor"],
                "duration_seconds": condition["duration_seconds"],
                "audio_source": condition["audio_source"],
                "variants": [compact(record) for record in condition_records],
            }
        )

    x2 = next(item for item in conditions if item["id"] == "speed-2p00")
    successful_x2 = [item for item in x2["variants"] if item["status"] == "SUCCESS"]
    best_x2 = min(successful_x2, key=lambda item: (item["character_error_rate_vs_reference"], item["total_stride_samples"]))
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "group_id": group_id,
        "method": "ordinary Python over SSH; no remote Codex Agent",
        "scientific_question": "Does reducing convolutional temporal downsampling change or improve fast-speech CTC output without retraining?",
        "intervention_boundary": "Only convolution strides are changed; pretrained weights are unchanged. Cross-rate hidden/logit distances use normalized-time interpolation.",
        "conditions": conditions,
        "x2_best_by_reference_cer": {
            "variant_id": best_x2["frontend_variant_id"],
            "label": best_x2["frontend_label"],
            "transcript": best_x2["adjusted_transcript"],
            "character_error_rate_vs_reference": best_x2["character_error_rate_vs_reference"],
        },
        "counts": {
            "conditions": len(conditions),
            "variants_per_condition": len(FRONTEND_VARIANTS),
            "unique_inferences": len(all_records),
            "failed": sum(record["status"] != "SUCCESS" for record in all_records),
            "text_changed_vs_standard": sum(record.get("transcript_edit_distance_from_baseline", 0) > 0 for record in all_records),
        },
    }
    write_json(group / "frontend_results.json", summary)
    write_text(group / "run.log", "\n".join(log_lines) + "\n")
    write_json(
        ROOT / "outputs" / TASK_ID / "current_frontend.json",
        {"relative_path": group.relative_to(ROOT).as_posix(), "group_id": group_id, "updated_at_utc": utc_now()},
    )
    data_dir = ROOT / "demo" / TASK_ID / "data"
    write_json(data_dir / "frontend_results.json", summary)
    write_text(
        data_dir / "frontend_data.js",
        "window.TB001_FRONTEND = " + json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n",
    )
    print(json.dumps({"status": "SUCCESS", **summary["counts"], "x2_best": summary["x2_best_by_reference_cer"], "group": str(group)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(main())
