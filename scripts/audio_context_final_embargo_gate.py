from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from m6a_public.audio_context_gate import (
    CANDIDATE_STATUS,
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
    MODEL_FILES,
    MODEL_ID,
    MODEL_REVISION_LABEL,
    MODEL_SAMPLE_RATE_HZ,
    REPORT_SCHEMA_VERSION,
    SOURCE_SAMPLE_RATE_HZ,
    audit_model_config,
    audit_pytorch_weight_file,
    build_split_guard_candidate,
    crop_mono_passage,
    derive_convolution_timing,
    finalize_candidate_report,
    frame_center_seconds,
    frozen_resampling_spec,
    load_isolated_passage,
    load_strict_json_object,
    resample_independent_passage,
    write_report,
)
from m6a_public.embargo_gate import evaluate_final_embargo_candidate


def _modified_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _read_model_file(path: Path) -> bool:
    if path.name in {"config.json", "preprocessor_config.json"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            audit_model_config(path)["status"] == "PASS"
            if path.name == "config.json"
            else payload.get("sampling_rate") == MODEL_SAMPLE_RATE_HZ
        )
    if path.name == "README.md":
        return bool(path.read_text(encoding="utf-8")[:256].strip())
    if path.name == "pytorch_model.bin":
        return audit_pytorch_weight_file(path)["status"] == "PASS"
    return False


def _run_split_sentinel() -> dict[str, Any]:
    store = {
        "train/passage.wav": np.linspace(-0.5, 0.5, SOURCE_SAMPLE_RATE_HZ // 10),
        "validation/sentinel.wav": np.full(SOURCE_SAMPLE_RATE_HZ // 10, 7.0),
        "test/sentinel.wav": np.full(SOURCE_SAMPLE_RATE_HZ // 10, 11.0),
    }
    read_paths: list[str] = []

    def reader(path: str) -> tuple[np.ndarray, int]:
        read_paths.append(path)
        return store[path], SOURCE_SAMPLE_RATE_HZ

    cropped, _ = load_isolated_passage(
        "train/passage.wav",
        "train/passage.wav",
        reader,
        100,
        len(store["train/passage.wav"]) - 100,
    )
    resampled = resample_independent_passage(cropped)
    heldout_read = any(path != "train/passage.wav" for path in read_paths)
    return {
        "status": "PASS" if read_paths == ["train/passage.wav"] and resampled.size > 0 else "FAIL",
        "read_paths": read_paths,
        "forbidden_read_count": sum(path != "train/passage.wav" for path in read_paths),
        "heldout_sentinel_observed": heldout_read,
        "output_samples": int(resampled.size),
    }


def _run_model_canary(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import Wav2Vec2Model

    loaded = Wav2Vec2Model.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    model, loading_info = cast(tuple[Any, dict[str, Any]], loaded)
    normalized_loading_info = {
        "missing_keys": sorted(str(key) for key in loading_info.get("missing_keys", [])),
        "unexpected_keys": sorted(str(key) for key in loading_info.get("unexpected_keys", [])),
        "mismatched_keys": sorted(str(key) for key in loading_info.get("mismatched_keys", [])),
        "error_msgs": [str(message) for message in loading_info.get("error_msgs", [])],
    }
    if normalized_loading_info != {
        "missing_keys": [],
        "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
        "mismatched_keys": [],
        "error_msgs": [],
    }:
        raise RuntimeError("base encoder loading information differs from the frozen profile")
    model.requires_grad_(False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # type: ignore[arg-type]

    duration_seconds = 1.5
    times = np.arange(round(duration_seconds * SOURCE_SAMPLE_RATE_HZ), dtype=np.float64) / SOURCE_SAMPLE_RATE_HZ
    source = 0.2 * np.sin(2.0 * math.pi * 440.0 * times)
    source[round(0.75 * SOURCE_SAMPLE_RATE_HZ)] += 0.5
    start_sample = round(0.25 * SOURCE_SAMPLE_RATE_HZ)
    end_sample = round(1.25 * SOURCE_SAMPLE_RATE_HZ)
    cropped = crop_mono_passage(source, start_sample, end_sample)
    resampled = resample_independent_passage(cropped)
    input_values = torch.from_numpy(resampled).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_values, dtype=torch.long)

    projected: list[Any] = []

    def capture_projection(_module: Any, _inputs: Any, output: Any) -> None:
        projected.append(output[0].detach())

    hook = model.feature_projection.register_forward_hook(capture_projection)
    try:
        with torch.inference_mode():
            output = model(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        hook.remove()
    if len(projected) != 1 or output.hidden_states is None or len(output.hidden_states) != 13:
        raise RuntimeError("unexpected wav2vec2 hidden-state behavior")
    selected_layers = [projected[0], *output.hidden_states[1:]]

    kernels = tuple(int(value) for value in model.config.conv_kernel)
    strides = tuple(int(value) for value in model.config.conv_stride)
    timing = derive_convolution_timing(kernels, strides)
    expected_frames = 0
    from m6a_public.audio_context_gate import expected_frame_count

    expected_frames = expected_frame_count(resampled.size, timing)
    observed_frames = int(selected_layers[0].shape[1])
    centers = frame_center_seconds(observed_frames, timing)
    all_outputs_finite = all(bool(torch.isfinite(layer).all().item()) for layer in selected_layers)

    canary = {
        "device": str(device),
        "local_files_only": True,
        "trust_remote_code": False,
        "repository_custom_code_executed": False,
        "loading_info": normalized_loading_info,
        "unexpected_key_semantics": (
            "PREDECLARED_PRETRAINING_HEADS_EXCLUDED_FROM_FROZEN_BASE_ENCODER"
        ),
        "model_eval": not model.training,
        "parameter_requires_grad_count": sum(int(parameter.requires_grad) for parameter in model.parameters()),
        "source_samples": int(source.size),
        "crop_start_sample": start_sample,
        "crop_end_sample_exclusive": end_sample,
        "cropped_samples": int(cropped.size),
        "crop_half_open_interval_verified": int(cropped.size) == end_sample - start_sample,
        "resampled_samples": int(resampled.size),
        "model_input_padding_samples": 0,
        "attention_mask_all_ones": bool(torch.all(attention_mask == 1).item()),
        "passage_local_resample_padding": True,
        "layer_keys": list(EXPECTED_LAYER_KEYS),
        "layer_shapes": [list(layer.shape) for layer in selected_layers],
        "all_outputs_finite": all_outputs_finite,
        "hidden_state_selection_semantics": (
            "feature_projection_output_plus_transformer_hidden_states_1_through_12; "
            "encoder hidden_states[0] is not relabeled as raw projection"
        ),
    }
    frame_timing = {
        "kernels": list(kernels),
        "strides": list(strides),
        "cumulative_stride_samples": timing.cumulative_stride_samples,
        "receptive_field_samples": timing.receptive_field_samples,
        "convolutional_receptive_field_seconds": timing.receptive_field_samples / MODEL_SAMPLE_RATE_HZ,
        "expected_frames": expected_frames,
        "observed_frames": observed_frames,
        "first_frame_center_seconds": float(centers[0]),
        "last_frame_center_seconds": float(centers[-1]),
        "frame_step_seconds": timing.cumulative_stride_samples / MODEL_SAMPLE_RATE_HZ,
        "timestamp_semantics": "SECONDS_FROM_ISOLATED_PASSAGE_START_AT_CONVOLUTION_SUPPORT_CENTER",
    }
    model_configuration = {
        "hidden_size": int(model.config.hidden_size),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "conv_kernel": list(kernels),
        "conv_stride": list(strides),
        "feat_extract_norm": str(model.config.feat_extract_norm),
        "do_stable_layer_norm": bool(model.config.do_stable_layer_norm),
        "model_input_sampling_rate_hz": MODEL_SAMPLE_RATE_HZ,
    }
    return canary, frame_timing, model_configuration


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the single M6A audio-context/final-embargo candidate gate.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_strict_json_object(args.config)
    model_dir = args.model_dir.resolve()
    configured_dir = Path(config["model"]["remote_cache"]).resolve()
    project_root = Path(config["resources"]["remote_project_root"]).resolve()
    if model_dir != configured_dir or model_dir == project_root or project_root not in model_dir.parents:
        raise ValueError("model-dir must equal the configured cache inside the dedicated project root")
    if config["model"]["model_id"] != MODEL_ID or config["model"]["revision_label"] != MODEL_REVISION_LABEL:
        raise ValueError("unexpected model identity")

    inventory = []
    for name in MODEL_FILES:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        inventory.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "modified_at_utc": _modified_at(path),
                "sample_readability": _read_model_file(path),
            }
        )

    canary, frame_timing, model_configuration = _run_model_canary(model_dir)
    split_guard = build_split_guard_candidate(args.split_csv, config)
    components = config["split"]["final_embargo_components_seconds"]
    evidence: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "model": {
            "model_id": MODEL_ID,
            "revision_label": MODEL_REVISION_LABEL,
            "resolved_at": config["model"]["resolved_at"],
            "revision_limitation": "MUTABLE_MAIN_LABEL_NON_CRYPTOGRAPHIC_REPRODUCIBILITY_ONLY",
            "source_endpoint": config["model"]["source_endpoint"],
            "source_endpoint_role": config["model"]["source_endpoint_role"],
            "source_endpoint_limitation": config["model"]["source_endpoint_limitation"],
            "declared_license": config["model"]["declared_license"],
            "trainable": False,
            "cache_state": config["model"]["cache_state"],
            "configuration": model_configuration,
            "config_semantic_audit": audit_model_config(model_dir / "config.json"),
            "weight_semantic_audit": audit_pytorch_weight_file(model_dir / "pytorch_model.bin"),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "soundfile": importlib.metadata.version("soundfile"),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        },
        "model_cache_path": str(model_dir),
        "model_cache_remote_only": True,
        "model_inventory": inventory,
        "model_inventory_total_bytes": sum(item["bytes"] for item in inventory),
        "input_semantics": {
            "source_sample_rate_hz": SOURCE_SAMPLE_RATE_HZ,
            "model_sample_rate_hz": MODEL_SAMPLE_RATE_HZ,
            "channels": 1,
            "mono_policy": "REQUIRE_MONO_NO_IMPLICIT_DOWNMIX",
            "passage_policy": "ONE_ELIGIBLE_PASSAGE_PER_INFERENCE_CALL",
            "crop_interval": "HALF_OPEN_INTEGER_SAMPLE_BOUNDS_WITHIN_STANDALONE_PASSAGE_FILE",
            "neighbor_audio_read_allowed": False,
            "batch_padding": "FORBIDDEN_PRIMARY_INFERENCE",
        },
        "resampling": frozen_resampling_spec().__dict__,
        "split_sentinel": _run_split_sentinel(),
        "model_canary": canary,
        "frame_timing": frame_timing,
        "context": {
            "transformer_attention_scope": "GLOBAL_WITHIN_SINGLE_PASSAGE",
            "transformer_local_receptive_field_claimed": False,
            "cross_split_input_overlap_measured": True,
            "audio_cross_split_context_overlap_seconds": 0.0,
            "basis": "REAL_319_ROW_AUDIO_IDENTITY_GATE_PLUS_SYNTHETIC_PATH_SENTINEL",
        },
        "embargo_components_seconds": components,
        "embargo_evaluation": evaluate_final_embargo_candidate(components),
        "split_guard": split_guard,
        "formal_feature_extraction_run": False,
        "real_neural_waveform_read": False,
        "baseline_run": False,
        "exchange_candidate_created": False,
        "scientific_result_claimed": False,
    }
    report = finalize_candidate_report(evidence, config)
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed_checks": report["failed_checks"],
                "model_inventory_total_bytes": report["model_inventory_total_bytes"],
                "audio_cross_split_context_overlap_seconds": report["context"][
                    "audio_cross_split_context_overlap_seconds"
                ],
                "final_embargo_candidate_seconds": report["embargo_evaluation"].get(
                    "final_embargo_candidate_seconds"
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == CANDIDATE_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
