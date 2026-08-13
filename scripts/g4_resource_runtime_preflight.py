from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from m6a_public.audio_context_gate import (
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
    audit_model_config,
    audit_pytorch_weight_file,
    derive_convolution_timing,
    expected_frame_count,
    load_strict_json_object,
)
from m6a_public.g4_preflight_gate import (
    EXPECTED_FRAME_COUNT,
    EXPECTED_SYNTHETIC_SAMPLE_COUNT,
    PREFLIGHT_REPORT_SCHEMA_VERSION,
    PREFLIGHT_STATUS,
    audit_longest_passage_manifest,
    conservative_runtime_estimate,
    finalize_preflight_report,
    validate_preflight_config,
)
from m6a_public.wav2vec2_preprocessing import (
    PREPROCESSOR_SEMANTICS,
    PREPROCESSOR_SOURCE_ENDPOINT,
    audit_preprocessor_config,
    normalize_passage_waveform,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _tree_regular_file_bytes(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"preflight inventory directory is missing: {resolved}")
    total = 0
    file_count = 0
    symlink_count = 0
    for directory, directory_names, file_names in os.walk(resolved, followlinks=False):
        base = Path(directory)
        retained_directories: list[str] = []
        for name in directory_names:
            candidate = base / name
            if candidate.is_symlink():
                symlink_count += 1
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in file_names:
            candidate = base / name
            if candidate.is_symlink():
                symlink_count += 1
                continue
            stat = candidate.stat()
            total += int(stat.st_size)
            file_count += 1
    return {
        "path": resolved.as_posix(),
        "exists": True,
        "bytes": total,
        "regular_file_count": file_count,
        "symlink_entry_count_not_followed": symlink_count,
        "modified_at_utc": datetime.fromtimestamp(
            resolved.stat().st_mtime, timezone.utc
        ).isoformat(),
    }


def _audit_storage(config: dict[str, Any]) -> dict[str, Any]:
    storage = cast(dict[str, Any], config["storage"])
    project_root = Path(str(storage["project_root"])).resolve()
    categories_config = cast(dict[str, list[str]], storage["categories"])
    categories: dict[str, Any] = {}
    selected_total = 0
    for name, configured_paths in categories_config.items():
        paths: list[dict[str, Any]] = []
        for configured in configured_paths:
            path = Path(configured).resolve()
            if path == project_root or not _inside(path, project_root):
                raise ValueError(f"inventory path escapes or equals project root: {path}")
            item = _tree_regular_file_bytes(path)
            paths.append(item)
        category_bytes = sum(int(item["bytes"]) for item in paths)
        categories[name] = {"paths": paths, "bytes": category_bytes}
        selected_total += category_bytes
    project = _tree_regular_file_bytes(project_root)
    project_total = int(project["bytes"])
    if selected_total > project_total:
        raise RuntimeError("selected category bytes exceed project-root bytes")
    data_bytes = int(categories["data"]["bytes"])
    cache_bytes = int(categories["cache"]["bytes"])
    estimated_new = int(storage["estimated_new_bytes_upper_bound"])
    return {
        "audited_at_utc": _utc_now(),
        "project_root": project_root.as_posix(),
        "measurement": "REGULAR_FILE_BYTES_WITHOUT_FOLLOWING_SYMLINKS",
        "categories": categories,
        "selected_category_total_bytes": selected_total,
        "project_root_total_bytes": project_total,
        "project_root_regular_file_count": project["regular_file_count"],
        "unclassified_other_bytes": project_total - selected_total,
        "actual_free_bytes": int(shutil.disk_usage(project_root).free),
        "estimated_new_bytes_upper_bound": estimated_new,
        "data_cache_plus_estimated_new_bytes": data_bytes + cache_bytes + estimated_new,
    }


def _synthetic_mono(sample_count: int) -> np.ndarray:
    if sample_count < 1:
        raise ValueError("synthetic sample count must be positive")
    index = np.arange(sample_count, dtype=np.float64)
    values = (
        0.1 * np.sin(2.0 * math.pi * 220.0 * index / 16_000.0)
        + 0.05 * np.sin(2.0 * math.pi * 440.0 * index / 16_000.0)
    ).astype(np.float32)
    if values.shape != (sample_count,) or not np.all(np.isfinite(values)):
        raise RuntimeError("deterministic synthetic mono generation failed")
    return values


def _normalize_loading_info(value: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "missing_keys": sorted(str(key) for key in value.get("missing_keys", [])),
        "unexpected_keys": sorted(str(key) for key in value.get("unexpected_keys", [])),
        "mismatched_keys": sorted(str(key) for key in value.get("mismatched_keys", [])),
        "error_msgs": [str(message) for message in value.get("error_msgs", [])],
    }


def _forward_layers(model: Any, input_values: Any, torch: Any) -> tuple[list[Any], Any]:
    projected: list[Any] = []

    def capture_projection(_module: Any, _inputs: Any, output: Any) -> None:
        projected.append(output[0].detach())

    hook = model.feature_projection.register_forward_hook(capture_projection)
    try:
        with torch.inference_mode():
            output = model(
                input_values=input_values,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        hook.remove()
    if len(projected) != 1 or output.hidden_states is None:
        raise RuntimeError("projected or hidden-state capture failed")
    return [projected[0], *output.hidden_states[1:]], output


def _feature_extractor_equivalence(
    extractor: Any, raw_values: np.ndarray, label: str
) -> tuple[np.ndarray, dict[str, Any]]:
    manual, normalization = normalize_passage_waveform(raw_values)
    encoded = extractor(
        raw_values,
        sampling_rate=16_000,
        padding=False,
        return_attention_mask=False,
        return_tensors="np",
    )
    if "attention_mask" in encoded:
        raise RuntimeError("no-padding feature extractor returned an attention mask")
    extracted = np.asarray(encoded["input_values"][0], dtype=np.float32)
    if extracted.shape != manual.shape or not np.all(np.isfinite(extracted)):
        raise RuntimeError("feature extractor output shape or finite gate failed")
    difference = np.abs(extracted.astype(np.float64) - manual.astype(np.float64))
    max_absolute_difference = float(difference.max(initial=0.0))
    equivalent = bool(np.allclose(extracted, manual, rtol=1e-7, atol=1e-7))
    if not equivalent:
        raise RuntimeError("manual preprocessing does not match feature extractor")
    return extracted, {
        "label": label,
        "status": "PASS",
        "sample_count": int(raw_values.size),
        "feature_extractor_output_shape": list(extracted.shape),
        "feature_extractor_output_dtype": str(extracted.dtype),
        "attention_mask_returned": False,
        "attention_mask_argument": "OMITTED",
        "padding_used": False,
        "absolute_tolerance": 1e-7,
        "relative_tolerance": 1e-7,
        "max_absolute_difference": max_absolute_difference,
        "equivalent": equivalent,
        "normalization": normalization,
    }


def _run_runtime_canary(
    config: dict[str, Any], model_dir: Path, mirror_audit: dict[str, Any]
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    model_config_audit = audit_model_config(model_dir / "config.json")
    preprocessor_config_audit = audit_preprocessor_config(
        model_dir / "preprocessor_config.json", expected_cache_root=model_dir
    )
    load_started = time.perf_counter()
    weight_audit = audit_pytorch_weight_file(model_dir / "pytorch_model.bin")
    if (
        model_config_audit.get("status") != "PASS"
        or preprocessor_config_audit.get("status") != "PASS"
        or weight_audit.get("status") != "PASS"
        or weight_audit.get("weights_only") is not True
        or weight_audit.get("tensor_only") is not True
    ):
        raise RuntimeError("frozen local model semantic audit failed")
    if (
        mirror_audit.get("status") != "PASS"
        or mirror_audit.get("source_endpoint") != PREPROCESSOR_SOURCE_ENDPOINT
        or mirror_audit.get("filename") != "preprocessor_config.json"
        or mirror_audit.get("http_status") != 200
        or mirror_audit.get("mirror_semantic_fields") != PREPROCESSOR_SEMANTICS
        or _mapping(mirror_audit.get("cache_audit")).get("bytes")
        != preprocessor_config_audit.get("bytes")
        or mirror_audit.get("cache_write_performed") is not False
        or mirror_audit.get("network_body_persisted") is not False
    ):
        raise RuntimeError("remote preprocessor mirror semantic audit failed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    extractor_semantics = {
        "feature_size": extractor.feature_size,
        "sampling_rate": extractor.sampling_rate,
        "padding_value": extractor.padding_value,
        "do_normalize": extractor.do_normalize,
        "return_attention_mask": extractor.return_attention_mask,
        "padding_side": extractor.padding_side,
    }
    if extractor_semantics != PREPROCESSOR_SEMANTICS:
        raise RuntimeError("local feature extractor semantics drifted")
    loaded = Wav2Vec2Model.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        output_loading_info=True,
    )
    model, loading_info = cast(tuple[Any, dict[str, Any]], loaded)
    normalized_loading = _normalize_loading_info(loading_info)
    if normalized_loading != {
        "missing_keys": [],
        "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
        "mismatched_keys": [],
        "error_msgs": [],
    }:
        raise RuntimeError("base encoder loading information drifted")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the runtime preflight")
    device = torch.device("cuda:0")
    model.requires_grad_(False)
    model.eval()
    model.to(device)  # type: ignore[arg-type]
    torch.cuda.synchronize(device)
    model_load_wall = time.perf_counter() - load_started

    warmup_raw = _synthetic_mono(16_000)
    warmup_np, warmup_preprocessing = _feature_extractor_equivalence(
        extractor, warmup_raw, "ONE_SECOND_WARMUP"
    )
    warmup_values = torch.from_numpy(warmup_np).unsqueeze(0).to(device)
    warmup_started = time.perf_counter()
    warmup_layers, _ = _forward_layers(model, warmup_values, torch)
    torch.cuda.synchronize(device)
    warmup_wall = time.perf_counter() - warmup_started
    warmup_ok = (
        len(warmup_layers) == 13
        and all(bool(torch.isfinite(layer).all().item()) for layer in warmup_layers)
    )
    del warmup_layers, warmup_values, warmup_np, warmup_raw
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    longest_raw = _synthetic_mono(EXPECTED_SYNTHETIC_SAMPLE_COUNT)
    longest_np, longest_preprocessing = _feature_extractor_equivalence(
        extractor, longest_raw, "LONGEST_G4_PASSAGE"
    )
    longest_values = torch.from_numpy(longest_np).unsqueeze(0).to(device)
    forward_started = time.perf_counter()
    layers, _ = _forward_layers(model, longest_values, torch)
    torch.cuda.synchronize(device)
    forward_wall = time.perf_counter() - forward_started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    kernels = tuple(int(value) for value in model.config.conv_kernel)
    strides = tuple(int(value) for value in model.config.conv_stride)
    timing = derive_convolution_timing(kernels, strides)
    expected_frames = expected_frame_count(EXPECTED_SYNTHETIC_SAMPLE_COUNT, timing)
    layer_shapes = [list(layer.shape) for layer in layers]
    all_finite = all(bool(torch.isfinite(layer).all().item()) for layer in layers)
    if (
        len(layers) != 13
        or expected_frames != EXPECTED_FRAME_COUNT
        or layer_shapes != [[1, EXPECTED_FRAME_COUNT, 768]] * 13
        or not all_finite
    ):
        raise RuntimeError("longest synthetic wav2vec2 shape/layer/finite gate failed")

    estimate = conservative_runtime_estimate(model_load_wall, forward_wall)
    estimate["duration_linear_scaling_assumed"] = False
    return {
        "status": "PASS",
        "model_id": "facebook/wav2vec2-base",
        "revision_label": "main",
        "device": str(device),
        "local_files_only": True,
        "trust_remote_code": False,
        "repository_custom_code_executed": False,
        "weights_only": True,
        "tensor_only": weight_audit.get("tensor_only"),
        "download_attempted": False,
        "model_eval": not model.training,
        "inference_mode": True,
        "parameter_requires_grad_count": sum(
            int(parameter.requires_grad) for parameter in model.parameters()
        ),
        "loading_info": normalized_loading,
        "model_load_wall_seconds": model_load_wall,
        "transformers_version": transformers.__version__,
        "preprocessor_remote_semantic_audit": {
            "status": mirror_audit.get("status"),
            "audited_at_utc": mirror_audit.get("audited_at_utc"),
            "source_endpoint": mirror_audit.get("source_endpoint"),
            "filename": mirror_audit.get("filename"),
            "prior_failed_audit": mirror_audit.get("prior_failed_audit"),
            "probes": mirror_audit.get("probes"),
            "http_status": mirror_audit.get("http_status"),
            "mirror_body_bytes": mirror_audit.get("mirror_body_bytes"),
            "mirror_semantic_fields": mirror_audit.get("mirror_semantic_fields"),
            "cache_bytes": _mapping(mirror_audit.get("cache_audit")).get("bytes"),
            "cache_modified_at_utc": _mapping(
                mirror_audit.get("cache_audit")
            ).get("modified_at_utc"),
            "remote_only": _mapping(mirror_audit.get("cache_audit")).get(
                "remote_only"
            ),
            "cache_write_performed": mirror_audit.get("cache_write_performed"),
            "network_body_persisted": mirror_audit.get("network_body_persisted"),
            "proxy_used": mirror_audit.get("proxy_used"),
        },
        "preprocessor_config_audit": preprocessor_config_audit,
        "feature_extractor_semantics": extractor_semantics,
        "feature_extractor_equivalence": {
            "status": "PASS",
            "warmup": warmup_preprocessing,
            "longest": longest_preprocessing,
        },
        "passage_wise_normalization_applied": True,
        "cross_passage_statistics_used": False,
        "train_fitted_preprocessing_statistics_used": False,
        "attention_mask_argument": "OMITTED",
        "attention_mask_tensor_created": False,
        "raw_input_canary_history": "NOT_RUN_NO_SUPERSEDED_PROVENANCE_CREATED",
        "input_source": "DETERMINISTIC_IN_MEMORY_FINITE_MONO_NO_FILE_PATH",
        "input_path": None,
        "real_audio_read": False,
        "real_edf_read": False,
        "synthetic_input_all_finite": bool(np.all(np.isfinite(longest_raw))),
        "preprocessed_input_all_finite": bool(np.all(np.isfinite(longest_np))),
        "synthetic_input_channels": 1,
        "warmup": {
            "status": "PASS" if warmup_ok else "FAIL",
            "sample_count": 16_000,
            "wall_seconds": warmup_wall,
        },
        "longest_forward": {
            "status": "PASS",
            "batch_size": 1,
            "sample_count": EXPECTED_SYNTHETIC_SAMPLE_COUNT,
            "frame_count": expected_frames,
            "layer_keys": list(EXPECTED_LAYER_KEYS),
            "layer_shapes": layer_shapes,
            "all_finite": all_finite,
            "attention_scope": "GLOBAL_WITHIN_ONE_SYNTHETIC_PASSAGE",
            "chunked_or_windowed_approximation_used": False,
            "oom": False,
            "wall_seconds": forward_wall,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
        },
        "execution_estimate": estimate,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite preflight report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-preflight-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"preflight report partial already exists: {partial}")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the M6A G4 storage and synthetic longest-passage runtime preflight."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preprocessor-mirror-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_strict_json_object(args.config)
    schema = load_strict_json_object(args.schema)
    main_config = load_strict_json_object(args.main_config)
    protocol = load_strict_json_object(args.protocol_config)
    config_errors = validate_preflight_config(config, schema, main_config, protocol)
    manifest_audit = audit_longest_passage_manifest(args.split_csv)
    project_root = Path(config["storage"]["project_root"]).resolve()
    model_dir = args.model_dir.resolve()
    expected_model_dir = Path(config["model"]["cache_path"]).resolve()
    output = args.output.resolve()
    output_root = project_root / "logs"
    preprocessor_mirror_audit_path = args.preprocessor_mirror_audit.resolve()
    if model_dir != expected_model_dir or not _inside(model_dir, project_root):
        config_errors.append("model-dir must equal the frozen cache inside project root")
    if not _inside(output, output_root) or output == output_root:
        config_errors.append("output must be a file inside the dedicated logs root")
    if (
        preprocessor_mirror_audit_path.parent != output_root
        or preprocessor_mirror_audit_path == output
    ):
        config_errors.append(
            "preprocessor mirror audit must be a distinct file in project logs"
        )

    storage_audit: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    runtime_error: dict[str, str] | None = None
    try:
        storage_audit = _audit_storage(config)
        if config_errors or manifest_audit.get("status") != "PASS":
            raise RuntimeError("pre-GPU config or manifest gate failed")
        mirror_audit = load_strict_json_object(preprocessor_mirror_audit_path)
        runtime = _run_runtime_canary(config, model_dir, mirror_audit)
    except Exception as error:  # noqa: BLE001 - fail evidence must survive runtime errors
        runtime_error = {"type": type(error).__name__, "message": str(error)}
        runtime = {
            "status": "FAIL",
            "oom": type(error).__name__ == "OutOfMemoryError",
            "download_attempted": False,
            "real_audio_read": False,
            "real_edf_read": False,
        }

    runtime_estimate = _mapping(runtime.get("execution_estimate"))
    evidence = {
        "report_schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "audited_at_utc": _utc_now(),
        "config_path": "configs/m6a_g4_resource_runtime_preflight_candidate.json",
        "schema_path": "schemas/m6a_g4_resource_runtime_preflight_candidate.schema.json",
        "config_errors": config_errors,
        "longest_passage_manifest_audit": manifest_audit,
        "storage_audit": storage_audit,
        "runtime_canary": {key: value for key, value in runtime.items() if key != "execution_estimate"},
        "runtime_error": runtime_error,
        "execution_estimate": runtime_estimate,
        "checkpoint_design_audit": {
            "execution_unit": "ONE_PASSAGE_PER_INVOCATION",
            "successful_passage_checkpoint_required": True,
            "atomic_final_rename_required": True,
            "resume_skips_only_validated_final_checkpoint": True,
            "partial_preserved_on_failure": True,
            "checkpoint_before_two_hour_limit": True,
        },
        "g4_execution_authorized": False,
        "new_real_edf_read": False,
        "new_real_audio_read": False,
        "real_feature_extraction_run": False,
        "ridge_run": False,
        "null_run": False,
        "metric_run": False,
        "scientific_result_claimed": False,
        "exchange_candidate_created": False,
    }
    report = finalize_preflight_report(evidence, config)
    if config_errors or runtime_error is not None:
        report["status"] = "FAIL"
        report["failed_checks"] = sorted(
            set(report.get("failed_checks", [])) | {"preflight_runtime_completed"}
        )
    _atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "required_checks": report["required_checks"],
                "failed_checks": report["failed_checks"],
                "output": output.as_posix(),
                "g4_execution_performed": False,
                "new_real_audio_read": False,
                "new_real_edf_read": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == PREFLIGHT_STATUS else 1


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
