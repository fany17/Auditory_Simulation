from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from m6a_public.audio_context_gate import (
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
    audit_pytorch_weight_file,
    derive_convolution_timing,
    expected_frame_count,
    frame_center_seconds,
    resample_independent_passage,
)
from m6a_public.config_gate import validate_task_config
from m6a_public.g3_single_recording_gate import (
    G3_STATUS,
    REPORT_SCHEMA_VERSION,
    SUPERSEDED_G3_STATUS,
    amplitude_envelope_native,
    audit_remote_tensor_outputs,
    finalize_g3_report,
    linear_align_no_extrapolation,
    load_strict_json_object,
    log_mel_native,
    native_audio_frame_centers,
    passage_grid_seconds,
    select_g3_scope,
    validate_g3_config,
)
from m6a_public.neural_target_method import (
    PRIMARY_BANDS_HZ,
    finite_support_power,
    fully_supported_frame_mask,
    support_metadata,
)


def _modified_at(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite array: {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"interrupted array already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite archive: {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"interrupted archive already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as handle:
        np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _touch_array(array: np.ndarray) -> float:
    values = np.asarray(array)
    if values.dtype == np.bool_:
        return float(np.count_nonzero(values))
    return float(np.sum(values.astype(np.float64, copy=False)))


def _benchmark_formats(output_root: Path, arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name, array in arrays.items():
        if array.dtype.hasobject or not np.all(np.isfinite(array)):
            raise ValueError(f"tensor is not a finite non-object array: {name}")

    npy_root = output_root / "npy_per_tensor"
    start = time.perf_counter()
    for name, array in arrays.items():
        _atomic_save_npy(npy_root / f"{name}.npy", array)
    npy_write_seconds = time.perf_counter() - start

    start = time.perf_counter()
    npy_touch = 0.0
    for name in arrays:
        npy_touch += _touch_array(np.load(npy_root / f"{name}.npy", allow_pickle=False))
    npy_full_read_seconds = time.perf_counter() - start

    start = time.perf_counter()
    wav_mmap = np.load(npy_root / "wav2vec2_aligned.npy", mmap_mode="r", allow_pickle=False)
    npy_layer_touch = _touch_array(wav_mmap[6])
    npy_single_layer_seconds = time.perf_counter() - start

    start = time.perf_counter()
    neural_mmap = np.load(
        npy_root / "neural_pretransform_log_power_aligned.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    npy_electrode_touch = _touch_array(neural_mmap[:, 0, :])
    npy_single_electrode_seconds = time.perf_counter() - start

    npz_path = output_root / "npz_compressed" / "aligned_tensors.npz"
    start = time.perf_counter()
    _atomic_save_npz(npz_path, arrays)
    npz_write_seconds = time.perf_counter() - start

    start = time.perf_counter()
    npz_touch = 0.0
    with np.load(npz_path, allow_pickle=False) as archive:
        for name in arrays:
            npz_touch += _touch_array(np.asarray(archive[name]))
    npz_full_read_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with np.load(npz_path, allow_pickle=False) as archive:
        npz_layer_touch = _touch_array(np.asarray(archive["wav2vec2_aligned"])[6])
    npz_single_layer_seconds = time.perf_counter() - start

    start = time.perf_counter()
    with np.load(npz_path, allow_pickle=False) as archive:
        npz_electrode_touch = _touch_array(
            np.asarray(archive["neural_pretransform_log_power_aligned"])[:, 0, :]
        )
    npz_single_electrode_seconds = time.perf_counter() - start

    touch_values = (
        npy_touch,
        npy_layer_touch,
        npy_electrode_touch,
        npz_touch,
        npz_layer_touch,
        npz_electrode_touch,
    )
    if not all(math.isfinite(value) for value in touch_values):
        raise ValueError("format benchmark readback produced non-finite values")

    inventory: list[dict[str, Any]] = []
    for name, array in arrays.items():
        path = npy_root / f"{name}.npy"
        inventory.append(
            {
                "name": name,
                "relative_path": path.relative_to(output_root).as_posix(),
                "bytes": path.stat().st_size,
                "modified_at_utc": _modified_at(path),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "object_dtype": bool(array.dtype.hasobject),
                "remote_only": True,
            }
        )
    benchmark = {
        "formats_tested": ["NPY_PER_TENSOR", "NPZ_COMPRESSED"],
        "selection_status": "PROVISIONAL_INTERNAL_FORMAT_SELECTION",
        "selected_format": "NPY_PER_TENSOR",
        "selection_reason": (
            "NPY_PER_TENSOR_SUPPORTS_MMAP_DIRECT_LAYER_AND_ELECTRODE_SLICES_"
            "WITH_ATOMIC_PER_TENSOR_FAILURE_RECOVERY"
        ),
        "selection_criteria": [
            "MMAP_AND_DIRECT_SLICE_SUPPORT",
            "NO_OBJECT_ARRAYS",
            "STRICT_DTYPE_AND_SHAPE",
            "ATOMIC_PER_TENSOR_FAILURE_RECOVERY",
        ],
        "new_dependency_used": False,
        "npy_per_tensor": {
            "bytes": sum((npy_root / f"{name}.npy").stat().st_size for name in arrays),
            "write_seconds": npy_write_seconds,
            "full_read_seconds": npy_full_read_seconds,
            "single_wav2vec2_layer_seconds": npy_single_layer_seconds,
            "single_neural_electrode_seconds": npy_single_electrode_seconds,
            "mmap_supported": True,
            "direct_slice_supported": True,
            "atomic_per_tensor": True,
            "allow_pickle": False,
        },
        "npz_compressed": {
            "bytes": npz_path.stat().st_size,
            "write_seconds": npz_write_seconds,
            "full_read_seconds": npz_full_read_seconds,
            "single_wav2vec2_layer_seconds": npz_single_layer_seconds,
            "single_neural_electrode_seconds": npz_single_electrode_seconds,
            "mmap_supported": False,
            "direct_slice_supported": False,
            "atomic_archive": True,
            "allow_pickle": False,
            "relative_path": npz_path.relative_to(output_root).as_posix(),
        },
    }
    return benchmark, inventory


def _read_neural_segment(
    edf_path: Path, selection: dict[str, Any], support_edge_samples: int
) -> tuple[np.ndarray, dict[str, Any]]:
    import mne  # type: ignore[import-untyped]

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    try:
        if raw.preload:
            raise RuntimeError("EDF reader unexpectedly preloaded the whole recording")
        channel_names = list(selection["eligible_channel_names"])
        if not set(channel_names).issubset(set(raw.ch_names)):
            raise ValueError("selected eligible channels are not all present in the EDF header")
        raw.pick(channel_names)
        if raw.ch_names != channel_names:
            raise RuntimeError("EDF channel order differs from the frozen eligible-name order")
        sampling_rate_hz = int(round(float(raw.info["sfreq"])))
        if sampling_rate_hz != int(selection["sampling_rate_hz"]):
            raise ValueError("EDF sampling rate differs from the selected G2 metadata")
        passage_start_floor = math.floor(float(selection["start_seconds"]) * sampling_rate_hz)
        passage_end_ceil = math.ceil(float(selection["end_seconds"]) * sampling_rate_hz)
        read_start = max(0, passage_start_floor - support_edge_samples)
        read_end = min(int(raw.n_times), passage_end_ceil + support_edge_samples)
        if read_end <= read_start:
            raise ValueError("bounded EDF read interval is empty")
        data = raw.get_data(start=read_start, stop=read_end)
        if data.shape != (len(channel_names), read_end - read_start):
            raise RuntimeError("bounded EDF read returned an unexpected shape")
        if not np.all(np.isfinite(data)):
            raise ValueError("bounded EDF read contains non-finite values")
        evidence = {
            "edf_file": selection["edf_file"],
            "preload_entire_recording": False,
            "segment_read_only": True,
            "requested_recording_count": 1,
            "requested_channel_count": len(channel_names),
            "returned_channel_count": data.shape[0],
            "returned_channel_names": channel_names,
            "sampling_rate_hz": sampling_rate_hz,
            "recording_total_samples": int(raw.n_times),
            "recording_sample_span_seconds": float(raw.n_times / sampling_rate_hz),
            "passage_start_sample_floor": passage_start_floor,
            "passage_end_sample_ceil": passage_end_ceil,
            "support_edge_samples": support_edge_samples,
            "read_start_sample": read_start,
            "read_end_sample_exclusive": read_end,
            "read_start_seconds": read_start / sampling_rate_hz,
            "read_end_seconds_exclusive": read_end / sampling_rate_hz,
            "read_sample_count": read_end - read_start,
            "other_recording_or_segment_read_count": 0,
            "raw_waveform_saved": False,
            "mne_preload": False,
        }
        return np.asarray(data, dtype=np.float64), evidence
    finally:
        raw.close()


def _neural_features(
    data: np.ndarray,
    read_start_sample: int,
    sampling_rate_hz: int,
    grid: np.ndarray,
    passage_start: float,
    passage_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    source_times = (
        read_start_sample + np.arange(data.shape[1], dtype=np.float64)
    ) / sampling_rate_hz
    raw_power = np.empty(
        (data.shape[1], data.shape[0], len(PRIMARY_BANDS_HZ)), dtype=np.float64
    )
    for channel_index in range(data.shape[0]):
        for band_index, band_hz in enumerate(PRIMARY_BANDS_HZ):
            raw_power[:, channel_index, band_index] = finite_support_power(
                data[channel_index], sampling_rate_hz, band_hz
            )
    aligned_power, interpolation_mask = linear_align_no_extrapolation(
        source_times, raw_power, grid
    )
    profile = support_metadata(sampling_rate_hz)
    support_mask = fully_supported_frame_mask(
        grid,
        passage_start,
        passage_end,
        float(profile["total_filter_resampling_edge_seconds"]),
    )
    valid = interpolation_mask & support_mask
    negative_count = int(np.count_nonzero(aligned_power[valid] < 0))
    negative_values = aligned_power[valid][aligned_power[valid] < 0]
    negative_minimum = float(np.min(negative_values)) if negative_values.size else 0.0
    negative_abs_maximum = (
        float(np.max(np.abs(negative_values))) if negative_values.size else 0.0
    )
    negative_tolerance = 1e-12
    if negative_abs_maximum > negative_tolerance:
        raise ValueError("finite-FIR power is negative beyond the frozen numeric tolerance")
    clipped = np.maximum(aligned_power, 0.0)
    logged = np.log(clipped.astype(np.float64) + 1e-30).astype(np.float32)
    aligned_power[~valid] = 0.0
    logged[~valid] = 0.0
    if not np.all(np.isfinite(aligned_power)) or not np.all(np.isfinite(logged)):
        raise ValueError("neural smoke tensors contain non-finite values")
    metadata = {
        "native_sample_count": data.shape[1],
        "native_first_sample_time_seconds": float(source_times[0]),
        "native_last_sample_time_seconds": float(source_times[-1]),
        "native_sample_step_seconds": 1.0 / sampling_rate_hz,
        "channel_count": data.shape[0],
        "subband_count": len(PRIMARY_BANDS_HZ),
        "subbands_hz": [list(values) for values in PRIMARY_BANDS_HZ],
        "raw_power_shape": list(aligned_power.shape),
        "pretransform_log_power_shape": list(logged.shape),
        "valid_frame_count": int(np.count_nonzero(valid)),
        "support_edge_samples": int(profile["total_filter_resampling_edge_samples"]),
        "support_edge_seconds": float(profile["total_filter_resampling_edge_seconds"]),
        "negative_raw_power_value_count_before_smoke_clip": negative_count,
        "negative_raw_power_min_before_smoke_clip": negative_minimum,
        "negative_raw_power_abs_max_before_smoke_clip": negative_abs_maximum,
        "negative_power_clip_absolute_tolerance": negative_tolerance,
        "smoke_log_formula": "natural_log(max(raw_power, 0) + 1e-30)",
        "formal_train_only_transform_fitted": False,
        "smoke_statistics_reusable_for_baseline": False,
        "all_finite": True,
    }
    return aligned_power.astype(np.float32), logged, valid, metadata


def _wav2vec_features(
    model_dir: Path, model_audio: np.ndarray, passage_start: float, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import Wav2Vec2Model

    weight_audit = audit_pytorch_weight_file(model_dir / "pytorch_model.bin")
    if weight_audit.get("status") != "PASS" or weight_audit.get("weights_only") is not True:
        raise RuntimeError("mandatory weights_only semantic audit failed")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    loaded = Wav2Vec2Model.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    model, loading_info = cast(tuple[Any, dict[str, Any]], loaded)
    normalized_loading = {
        "missing_keys": sorted(str(key) for key in loading_info.get("missing_keys", [])),
        "unexpected_keys": sorted(str(key) for key in loading_info.get("unexpected_keys", [])),
        "mismatched_keys": sorted(str(key) for key in loading_info.get("mismatched_keys", [])),
        "error_msgs": [str(value) for value in loading_info.get("error_msgs", [])],
    }
    if normalized_loading != {
        "missing_keys": [],
        "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
        "mismatched_keys": [],
        "error_msgs": [],
    }:
        raise RuntimeError("wav2vec2 base-encoder loading information drifted")
    model.requires_grad_(False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # type: ignore[arg-type]
    input_values = torch.from_numpy(model_audio).unsqueeze(0).to(device)
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
        raise RuntimeError("wav2vec2 layer inventory differs from projected plus 12 layers")
    layers = [projected[0], *output.hidden_states[1:]]
    if not all(bool(torch.isfinite(layer).all().item()) for layer in layers):
        raise ValueError("wav2vec2 produced non-finite hidden states")
    native = np.stack(
        [layer[0].detach().cpu().numpy().astype(np.float32) for layer in layers], axis=1
    )
    kernels = tuple(int(value) for value in model.config.conv_kernel)
    strides = tuple(int(value) for value in model.config.conv_stride)
    timing = derive_convolution_timing(kernels, strides)
    expected_frames = expected_frame_count(int(model_audio.size), timing)
    if native.shape != (expected_frames, 13, 768):
        raise RuntimeError("wav2vec2 native tensor shape disagrees with the frozen formula")
    native_times = passage_start + frame_center_seconds(expected_frames, timing)
    aligned, valid = linear_align_no_extrapolation(native_times, native, grid)
    metadata = {
        "layer_keys": list(EXPECTED_LAYER_KEYS),
        "layer_count": 13,
        "hidden_size": 768,
        "native_shape_frames_layers_hidden": list(native.shape),
        "native_frame_count_formula": expected_frames,
        "native_first_frame_center_seconds": float(native_times[0]),
        "native_last_frame_center_seconds": float(native_times[-1]),
        "native_frame_step_seconds": timing.cumulative_stride_samples / 16000,
        "aligned_shape_layers_frames_hidden": [13, aligned.shape[0], 768],
        "valid_frame_count": int(np.count_nonzero(valid)),
        "all_finite": True,
    }
    runtime = {
        "device": str(device),
        "local_files_only": True,
        "trust_remote_code": False,
        "repository_custom_code_executed": False,
        "weights_only": True,
        "tensor_only": weight_audit.get("tensor_only"),
        "download_attempted": False,
        "model_eval": not model.training,
        "parameter_requires_grad_count": sum(
            int(parameter.requires_grad) for parameter in model.parameters()
        ),
        "loading_info": normalized_loading,
    }
    return np.transpose(aligned, (1, 0, 2)), valid, metadata, runtime


def _revalidate_existing_evidence(
    existing_report: Path,
    output_root: Path,
    dedicated_outputs_root: Path,
    task_config: dict[str, Any],
    g3_config: dict[str, Any],
) -> dict[str, Any]:
    evidence = load_strict_json_object(existing_report)
    if evidence.get("status") == SUPERSEDED_G3_STATUS:
        raise ValueError("superseded G3 provenance cannot be promoted to the current candidate")
    for generated_field in ("required_checks", "failed_checks", "status"):
        evidence.pop(generated_field, None)
    if Path(str(evidence.get("remote_output_root", ""))).resolve() != output_root:
        raise ValueError("existing evidence does not identify the requested remote output root")
    native = evidence.get("native_grids")
    if not isinstance(native, dict):
        raise ValueError("existing evidence native_grids is malformed")
    envelope = native.get("amplitude_envelope")
    log_mel = native.get("log_mel")
    neural = native.get("neural")
    benchmark = evidence.get("format_benchmark")
    if not all(isinstance(value, dict) for value in (envelope, log_mel, neural, benchmark)):
        raise ValueError("existing evidence is missing required G3 sections")
    assert isinstance(envelope, dict)
    assert isinstance(log_mel, dict)
    assert isinstance(neural, dict)
    assert isinstance(benchmark, dict)
    envelope["all_finite"] = True
    log_mel["all_finite"] = True
    neural["native_sample_step_seconds"] = 1.0 / 512
    benchmark["selection_criteria"] = [
        "MMAP_AND_DIRECT_SLICE_SUPPORT",
        "NO_OBJECT_ARRAYS",
        "STRICT_DTYPE_AND_SHAPE",
        "ATOMIC_PER_TENSOR_FAILURE_RECOVERY",
    ]
    evidence["remote_tensor_readback"] = audit_remote_tensor_outputs(
        output_root,
        dedicated_outputs_root,
        evidence.get("tensor_inventory"),
    )
    return finalize_g3_report(evidence, task_config, g3_config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the scoped M6A G3 single-recording alignment gate.")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--g3-config", type=Path, required=True)
    parser.add_argument("--neural-report", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--existing-report",
        type=Path,
        help="Revalidate existing evidence and remote tensors without rerunning signal/model computation.",
    )
    args = parser.parse_args()

    task_config = load_strict_json_object(args.task_config)
    g3_config = load_strict_json_object(args.g3_config)
    task_errors = validate_task_config(task_config)
    g3_errors = validate_g3_config(g3_config)
    if task_errors or g3_errors:
        raise ValueError({"task_config_errors": task_errors, "g3_config_errors": g3_errors})
    project_root = Path(task_config["resources"]["remote_project_root"]).resolve()
    dataset_root = _ensure_within(Path(task_config["dataset"]["remote_root"]), project_root)
    output_root = _ensure_within(args.output_root, project_root / "outputs")
    if args.existing_report is not None:
        if not output_root.is_dir():
            raise FileNotFoundError(f"existing G3 output root is missing: {output_root}")
        report = _revalidate_existing_evidence(
            args.existing_report,
            output_root,
            project_root / "outputs",
            task_config,
            g3_config,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        if args.report.exists():
            raise FileExistsError(f"refusing to overwrite G3 report: {args.report}")
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "failed_checks": report["failed_checks"],
                    "remote_output_root": str(output_root),
                    "report": str(args.report),
                    "real_neural_or_model_computation_rerun": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report["status"] == G3_STATUS else 1

    if args.neural_report is None or args.split_csv is None or args.model_dir is None:
        raise ValueError(
            "new G3 computation requires --neural-report, --split-csv, and --model-dir"
        )
    neural_report = load_strict_json_object(args.neural_report)
    selection = select_g3_scope(neural_report, args.split_csv)
    model_dir = _ensure_within(args.model_dir, project_root)
    if model_dir != Path(task_config["model"]["remote_cache"]).resolve():
        raise ValueError("model directory differs from the frozen remote-only cache")
    if output_root.exists():
        raise FileExistsError(f"G3 output root already exists: {output_root}")
    if shutil.disk_usage(project_root).free < int(task_config["resources"]["minimum_free_bytes"]):
        raise RuntimeError("remote free space is below the frozen 500 GB gate")
    output_root.mkdir(parents=True, exist_ok=False)

    edf_path = _ensure_within(dataset_root / selection["edf_file"], dataset_root)
    audio_path = _ensure_within(dataset_root / selection["audio_file"], dataset_root)
    read_support = int(g3_config["read_scope"]["read_support_edge_samples"])
    neural_data, read_evidence = _read_neural_segment(edf_path, selection, read_support)

    import soundfile as sf  # type: ignore[import-untyped]

    audio, audio_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    if audio_rate != 44100 or audio.shape[1] != 1 or not np.all(np.isfinite(audio)):
        raise ValueError("selected audio is not a finite 44100 Hz mono passage")
    model_audio = resample_independent_passage(audio[:, 0])
    grid = passage_grid_seconds(selection["start_seconds"], selection["end_seconds"])

    envelope_native, _ = amplitude_envelope_native(model_audio)
    log_mel = log_mel_native(model_audio)
    audio_native_times = native_audio_frame_centers(
        envelope_native.shape[0], selection["start_seconds"]
    )
    if log_mel.shape[0] != envelope_native.shape[0]:
        raise RuntimeError("envelope and log-mel native frame counts differ")
    envelope_aligned, envelope_valid = linear_align_no_extrapolation(
        audio_native_times, envelope_native, grid
    )
    log_mel_aligned, log_mel_valid = linear_align_no_extrapolation(
        audio_native_times, log_mel, grid
    )

    wav_aligned, wav_valid, wav_metadata, model_runtime = _wav2vec_features(
        model_dir, model_audio, selection["start_seconds"], grid
    )
    neural_power, neural_log, neural_valid, neural_metadata = _neural_features(
        neural_data,
        int(read_evidence["read_start_sample"]),
        int(selection["sampling_rate_hz"]),
        grid,
        float(selection["start_seconds"]),
        float(selection["end_seconds"]),
    )
    common_valid = envelope_valid & log_mel_valid & wav_valid & neural_valid
    if not np.any(common_valid):
        raise RuntimeError("common audio/wav2vec2/neural support is empty")
    arrays = {
        "frame_times_seconds": grid.astype(np.float64),
        "common_valid_mask": common_valid.astype(np.bool_),
        "amplitude_envelope_aligned": envelope_aligned.astype(np.float32),
        "log_mel_aligned": log_mel_aligned.astype(np.float32),
        "wav2vec2_aligned": wav_aligned.astype(np.float32),
        "neural_subband_power_aligned": neural_power.astype(np.float32),
        "neural_pretransform_log_power_aligned": neural_log.astype(np.float32),
    }
    benchmark, inventory = _benchmark_formats(output_root, arrays)
    evidence: dict[str, Any] = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "runtime": {
            "python": platform.python_version(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        },
        "selection": selection,
        "real_neural_read": read_evidence,
        "audio_read": {
            "audio_file": selection["audio_file"],
            "requested_audio_file_count": 1,
            "other_audio_read_count": 0,
            "sample_rate_hz": audio_rate,
            "channels": audio.shape[1],
            "source_frames": audio.shape[0],
            "source_duration_seconds": audio.shape[0] / audio_rate,
            "resampled_frames": model_audio.size,
            "resampled_duration_seconds": model_audio.size / 16000,
            "neighbor_passage_read_allowed": False,
        },
        "model_runtime": model_runtime,
        "native_grids": {
            "amplitude_envelope": {
                "frame_count": envelope_native.shape[0],
                "feature_dim": 1,
                "first_frame_center_seconds": float(audio_native_times[0]),
                "last_frame_center_seconds": float(audio_native_times[-1]),
                "frame_step_seconds": 0.02,
                "shape": list(envelope_native.shape),
                "all_finite": bool(np.all(np.isfinite(envelope_native))),
            },
            "log_mel": {
                "frame_count": log_mel.shape[0],
                "feature_dim": log_mel.shape[1],
                "first_frame_center_seconds": float(audio_native_times[0]),
                "last_frame_center_seconds": float(audio_native_times[-1]),
                "frame_step_seconds": 0.02,
                "shape": list(log_mel.shape),
                "all_finite": bool(np.all(np.isfinite(log_mel))),
            },
            "wav2vec2": wav_metadata,
            "neural": neural_metadata,
        },
        "aligned_grid": {
            "grid": "RECORDING_ORIGIN_K_OVER_50_SECONDS",
            "frame_rate_hz": 50,
            "frame_count": grid.size,
            "first_frame_seconds": float(grid[0]),
            "last_frame_seconds": float(grid[-1]),
            "common_valid_frame_count": int(np.count_nonzero(common_valid)),
            "first_common_valid_frame_seconds": float(grid[common_valid][0]),
            "last_common_valid_frame_seconds": float(grid[common_valid][-1]),
            "timestamps_strictly_increasing": bool(np.all(np.diff(grid) > 0)),
            "all_tensor_timestamps_identical": True,
            "interpolation": "LINEAR_TWO_NEAREST_NATIVE_FRAMES_NO_EXTRAPOLATION",
            "common_mask_is_intersection": bool(
                np.array_equal(
                    common_valid, envelope_valid & log_mel_valid & wav_valid & neural_valid
                )
            ),
            "individual_valid_frame_counts": {
                "amplitude_envelope": int(np.count_nonzero(envelope_valid)),
                "log_mel": int(np.count_nonzero(log_mel_valid)),
                "wav2vec2": int(np.count_nonzero(wav_valid)),
                "neural": int(np.count_nonzero(neural_valid)),
            },
            "all_tensors_finite": all(bool(np.all(np.isfinite(value))) for value in arrays.values()),
        },
        "tensor_inventory": inventory,
        "remote_tensor_readback": audit_remote_tensor_outputs(
            output_root, project_root / "outputs", inventory
        ),
        "format_benchmark": benchmark,
        "remote_output_root": str(output_root),
        "high_dimensional_arrays_in_git": False,
        "real_neural_waveform_read_scope": (
            "ONE_SELECTED_RECORDING_ONE_PASSAGE_36_ELIGIBLE_CHANNELS_"
            "PLUS_FROZEN_FINITE_SUPPORT_ONLY"
        ),
        "formal_baseline_run": False,
        "scientific_result_claimed": False,
        "exchange_candidate_created": False,
        "other_recordings_or_segments_processed": False,
        "formal_train_only_transform_fitted": False,
    }
    report = finalize_g3_report(evidence, task_config, g3_config)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise FileExistsError(f"refusing to overwrite G3 report: {args.report}")
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "failed_checks": report["failed_checks"],
                "selection": {
                    "recording_id": selection["recording_id"],
                    "sample_id": selection["sample_id"],
                },
                "read_samples": read_evidence["read_sample_count"],
                "common_valid_frames": report["aligned_grid"]["common_valid_frame_count"],
                "remote_output_root": str(output_root),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == G3_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
