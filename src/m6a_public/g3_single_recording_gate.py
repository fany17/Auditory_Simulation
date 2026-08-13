from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, NoReturn

import numpy as np

from m6a_public.audio_context_gate import (
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
    nonfinite_numeric_paths,
    normalize_relative_audio_path,
)
from m6a_public.config_gate import find_forbidden_fields


G3_STATUS = "G3_SINGLE_RECORDING_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
G3_ACCEPTED_STATUS = "G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY"
SUPERSEDED_G3_STATUS = "SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE"
G3_CONFIG_STATUS = "G3_SINGLE_RECORDING_ALIGNMENT_AUTHORIZED_SCOPED"
REPORT_SCHEMA_VERSION = "m6a-g3-single-recording-alignment-candidate-v1"
EXPECTED_RECORDING_ID = "sub-SD012_ses-02_task-PassiveListen"
EXPECTED_SAMPLE_ID = "sub-SD012_ses-02_task-PassiveListen__seg-004"
EXPECTED_STIMULUS_ID = "s4002b-ex01"
EXPECTED_AUDIO_FILE = "stimuli/excerpts/Block 1/s4002b-ex01_normed.wav"
EXPECTED_EDF_FILE = (
    "sub-SD012/ses-02/ieeg/sub-SD012_ses-02_task-PassiveListen_ieeg.edf"
)
EXPECTED_START_SECONDS = 126.519807944
EXPECTED_END_SECONDS = 161.15130454263945
EXPECTED_SAMPLING_RATE_HZ = 512
EXPECTED_CHANNEL_COUNT = 36
EXPECTED_REFERENCE = "scalp electrode, not included with data"
EXPECTED_GRID_RATE_HZ = 50
EXPECTED_MEL_BINS = 80
EXPECTED_HIDDEN_SIZE = 768
EXPECTED_SUBBAND_COUNT = 6
EXPECTED_AUDIO_SOURCE_FRAMES = 1_527_249
EXPECTED_AUDIO_RESAMPLED_FRAMES = 554_104
EXPECTED_RECORDING_TOTAL_SAMPLES = 1_331_072
EXPECTED_PASSAGE_START_SAMPLE_FLOOR = 64_778
EXPECTED_PASSAGE_END_SAMPLE_CEIL = 82_510
EXPECTED_SUPPORT_EDGE_SAMPLES = 559
EXPECTED_READ_START_SAMPLE = 64_219
EXPECTED_READ_END_SAMPLE_EXCLUSIVE = 83_069
EXPECTED_READ_SAMPLE_COUNT = 18_850
EXPECTED_ALIGNED_FRAME_COUNT = 1_732
EXPECTED_COMMON_VALID_FRAME_COUNT = 1_622
EXPECTED_AUDIO_NATIVE_FRAME_COUNT = 1_731
EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS = 126.532276694
EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS = 161.132276694
EXPECTED_AUDIO_NATIVE_STEP_SECONDS = 0.02
EXPECTED_GRID_FIRST_SECONDS = 126.52
EXPECTED_GRID_LAST_SECONDS = 161.14
EXPECTED_COMMON_FIRST_SECONDS = 127.62
EXPECTED_COMMON_LAST_SECONDS = 160.04
EXPECTED_NEURAL_NATIVE_FIRST_SECONDS = 125.427734375
EXPECTED_NEURAL_NATIVE_LAST_SECONDS = 162.2421875
EXPECTED_NEURAL_NATIVE_STEP_SECONDS = 1.0 / EXPECTED_SAMPLING_RATE_HZ
EXPECTED_REMOTE_OUTPUTS_ROOT = PurePosixPath(
    "/home/fanyu/auditory_simulation_m6a/outputs"
)
EXPECTED_FORMAT_SELECTION_REASON = (
    "NPY_PER_TENSOR_SUPPORTS_MMAP_DIRECT_LAYER_AND_ELECTRODE_SLICES_"
    "WITH_ATOMIC_PER_TENSOR_FAILURE_RECOVERY"
)
EXPECTED_FORMAT_SELECTION_CRITERIA = [
    "MMAP_AND_DIRECT_SLICE_SUPPORT",
    "NO_OBJECT_ARRAYS",
    "STRICT_DTYPE_AND_SHAPE",
    "ATOMIC_PER_TENSOR_FAILURE_RECOVERY",
]
EXPECTED_TENSOR_SPECS: dict[str, tuple[str, str, list[int]]] = {
    "frame_times_seconds": (
        "npy_per_tensor/frame_times_seconds.npy",
        "float64",
        [EXPECTED_ALIGNED_FRAME_COUNT],
    ),
    "common_valid_mask": (
        "npy_per_tensor/common_valid_mask.npy",
        "bool",
        [EXPECTED_ALIGNED_FRAME_COUNT],
    ),
    "amplitude_envelope_aligned": (
        "npy_per_tensor/amplitude_envelope_aligned.npy",
        "float32",
        [EXPECTED_ALIGNED_FRAME_COUNT, 1],
    ),
    "log_mel_aligned": (
        "npy_per_tensor/log_mel_aligned.npy",
        "float32",
        [EXPECTED_ALIGNED_FRAME_COUNT, EXPECTED_MEL_BINS],
    ),
    "wav2vec2_aligned": (
        "npy_per_tensor/wav2vec2_aligned.npy",
        "float32",
        [13, EXPECTED_ALIGNED_FRAME_COUNT, EXPECTED_HIDDEN_SIZE],
    ),
    "neural_subband_power_aligned": (
        "npy_per_tensor/neural_subband_power_aligned.npy",
        "float32",
        [EXPECTED_ALIGNED_FRAME_COUNT, EXPECTED_CHANNEL_COUNT, EXPECTED_SUBBAND_COUNT],
    ),
    "neural_pretransform_log_power_aligned": (
        "npy_per_tensor/neural_pretransform_log_power_aligned.npy",
        "float32",
        [EXPECTED_ALIGNED_FRAME_COUNT, EXPECTED_CHANNEL_COUNT, EXPECTED_SUBBAND_COUNT],
    ),
}


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_strict_json_object(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonstandard_json_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _same_number(value: Any, expected: float, tolerance: float = 1e-12) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=tolerance
    )


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _aware_iso_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _nonnegative_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _unique_nonempty_strings(values: Any, expected_count: int) -> bool:
    return (
        isinstance(values, list)
        and len(values) == expected_count
        and all(isinstance(value, str) and bool(value) for value in values)
        and len(set(values)) == expected_count
    )


def _remote_output_root_is_dedicated(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    candidate = PurePosixPath(value)
    if not candidate.is_absolute() or candidate == EXPECTED_REMOTE_OUTPUTS_ROOT:
        return False
    try:
        candidate.relative_to(EXPECTED_REMOTE_OUTPUTS_ROOT)
    except ValueError:
        return False
    return ".." not in candidate.parts


def _inventory_is_exact(tensors: Any) -> bool:
    if not isinstance(tensors, list) or len(tensors) != len(EXPECTED_TENSOR_SPECS):
        return False
    if [item.get("name") for item in tensors if isinstance(item, dict)] != list(
        EXPECTED_TENSOR_SPECS
    ):
        return False
    required_fields = {
        "name",
        "relative_path",
        "bytes",
        "modified_at_utc",
        "dtype",
        "shape",
        "object_dtype",
        "remote_only",
    }
    for item in tensors:
        if not isinstance(item, dict) or set(item) != required_fields:
            return False
        name = item.get("name")
        if not isinstance(name, str) or name not in EXPECTED_TENSOR_SPECS:
            return False
        relative_path, dtype, shape = EXPECTED_TENSOR_SPECS[name]
        if (
            item.get("relative_path") != relative_path
            or item.get("dtype") != dtype
            or item.get("shape") != shape
            or not _positive_int(item.get("bytes"))
            or item.get("object_dtype") is not False
            or item.get("remote_only") is not True
            or not _aware_iso_timestamp(item.get("modified_at_utc"))
        ):
            return False
    return True


def _inventory_byte_total(tensors: Any) -> int:
    if not _inventory_is_exact(tensors):
        return -1
    assert isinstance(tensors, list)
    total = 0
    for item in tensors:
        assert isinstance(item, dict)
        value = item.get("bytes")
        assert isinstance(value, int) and not isinstance(value, bool)
        total += value
    return total


def _readback_is_exact(
    readback: Any, evidence_output_root: Any, tensors: Any
) -> bool:
    if not isinstance(readback, dict) or not _remote_output_root_is_dedicated(
        evidence_output_root
    ) or not _inventory_is_exact(tensors):
        return False
    assert isinstance(tensors, list)
    inventory_by_name = {
        item["name"]: item for item in tensors if isinstance(item, dict)
    }
    expected_fields = {
        "output_root",
        "dedicated_outputs_root",
        "output_root_within_dedicated_outputs",
        "active_partial_count",
        "npy_file_count",
        "unexpected_npy_files",
        "all_files_present",
        "all_headers_match_inventory",
        "all_arrays_finite",
        "allow_pickle",
        "common_valid_true_count",
        "frame_times_strictly_increasing",
        "first_frame_seconds",
        "last_frame_seconds",
        "tensor_checks",
    }
    if set(readback) != expected_fields:
        return False
    if (
        readback.get("output_root") != evidence_output_root
        or readback.get("dedicated_outputs_root") != str(EXPECTED_REMOTE_OUTPUTS_ROOT)
        or readback.get("output_root_within_dedicated_outputs") is not True
        or readback.get("active_partial_count") != 0
        or readback.get("npy_file_count") != len(EXPECTED_TENSOR_SPECS)
        or readback.get("unexpected_npy_files") != []
        or readback.get("all_files_present") is not True
        or readback.get("all_headers_match_inventory") is not True
        or readback.get("all_arrays_finite") is not True
        or readback.get("allow_pickle") is not False
        or readback.get("common_valid_true_count") != EXPECTED_COMMON_VALID_FRAME_COUNT
        or readback.get("frame_times_strictly_increasing") is not True
        or not _same_number(
            readback.get("first_frame_seconds"), EXPECTED_GRID_FIRST_SECONDS
        )
        or not _same_number(
            readback.get("last_frame_seconds"), EXPECTED_GRID_LAST_SECONDS
        )
    ):
        return False
    checks = readback.get("tensor_checks")
    if not isinstance(checks, list) or len(checks) != len(EXPECTED_TENSOR_SPECS):
        return False
    expected_check_fields = {
        "name",
        "relative_path",
        "bytes",
        "dtype",
        "shape",
        "object_dtype",
        "all_finite",
    }
    for check, (name, spec) in zip(checks, EXPECTED_TENSOR_SPECS.items(), strict=True):
        if not isinstance(check, dict) or set(check) != expected_check_fields:
            return False
        relative_path, dtype, shape = spec
        if (
            check.get("name") != name
            or check.get("relative_path") != relative_path
            or check.get("dtype") != dtype
            or check.get("shape") != shape
            or not _positive_int(check.get("bytes"))
            or check.get("bytes") != inventory_by_name[name].get("bytes")
            or check.get("object_dtype") is not False
            or check.get("all_finite") is not True
        ):
            return False
    return True


def audit_remote_tensor_outputs(
    output_root: str | Path,
    dedicated_outputs_root: str | Path,
    inventory: Any,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    dedicated = Path(dedicated_outputs_root).resolve()
    if root == dedicated:
        raise ValueError("G3 tensor output root cannot be the dedicated outputs root")
    try:
        root.relative_to(dedicated)
    except ValueError as error:
        raise ValueError("G3 tensor output root is outside the dedicated outputs root") from error
    if not _inventory_is_exact(inventory):
        raise ValueError("G3 tensor inventory is not exact before remote readback")

    active_partials = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".partial" in path.name
    )
    npy_root = root / "npy_per_tensor"
    expected_paths = {
        (root / spec[0]).resolve() for spec in EXPECTED_TENSOR_SPECS.values()
    }
    actual_paths = {path.resolve() for path in npy_root.glob("*.npy") if path.is_file()}
    unexpected = sorted(path.relative_to(root).as_posix() for path in actual_paths - expected_paths)
    missing = expected_paths - actual_paths
    tensor_checks: list[dict[str, Any]] = []
    all_headers_match = not missing and not unexpected
    all_finite = True
    common_valid_true_count = -1
    frame_times_strictly_increasing = False
    first_frame_seconds = math.nan
    last_frame_seconds = math.nan
    inventory_by_name = {
        item["name"]: item for item in inventory if isinstance(item, dict)
    }
    for name, (relative_path, dtype, shape) in EXPECTED_TENSOR_SPECS.items():
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("G3 tensor path escaped the output root") from error
        if not path.is_file():
            all_headers_match = False
            all_finite = False
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        item = inventory_by_name[name]
        item_finite = bool(np.all(np.isfinite(array)))
        item_match = (
            str(array.dtype) == dtype
            and list(array.shape) == shape
            and not array.dtype.hasobject
            and path.stat().st_size == item["bytes"]
            and item["relative_path"] == relative_path
            and item["dtype"] == dtype
            and item["shape"] == shape
        )
        all_headers_match = all_headers_match and item_match
        all_finite = all_finite and item_finite
        tensor_checks.append(
            {
                "name": name,
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "object_dtype": bool(array.dtype.hasobject),
                "all_finite": item_finite,
            }
        )
        if name == "common_valid_mask":
            common_valid_true_count = int(np.count_nonzero(array))
        elif name == "frame_times_seconds":
            values = np.asarray(array, dtype=np.float64)
            frame_times_strictly_increasing = bool(np.all(np.diff(values) > 0))
            first_frame_seconds = float(values[0])
            last_frame_seconds = float(values[-1])

    return {
        "output_root": root.as_posix(),
        "dedicated_outputs_root": dedicated.as_posix(),
        "output_root_within_dedicated_outputs": True,
        "active_partial_count": len(active_partials),
        "npy_file_count": len(actual_paths),
        "unexpected_npy_files": unexpected,
        "all_files_present": not missing,
        "all_headers_match_inventory": all_headers_match,
        "all_arrays_finite": all_finite,
        "allow_pickle": False,
        "common_valid_true_count": common_valid_true_count,
        "frame_times_strictly_increasing": frame_times_strictly_increasing,
        "first_frame_seconds": first_frame_seconds,
        "last_frame_seconds": last_frame_seconds,
        "tensor_checks": tensor_checks,
    }


def validate_g3_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(config) != {
        "schema_version",
        "task_id",
        "status",
        "selection",
        "read_scope",
        "audio_features",
        "neural_smoke",
        "alignment",
        "format_benchmark",
        "execution",
    }:
        errors.append("G3 config top-level fields are not exact")
    if config.get("schema_version") != "m6a-g3-single-recording-candidate-v1":
        errors.append("G3 config schema version is not frozen")
    if config.get("task_id") != "M6A-PUBLIC-001" or config.get("status") != G3_CONFIG_STATUS:
        errors.append("G3 config identity or scoped status is invalid")

    selection = config.get("selection", {})
    expected_selection = {
        "metadata_report": "reports/ds004703_neural_metadata_g2_candidate.json",
        "split_csv": "reports/ds004703_primary_split.csv",
        "recording_rule": (
            "G2_HEADER_READABLE_AND_HEADER_CHANNEL_COUNT_MATCHES_TSV_AND_ELIGIBLE_"
            "NAMES_ALL_IN_EDF_AND_REFERENCE_PASS_THEN_MIN_ELIGIBLE_CHANNEL_COUNT_"
            "THEN_RECORDING_ID_LEXICOGRAPHIC"
        ),
        "passage_rule": "EARLIEST_ELIGIBLE_TRAIN_PASSAGE_BY_START_SEC_THEN_SAMPLE_ID",
        "expected_recording_id": EXPECTED_RECORDING_ID,
        "expected_sample_id": EXPECTED_SAMPLE_ID,
        "expected_stimulus_id": EXPECTED_STIMULUS_ID,
        "expected_audio_file": EXPECTED_AUDIO_FILE,
        "expected_edf_file": EXPECTED_EDF_FILE,
        "expected_split": "train",
        "expected_start_seconds": EXPECTED_START_SECONDS,
        "expected_end_seconds": EXPECTED_END_SECONDS,
        "expected_sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
        "expected_eligible_channel_count": EXPECTED_CHANNEL_COUNT,
        "result_based_selection_allowed": False,
    }
    if selection != expected_selection:
        errors.append("G3 selection rule and expected identity must remain exact")

    read_scope = config.get("read_scope", {})
    expected_read_scope = {
        "dataset_root": "/home/fanyu/auditory_simulation_m6a/data/ds004703/v1.1.0",
        "allowed_recording_count": 1,
        "allowed_passage_count": 1,
        "allowed_channel_count": EXPECTED_CHANNEL_COUNT,
        "preload_entire_recording": False,
        "segment_read_only": True,
        "read_support_profile_hz": EXPECTED_SAMPLING_RATE_HZ,
        "read_support_edge_samples": 559,
        "read_support_edge_seconds": 1.091796875,
        "read_rounding_tolerance_samples": 1,
        "other_recordings_allowed": False,
        "other_segments_allowed": False,
        "stn_or_patient_data_allowed": False,
    }
    if read_scope != expected_read_scope:
        errors.append("G3 real-waveform read scope must remain exact and bounded")

    audio = _mapping(config.get("audio_features"))
    envelope = _mapping(audio.get("amplitude_envelope"))
    log_mel = _mapping(audio.get("log_mel"))
    wav2vec = _mapping(audio.get("wav2vec2"))
    if audio.get("source_sample_rate_hz") != 44100 or audio.get("model_sample_rate_hz") != 16000:
        errors.append("G3 audio sample rates are not frozen")
    if audio.get("channels") != 1 or audio.get("independent_passage_inference") is not True:
        errors.append("G3 audio must remain mono and passage-isolated")
    if audio.get("neighbor_passage_read_allowed") is not False:
        errors.append("G3 cannot read neighboring passages")
    if envelope != {
        "method": "HANN_WEIGHTED_ROOT_MEAN_SQUARE",
        "window_samples": 400,
        "hop_samples": 320,
        "window": "PERIODIC_HANN",
        "center": False,
        "padding": "NONE",
        "frame_timestamp": "PASSAGE_START_PLUS_WINDOW_SAMPLE_CENTER",
    }:
        errors.append("amplitude-envelope parameters are not frozen")
    if log_mel != {
        "method": "HTK_TRIANGULAR_MEL_POWER",
        "n_fft": 512,
        "window_samples": 400,
        "hop_samples": 320,
        "mel_bins": EXPECTED_MEL_BINS,
        "fmin_hz": 50,
        "fmax_hz": 7600,
        "window": "PERIODIC_HANN",
        "center": False,
        "padding": "NONE",
        "power": 2,
        "log_formula": "natural_log(mel_power + 1e-10)",
        "frame_timestamp": "PASSAGE_START_PLUS_WINDOW_SAMPLE_CENTER",
    }:
        errors.append("raw log-mel parameters are not frozen")
    if wav2vec != {
        "model_id": "facebook/wav2vec2-base",
        "cache_state": "SEMANTICALLY_VALIDATED_REMOTE_ONLY",
        "download_allowed": False,
        "local_files_only": True,
        "trust_remote_code": False,
        "weights_only_required": True,
        "layer_inventory": "PROJECTED_PLUS_12_TRANSFORMER_LAYERS",
        "expected_layer_count": 13,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "transformer_attention_scope": "GLOBAL_WITHIN_SINGLE_PASSAGE",
    }:
        errors.append("wav2vec2 G3 runtime gate is not frozen")

    neural = config.get("neural_smoke", {})
    expected_neural = {
        "method_status": "METHOD_FROZEN_COORDINATOR_ACCEPTED",
        "reference": "AS_RECORDED_SCALP_REFERENCE",
        "subband_count": EXPECTED_SUBBAND_COUNT,
        "output": "RAW_FINITE_FIR_POWER_AND_PRETRANSFORM_LOG_POWER_BY_CHANNEL_AND_SUBBAND",
        "smoke_log_formula": "natural_log(max(raw_power, 0) + 1e-30)",
        "negative_power_clip_absolute_tolerance": 1e-12,
        "smoke_log_limitation": (
            "ENGINEERING_FINITE_READABILITY_ONLY_NOT_THE_TRAIN_ONLY_TARGET_TRANSFORM"
        ),
        "formal_train_only_transform_fitted": False,
        "smoke_statistics_reusable_for_baseline": False,
        "target_frame_rate_hz": EXPECTED_GRID_RATE_HZ,
        "grid_origin_seconds": 0,
        "insufficient_support_action": "MASK",
        "real_waveform_read_scope": (
            "ONE_SELECTED_RECORDING_ONE_PASSAGE_36_ELIGIBLE_CHANNELS_"
            "PLUS_FROZEN_FINITE_SUPPORT_ONLY"
        ),
    }
    if neural != expected_neural:
        errors.append("neural smoke is not exact or attempts a formal train-only transform")

    alignment = config.get("alignment", {})
    if alignment != {
        "grid": "RECORDING_ORIGIN_K_OVER_50_SECONDS",
        "interpolation": "LINEAR_TWO_NEAREST_NATIVE_FRAMES_NO_EXTRAPOLATION",
        "native_and_aligned_grids_reported_separately": True,
        "common_valid_mask": (
            "INTERSECTION_OF_ENVELOPE_LOG_MEL_WAV2VEC2_AND_NEURAL_COMPLETE_SUPPORT"
        ),
        "outside_valid_mask_fill": 0,
        "timestamps_strictly_identical": True,
    }:
        errors.append("G3 aligned-grid semantics are not frozen")

    benchmark = config.get("format_benchmark", {})
    if benchmark != {
        "formats": ["NPY_PER_TENSOR", "NPZ_COMPRESSED"],
        "new_dependency_allowed": False,
        "selection_criteria": [
            "MMAP_AND_DIRECT_SLICE_SUPPORT",
            "NO_OBJECT_ARRAYS",
            "STRICT_DTYPE_AND_SHAPE",
            "ATOMIC_PER_TENSOR_FAILURE_RECOVERY",
        ],
        "provisional_preference": "NPY_PER_TENSOR",
        "required_access_benchmarks": [
            "FULL_READ",
            "SINGLE_WAV2VEC2_LAYER",
            "SINGLE_NEURAL_ELECTRODE",
        ],
    }:
        errors.append("G3 format benchmark criteria are not predeclared")

    execution = config.get("execution", {})
    if execution != {
        "remote_only_arrays": True,
        "raw_waveform_saved": False,
        "formal_baseline_run": False,
        "scientific_result_claimed": False,
        "exchange_candidate_created": False,
        "expand_other_recordings_or_segments_before_review": False,
        "allowed_output_status": G3_STATUS,
    }:
        errors.append("G3 execution boundary is not fail-closed")
    if nonfinite_numeric_paths(config):
        errors.append("G3 config contains non-finite numeric values")
    if find_forbidden_fields(config):
        errors.append("G3 config contains forbidden integrity fields")
    return errors


def select_g3_scope(
    neural_report: dict[str, Any], split_csv: str | Path
) -> dict[str, Any]:
    recordings = neural_report.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("neural metadata report recordings must be a list")
    eligible: list[dict[str, Any]] = []
    for recording in recordings:
        if not isinstance(recording, dict):
            continue
        header = recording.get("edf_header")
        channels = recording.get("channels")
        names = channels.get("analysis_eligible_neural_names") if isinstance(channels, dict) else None
        count = (
            channels.get("analysis_eligible_neural_channel_count")
            if isinstance(channels, dict)
            else None
        )
        header_names = header.get("channel_names") if isinstance(header, dict) else None
        valid = (
            isinstance(header, dict)
            and isinstance(header.get("path"), str)
            and bool(header.get("path"))
            and isinstance(header_names, list)
            and isinstance(names, list)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and len(names) == count
            and len(names) == len(set(names))
            and all(isinstance(name, str) and bool(name.strip()) for name in names)
            and set(names).issubset(set(header_names))
            and recording.get("edf_header_channel_count_matches_tsv") is True
            and recording.get("edf_header_sampling_rate_matches_sidecar") is True
            and recording.get("analysis_eligible_neural_channels_missing_from_edf") == []
            and recording.get("iEEGReference") == EXPECTED_REFERENCE
        )
        if valid:
            eligible.append(recording)
    if not eligible:
        raise ValueError("no recording satisfies the frozen G3 metadata selection gate")
    selected = min(
        eligible,
        key=lambda item: (
            int(item["channels"]["analysis_eligible_neural_channel_count"]),
            str(item.get("recording_id")),
        ),
    )

    passage_rows: list[dict[str, str]] = []
    with Path(split_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("recording_id") == selected.get("recording_id")
                and row.get("split") == "train"
                and row.get("analysis_eligible") == "True"
            ):
                passage_rows.append(row)
    if not passage_rows:
        raise ValueError("selected recording has no eligible train passage")
    passage = min(passage_rows, key=lambda row: (float(row["start_sec"]), row["sample_id"]))
    channel_names = list(selected["channels"]["analysis_eligible_neural_names"])
    result = {
        "recording_id": selected["recording_id"],
        "sample_id": passage["sample_id"],
        "stimulus_id": passage["stimulus_id"],
        "audio_file": normalize_relative_audio_path(passage["audio_file"]),
        "edf_file": normalize_relative_audio_path(selected["edf_file"]),
        "split": passage["split"],
        "start_seconds": float(passage["start_sec"]),
        "end_seconds": float(passage["end_sec"]),
        "sampling_rate_hz": int(float(selected["sampling_rate_hz"])),
        "eligible_channel_count": len(channel_names),
        "eligible_channel_names": channel_names,
        "reference": selected["iEEGReference"],
        "g2_declared_recording_duration_seconds": float(
            selected["recording_duration_seconds"]
        ),
        "g2_edf_header_last_sample_time_seconds": float(
            selected["edf_header"]["duration_seconds"]
        ),
        "selection_candidate_recording_count": len(eligible),
        "result_based_selection_used": False,
    }
    expected = {
        "recording_id": EXPECTED_RECORDING_ID,
        "sample_id": EXPECTED_SAMPLE_ID,
        "stimulus_id": EXPECTED_STIMULUS_ID,
        "audio_file": EXPECTED_AUDIO_FILE,
        "edf_file": EXPECTED_EDF_FILE,
        "split": "train",
        "start_seconds": EXPECTED_START_SECONDS,
        "end_seconds": EXPECTED_END_SECONDS,
        "sampling_rate_hz": EXPECTED_SAMPLING_RATE_HZ,
        "eligible_channel_count": EXPECTED_CHANNEL_COUNT,
        "reference": EXPECTED_REFERENCE,
        "result_based_selection_used": False,
    }
    for key, expected_value in expected.items():
        observed = result.get(key)
        if isinstance(expected_value, float):
            if not _same_number(observed, expected_value):
                raise ValueError(f"selected scope drifted at {key}")
        elif observed != expected_value:
            raise ValueError(f"selected scope drifted at {key}")
    return result


def passage_grid_seconds(start_seconds: float, end_seconds: float) -> np.ndarray:
    if not all(math.isfinite(value) for value in (start_seconds, end_seconds)):
        raise ValueError("passage bounds must be finite")
    if end_seconds <= start_seconds:
        raise ValueError("passage interval must have positive duration")
    first_k = math.ceil(start_seconds * EXPECTED_GRID_RATE_HZ)
    stop_k = math.ceil(end_seconds * EXPECTED_GRID_RATE_HZ)
    grid = np.arange(first_k, stop_k, dtype=np.float64) / EXPECTED_GRID_RATE_HZ
    if grid.size == 0 or np.any(np.diff(grid) <= 0) or grid[-1] >= end_seconds:
        raise ValueError("recording-origin 50 Hz grid is invalid")
    return grid


def frame_audio_16khz(values: np.ndarray, window_samples: int = 400, hop_samples: int = 320) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or not np.all(np.isfinite(signal)):
        raise ValueError("audio must be a finite mono vector")
    if signal.size < window_samples or window_samples <= 0 or hop_samples <= 0:
        raise ValueError("audio is too short for the frozen frame definition")
    frames = np.lib.stride_tricks.sliding_window_view(signal, window_samples)[::hop_samples]
    expected = 1 + (signal.size - window_samples) // hop_samples
    if frames.shape != (expected, window_samples):
        raise RuntimeError("audio framing shape disagrees with the frozen formula")
    return np.asarray(frames, dtype=np.float64)


def native_audio_frame_centers(
    frame_count: int, passage_start_seconds: float, window_samples: int = 400, hop_samples: int = 320
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame count must be positive")
    return passage_start_seconds + (
        (window_samples - 1) / 2.0 + np.arange(frame_count, dtype=np.float64) * hop_samples
    ) / 16000.0


def amplitude_envelope_native(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = frame_audio_16khz(values)
    window = np.hanning(401)[:-1]
    denominator = float(np.sum(window))
    envelope = np.sqrt(np.sum((frames * frames) * window[None, :], axis=1) / denominator)
    if not np.all(np.isfinite(envelope)):
        raise ValueError("non-finite amplitude envelope")
    return envelope[:, None].astype(np.float32), window


def _hz_to_htk_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz, dtype=np.float64) / 700.0)


def _htk_mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def frozen_mel_filterbank() -> np.ndarray:
    n_fft = 512
    sample_rate_hz = 16000
    mel_points = np.linspace(
        float(_hz_to_htk_mel(50.0)), float(_hz_to_htk_mel(7600.0)), EXPECTED_MEL_BINS + 2
    )
    hz_points = _htk_mel_to_hz(mel_points)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
    bank = np.zeros((EXPECTED_MEL_BINS, frequencies.size), dtype=np.float64)
    for index in range(EXPECTED_MEL_BINS):
        left, center, right = hz_points[index : index + 3]
        bank[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center),
            ),
        )
    if not np.all(np.isfinite(bank)) or np.any(np.sum(bank, axis=1) <= 0):
        raise ValueError("invalid frozen HTK mel filterbank")
    return bank


def log_mel_native(values: np.ndarray) -> np.ndarray:
    frames = frame_audio_16khz(values)
    window = np.hanning(401)[:-1]
    spectrum = np.fft.rfft(frames * window[None, :], n=512, axis=1)
    power = np.abs(spectrum) ** 2
    mel_power = power @ frozen_mel_filterbank().T
    output = np.log(mel_power + 1e-10)
    if output.shape[1] != EXPECTED_MEL_BINS or not np.all(np.isfinite(output)):
        raise ValueError("raw log-mel output is invalid")
    return output.astype(np.float32)


def linear_align_no_extrapolation(
    native_times: np.ndarray, values: np.ndarray, aligned_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(native_times, dtype=np.float64)
    data = np.asarray(values)
    target = np.asarray(aligned_times, dtype=np.float64)
    if (
        times.ndim != 1
        or target.ndim != 1
        or data.ndim < 1
        or data.shape[0] != times.size
        or times.size < 2
        or not np.all(np.isfinite(times))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(data))
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("native values and timestamps are invalid")
    valid = (target >= times[0]) & (target <= times[-1])
    output = np.zeros((target.size,) + data.shape[1:], dtype=np.float32)
    target_valid = target[valid]
    right = np.searchsorted(times, target_valid, side="left")
    right = np.clip(right, 1, times.size - 1)
    left = right - 1
    width = times[right] - times[left]
    alpha = (target_valid - times[left]) / width
    reshape = (alpha.size,) + (1,) * (data.ndim - 1)
    interpolated = data[left] * (1.0 - alpha.reshape(reshape)) + data[right] * alpha.reshape(reshape)
    output[valid] = interpolated.astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("linear alignment produced non-finite values")
    return output, valid


def validate_g3_evidence(
    evidence: dict[str, Any], task_config: dict[str, Any], g3_config: dict[str, Any]
) -> dict[str, bool]:
    payload = _mapping(evidence)
    task = _mapping(task_config)
    selection = _mapping(payload.get("selection"))
    read = _mapping(payload.get("real_neural_read"))
    audio_read = _mapping(payload.get("audio_read"))
    model_runtime = _mapping(payload.get("model_runtime"))
    loading_info = _mapping(model_runtime.get("loading_info"))
    native = _mapping(payload.get("native_grids"))
    envelope = _mapping(native.get("amplitude_envelope"))
    log_mel = _mapping(native.get("log_mel"))
    wav2vec2 = _mapping(native.get("wav2vec2"))
    neural = _mapping(native.get("neural"))
    aligned = _mapping(payload.get("aligned_grid"))
    tensors = payload.get("tensor_inventory")
    benchmark = _mapping(payload.get("format_benchmark"))
    npy_benchmark = _mapping(benchmark.get("npy_per_tensor"))
    npz_benchmark = _mapping(benchmark.get("npz_compressed"))
    channel_names = selection.get("eligible_channel_names")
    try:
        scoped_config_valid = validate_g3_config(g3_config) == []
    except (AttributeError, TypeError, ValueError):
        scoped_config_valid = False
    try:
        forbidden_absent = find_forbidden_fields(payload) == []
        nonfinite_absent = nonfinite_numeric_paths(payload) == []
    except (AttributeError, TypeError, ValueError):
        forbidden_absent = False
        nonfinite_absent = False
    audio_frame_formula = 1 + (
        EXPECTED_AUDIO_RESAMPLED_FRAMES - 400
    ) // 320
    source_frames = audio_read.get("source_frames")
    resampled_frame_formula = (
        math.ceil(source_frames * 160 / 441)
        if isinstance(source_frames, int)
        and not isinstance(source_frames, bool)
        and source_frames > 0
        else -1
    )
    passage_start_floor = math.floor(EXPECTED_START_SECONDS * EXPECTED_SAMPLING_RATE_HZ)
    passage_end_ceil = math.ceil(EXPECTED_END_SECONDS * EXPECTED_SAMPLING_RATE_HZ)
    read_start = max(0, passage_start_floor - EXPECTED_SUPPORT_EDGE_SAMPLES)
    read_end = min(
        EXPECTED_RECORDING_TOTAL_SAMPLES,
        passage_end_ceil + EXPECTED_SUPPORT_EDGE_SAMPLES,
    )
    benchmark_timings = [
        npy_benchmark.get("write_seconds"),
        npy_benchmark.get("full_read_seconds"),
        npy_benchmark.get("single_wav2vec2_layer_seconds"),
        npy_benchmark.get("single_neural_electrode_seconds"),
        npz_benchmark.get("write_seconds"),
        npz_benchmark.get("full_read_seconds"),
        npz_benchmark.get("single_wav2vec2_layer_seconds"),
        npz_benchmark.get("single_neural_electrode_seconds"),
    ]
    inventory_byte_total = _inventory_byte_total(tensors)
    negative_power_count = neural.get(
        "negative_raw_power_value_count_before_smoke_clip"
    )
    negative_power_abs_max = neural.get(
        "negative_raw_power_abs_max_before_smoke_clip"
    )
    negative_power_count_valid = (
        isinstance(negative_power_count, int)
        and not isinstance(negative_power_count, bool)
        and negative_power_count >= 0
    )
    negative_power_abs_max_valid = (
        not isinstance(negative_power_abs_max, bool)
        and isinstance(negative_power_abs_max, (int, float))
        and math.isfinite(float(negative_power_abs_max))
        and float(negative_power_abs_max) <= 1e-12
    )
    checks = {
        "report_identity": (
            payload.get("report_schema_version") == REPORT_SCHEMA_VERSION
            and payload.get("task_id") == "M6A-PUBLIC-001"
            and payload.get("status") in {None, G3_STATUS, G3_ACCEPTED_STATUS}
        ),
        "non_hash_and_finite_evidence": (
            payload.get("integrity_policy") == "NON_HASH_AUDIT"
            and payload.get("cryptographic_integrity_claim") is False
            and forbidden_absent
            and nonfinite_absent
        ),
        "scoped_config_valid": scoped_config_valid,
        "accepted_split_not_whole_m6a": (
            _mapping(task.get("split")).get("final_embargo_seconds") == 2.0
            and _mapping(task.get("split")).get("final_embargo_status")
            == "FINAL_EMBARGO_COORDINATOR_ACCEPTED"
            and _mapping(task.get("split")).get("split_status")
            == "BASELINE_FINAL_COORDINATOR_ACCEPTED"
            and _mapping(task.get("split")).get("baseline_final") is True
            and _mapping(task.get("g2")).get("whole_m6a_pass_claimed") is False
        ),
        "model_runtime_and_cache_frozen": (
            _mapping(task.get("model")).get("cache_state")
            == "SEMANTICALLY_VALIDATED_REMOTE_ONLY"
            and _mapping(task.get("model")).get("download_allowed") is False
            and model_runtime.get("local_files_only") is True
            and model_runtime.get("trust_remote_code") is False
            and model_runtime.get("repository_custom_code_executed") is False
            and model_runtime.get("weights_only") is True
            and model_runtime.get("tensor_only") is True
            and model_runtime.get("download_attempted") is False
            and model_runtime.get("model_eval") is True
            and model_runtime.get("parameter_requires_grad_count") == 0
            and loading_info
            == {
                "missing_keys": [],
                "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
                "mismatched_keys": [],
                "error_msgs": [],
            }
        ),
        "selection_rule_reexecuted": (
            selection.get("recording_id") == EXPECTED_RECORDING_ID
            and selection.get("sample_id") == EXPECTED_SAMPLE_ID
            and selection.get("stimulus_id") == EXPECTED_STIMULUS_ID
            and selection.get("audio_file") == EXPECTED_AUDIO_FILE
            and selection.get("edf_file") == EXPECTED_EDF_FILE
            and selection.get("split") == "train"
            and _same_number(selection.get("start_seconds"), EXPECTED_START_SECONDS)
            and _same_number(selection.get("end_seconds"), EXPECTED_END_SECONDS)
            and selection.get("sampling_rate_hz") == EXPECTED_SAMPLING_RATE_HZ
            and selection.get("eligible_channel_count") == EXPECTED_CHANNEL_COUNT
            and _unique_nonempty_strings(channel_names, EXPECTED_CHANNEL_COUNT)
            and selection.get("reference") == EXPECTED_REFERENCE
            and selection.get("result_based_selection_used") is False
        ),
        "audio_read_exact_and_isolated": (
            audio_read.get("audio_file") == EXPECTED_AUDIO_FILE
            and audio_read.get("requested_audio_file_count") == 1
            and audio_read.get("other_audio_read_count") == 0
            and audio_read.get("sample_rate_hz") == 44_100
            and audio_read.get("channels") == 1
            and audio_read.get("source_frames") == EXPECTED_AUDIO_SOURCE_FRAMES
            and audio_read.get("resampled_frames") == EXPECTED_AUDIO_RESAMPLED_FRAMES
            and audio_read.get("resampled_frames") == resampled_frame_formula
            and _same_number(
                audio_read.get("source_duration_seconds"),
                EXPECTED_AUDIO_SOURCE_FRAMES / 44_100,
            )
            and _same_number(
                audio_read.get("resampled_duration_seconds"),
                EXPECTED_AUDIO_RESAMPLED_FRAMES / 16_000,
            )
            and audio_read.get("neighbor_passage_read_allowed") is False
        ),
        "bounded_segment_read": (
            read.get("edf_file") == EXPECTED_EDF_FILE
            and read.get("sampling_rate_hz") == EXPECTED_SAMPLING_RATE_HZ
            and read.get("mne_preload") is False
            and read.get("preload_entire_recording") is False
            and read.get("segment_read_only") is True
            and read.get("requested_recording_count") == 1
            and read.get("requested_channel_count") == EXPECTED_CHANNEL_COUNT
            and read.get("returned_channel_count") == EXPECTED_CHANNEL_COUNT
            and read.get("returned_channel_names") == channel_names
            and read.get("other_recording_or_segment_read_count") == 0
            and read.get("recording_total_samples") == EXPECTED_RECORDING_TOTAL_SAMPLES
            and passage_start_floor == EXPECTED_PASSAGE_START_SAMPLE_FLOOR
            and passage_end_ceil == EXPECTED_PASSAGE_END_SAMPLE_CEIL
            and read_start == EXPECTED_READ_START_SAMPLE
            and read_end == EXPECTED_READ_END_SAMPLE_EXCLUSIVE
            and read.get("passage_start_sample_floor") == passage_start_floor
            and read.get("passage_end_sample_ceil") == passage_end_ceil
            and read.get("support_edge_samples") == EXPECTED_SUPPORT_EDGE_SAMPLES
            and read.get("read_start_sample") == read_start
            and read.get("read_end_sample_exclusive") == read_end
            and read.get("read_sample_count") == EXPECTED_READ_SAMPLE_COUNT
            and read.get("read_sample_count") == read_end - read_start
            and _same_number(
                read.get("read_start_seconds"), read_start / EXPECTED_SAMPLING_RATE_HZ
            )
            and _same_number(
                read.get("read_end_seconds_exclusive"),
                read_end / EXPECTED_SAMPLING_RATE_HZ,
            )
            and _same_number(
                selection.get("g2_declared_recording_duration_seconds"), 3552.75
            )
            and _same_number(
                selection.get("g2_edf_header_last_sample_time_seconds"),
                2599.748046875,
            )
            and _same_number(
                read.get("recording_sample_span_seconds"),
                EXPECTED_RECORDING_TOTAL_SAMPLES / EXPECTED_SAMPLING_RATE_HZ,
            )
            and read.get("raw_waveform_saved") is False
        ),
        "native_feature_shapes_times_and_finiteness": (
            audio_frame_formula == EXPECTED_AUDIO_NATIVE_FRAME_COUNT
            and envelope.get("frame_count") == EXPECTED_AUDIO_NATIVE_FRAME_COUNT
            and envelope.get("feature_dim") == 1
            and envelope.get("shape") == [EXPECTED_AUDIO_NATIVE_FRAME_COUNT, 1]
            and _same_number(
                envelope.get("first_frame_center_seconds"),
                EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS,
            )
            and _same_number(
                envelope.get("last_frame_center_seconds"),
                EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS,
            )
            and _same_number(
                envelope.get("frame_step_seconds"), EXPECTED_AUDIO_NATIVE_STEP_SECONDS
            )
            and envelope.get("all_finite") is True
            and log_mel.get("frame_count") == EXPECTED_AUDIO_NATIVE_FRAME_COUNT
            and log_mel.get("feature_dim") == EXPECTED_MEL_BINS
            and log_mel.get("shape")
            == [EXPECTED_AUDIO_NATIVE_FRAME_COUNT, EXPECTED_MEL_BINS]
            and _same_number(
                log_mel.get("first_frame_center_seconds"),
                EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS,
            )
            and _same_number(
                log_mel.get("last_frame_center_seconds"),
                EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS,
            )
            and _same_number(
                log_mel.get("frame_step_seconds"), EXPECTED_AUDIO_NATIVE_STEP_SECONDS
            )
            and log_mel.get("all_finite") is True
            and wav2vec2.get("layer_keys") == list(EXPECTED_LAYER_KEYS)
            and wav2vec2.get("layer_count") == 13
            and wav2vec2.get("hidden_size") == EXPECTED_HIDDEN_SIZE
            and wav2vec2.get("native_shape_frames_layers_hidden")
            == [EXPECTED_AUDIO_NATIVE_FRAME_COUNT, 13, EXPECTED_HIDDEN_SIZE]
            and wav2vec2.get("native_frame_count_formula")
            == EXPECTED_AUDIO_NATIVE_FRAME_COUNT
            and _same_number(
                wav2vec2.get("native_first_frame_center_seconds"),
                EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS,
            )
            and _same_number(
                wav2vec2.get("native_last_frame_center_seconds"),
                EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS,
            )
            and _same_number(
                wav2vec2.get("native_frame_step_seconds"),
                EXPECTED_AUDIO_NATIVE_STEP_SECONDS,
            )
            and wav2vec2.get("aligned_shape_layers_frames_hidden")
            == [13, EXPECTED_ALIGNED_FRAME_COUNT, EXPECTED_HIDDEN_SIZE]
            and wav2vec2.get("valid_frame_count") == 1_730
            and wav2vec2.get("all_finite") is True
            and neural.get("native_sample_count") == EXPECTED_READ_SAMPLE_COUNT
            and _same_number(
                neural.get("native_first_sample_time_seconds"),
                EXPECTED_NEURAL_NATIVE_FIRST_SECONDS,
            )
            and _same_number(
                neural.get("native_last_sample_time_seconds"),
                EXPECTED_NEURAL_NATIVE_LAST_SECONDS,
            )
            and _same_number(
                neural.get("native_sample_step_seconds"),
                EXPECTED_NEURAL_NATIVE_STEP_SECONDS,
            )
            and neural.get("channel_count") == EXPECTED_CHANNEL_COUNT
            and neural.get("subband_count") == EXPECTED_SUBBAND_COUNT
            and neural.get("subbands_hz")
            == [
                [70.0, 80.0],
                [80.0, 90.0],
                [90.0, 100.0],
                [100.0, 110.0],
                [130.0, 140.0],
                [140.0, 150.0],
            ]
            and neural.get("raw_power_shape")
            == [EXPECTED_ALIGNED_FRAME_COUNT, EXPECTED_CHANNEL_COUNT, 6]
            and neural.get("pretransform_log_power_shape")
            == [EXPECTED_ALIGNED_FRAME_COUNT, EXPECTED_CHANNEL_COUNT, 6]
            and neural.get("valid_frame_count") == EXPECTED_COMMON_VALID_FRAME_COUNT
            and neural.get("support_edge_samples") == EXPECTED_SUPPORT_EDGE_SAMPLES
            and _same_number(neural.get("support_edge_seconds"), 1.091796875)
            and neural.get("formal_train_only_transform_fitted") is False
            and neural.get("smoke_statistics_reusable_for_baseline") is False
            and neural.get("smoke_log_formula")
            == "natural_log(max(raw_power, 0) + 1e-30)"
            and negative_power_count_valid
            and _same_number(
                neural.get("negative_power_clip_absolute_tolerance"), 1e-12
            )
            and negative_power_abs_max_valid
            and neural.get("all_finite") is True
        ),
        "aligned_grid_shapes_times_and_common_mask": (
            aligned.get("grid") == "RECORDING_ORIGIN_K_OVER_50_SECONDS"
            and aligned.get("frame_rate_hz") == EXPECTED_GRID_RATE_HZ
            and aligned.get("frame_count") == EXPECTED_ALIGNED_FRAME_COUNT
            and _same_number(
                aligned.get("first_frame_seconds"), EXPECTED_GRID_FIRST_SECONDS
            )
            and _same_number(
                aligned.get("last_frame_seconds"), EXPECTED_GRID_LAST_SECONDS
            )
            and aligned.get("common_valid_frame_count")
            == EXPECTED_COMMON_VALID_FRAME_COUNT
            and _same_number(
                aligned.get("first_common_valid_frame_seconds"),
                EXPECTED_COMMON_FIRST_SECONDS,
            )
            and _same_number(
                aligned.get("last_common_valid_frame_seconds"),
                EXPECTED_COMMON_LAST_SECONDS,
            )
            and aligned.get("timestamps_strictly_increasing") is True
            and aligned.get("all_tensor_timestamps_identical") is True
            and aligned.get("interpolation")
            == "LINEAR_TWO_NEAREST_NATIVE_FRAMES_NO_EXTRAPOLATION"
            and aligned.get("common_mask_is_intersection") is True
            and aligned.get("individual_valid_frame_counts")
            == {
                "amplitude_envelope": 1_730,
                "log_mel": 1_730,
                "wav2vec2": 1_730,
                "neural": EXPECTED_COMMON_VALID_FRAME_COUNT,
            }
            and aligned.get("all_tensors_finite") is True
        ),
        "tensor_inventory_and_remote_readback_exact": (
            _inventory_is_exact(tensors)
            and _readback_is_exact(
                payload.get("remote_tensor_readback"),
                payload.get("remote_output_root"),
                tensors,
            )
            and payload.get("high_dimensional_arrays_in_git") is False
        ),
        "format_benchmark_and_selection": (
            benchmark.get("formats_tested") == ["NPY_PER_TENSOR", "NPZ_COMPRESSED"]
            and benchmark.get("selected_format") == "NPY_PER_TENSOR"
            and benchmark.get("selection_status") == "PROVISIONAL_INTERNAL_FORMAT_SELECTION"
            and benchmark.get("selection_reason") == EXPECTED_FORMAT_SELECTION_REASON
            and benchmark.get("selection_criteria") == EXPECTED_FORMAT_SELECTION_CRITERIA
            and benchmark.get("new_dependency_used") is False
            and npy_benchmark.get("bytes") == inventory_byte_total
            and _positive_int(npy_benchmark.get("bytes"))
            and all(_nonnegative_finite_number(value) for value in benchmark_timings)
            and npy_benchmark.get("mmap_supported") is True
            and npy_benchmark.get("direct_slice_supported") is True
            and npy_benchmark.get("atomic_per_tensor") is True
            and npy_benchmark.get("allow_pickle") is False
            and _positive_int(npz_benchmark.get("bytes"))
            and npz_benchmark.get("mmap_supported") is False
            and npz_benchmark.get("direct_slice_supported") is False
            and npz_benchmark.get("atomic_archive") is True
            and npz_benchmark.get("allow_pickle") is False
            and npz_benchmark.get("relative_path")
            == "npz_compressed/aligned_tensors.npz"
        ),
        "engineering_only_claim_boundary": (
            payload.get("real_neural_waveform_read_scope")
            == "ONE_SELECTED_RECORDING_ONE_PASSAGE_36_ELIGIBLE_CHANNELS_PLUS_FROZEN_FINITE_SUPPORT_ONLY"
            and payload.get("formal_baseline_run") is False
            and payload.get("scientific_result_claimed") is False
            and payload.get("exchange_candidate_created") is False
            and payload.get("other_recordings_or_segments_processed") is False
            and payload.get("formal_train_only_transform_fitted") is False
        ),
    }
    return checks


def finalize_g3_report(
    evidence: dict[str, Any], task_config: dict[str, Any], g3_config: dict[str, Any]
) -> dict[str, Any]:
    checks = validate_g3_evidence(evidence, task_config, g3_config)
    failed = [name for name, passed in checks.items() if passed is not True]
    report = dict(evidence)
    report["required_checks"] = checks
    report["failed_checks"] = failed
    report["status"] = G3_STATUS if not failed else "FAIL"
    return report
