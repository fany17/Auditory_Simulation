from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, NoReturn, Sequence

import numpy as np
from scipy import signal

from m6a_public.config_gate import find_forbidden_fields
from m6a_public.split_guard import read_assignments, summarize_assignments, validate_assignments


MODEL_ID = "facebook/wav2vec2-base"
MODEL_REVISION_LABEL = "main"
SOURCE_SAMPLE_RATE_HZ = 44_100
MODEL_SAMPLE_RATE_HZ = 16_000
RESAMPLE_FILTER_TAPS = 8_821
RESAMPLE_KAISER_BETA = 5.0
RESAMPLE_INPUT_RADIUS_SAMPLES = 28
RESAMPLE_OUTPUT_RADIUS_SAMPLES = 10
RESAMPLE_EDGE_SECONDS = RESAMPLE_INPUT_RADIUS_SAMPLES / SOURCE_SAMPLE_RATE_HZ
EXPECTED_HIDDEN_SIZE = 768
EXPECTED_LAYER_KEYS = ("projected",) + tuple(f"transformer_{index:02d}" for index in range(1, 13))
EXPECTED_SPLIT_COUNTS = {"train": 223, "validation": 48, "test": 48}
EXPECTED_BLOCK_ASSIGNMENTS = {
    "block-01": "train",
    "block-02": "train",
    "block-03": "validation",
    "block-04": "test",
    "block-05": "train",
    "block-06": "train",
}
EXPECTED_UNIQUE_AUDIO_FILE_COUNT = 48
EXPECTED_AUDIO_SOURCE_STATUS = "BUNDLED_BLOCK_AUDIO"
REQUIRED_AUDIO_IDENTITY_COLUMNS = {
    "audio_file",
    "stimulus_id",
    "block_id",
    "split",
    "audio_sample_rate_hz",
    "audio_channels",
    "audio_source_status",
}
MODEL_FILES = ("README.md", "config.json", "preprocessor_config.json", "pytorch_model.bin")
CANDIDATE_STATUS = "FINAL_EMBARGO_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
REPORT_SCHEMA_VERSION = "m6a-audio-context-final-embargo-candidate-v1"
EXPECTED_CONFIG_PROFILE = {
    "model_type": "wav2vec2",
    "architectures": ["Wav2Vec2ForPreTraining"],
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "conv_dim": [512, 512, 512, 512, 512, 512, 512],
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
}
EXPECTED_WEIGHT_SUFFIX_SHAPES = {
    "feature_extractor.conv_layers.0.conv.weight": [512, 1, 10],
    "feature_projection.projection.weight": [768, 512],
    "encoder.layers.0.attention.q_proj.weight": [768, 768],
    "encoder.layers.11.attention.q_proj.weight": [768, 768],
}
EXPECTED_PRETRAINING_HEAD_KEYS = (
    "project_hid.bias",
    "project_hid.weight",
    "project_q.bias",
    "project_q.weight",
    "quantizer.codevectors",
    "quantizer.weight_proj.bias",
    "quantizer.weight_proj.weight",
)


@dataclass(frozen=True)
class ResamplingSpec:
    source_sample_rate_hz: int
    target_sample_rate_hz: int
    up: int
    down: int
    filter_design: str
    filter_taps: int
    kaiser_beta: float
    padtype: str
    input_support_radius_samples: int
    output_support_radius_samples: int
    edge_seconds: float


@dataclass(frozen=True)
class ConvolutionTiming:
    kernels: tuple[int, ...]
    strides: tuple[int, ...]
    cumulative_stride_samples: int
    receptive_field_samples: int
    first_frame_center_samples: float


AudioReader = Callable[[str], tuple[np.ndarray, int]]


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_strict_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonstandard_json_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_model_config(path: str | Path) -> dict[str, Any]:
    payload = load_strict_json_object(path)
    observed = {key: payload.get(key) for key in EXPECTED_CONFIG_PROFILE}
    errors = [
        f"config {key} does not match the frozen profile"
        for key, expected in EXPECTED_CONFIG_PROFILE.items()
        if observed[key] != expected
    ]
    return {
        "status": "PASS" if not errors else "FAIL",
        "observed": observed,
        "errors": errors,
    }


def audit_pytorch_weight_file(path: str | Path) -> dict[str, Any]:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("runtime does not support mandatory weights_only=True") from exc
    errors: list[str] = []
    if not isinstance(state, dict) or not state:
        return {
            "status": "FAIL",
            "weights_only": True,
            "tensor_only": False,
            "tensor_count": 0,
            "key_parameter_shapes": {},
            "errors": ["weight container must be a non-empty mapping"],
        }
    non_tensor_keys = [str(key) for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        errors.append("state dict contains non-tensor values")
    selected: dict[str, list[int]] = {}
    for suffix, expected_shape in EXPECTED_WEIGHT_SUFFIX_SHAPES.items():
        matches = [str(key) for key in state if str(key).endswith(suffix)]
        if len(matches) != 1:
            errors.append(f"expected exactly one state-dict key ending with {suffix}")
            continue
        key = matches[0]
        value = state[key]
        if not isinstance(value, torch.Tensor):
            errors.append(f"critical state-dict entry is not a tensor: {key}")
            continue
        observed_shape = list(value.shape)
        selected[key] = observed_shape
        if observed_shape != expected_shape:
            errors.append(f"critical parameter shape mismatch: {key}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "weights_only": True,
        "tensor_only": not non_tensor_keys,
        "tensor_count": len(state),
        "key_parameter_shapes": selected,
        "errors": errors,
    }


def nonfinite_numeric_paths(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(path)
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(nonfinite_numeric_paths(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            errors.extend(nonfinite_numeric_paths(child, f"{path}[{index}]"))
    return errors


def frozen_resampling_spec() -> ResamplingSpec:
    return ResamplingSpec(
        source_sample_rate_hz=SOURCE_SAMPLE_RATE_HZ,
        target_sample_rate_hz=MODEL_SAMPLE_RATE_HZ,
        up=160,
        down=441,
        filter_design="scipy.signal.firwin_kaiser_symmetric_finite_polyphase",
        filter_taps=RESAMPLE_FILTER_TAPS,
        kaiser_beta=RESAMPLE_KAISER_BETA,
        padtype="constant_zero_passage_local",
        input_support_radius_samples=RESAMPLE_INPUT_RADIUS_SAMPLES,
        output_support_radius_samples=RESAMPLE_OUTPUT_RADIUS_SAMPLES,
        edge_seconds=RESAMPLE_EDGE_SECONDS,
    )


def design_resampling_filter(spec: ResamplingSpec | None = None) -> np.ndarray:
    frozen = frozen_resampling_spec() if spec is None else spec
    coefficients = signal.firwin(
        frozen.filter_taps,
        cutoff=1.0 / max(frozen.up, frozen.down),
        window=("kaiser", frozen.kaiser_beta),
    )
    if coefficients.shape != (frozen.filter_taps,):
        raise ValueError("unexpected resampling-filter shape")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("non-finite resampling-filter coefficients")
    return coefficients


def _as_finite_mono(samples: Any) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("audio must be mono; implicit downmixing is forbidden")
    if array.size == 0:
        raise ValueError("audio passage is empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("audio passage contains non-finite values")
    return array


def crop_mono_passage(samples: Any, start_sample: int, end_sample: int) -> np.ndarray:
    array = _as_finite_mono(samples)
    if isinstance(start_sample, bool) or isinstance(end_sample, bool):
        raise ValueError("crop bounds must be integer sample indices")
    if not isinstance(start_sample, int) or not isinstance(end_sample, int):
        raise ValueError("crop bounds must be integer sample indices")
    if start_sample < 0 or end_sample <= start_sample or end_sample > array.size:
        raise ValueError("crop bounds must define a non-empty in-passage half-open interval")
    return np.array(array[start_sample:end_sample], dtype=np.float64, copy=True)


def resample_independent_passage(samples: Any) -> np.ndarray:
    array = _as_finite_mono(samples)
    spec = frozen_resampling_spec()
    if array.size <= 2 * spec.input_support_radius_samples:
        raise ValueError("audio passage is too short for the frozen resampling support")
    coefficients = design_resampling_filter(spec)
    output = signal.resample_poly(
        array,
        spec.up,
        spec.down,
        window=coefficients.tolist(),
        padtype="constant",
        cval=0.0,
    )
    expected_length = math.ceil(array.size * spec.up / spec.down)
    if output.shape != (expected_length,):
        raise ValueError("resampling changed length contrary to the frozen rule")
    if not np.all(np.isfinite(output)):
        raise ValueError("resampling produced non-finite values")
    return np.asarray(output, dtype=np.float32)


def normalize_relative_audio_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ValueError("audio path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("audio path must remain inside the passage root")
    return path.as_posix()


def load_isolated_passage(
    requested_relative_path: str,
    allowed_relative_path: str,
    reader: AudioReader,
    start_sample: int,
    end_sample: int,
) -> tuple[np.ndarray, int]:
    requested = normalize_relative_audio_path(requested_relative_path)
    allowed = normalize_relative_audio_path(allowed_relative_path)
    if requested != allowed:
        raise ValueError("requested audio is not the single authorized passage")
    samples, sample_rate_hz = reader(requested)
    if sample_rate_hz != SOURCE_SAMPLE_RATE_HZ:
        raise ValueError("eligible passage sampling rate must be 44100 Hz")
    return crop_mono_passage(samples, start_sample, end_sample), sample_rate_hz


def derive_convolution_timing(kernels: Sequence[int], strides: Sequence[int]) -> ConvolutionTiming:
    if len(kernels) == 0 or len(kernels) != len(strides):
        raise ValueError("convolution kernels and strides must be non-empty and aligned")
    cumulative_stride = 1
    receptive_field = 1
    normalized_kernels: list[int] = []
    normalized_strides: list[int] = []
    for kernel, stride in zip(kernels, strides, strict=True):
        if isinstance(kernel, bool) or isinstance(stride, bool):
            raise ValueError("convolution kernels and strides must be positive integers")
        if not isinstance(kernel, int) or not isinstance(stride, int) or kernel <= 0 or stride <= 0:
            raise ValueError("convolution kernels and strides must be positive integers")
        receptive_field += (kernel - 1) * cumulative_stride
        cumulative_stride *= stride
        normalized_kernels.append(kernel)
        normalized_strides.append(stride)
    return ConvolutionTiming(
        kernels=tuple(normalized_kernels),
        strides=tuple(normalized_strides),
        cumulative_stride_samples=cumulative_stride,
        receptive_field_samples=receptive_field,
        first_frame_center_samples=(receptive_field - 1) / 2.0,
    )


def expected_frame_count(input_samples: int, timing: ConvolutionTiming) -> int:
    if isinstance(input_samples, bool) or not isinstance(input_samples, int) or input_samples < 0:
        raise ValueError("input_samples must be a non-negative integer")
    frames = input_samples
    for kernel, stride in zip(timing.kernels, timing.strides, strict=True):
        if frames < kernel:
            return 0
        frames = 1 + (frames - kernel) // stride
    return frames


def frame_center_seconds(frame_count: int, timing: ConvolutionTiming) -> np.ndarray:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
        raise ValueError("frame_count must be a non-negative integer")
    centers = (
        timing.first_frame_center_samples
        + np.arange(frame_count, dtype=np.float64) * timing.cumulative_stride_samples
    ) / MODEL_SAMPLE_RATE_HZ
    if not np.all(np.isfinite(centers)):
        raise ValueError("non-finite frame-center timestamps")
    return centers


def audit_split_audio_identity(split_csv: str | Path) -> dict[str, Any]:
    issues: list[str] = []
    grouped: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"stimulus_ids": set(), "block_ids": set(), "splits": set()}
    )
    sample_rate_values: set[int] = set()
    channel_values: set[int] = set()
    source_status_values: set[str] = set()
    row_count = 0
    empty_audio_file_rows = 0

    with Path(split_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_AUDIO_IDENTITY_COLUMNS - columns)
        if missing:
            return {
                "status": "FAIL",
                "row_count": 0,
                "audio_file_nonempty": False,
                "empty_audio_file_rows": 0,
                "unique_audio_file_count": 0,
                "one_stimulus_per_audio_file": False,
                "one_block_per_audio_file": False,
                "one_split_per_audio_file": False,
                "audio_file_cross_split_count": 0,
                "audio_files_crossing_splits": [],
                "sample_rate_hz_values": [],
                "channel_values": [],
                "audio_source_status_values": [],
                "audio_file_assignments": [],
                "issues": ["missing audio identity columns: " + ", ".join(missing)],
            }
        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            audio_file = (row.get("audio_file") or "").strip()
            if not audio_file:
                empty_audio_file_rows += 1
                issues.append(f"empty audio_file at line {line_number}")
                continue
            try:
                normalized_audio_file = normalize_relative_audio_path(audio_file)
            except ValueError as exc:
                issues.append(f"invalid audio_file at line {line_number}: {exc}")
                continue
            stimulus_id = (row.get("stimulus_id") or "").strip()
            block_id = (row.get("block_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not stimulus_id or not block_id or not split:
                issues.append(f"incomplete audio identity at line {line_number}")
            grouped[normalized_audio_file]["stimulus_ids"].add(stimulus_id)
            grouped[normalized_audio_file]["block_ids"].add(block_id)
            grouped[normalized_audio_file]["splits"].add(split)
            try:
                sample_rate = int((row.get("audio_sample_rate_hz") or "").strip())
                channels = int((row.get("audio_channels") or "").strip())
            except ValueError:
                issues.append(f"invalid audio sampling identity at line {line_number}")
            else:
                sample_rate_values.add(sample_rate)
                channel_values.add(channels)
            source_status = (row.get("audio_source_status") or "").strip()
            if not source_status:
                issues.append(f"empty audio_source_status at line {line_number}")
            source_status_values.add(source_status)

    assignments: list[dict[str, Any]] = []
    for audio_file, identity in sorted(grouped.items()):
        assignments.append(
            {
                "audio_file": audio_file,
                "stimulus_ids": sorted(identity["stimulus_ids"]),
                "block_ids": sorted(identity["block_ids"]),
                "splits": sorted(identity["splits"]),
            }
        )
    crossing = [item["audio_file"] for item in assignments if len(item["splits"]) > 1]
    one_stimulus = all(len(item["stimulus_ids"]) == 1 for item in assignments)
    one_block = all(len(item["block_ids"]) == 1 for item in assignments)
    one_split = all(len(item["splits"]) == 1 for item in assignments)
    if row_count != 319:
        issues.append(f"audio identity row count must be 319, got {row_count}")
    if len(assignments) != EXPECTED_UNIQUE_AUDIO_FILE_COUNT:
        issues.append(
            "unique audio_file count must be "
            f"{EXPECTED_UNIQUE_AUDIO_FILE_COUNT}, got {len(assignments)}"
        )
    if not one_stimulus:
        issues.append("an audio_file maps to multiple stimulus_id values")
    if not one_block:
        issues.append("an audio_file maps to multiple block_id values")
    if not one_split:
        issues.append("an audio_file maps to multiple split values")
    if sample_rate_values != {SOURCE_SAMPLE_RATE_HZ}:
        issues.append("audio sample rates must all be 44100 Hz")
    if channel_values != {1}:
        issues.append("audio channels must all equal 1")
    if source_status_values != {EXPECTED_AUDIO_SOURCE_STATUS}:
        issues.append("audio_source_status must be BUNDLED_BLOCK_AUDIO")

    return {
        "status": "PASS" if not issues else "FAIL",
        "row_count": row_count,
        "audio_file_nonempty": empty_audio_file_rows == 0,
        "empty_audio_file_rows": empty_audio_file_rows,
        "unique_audio_file_count": len(assignments),
        "one_stimulus_per_audio_file": one_stimulus,
        "one_block_per_audio_file": one_block,
        "one_split_per_audio_file": one_split,
        "audio_file_cross_split_count": len(crossing),
        "audio_files_crossing_splits": crossing,
        "sample_rate_hz_values": sorted(sample_rate_values),
        "channel_values": sorted(channel_values),
        "audio_source_status_values": sorted(source_status_values),
        "audio_file_assignments": assignments,
        "issues": sorted(set(issues)),
    }


def build_split_guard_candidate(split_csv: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    split = config["split"]
    candidate_seconds = split["final_embargo_candidate_seconds"]
    rows = read_assignments(split_csv)
    issues = validate_assignments(
        rows,
        required_group_keys=split["required_group_keys"],
        optional_group_keys=split.get("optional_group_keys", []),
        stratification_keys=split.get("stratification_keys", []),
        temporal_context_key=split["temporal_context_key"],
        temporal_embargo_seconds=float(candidate_seconds),
    )
    audio_identity = audit_split_audio_identity(split_csv)
    issues.extend(f"audio identity: {issue}" for issue in audio_identity["issues"])
    issues = sorted(set(issues))
    report: dict[str, Any] = {
        "report_schema_version": "m6a-split-guard-final-embargo-candidate-v1",
        "task_id": "M6A-PUBLIC-001",
        "dataset_id": "ds004703",
        "dataset_version": "1.1.0",
        "status": "PASS" if not issues else "FAIL",
        "rows": len(rows),
        "issues": issues,
        "embargo_status": "FINAL_EMBARGO_CANDIDATE_ONLY",
        "final_embargo_candidate_seconds": candidate_seconds,
        "baseline_final": False,
        "audio_identity": audio_identity,
    }
    report.update(summarize_assignments(rows))
    return report


def _get(mapping: Any, *keys: str) -> Any:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _same_finite_number(value: Any, expected: float, tolerance: float = 1e-15) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and math.isclose(
        number, expected, rel_tol=0.0, abs_tol=tolerance
    )


def _valid_inventory(inventory: Any) -> tuple[bool, list[str], list[int]]:
    if not isinstance(inventory, list) or not all(isinstance(item, dict) for item in inventory):
        return False, [], []
    names = [item.get("path") for item in inventory]
    sizes = [item.get("bytes") for item in inventory]
    valid = (
        all(isinstance(name, str) for name in names)
        and names == list(MODEL_FILES)
        and len(names) == len(set(names))
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in sizes
        )
        and all(item.get("sample_readability") is True for item in inventory)
        and all(
            isinstance(item.get("modified_at_utc"), str)
            and bool(item.get("modified_at_utc"))
            for item in inventory
        )
    )
    return valid, names, sizes if valid else []


def _valid_layer_shapes(layer_shapes: Any) -> bool:
    return (
        isinstance(layer_shapes, list)
        and len(layer_shapes) == 13
        and all(
            isinstance(shape, list)
            and len(shape) == 3
            and shape[0] == 1
            and isinstance(shape[1], int)
            and not isinstance(shape[1], bool)
            and shape[1] > 0
            and shape[2] == EXPECTED_HIDDEN_SIZE
            for shape in layer_shapes
        )
    )


def _valid_audio_identity_assignments(assignments: Any) -> bool:
    if not isinstance(assignments, list) or len(assignments) != EXPECTED_UNIQUE_AUDIO_FILE_COUNT:
        return False
    observed_paths: list[str] = []
    for item in assignments:
        if not isinstance(item, dict):
            return False
        audio_file_value = item.get("audio_file")
        if not isinstance(audio_file_value, str):
            return False
        try:
            audio_file = normalize_relative_audio_path(audio_file_value)
        except ValueError:
            return False
        stimulus_ids = item.get("stimulus_ids")
        block_ids = item.get("block_ids")
        splits = item.get("splits")
        for values in (stimulus_ids, block_ids, splits):
            if (
                not isinstance(values, list)
                or len(values) != 1
                or not isinstance(values[0], str)
                or not values[0].strip()
            ):
                return False
        if not isinstance(splits, list):
            return False
        if splits[0] not in EXPECTED_SPLIT_COUNTS:
            return False
        observed_paths.append(audio_file)
    return len(observed_paths) == len(set(observed_paths))


def validate_candidate_evidence(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    spec = frozen_resampling_spec()
    timing = derive_convolution_timing((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2))
    split_report_value = evidence.get("split_guard", {})
    split_report = split_report_value if isinstance(split_report_value, dict) else {}
    embargo_value = evidence.get("embargo_evaluation", {})
    embargo = embargo_value if isinstance(embargo_value, dict) else {}
    inventory = evidence.get("model_inventory", [])
    inventory_valid, inventory_names, inventory_bytes = _valid_inventory(inventory)
    layer_shapes = _get(evidence, "model_canary", "layer_shapes")
    audio_identity_value = split_report.get("audio_identity", {})
    audio_identity = audio_identity_value if isinstance(audio_identity_value, dict) else {}
    audio_assignments_value = audio_identity.get("audio_file_assignments", [])
    audio_assignments = audio_assignments_value if isinstance(audio_assignments_value, list) else []
    checks = {
        "report_identity": (
            evidence.get("report_schema_version") == REPORT_SCHEMA_VERSION
            and evidence.get("task_id") == "M6A-PUBLIC-001"
        ),
        "non_hash_policy": (
            evidence.get("integrity_policy") == "NON_HASH_AUDIT"
            and evidence.get("cryptographic_integrity_claim") is False
            and find_forbidden_fields(evidence) == []
        ),
        "finite_numeric_values": nonfinite_numeric_paths(evidence) == [],
        "g2_coordinator_acceptance": (
            _get(config, "g2", "status") == "G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE"
            and _get(config, "g2", "coordinator_review") == "ACCEPT"
            and _get(config, "g2", "whole_m6a_pass_claimed") is False
        ),
        "single_frozen_model_identity": (
            _get(evidence, "model", "model_id") == MODEL_ID
            and _get(evidence, "model", "revision_label") == MODEL_REVISION_LABEL
            and _get(evidence, "model", "revision_limitation")
            == "MUTABLE_MAIN_LABEL_NON_CRYPTOGRAPHIC_REPRODUCIBILITY_ONLY"
            and _get(evidence, "model", "source_endpoint") == "https://hf-mirror.com"
            and _get(evidence, "model", "source_endpoint_role")
            == "PUBLIC_HUGGING_FACE_ENDPOINT_MIRROR"
            and _get(evidence, "model", "source_endpoint_limitation")
            == "THIRD_PARTY_MIRROR_PLUS_MUTABLE_MAIN_AND_NO_HASH_POLICY_DO_NOT_PROVIDE_CRYPTOGRAPHIC_INTEGRITY_OR_IMMUTABLE_PROVENANCE"
            and _get(evidence, "model", "trainable") is False
            and _get(config, "model", "model_id") == MODEL_ID
            and _get(evidence, "model", "cache_state")
            == "SEMANTICALLY_VALIDATED_REMOTE_ONLY"
            and _get(config, "model", "cache_state")
            == "SEMANTICALLY_VALIDATED_REMOTE_ONLY"
            and _get(config, "model", "download_allowed") is False
        ),
        "model_config_frozen_profile": (
            _get(evidence, "model", "config_semantic_audit", "status") == "PASS"
            and _get(evidence, "model", "config_semantic_audit", "observed")
            == EXPECTED_CONFIG_PROFILE
            and _get(evidence, "model", "config_semantic_audit", "errors") == []
        ),
        "weight_container_safe_and_shaped": (
            _get(evidence, "model", "weight_semantic_audit", "status") == "PASS"
            and _get(evidence, "model", "weight_semantic_audit", "weights_only") is True
            and _get(evidence, "model", "weight_semantic_audit", "tensor_only") is True
            and isinstance(_get(evidence, "model", "weight_semantic_audit", "tensor_count"), int)
            and not isinstance(_get(evidence, "model", "weight_semantic_audit", "tensor_count"), bool)
            and _get(evidence, "model", "weight_semantic_audit", "tensor_count") > 0
            and len(_get(evidence, "model", "weight_semantic_audit", "key_parameter_shapes") or {})
            == len(EXPECTED_WEIGHT_SUFFIX_SHAPES)
            and _get(evidence, "model", "weight_semantic_audit", "errors") == []
        ),
        "runtime_profile_present": (
            all(
                isinstance(_get(evidence, "runtime", key), str)
                and bool(_get(evidence, "runtime", key))
                for key in ("python", "torch", "transformers", "numpy", "scipy", "soundfile")
            )
            and _get(evidence, "runtime", "conda_environment") == "auditory_m6a_public_001"
        ),
        "model_inventory_exact_and_readable": (
            inventory_valid
            and inventory_names == list(MODEL_FILES)
            and evidence.get("model_inventory_total_bytes") == sum(inventory_bytes)
            and evidence.get("model_cache_remote_only") is True
        ),
        "input_semantics_frozen": (
            _get(evidence, "input_semantics", "source_sample_rate_hz") == SOURCE_SAMPLE_RATE_HZ
            and _get(evidence, "input_semantics", "model_sample_rate_hz") == MODEL_SAMPLE_RATE_HZ
            and _get(evidence, "input_semantics", "channels") == 1
            and _get(evidence, "input_semantics", "passage_policy")
            == "ONE_ELIGIBLE_PASSAGE_PER_INFERENCE_CALL"
            and _get(evidence, "input_semantics", "neighbor_audio_read_allowed") is False
            and _get(evidence, "input_semantics", "batch_padding") == "FORBIDDEN_PRIMARY_INFERENCE"
        ),
        "finite_resampling_support": (
            _get(evidence, "resampling") == asdict(spec)
            and _same_finite_number(
                _get(config, "split", "final_embargo_components_seconds", "audio_resampling_edge_seconds"),
                spec.edge_seconds,
            )
        ),
        "synthetic_split_sentinel": (
            _get(evidence, "split_sentinel", "status") == "PASS"
            and _get(evidence, "split_sentinel", "read_paths") == ["train/passage.wav"]
            and _get(evidence, "split_sentinel", "forbidden_read_count") == 0
            and _get(evidence, "split_sentinel", "heldout_sentinel_observed") is False
        ),
        "model_load_and_frozen_eval": (
            _get(evidence, "model_canary", "local_files_only") is True
            and _get(evidence, "model_canary", "trust_remote_code") is False
            and _get(evidence, "model_canary", "repository_custom_code_executed") is False
            and _get(evidence, "model_canary", "model_eval") is True
            and _get(evidence, "model_canary", "parameter_requires_grad_count") == 0
            and _get(evidence, "model_canary", "all_outputs_finite") is True
        ),
        "base_encoder_load_is_complete": (
            _get(evidence, "model_canary", "loading_info", "missing_keys") == []
            and _get(evidence, "model_canary", "loading_info", "mismatched_keys") == []
            and _get(evidence, "model_canary", "loading_info", "error_msgs") == []
            and _get(evidence, "model_canary", "loading_info", "unexpected_keys")
            == list(EXPECTED_PRETRAINING_HEAD_KEYS)
            and _get(evidence, "model_canary", "unexpected_key_semantics")
            == "PREDECLARED_PRETRAINING_HEADS_EXCLUDED_FROM_FROZEN_BASE_ENCODER"
        ),
        "padding_and_crop_canary": (
            _get(evidence, "model_canary", "crop_half_open_interval_verified") is True
            and _get(evidence, "model_canary", "model_input_padding_samples") == 0
            and _get(evidence, "model_canary", "attention_mask_all_ones") is True
            and _get(evidence, "model_canary", "passage_local_resample_padding") is True
        ),
        "projected_plus_12_layers": (
            _get(evidence, "model_canary", "layer_keys") == list(EXPECTED_LAYER_KEYS)
            and _valid_layer_shapes(layer_shapes)
        ),
        "frame_count_and_center_mapping": (
            _get(evidence, "frame_timing", "kernels") == list(timing.kernels)
            and _get(evidence, "frame_timing", "strides") == list(timing.strides)
            and _get(evidence, "frame_timing", "cumulative_stride_samples")
            == timing.cumulative_stride_samples
            and _get(evidence, "frame_timing", "receptive_field_samples")
            == timing.receptive_field_samples
            and _get(evidence, "frame_timing", "observed_frames")
            == _get(evidence, "frame_timing", "expected_frames")
            and _same_finite_number(
                _get(evidence, "frame_timing", "first_frame_center_seconds"),
                timing.first_frame_center_samples / MODEL_SAMPLE_RATE_HZ,
                1e-12,
            )
            and _same_finite_number(
                _get(evidence, "frame_timing", "frame_step_seconds"),
                timing.cumulative_stride_samples / MODEL_SAMPLE_RATE_HZ,
                1e-12,
            )
        ),
        "context_scope_distinction": (
            _get(evidence, "context", "transformer_attention_scope") == "GLOBAL_WITHIN_SINGLE_PASSAGE"
            and _get(evidence, "context", "transformer_local_receptive_field_claimed") is False
            and _get(evidence, "context", "cross_split_input_overlap_measured") is True
            and _get(evidence, "context", "audio_cross_split_context_overlap_seconds") == 0.0
            and _get(config, "split", "final_embargo_components_seconds", "audio_cross_split_context_overlap_seconds")
            == 0.0
            and _get(evidence, "context", "basis")
            == "REAL_319_ROW_AUDIO_IDENTITY_GATE_PLUS_SYNTHETIC_PATH_SENTINEL"
        ),
        "real_manifest_audio_identity": (
            audio_identity.get("status") == "PASS"
            and audio_identity.get("row_count") == 319
            and audio_identity.get("audio_file_nonempty") is True
            and audio_identity.get("empty_audio_file_rows") == 0
            and audio_identity.get("unique_audio_file_count")
            == EXPECTED_UNIQUE_AUDIO_FILE_COUNT
            and audio_identity.get("one_stimulus_per_audio_file") is True
            and audio_identity.get("one_block_per_audio_file") is True
            and audio_identity.get("one_split_per_audio_file") is True
            and audio_identity.get("audio_file_cross_split_count") == 0
            and audio_identity.get("audio_files_crossing_splits") == []
            and audio_identity.get("sample_rate_hz_values") == [SOURCE_SAMPLE_RATE_HZ]
            and audio_identity.get("channel_values") == [1]
            and audio_identity.get("audio_source_status_values")
            == [EXPECTED_AUDIO_SOURCE_STATUS]
            and _valid_audio_identity_assignments(audio_assignments)
            and audio_identity.get("issues") == []
        ),
        "final_embargo_candidate": (
            embargo.get("status") == "FINAL_EMBARGO_CANDIDATE_READY"
            and embargo.get("final_embargo_candidate_seconds") == 2.0
            and embargo.get("baseline_final") is False
            and _get(config, "split", "final_embargo_candidate_seconds") == 2.0
            and _get(config, "split", "final_embargo_seconds") is None
            and _get(config, "split", "baseline_final") is False
        ),
        "real_lightweight_split_guard": (
            split_report.get("report_schema_version")
            == "m6a-split-guard-final-embargo-candidate-v1"
            and split_report.get("status") == "PASS"
            and split_report.get("rows") == 319
            and split_report.get("issues") == []
            and split_report.get("final_embargo_candidate_seconds") == 2.0
            and split_report.get("baseline_final") is False
            and split_report.get("split_counts") == EXPECTED_SPLIT_COUNTS
            and split_report.get("block_assignments") == EXPECTED_BLOCK_ASSIGNMENTS
            and split_report.get("language_counts") == {"en": 319}
            and split_report.get("catalan_rows") == 0
        ),
        "execution_remains_blocked": (
            _get(config, "neural_target", "neural_extraction_allowed") is False
            and _get(config, "split", "baseline_final") is False
            and _get(config, "artifact", "exchange_candidate_exists") is False
            and evidence.get("formal_feature_extraction_run") is False
            and evidence.get("real_neural_waveform_read") is False
            and evidence.get("baseline_run") is False
            and evidence.get("exchange_candidate_created") is False
            and evidence.get("scientific_result_claimed") is False
        ),
    }
    return checks


def finalize_candidate_report(evidence: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    checks = validate_candidate_evidence(evidence, config)
    failed = [name for name, passed in checks.items() if passed is not True]
    report = dict(evidence)
    report["required_checks"] = checks
    report["failed_checks"] = failed
    report["status"] = CANDIDATE_STATUS if not failed else "FAIL"
    return report


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
