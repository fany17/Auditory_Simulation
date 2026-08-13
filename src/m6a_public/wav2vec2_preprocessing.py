from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


NORMALIZATION_EPSILON = 1e-7
PREPROCESSOR_FILENAME = "preprocessor_config.json"
PREPROCESSOR_SOURCE_ENDPOINT = "https://hf-mirror.com"
PREPROCESSOR_SEMANTICS: dict[str, Any] = {
    "feature_size": 1,
    "sampling_rate": 16_000,
    "padding_value": 0.0,
    "do_normalize": True,
    "return_attention_mask": False,
    "padding_side": "right",
}
WAV2VEC2_INPUT_PREPROCESSING_CONTRACT: dict[str, Any] = {
    "preprocessor_config_filename": PREPROCESSOR_FILENAME,
    "preprocessor_config_source_endpoint": PREPROCESSOR_SOURCE_ENDPOINT,
    "preprocessor_source_probe": "reports/wav2vec2_model_mirror_probe_20260813.json",
    "preprocessor_config_bytes": 159,
    "cache_state": "SEMANTICALLY_VALIDATED_REMOTE_ONLY",
    "feature_size": 1,
    "sampling_rate_hz": 16_000,
    "do_normalize": True,
    "return_attention_mask": False,
    "padding_value": 0.0,
    "padding_side": "right",
    "input_dtype": "float32",
    "passage_scope": "EACH_PASSAGE_INDEPENDENT_NO_CROSS_PASSAGE_STATISTICS",
    "normalization_formula": (
        "(x - mean_float32(x)) / "
        "sqrt(population_variance_float32_ddof0(x) + 1e-7)"
    ),
    "normalization_epsilon": NORMALIZATION_EPSILON,
    "constant_input_action": "FAIL_BEFORE_FEATURE_EXTRACTOR",
    "nonfinite_input_action": "FAIL_BEFORE_FEATURE_EXTRACTOR",
    "no_padding_policy": "SINGLE_PASSAGE_NO_PADDING",
    "attention_mask_argument": "OMITTED",
    "validation_test_scope": "SAME_FIXED_PARAMETER_FREE_PASSAGE_WISE_RULE",
    "train_fitted_statistics_used": False,
    "cross_passage_statistics_allowed": False,
}


def _reject_parse_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def _load_strict_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_parse_constant)
    if not isinstance(payload, dict):
        raise ValueError("preprocessor config must contain one JSON object")
    return payload


def audit_preprocessor_config(
    path: str | Path, *, expected_cache_root: str | Path | None = None
) -> dict[str, Any]:
    candidate = Path(path).resolve()
    issues: list[str] = []
    if candidate.name != PREPROCESSOR_FILENAME or not candidate.is_file():
        issues.append("preprocessor_config.json is missing or misnamed")
    if candidate.is_symlink():
        issues.append("preprocessor config must not be a symlink")
    if expected_cache_root is not None:
        cache_root = Path(expected_cache_root).resolve()
        if candidate.parent != cache_root:
            issues.append("preprocessor config is outside the frozen model cache root")

    payload: dict[str, Any] = {}
    try:
        payload = _load_strict_json_object(candidate)
    except (OSError, TypeError, ValueError) as error:
        issues.append(f"preprocessor config is not strict readable JSON: {error}")

    semantic_fields = {
        key: payload.get(key) for key in PREPROCESSOR_SEMANTICS
    }
    if semantic_fields != PREPROCESSOR_SEMANTICS:
        issues.append("preprocessor semantic fields drifted")
    unexpected_fields = sorted(set(payload) - set(PREPROCESSOR_SEMANTICS))
    missing_fields = sorted(set(PREPROCESSOR_SEMANTICS) - set(payload))
    if unexpected_fields or missing_fields:
        issues.append("preprocessor config field inventory drifted")

    file_bytes = -1
    modified_at_utc: str | None = None
    try:
        stat = candidate.stat()
        file_bytes = int(stat.st_size)
        modified_at_utc = datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat()
    except OSError as error:
        issues.append(f"preprocessor file metadata is unreadable: {error}")
    if file_bytes <= 0:
        issues.append("preprocessor config must have positive bytes")

    return {
        "status": "PASS" if not issues else "FAIL",
        "path": candidate.as_posix(),
        "filename": candidate.name,
        "bytes": file_bytes,
        "modified_at_utc": modified_at_utc,
        "semantic_fields": semantic_fields,
        "missing_fields": missing_fields,
        "unexpected_fields": unexpected_fields,
        "remote_only": True,
        "issues": issues,
    }


def normalize_passage_waveform(values: Any) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        waveform = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("waveform must be convertible to float32") from error
    if waveform.ndim != 1 or waveform.size < 1:
        raise ValueError("waveform must be a non-empty one-dimensional passage")
    if not bool(np.all(np.isfinite(waveform))):
        raise ValueError("waveform contains a non-finite value")

    mean = waveform.mean()
    variance = waveform.var()
    if not math.isfinite(float(mean)) or not math.isfinite(float(variance)):
        raise ValueError("waveform mean or variance is non-finite")
    if float(variance) <= 0.0:
        raise ValueError("constant waveform is forbidden by the G4 input contract")

    normalized = (waveform - mean) / np.sqrt(variance + NORMALIZATION_EPSILON)
    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.shape != waveform.shape or not bool(np.all(np.isfinite(normalized))):
        raise ValueError("normalized waveform failed shape or finite validation")
    return normalized, {
        "sample_count": int(waveform.size),
        "input_dtype": str(waveform.dtype),
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "pre_all_finite": True,
        "pre_mean": float(mean),
        "pre_population_variance": float(variance),
        "pre_population_std": float(np.sqrt(variance)),
        "post_all_finite": True,
        "post_mean": float(normalized.mean()),
        "post_population_variance": float(normalized.var()),
        "post_population_std": float(normalized.std()),
    }


__all__ = [
    "NORMALIZATION_EPSILON",
    "PREPROCESSOR_FILENAME",
    "PREPROCESSOR_SEMANTICS",
    "PREPROCESSOR_SOURCE_ENDPOINT",
    "WAV2VEC2_INPUT_PREPROCESSING_CONTRACT",
    "audit_preprocessor_config",
    "normalize_passage_waveform",
]
