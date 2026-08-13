from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from jsonschema import Draft202012Validator

from m6a_public.audio_context_gate import (
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
    nonfinite_numeric_paths,
)
from m6a_public.config_gate import find_forbidden_fields
from m6a_public.wav2vec2_preprocessing import (
    PREPROCESSOR_FILENAME,
    PREPROCESSOR_SEMANTICS,
    PREPROCESSOR_SOURCE_ENDPOINT,
    WAV2VEC2_INPUT_PREPROCESSING_CONTRACT,
)


PREFLIGHT_STATUS = (
    "G4_RESOURCE_AND_RUNTIME_PREFLIGHT_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
)
PREFLIGHT_SCHEMA_VERSION = "m6a-g4-resource-runtime-preflight-candidate-v1"
PREFLIGHT_REPORT_SCHEMA_VERSION = "m6a-g4-resource-runtime-preflight-report-v1"
G4_PROTOCOL_STATUS = (
    "G4_PROTOCOL_AMENDMENT_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
)
EXPECTED_RECORDING = "sub-SD012_ses-02_task-PassiveListen"
EXPECTED_LONGEST_SAMPLE = "sub-SD012_ses-02_task-PassiveListen__seg-028"
EXPECTED_LONGEST_STIMULUS = "s3201a-ex01"
EXPECTED_LONGEST_AUDIO = "stimuli/excerpts/Block 3/s3201a-ex01_normed.wav"
EXPECTED_LONGEST_DURATION_SECONDS = 77.08981859410432
EXPECTED_SYNTHETIC_SAMPLE_COUNT = 1_233_438
EXPECTED_FRAME_COUNT = 3_854
EXPECTED_PASSAGE_COUNT = 40
EXPECTED_CATEGORY_PATHS = {
    "data": ["/home/fanyu/auditory_simulation_m6a/data"],
    "cache": ["/home/fanyu/auditory_simulation_m6a/cache"],
    "outputs": ["/home/fanyu/auditory_simulation_m6a/outputs"],
    "log": [
        "/home/fanyu/auditory_simulation_m6a/log",
        "/home/fanyu/auditory_simulation_m6a/logs",
    ],
    "code": ["/home/fanyu/auditory_simulation_m6a/code_snapshot"],
}


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (float(value) > 0 if positive else float(value) >= 0)
    )


def _finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _int_or_negative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def _aware_iso(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def audit_longest_passage_manifest(
    split_csv: str | Path, recording_id: str = EXPECTED_RECORDING
) -> dict[str, Any]:
    required = {
        "sample_id",
        "recording_id",
        "stimulus_id",
        "split",
        "audio_file",
        "audio_duration_seconds",
        "analysis_eligible",
        "audio_source_status",
    }
    issues: list[str] = []
    rows: list[dict[str, str]] = []
    with Path(split_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            return {"status": "FAIL", "issues": ["split manifest columns drifted"]}
        rows = [row for row in reader if row.get("recording_id") == recording_id]
    if len(rows) != EXPECTED_PASSAGE_COUNT:
        issues.append("G4 recording must contain exactly 40 passages")

    ranked: list[tuple[float, str, dict[str, str]]] = []
    for row in rows:
        try:
            duration = float(row.get("audio_duration_seconds", ""))
        except ValueError:
            duration = math.nan
        sample_id = row.get("sample_id", "")
        if not math.isfinite(duration) or duration <= 0 or not sample_id:
            issues.append("G4 passage duration/sample identity is invalid")
        ranked.append((duration, sample_id, row))
    if issues or not ranked:
        return {"status": "FAIL", "passage_count": len(rows), "issues": issues}
    ranked.sort(key=lambda item: (-item[0], item[1]))
    duration, _, longest = ranked[0]
    sample_count = math.ceil(duration * 16_000)
    observed = {
        "sample_id": longest.get("sample_id"),
        "stimulus_id": longest.get("stimulus_id"),
        "split": longest.get("split"),
        "audio_file_metadata_only": longest.get("audio_file"),
        "audio_duration_seconds": duration,
        "synthetic_sample_count_formula": "ceil(audio_duration_seconds * 16000)",
        "synthetic_sample_count": sample_count,
        "analysis_eligible": longest.get("analysis_eligible"),
        "audio_source_status": longest.get("audio_source_status"),
    }
    if (
        observed["sample_id"] != EXPECTED_LONGEST_SAMPLE
        or observed["stimulus_id"] != EXPECTED_LONGEST_STIMULUS
        or observed["split"] != "validation"
        or observed["audio_file_metadata_only"] != EXPECTED_LONGEST_AUDIO
        or not math.isclose(
            duration, EXPECTED_LONGEST_DURATION_SECONDS, rel_tol=0.0, abs_tol=1e-12
        )
        or sample_count != EXPECTED_SYNTHETIC_SAMPLE_COUNT
        or observed["analysis_eligible"] != "True"
        or observed["audio_source_status"] != "BUNDLED_BLOCK_AUDIO"
    ):
        issues.append("longest G4 passage metadata drifted")
    return {
        "status": "PASS" if not issues else "FAIL",
        "manifest_path": Path(split_csv).as_posix(),
        "recording_id": recording_id,
        "selection_rule": "MAX_AUDIO_DURATION_SECONDS_THEN_SAMPLE_ID_LEXICOGRAPHIC",
        "passage_count": len(rows),
        "longest": observed,
        "real_audio_read": False,
        "issues": issues,
    }


def validate_preflight_config(
    config: dict[str, Any],
    schema: dict[str, Any],
    main_config: dict[str, Any],
    protocol_config: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    try:
        validator = Draft202012Validator(schema)
        errors.extend(
            "schema: " + error.message
            for error in sorted(validator.iter_errors(config), key=lambda item: list(item.path))
        )
    except (TypeError, ValueError) as error:
        errors.append(f"schema validation could not run: {error}")
    try:
        if find_forbidden_fields(config):
            errors.append("preflight config contains forbidden integrity fields")
        if nonfinite_numeric_paths(config):
            errors.append("preflight config contains non-finite numbers")
    except (AttributeError, TypeError, ValueError):
        errors.append("preflight config is malformed")

    if (
        config.get("schema_version") != PREFLIGHT_SCHEMA_VERSION
        or config.get("task_id") != "M6A-PUBLIC-001"
        or config.get("status") != PREFLIGHT_STATUS
        or config.get("integrity_policy") != "NON_HASH_AUDIT"
    ):
        errors.append("preflight identity/status drifted")

    dependency = _mapping(config.get("dependency"))
    if dependency != {
        "g4_protocol_status": G4_PROTOCOL_STATUS,
        "g4_protocol_review": "PENDING",
        "g4_protocol_reviewed_on": None,
        "g4_protocol_config": "configs/m6a_g4_protocol_candidate.json",
        "g4_protocol_report": "reports/g4_protocol_amendment_candidate_20260813_v2.json",
        "prior_g4_protocol_status": "G4_PROTOCOL_COORDINATOR_ACCEPTED",
        "prior_g4_protocol_report": "reports/g4_protocol_candidate_20260813_v3.json",
        "amendment_scope": "WAV2VEC2_PREPROCESSING_INPUT_CONTRACT_ONLY",
        "g4_scientific_result_claimed": False,
    }:
        errors.append("accepted G4 protocol dependency drifted")
    if (
        protocol_config.get("status") != G4_PROTOCOL_STATUS
        or protocol_config.get("coordinator_review") != "PENDING"
        or protocol_config.get("reviewed_on") is not None
        or protocol_config.get("scientific_result_claimed") is not False
    ):
        errors.append("G4 preprocessing protocol amendment is not an unaccepted candidate")

    main_model = _mapping(main_config.get("model"))
    model = _mapping(config.get("model"))
    preprocessing = _mapping(model.get("input_preprocessing"))
    main_preprocessing = _mapping(
        _mapping(main_model.get("inference_input")).get("preprocessing")
    )
    protocol_preprocessing = _mapping(
        _mapping(_mapping(protocol_config.get("features")).get("wav2vec2")).get(
            "input_preprocessing"
        )
    )
    if (
        model.get("model_id") != main_model.get("model_id")
        or model.get("revision_label") != main_model.get("revision_label")
        or model.get("cache_path") != main_model.get("remote_cache")
        or model.get("cache_state") != main_model.get("cache_state")
        or main_model.get("download_allowed") is not False
        or model.get("download_allowed") is not False
        or model.get("local_files_only") is not True
        or model.get("trust_remote_code") is not False
        or model.get("weights_only_required") is not True
        or model.get("tensor_only_required") is not True
        or model.get("model_eval_required") is not True
        or model.get("inference_mode_required") is not True
        or model.get("layer_keys") != list(EXPECTED_LAYER_KEYS)
        or preprocessing != WAV2VEC2_INPUT_PREPROCESSING_CONTRACT
        or main_preprocessing != WAV2VEC2_INPUT_PREPROCESSING_CONTRACT
        or protocol_preprocessing != WAV2VEC2_INPUT_PREPROCESSING_CONTRACT
    ):
        errors.append("frozen model cache/runtime boundary drifted")

    runtime_config = _mapping(config.get("runtime_canary"))
    if (
        runtime_config.get("feature_extractor_equivalence_absolute_tolerance")
        != 1e-7
        or runtime_config.get("feature_extractor_equivalence_relative_tolerance")
        != 1e-7
        or runtime_config.get("attention_mask_argument") != "OMITTED"
        or runtime_config.get("model_input_padding_samples") != 0
        or runtime_config.get("measurements")
        != [
            "MODEL_LOAD_WALL_SECONDS",
            "WARMUP_WALL_SECONDS",
            "LONGEST_FORWARD_WALL_SECONDS",
            "CUDA_PEAK_ALLOCATED_BYTES",
            "CUDA_PEAK_RESERVED_BYTES",
            "THIRTEEN_LAYER_SHAPES_AND_FINITE",
            "PREPROCESSOR_CONFIG_AND_FEATURE_EXTRACTOR_EQUIVALENCE",
        ]
    ):
        errors.append("feature-extractor equivalence or attention-mask contract drifted")

    storage = _mapping(config.get("storage"))
    categories = _mapping(storage.get("categories"))
    project_root = PurePosixPath(str(storage.get("project_root", "")))
    paths_are_scoped = project_root.is_absolute()
    for values in categories.values():
        if not isinstance(values, list):
            paths_are_scoped = False
            continue
        for value in values:
            path = PurePosixPath(str(value))
            if not path.is_absolute() or path == project_root or project_root not in path.parents:
                paths_are_scoped = False
    if categories != EXPECTED_CATEGORY_PATHS or not paths_are_scoped:
        errors.append("storage categories must be exact absolute paths inside project root")

    longest = _mapping(config.get("longest_passage"))
    expected_sample_count = math.ceil(
        float(longest.get("audio_duration_seconds", math.nan)) * 16_000
    ) if _finite_number(longest.get("audio_duration_seconds"), positive=True) else -1
    if (
        longest.get("sample_id") != EXPECTED_LONGEST_SAMPLE
        or longest.get("stimulus_id") != EXPECTED_LONGEST_STIMULUS
        or longest.get("audio_file_metadata_only") != EXPECTED_LONGEST_AUDIO
        or expected_sample_count != EXPECTED_SYNTHETIC_SAMPLE_COUNT
        or longest.get("expected_synthetic_sample_count") != expected_sample_count
        or longest.get("expected_frame_count") != EXPECTED_FRAME_COUNT
        or longest.get("real_audio_read_allowed") is not False
    ):
        errors.append("longest-passage metadata or frozen 16 kHz shape drifted")

    synthetic = _mapping(config.get("synthetic_input"))
    if (
        synthetic.get("source") != "DETERMINISTIC_IN_MEMORY_FINITE_MONO_NO_FILE_PATH"
        or synthetic.get("input_path") is not None
        or synthetic.get("finite_required") is not True
        or synthetic.get("channels") != 1
        or synthetic.get("neighbor_input_allowed") is not False
    ):
        errors.append("preflight input must remain deterministic synthetic mono without a path")

    estimate = _mapping(config.get("execution_estimate"))
    checkpoint = _mapping(config.get("checkpoint_design"))
    if (
        estimate.get("passage_count") != EXPECTED_PASSAGE_COUNT
        or estimate.get("model_load_safety_multiplier") != 2.0
        or estimate.get("forward_safety_multiplier") != 2.0
        or estimate.get("model_load_minimum_upper_bound_seconds") != 60.0
        or estimate.get("forward_minimum_upper_bound_seconds") != 60.0
        or estimate.get("checkpoint_and_process_overhead_upper_bound_seconds") != 30.0
        or estimate.get("single_invocation_gpu_hours_strictly_below") != 2.0
        or estimate.get("total_gpu_hours_maximum") != 4.0
        or estimate.get("duration_linear_scaling_assumed") is not False
        or checkpoint.get("execution_unit") != "ONE_PASSAGE_PER_INVOCATION"
        or checkpoint.get("successful_passage_checkpoint_required") is not True
        or checkpoint.get("resume_skips_only_validated_final_checkpoint") is not True
        or checkpoint.get("partial_preserved_on_failure") is not True
        or checkpoint.get("delete_on_failure_allowed") is not False
        or checkpoint.get("checkpoint_before_two_hour_limit") is not True
    ):
        errors.append("runtime estimate or recoverable checkpoint design drifted")

    execution = _mapping(config.get("execution"))
    required_false = {
        "g4_execution_authorized", "new_real_edf_read", "new_real_audio_read",
        "real_feature_extraction_run", "ridge_run", "null_run", "metric_run",
        "scientific_result_claimed", "exchange_candidate_created",
    }
    if (
        execution.get("preflight_only") is not True
        or any(execution.get(key) is not False for key in required_false)
        or execution.get("allowed_output_status") != PREFLIGHT_STATUS
        or _mapping(main_config.get("g4_protocol")).get("g4_execution_authorized")
        is not False
    ):
        errors.append("preflight must keep G4 execution and scientific claims closed")
    return errors


def conservative_runtime_estimate(
    model_load_wall_seconds: float, longest_forward_wall_seconds: float
) -> dict[str, float]:
    if not _finite_number(model_load_wall_seconds, positive=True) or not _finite_number(
        longest_forward_wall_seconds, positive=True
    ):
        raise ValueError("measured runtime values must be finite and positive")
    load_upper = max(60.0, 2.0 * model_load_wall_seconds)
    forward_upper = max(60.0, 2.0 * longest_forward_wall_seconds)
    single_seconds = load_upper + forward_upper + 30.0
    single_hours = single_seconds / 3600.0
    total_hours = EXPECTED_PASSAGE_COUNT * single_hours
    return {
        "model_load_upper_bound_seconds": load_upper,
        "longest_forward_upper_bound_seconds": forward_upper,
        "checkpoint_and_process_overhead_upper_bound_seconds": 30.0,
        "single_invocation_upper_bound_seconds": single_seconds,
        "single_invocation_upper_bound_gpu_hours": single_hours,
        "total_40_passage_upper_bound_gpu_hours": total_hours,
    }


def validate_preflight_evidence(
    evidence: dict[str, Any], config: dict[str, Any]
) -> dict[str, bool]:
    storage = _mapping(evidence.get("storage_audit"))
    categories = _mapping(storage.get("categories"))
    runtime = _mapping(evidence.get("runtime_canary"))
    warmup = _mapping(runtime.get("warmup"))
    longest = _mapping(runtime.get("longest_forward"))
    estimate = _mapping(evidence.get("execution_estimate"))
    manifest = _mapping(evidence.get("longest_passage_manifest_audit"))
    checkpoint = _mapping(evidence.get("checkpoint_design_audit"))
    expected_shape = [1, EXPECTED_FRAME_COUNT, 768]
    layer_shapes = longest.get("layer_shapes")
    expected_paths = EXPECTED_CATEGORY_PATHS

    category_identity_ok = set(categories) == set(expected_paths)
    category_total = 0
    if category_identity_ok:
        for name, expected in expected_paths.items():
            category = _mapping(categories.get(name))
            paths = category.get("paths")
            if not isinstance(paths, list) or [item.get("path") for item in paths if isinstance(item, dict)] != expected:
                category_identity_ok = False
                break
            if not all(
                item.get("exists") is True
                and _nonnegative_int(item.get("bytes"))
                and _aware_iso(item.get("modified_at_utc"))
                for item in paths
                if isinstance(item, dict)
            ):
                category_identity_ok = False
                break
            category_bytes = category.get("bytes")
            if not _nonnegative_int(category_bytes):
                category_identity_ok = False
                break
            if category_bytes != sum(item["bytes"] for item in paths):
                category_identity_ok = False
                break
            category_total += int(category_bytes)

    data_bytes = _mapping(categories.get("data")).get("bytes", -1)
    cache_bytes = _mapping(categories.get("cache")).get("bytes", -1)
    estimated_new = _mapping(config.get("storage")).get(
        "estimated_new_bytes_upper_bound", -1
    )
    loading_info = _mapping(runtime.get("loading_info"))
    mirror_audit = _mapping(runtime.get("preprocessor_remote_semantic_audit"))
    mirror_probes = mirror_audit.get("probes")
    preprocessor_audit = _mapping(runtime.get("preprocessor_config_audit"))
    extractor_semantics = _mapping(runtime.get("feature_extractor_semantics"))
    equivalence = _mapping(runtime.get("feature_extractor_equivalence"))
    warmup_equivalence = _mapping(equivalence.get("warmup"))
    longest_equivalence = _mapping(equivalence.get("longest"))

    def equivalence_is_valid(item: dict[str, Any], sample_count: int) -> bool:
        normalization = _mapping(item.get("normalization"))
        numeric_fields = (
            "pre_mean",
            "pre_population_variance",
            "pre_population_std",
            "post_mean",
            "post_population_variance",
            "post_population_std",
        )
        return (
            item.get("status") == "PASS"
            and item.get("sample_count") == sample_count
            and item.get("feature_extractor_output_shape") == [sample_count]
            and item.get("feature_extractor_output_dtype") == "float32"
            and item.get("attention_mask_returned") is False
            and item.get("attention_mask_argument") == "OMITTED"
            and item.get("padding_used") is False
            and item.get("absolute_tolerance") == 1e-7
            and item.get("relative_tolerance") == 1e-7
            and _finite_number(item.get("max_absolute_difference"))
            and float(item.get("max_absolute_difference", math.inf)) <= 1e-7
            and item.get("equivalent") is True
            and normalization.get("sample_count") == sample_count
            and normalization.get("input_dtype") == "float32"
            and normalization.get("normalization_epsilon") == 1e-7
            and normalization.get("pre_all_finite") is True
            and normalization.get("post_all_finite") is True
            and all(_finite_real(normalization.get(name)) for name in numeric_fields)
            and float(normalization.get("pre_population_variance", 0.0)) > 0.0
            and float(normalization.get("pre_population_std", 0.0)) > 0.0
            and float(normalization.get("post_population_variance", 0.0)) > 0.0
            and float(normalization.get("post_population_std", 0.0)) > 0.0
        )
    runtime_estimate_expected: dict[str, float] | None = None
    model_load_wall = runtime.get("model_load_wall_seconds")
    longest_wall = longest.get("wall_seconds")
    if _finite_number(model_load_wall, positive=True) and _finite_number(
        longest_wall, positive=True
    ):
        runtime_estimate_expected = conservative_runtime_estimate(
            float(cast(int | float, model_load_wall)),
            float(cast(int | float, longest_wall)),
        )
    project_total_bytes = storage.get("project_root_total_bytes")
    unclassified_other_bytes = storage.get("unclassified_other_bytes")
    actual_free_bytes = storage.get("actual_free_bytes")
    project_total_int = _int_or_negative(project_total_bytes)
    unclassified_other_int = _int_or_negative(unclassified_other_bytes)
    actual_free_int = _int_or_negative(actual_free_bytes)

    checks = {
        "preflight_pipeline_completed": (
            evidence.get("config_errors") == []
            and evidence.get("runtime_error") is None
            and runtime.get("status") == "PASS"
            and _aware_iso(evidence.get("audited_at_utc"))
            and _aware_iso(storage.get("audited_at_utc"))
        ),
        "identity_and_non_hash_policy": (
            evidence.get("report_schema_version") == PREFLIGHT_REPORT_SCHEMA_VERSION
            and evidence.get("task_id") == "M6A-PUBLIC-001"
            and evidence.get("integrity_policy") == "NON_HASH_AUDIT"
            and evidence.get("cryptographic_integrity_claim") is False
            and not find_forbidden_fields(evidence)
            and not nonfinite_numeric_paths(evidence)
        ),
        "longest_passage_from_lightweight_manifest": (
            manifest.get("status") == "PASS"
            and manifest.get("passage_count") == EXPECTED_PASSAGE_COUNT
            and _mapping(manifest.get("longest")).get("sample_id")
            == EXPECTED_LONGEST_SAMPLE
            and _mapping(manifest.get("longest")).get("synthetic_sample_count")
            == EXPECTED_SYNTHETIC_SAMPLE_COUNT
            and manifest.get("real_audio_read") is False
        ),
        "storage_category_identity_and_bytes": (
            category_identity_ok
            and storage.get("measurement")
            == "REGULAR_FILE_BYTES_WITHOUT_FOLLOWING_SYMLINKS"
            and storage.get("selected_category_total_bytes") == category_total
            and _nonnegative_int(project_total_bytes)
            and project_total_int >= category_total
            and _nonnegative_int(unclassified_other_bytes)
            and unclassified_other_int == project_total_int - category_total
        ),
        "free_space_at_least_500gb": (
            _nonnegative_int(actual_free_bytes)
            and actual_free_int >= 500_000_000_000
        ),
        "data_cache_plus_new_strictly_below_500gb": (
            isinstance(data_bytes, int)
            and isinstance(cache_bytes, int)
            and isinstance(estimated_new, int)
            and storage.get("data_cache_plus_estimated_new_bytes")
            == data_bytes + cache_bytes + estimated_new
            and data_bytes + cache_bytes + estimated_new < 500_000_000_000
        ),
        "offline_frozen_model_security": (
            runtime.get("model_id") == "facebook/wav2vec2-base"
            and runtime.get("revision_label") == "main"
            and runtime.get("local_files_only") is True
            and runtime.get("trust_remote_code") is False
            and runtime.get("repository_custom_code_executed") is False
            and runtime.get("weights_only") is True
            and runtime.get("tensor_only") is True
            and runtime.get("download_attempted") is False
            and runtime.get("model_eval") is True
            and runtime.get("inference_mode") is True
            and runtime.get("parameter_requires_grad_count") == 0
            and loading_info
            == {
                "missing_keys": [],
                "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
                "mismatched_keys": [],
                "error_msgs": [],
            }
        ),
        "remote_preprocessor_config_semantic_audit": (
            mirror_audit.get("status") == "PASS"
            and _aware_iso(mirror_audit.get("audited_at_utc"))
            and mirror_audit.get("source_endpoint") == PREPROCESSOR_SOURCE_ENDPOINT
            and mirror_audit.get("filename") == PREPROCESSOR_FILENAME
            and isinstance(mirror_probes, list)
            and len(mirror_probes) == 2
            and _mapping(mirror_probes[0]).get("endpoint")
            == "https://mirrors.tuna.tsinghua.edu.cn"
            and _mapping(mirror_probes[0]).get("http_status") == 404
            and _mapping(mirror_probes[0]).get("semantic_match") is False
            and _mapping(mirror_probes[0]).get("proxy_used") is False
            and _mapping(mirror_probes[1]).get("endpoint")
            == PREPROCESSOR_SOURCE_ENDPOINT
            and _mapping(mirror_probes[1]).get("http_status") == 200
            and _mapping(mirror_probes[1]).get("semantic_match") is True
            and _mapping(mirror_probes[1]).get("proxy_used") is False
            and mirror_audit.get("http_status") == 200
            and mirror_audit.get("mirror_body_bytes") == 159
            and mirror_audit.get("cache_bytes") == 159
            and _aware_iso(mirror_audit.get("cache_modified_at_utc"))
            and mirror_audit.get("mirror_semantic_fields")
            == PREPROCESSOR_SEMANTICS
            and mirror_audit.get("remote_only") is True
            and mirror_audit.get("cache_write_performed") is False
            and mirror_audit.get("network_body_persisted") is False
            and mirror_audit.get("proxy_used") is False
            and preprocessor_audit.get("status") == "PASS"
            and preprocessor_audit.get("filename") == PREPROCESSOR_FILENAME
            and preprocessor_audit.get("bytes") == 159
            and _aware_iso(preprocessor_audit.get("modified_at_utc"))
            and preprocessor_audit.get("semantic_fields")
            == PREPROCESSOR_SEMANTICS
            and preprocessor_audit.get("missing_fields") == []
            and preprocessor_audit.get("unexpected_fields") == []
            and preprocessor_audit.get("remote_only") is True
            and preprocessor_audit.get("issues") == []
        ),
        "passage_wise_feature_extractor_equivalence": (
            extractor_semantics == PREPROCESSOR_SEMANTICS
            and equivalence.get("status") == "PASS"
            and equivalence_is_valid(warmup_equivalence, 16_000)
            and equivalence_is_valid(
                longest_equivalence, EXPECTED_SYNTHETIC_SAMPLE_COUNT
            )
            and runtime.get("passage_wise_normalization_applied") is True
            and runtime.get("cross_passage_statistics_used") is False
            and runtime.get("train_fitted_preprocessing_statistics_used") is False
            and runtime.get("attention_mask_argument") == "OMITTED"
            and runtime.get("attention_mask_tensor_created") is False
            and runtime.get("raw_input_canary_history")
            == "NOT_RUN_NO_SUPERSEDED_PROVENANCE_CREATED"
        ),
        "synthetic_input_only": (
            runtime.get("input_source")
            == "DETERMINISTIC_IN_MEMORY_FINITE_MONO_NO_FILE_PATH"
            and runtime.get("input_path") is None
            and runtime.get("real_audio_read") is False
            and runtime.get("real_edf_read") is False
            and runtime.get("synthetic_input_all_finite") is True
            and runtime.get("preprocessed_input_all_finite") is True
            and runtime.get("synthetic_input_channels") == 1
        ),
        "one_second_warmup": (
            warmup.get("status") == "PASS"
            and warmup.get("sample_count") == 16_000
            and _finite_number(warmup.get("wall_seconds"), positive=True)
        ),
        "longest_passage_global_forward": (
            longest.get("status") == "PASS"
            and longest.get("batch_size") == 1
            and longest.get("sample_count") == EXPECTED_SYNTHETIC_SAMPLE_COUNT
            and longest.get("frame_count") == EXPECTED_FRAME_COUNT
            and longest.get("layer_keys") == list(EXPECTED_LAYER_KEYS)
            and isinstance(layer_shapes, list)
            and layer_shapes == [expected_shape] * 13
            and longest.get("all_finite") is True
            and longest.get("attention_scope")
            == "GLOBAL_WITHIN_ONE_SYNTHETIC_PASSAGE"
            and longest.get("chunked_or_windowed_approximation_used") is False
            and longest.get("oom") is False
            and _finite_number(longest.get("wall_seconds"), positive=True)
            and _positive_int(longest.get("cuda_peak_allocated_bytes"))
            and _positive_int(longest.get("cuda_peak_reserved_bytes"))
        ),
        "conservative_runtime_bounds": (
            runtime_estimate_expected is not None
            and all(
                math.isclose(
                    float(estimate.get(key, math.nan)), value, rel_tol=0.0, abs_tol=1e-9
                )
                for key, value in runtime_estimate_expected.items()
            )
            and float(estimate.get("model_load_upper_bound_seconds", -1))
            >= float(runtime.get("model_load_wall_seconds", math.inf))
            and float(estimate.get("longest_forward_upper_bound_seconds", -1))
            >= float(longest.get("wall_seconds", math.inf))
            and float(estimate.get("single_invocation_upper_bound_gpu_hours", math.inf))
            < 2.0
            and float(estimate.get("total_40_passage_upper_bound_gpu_hours", math.inf))
            <= 4.0
            and estimate.get("duration_linear_scaling_assumed") is False
        ),
        "recoverable_one_passage_checkpoint_design": (
            checkpoint.get("execution_unit") == "ONE_PASSAGE_PER_INVOCATION"
            and checkpoint.get("successful_passage_checkpoint_required") is True
            and checkpoint.get("atomic_final_rename_required") is True
            and checkpoint.get("resume_skips_only_validated_final_checkpoint") is True
            and checkpoint.get("partial_preserved_on_failure") is True
            and checkpoint.get("checkpoint_before_two_hour_limit") is True
        ),
        "execution_and_claims_remain_blocked": (
            evidence.get("g4_execution_authorized") is False
            and evidence.get("new_real_edf_read") is False
            and evidence.get("new_real_audio_read") is False
            and evidence.get("real_feature_extraction_run") is False
            and evidence.get("ridge_run") is False
            and evidence.get("null_run") is False
            and evidence.get("metric_run") is False
            and evidence.get("scientific_result_claimed") is False
            and evidence.get("exchange_candidate_created") is False
        ),
    }
    return checks


def finalize_preflight_report(
    evidence: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    checks = validate_preflight_evidence(evidence, config)
    failed = [name for name, passed in checks.items() if passed is not True]
    report = dict(evidence)
    report["required_checks"] = checks
    report["failed_checks"] = failed
    report["status"] = PREFLIGHT_STATUS if not failed else "FAIL"
    report["current_candidate"] = not failed
    return report


__all__ = [
    "EXPECTED_FRAME_COUNT",
    "EXPECTED_LONGEST_DURATION_SECONDS",
    "EXPECTED_LONGEST_SAMPLE",
    "EXPECTED_SYNTHETIC_SAMPLE_COUNT",
    "PREFLIGHT_REPORT_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "PREFLIGHT_STATUS",
    "audit_longest_passage_manifest",
    "conservative_runtime_estimate",
    "finalize_preflight_report",
    "validate_preflight_config",
    "validate_preflight_evidence",
]
