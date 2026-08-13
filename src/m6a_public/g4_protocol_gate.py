from __future__ import annotations

import csv
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from m6a_public.audio_context_gate import EXPECTED_LAYER_KEYS, nonfinite_numeric_paths
from m6a_public.config_gate import find_forbidden_fields
from m6a_public.g3_single_recording_gate import load_strict_json_object
from m6a_public.wav2vec2_preprocessing import (
    WAV2VEC2_INPUT_PREPROCESSING_CONTRACT,
)


G3_ACCEPTED_STATUS = "G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY"
G4_STATUS = "G4_PROTOCOL_AMENDMENT_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
G4_SCHEMA_VERSION = "m6a-g4-protocol-candidate-v1"
G4_REPORT_SCHEMA_VERSION = "m6a-g4-protocol-gate-report-v1"
EXPECTED_PARTICIPANT = "sub-SD012"
EXPECTED_SESSION = "ses-02"
EXPECTED_RECORDING = "sub-SD012_ses-02_task-PassiveListen"
EXPECTED_SPLIT_COUNTS = {"train": 24, "validation": 8, "test": 8}
EXPECTED_BLOCK_ASSIGNMENTS = {
    "block-01": "train",
    "block-02": "train",
    "block-03": "validation",
    "block-04": "test",
    "block-05": "train",
}
EXPECTED_LAGS_SECONDS = [
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
]
EXPECTED_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
EXPECTED_SUBBAND_WEIGHTS = [1.0 / 6.0] * 6
EXPECTED_TEST_DERANGEMENT_COUNT = 14_833
EXPECTED_SMOKE_PERMUTATIONS = 20
EXPECTED_MANIFEST_SECONDS = 1532.45596371882
EXPECTED_CORE_TENSOR_BYTES = 3_151_044_252


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _has_exact_keys(value: dict[str, Any], expected: set[str]) -> bool:
    return set(value) == expected


def _same_number(value: Any, expected: float, tolerance: float = 1e-12) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tolerance)
    )


def reused_svd_ridge_path(
    features: np.ndarray,
    targets: np.ndarray,
    alphas: list[float] | tuple[float, ...],
) -> dict[float, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("ridge features and multi-target matrix must be compatible 2D arrays")
    if x.shape[1] < 1 or y.shape[1] < 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("ridge inputs must be finite and nonempty")
    if not alphas or any(
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
        for alpha in alphas
    ):
        raise ValueError("ridge alpha grid must be finite, positive, and nonempty")
    u, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    projected_targets = u.T @ y
    return {
        float(alpha): vt.T
        @ (
            (singular_values / (singular_values * singular_values + float(alpha)))[
                :, np.newaxis
            ]
            * projected_targets
        )
        for alpha in alphas
    }


def audit_pca20_train_matrix(train_matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(train_matrix, dtype=np.float64)
    issues: list[str] = []
    if matrix.ndim != 2:
        return {
            "status": "NOT_ESTIMABLE",
            "issues": ["log-mel PCA train matrix must be two-dimensional"],
        }
    sample_count, feature_count = matrix.shape
    if sample_count < 20:
        issues.append("log-mel PCA requires at least 20 train samples")
    if feature_count != 80:
        issues.append("log-mel PCA requires exactly 80 input bins")
    if not np.all(np.isfinite(matrix)):
        issues.append("log-mel PCA train matrix must be finite")

    singular_values: np.ndarray | None = None
    rank_tolerance: float | None = None
    numeric_rank: int | None = None
    if not issues:
        singular_values = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
        if singular_values.size < 20 or not np.all(np.isfinite(singular_values[:20])):
            issues.append("first 20 log-mel singular values must be finite")
        else:
            rank_tolerance = (
                max(sample_count, feature_count)
                * np.finfo(np.float64).eps
                * float(singular_values[0])
            )
            numeric_rank = int(np.count_nonzero(singular_values > rank_tolerance))
            if numeric_rank < 20 or float(singular_values[19]) <= rank_tolerance:
                issues.append("log-mel PCA numeric rank is below the frozen 20 dimensions")

    return {
        "status": "PASS" if not issues else "NOT_ESTIMABLE",
        "sample_count": sample_count,
        "feature_count": feature_count,
        "numeric_rank": numeric_rank,
        "rank_tolerance": rank_tolerance,
        "twentieth_singular_value": (
            float(singular_values[19])
            if singular_values is not None and singular_values.size >= 20
            else None
        ),
        "automatic_dimension_reduction_used": False,
        "issues": issues,
    }


def evaluate_g4_resource_preflight(
    *,
    actual_free_bytes: int,
    estimated_new_bytes: int,
    project_data_and_cache_bytes: int,
    estimated_single_invocation_gpu_hours: float,
    estimated_total_gpu_hours: float,
    continuous_formal_gpu_hours: float,
) -> dict[str, Any]:
    numeric_values = (
        actual_free_bytes,
        estimated_new_bytes,
        project_data_and_cache_bytes,
        estimated_single_invocation_gpu_hours,
        estimated_total_gpu_hours,
        continuous_formal_gpu_hours,
    )
    finite_nonnegative = all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
        for value in numeric_values
    )
    checks = {
        "all_inputs_finite_nonnegative": finite_nonnegative,
        "actual_free_bytes_at_least_500gb": (
            finite_nonnegative and actual_free_bytes >= 500_000_000_000
        ),
        "estimated_new_bytes_at_most_20gb": (
            finite_nonnegative and estimated_new_bytes <= 20_000_000_000
        ),
        "project_data_cache_plus_new_strictly_below_500gb": (
            finite_nonnegative
            and project_data_and_cache_bytes + estimated_new_bytes
            < 500_000_000_000
        ),
        "single_gpu_invocation_strictly_below_2h": (
            finite_nonnegative and estimated_single_invocation_gpu_hours < 2.0
        ),
        "total_gpu_estimate_at_most_4h": (
            finite_nonnegative
            and estimated_total_gpu_hours >= estimated_single_invocation_gpu_hours
            and estimated_total_gpu_hours <= 4.0
        ),
        "continuous_formal_gpu_not_above_24h_report_threshold": (
            finite_nonnegative and continuous_formal_gpu_hours <= 24.0
        ),
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    return {"status": "PASS" if not failed else "FAIL", "checks": checks, "failed": failed}


def validate_g4_protocol(
    config: dict[str, Any], schema: dict[str, Any]
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

    if find_forbidden_fields(config):
        errors.append("protocol contains forbidden integrity fields")
    if nonfinite_numeric_paths(config):
        errors.append("protocol contains non-finite numeric values")
    if config.get("schema_version") != G4_SCHEMA_VERSION:
        errors.append("G4 protocol schema version drifted")
    if config.get("task_id") != "M6A-PUBLIC-001" or config.get("status") != G4_STATUS:
        errors.append("G4 protocol identity or candidate status drifted")
    if (
        config.get("coordinator_review") != "PENDING"
        or config.get("reviewed_on") is not None
        or config.get("scientific_result_claimed") is not False
    ):
        errors.append("G4 protocol amendment review or claim boundary drifted")
    if config.get("integrity_policy") != "NON_HASH_AUDIT":
        errors.append("G4 protocol integrity policy must remain non-hash")

    if _mapping(config.get("prior_protocol_acceptance")) != {
        "status": "G4_PROTOCOL_COORDINATOR_ACCEPTED",
        "coordinator_review": "ACCEPT",
        "reviewed_on": "2026-08-13",
        "report": "reports/g4_protocol_candidate_20260813_v3.json",
        "preserved_as_historical_provenance": True,
    }:
        errors.append("prior accepted G4 protocol provenance drifted")
    if _mapping(config.get("amendment")) != {
        "scope": "WAV2VEC2_PREPROCESSING_INPUT_CONTRACT_ONLY",
        "reason": "PREPROCESSOR_FEATURE_EXTRACTOR_CONTRACT_WAS_MISSING",
        "g3_raw_input_representation_status": "ENGINEERING_SHAPE_TIME_EVIDENCE_ONLY",
        "g3_raw_input_representation_reuse_for_g4_scientific_baseline_allowed": False,
        "g3_recompute_required": False,
    }:
        errors.append("G4 preprocessing amendment scope or G3 reuse boundary drifted")

    dependency = _mapping(config.get("dependency_gates"))
    if dependency != {
        "g3_status": G3_ACCEPTED_STATUS,
        "g3_coordinator_review": "ACCEPT",
        "g3_reviewed_on": "2026-08-13",
        "g3_scientific_result_claimed": False,
        "final_embargo_status": "FINAL_EMBARGO_COORDINATOR_ACCEPTED",
        "split_status": "BASELINE_FINAL_COORDINATOR_ACCEPTED",
        "neural_method_status": "METHOD_FROZEN_COORDINATOR_ACCEPTED",
        "anatomy_status": "ANATOMY_MAPPING_NOT_READY",
    }:
        errors.append("G4 dependency gates are not exact")

    scope = _mapping(config.get("scope"))
    if (
        scope.get("split_csv") != "reports/ds004703_primary_split.csv"
        or scope.get("participant_id") != EXPECTED_PARTICIPANT
        or scope.get("session_id") != EXPECTED_SESSION
        or scope.get("excluded_sessions") != ["ses-01"]
        or scope.get("recording_id") != EXPECTED_RECORDING
        or scope.get("expected_passage_counts") != EXPECTED_SPLIT_COUNTS
        or scope.get("expected_total_passages") != 40
        or scope.get("expected_block_assignments") != EXPECTED_BLOCK_ASSIGNMENTS
        or scope.get("unique_stimulus_per_passage_required") is not True
        or scope.get("unique_audio_file_per_passage_required") is not True
        or scope.get("result_based_selection_allowed") is not False
        or scope.get("other_subjects_allowed") is not False
        or scope.get("other_sessions_allowed") is not False
        or scope.get("other_recordings_allowed") is not False
    ):
        errors.append("G4 scope is not exact, metadata-only, and single-recording")

    time = _mapping(config.get("time_alignment"))
    lags = time.get("lags_seconds")
    lags_are_exact = (
        isinstance(lags, list)
        and len(lags) == len(EXPECTED_LAGS_SECONDS)
        and all(
            _same_number(actual, expected)
            for actual, expected in zip(lags, EXPECTED_LAGS_SECONDS, strict=True)
        )
    )
    if (
        time.get("feature_grid") != "RECORDING_ORIGIN_K_OVER_50_SECONDS"
        or time.get("feature_rate_hz") != 50
        or time.get("feature_timestamp_semantics") != "AUDIO_FEATURE_AT_T_SECONDS"
        or not lags_are_exact
        or time.get("lag_semantics")
        != "AUDIO_AT_T_PREDICTS_NEURAL_TARGET_AT_T_PLUS_LAG"
        or time.get("fractional_grid_lag_policy")
        != "SAMPLE_FINITE_FILTERED_NATIVE_NEURAL_POWER_AT_EXACT_T_PLUS_LAG_NO_FRAME_ROUNDING"
        or time.get("target_timestamp_semantics") != "EXPLICIT_T_PLUS_LAG_SECONDS"
        or time.get("target_sampling")
        != "LINEAR_BETWEEN_TWO_NATIVE_NEURAL_SAMPLES_NO_EXTRAPOLATION"
        or time.get("target_source")
        != "FINITE_SUPPORT_LOW_PASS_SMOOTHED_NATIVE_NEURAL_POWER"
        or time.get("common_frame_set")
        != "INTERSECTION_OF_ALL_FEATURE_COMPLETE_SUPPORT_AND_ALL_LAG_TARGET_COMPLETE_SUPPORT_WITHIN_EACH_PASSAGE"
        or time.get("same_feature_frames_for_all_lags") is not True
        or time.get("passage_boundary_crossing_allowed") is not False
        or time.get("split_boundary_crossing_allowed") is not False
        or time.get("padding_across_passage_allowed") is not False
        or time.get("extrapolation_allowed") is not False
        or time.get("incomplete_support_action") != "MASK_BEFORE_ANY_FIT_OR_METRIC"
    ):
        errors.append("G4 exact-lag and common-support semantics drifted")
    if not _same_number(0.05 * 50, 2.5) or float(0.05 * 50).is_integer():
        errors.append("0.05-second lag must remain explicitly non-integral on the 50 Hz grid")

    target = _mapping(config.get("target_transform"))
    if (
        target.get("fit_partition") != "train"
        or target.get("fit_scope")
        != "ONE_DEDUPLICATED_PHYSICAL_TIME_POPULATION_ACROSS_ALL_11_LAGS_AND_24_TRAIN_PASSAGES"
        or target.get("fit_population")
        != "UNION_OF_ALL_EXACT_T_PLUS_LAG_TARGET_TIMES_FROM_COMMON_X_TRAIN_FRAMES_DEDUPLICATED_BY_RECORDING_ORIGIN_10MS_INTEGER_TICK"
        or target.get("fit_unit") != "CHANNEL_BY_SUBBAND"
        or target.get("parameters_shared_across_all_lags") is not True
        or target.get("lag_specific_refit_allowed") is not False
        or target.get("validation_contributes_to_fit") is not False
        or target.get("test_contributes_to_fit") is not False
        or target.get("negative_power_policy")
        != "FAIL_IF_BELOW_MINUS_1E_12_ELSE_CLIP_TO_ZERO"
        or target.get("epsilon_formula")
        != "MAX_1E_30_AND_1E_6_TIMES_MEDIAN_POSITIVE_TRAIN_POWER_PER_CHANNEL_SUBBAND"
        or target.get("no_positive_train_power_action") != "FAIL"
        or target.get("log_formula")
        != "NATURAL_LOG_OF_CLIPPED_POWER_PLUS_TRAIN_ONLY_EPSILON"
        or target.get("center") != "TRAIN_MEAN_OF_LOG_POWER_PER_CHANNEL_SUBBAND"
        or target.get("scale")
        != "TRAIN_POPULATION_STANDARD_DEVIATION_DDOF_0_PER_CHANNEL_SUBBAND"
        or not _same_number(target.get("minimum_scale"), 1e-12)
        or target.get("nonfinite_action") != "FAIL"
        or target.get("apply_frozen_train_parameters_to")
        != ["train", "validation", "test"]
        or target.get("subband_standardization_precedes_aggregation") is not True
        or target.get("subband_weights") != EXPECTED_SUBBAND_WEIGHTS
        or not _same_number(sum(target.get("subband_weights", [])), 1.0)
        or target.get("aggregation")
        != "EQUAL_WEIGHT_MEAN_OF_SIX_STANDARDIZED_SUBBANDS_PER_CHANNEL"
    ):
        errors.append("train-only target transform or equal subband aggregation drifted")

    features = _mapping(config.get("features"))
    envelope = _mapping(features.get("amplitude_envelope"))
    log_mel = _mapping(features.get("log_mel"))
    wav2vec2 = _mapping(features.get("wav2vec2"))
    if (
        features.get("shared_statistics_frame_set")
        != "COMMON_SUPPORT_TRAIN_FRAMES_SHARED_ACROSS_ALL_LAGS"
        or features.get("validation_contributes_to_fit") is not False
        or features.get("test_contributes_to_fit") is not False
        or features.get("scale_definition") != "TRAIN_SAMPLE_STANDARD_DEVIATION_DDOF_1"
        or not _same_number(features.get("minimum_scale"), 1e-12)
        or not _has_exact_keys(
            envelope, {"source", "output_dimension", "standardization"}
        )
        or envelope.get("source") != "FROZEN_G3_HANN_WEIGHTED_ROOT_MEAN_SQUARE"
        or envelope.get("output_dimension") != 1
        or envelope.get("standardization") != "TRAIN_ONLY_PER_DIMENSION_ZSCORE"
        or not _has_exact_keys(
            log_mel,
            {
                "source",
                "input_dimension",
                "pre_pca_standardization",
                "pca_dimension",
                "pca_dimension_selection",
                "pca_fit_partition",
                "pca_solver",
                "component_order",
                "component_sign",
                "score_standardization",
                "minimum_train_samples",
                "minimum_numeric_rank",
                "singular_value_rank_tolerance",
                "first_20_singular_values_must_be_finite",
                "twentieth_singular_value_must_exceed_rank_tolerance",
                "automatic_dimension_reduction_allowed",
                "rank_failure_action",
            },
        )
        or log_mel.get("source") != "FROZEN_G3_RAW_LOG_MEL_80"
        or log_mel.get("input_dimension") != 80
        or log_mel.get("pre_pca_standardization")
        != "TRAIN_ONLY_PER_BIN_ZSCORE"
        or log_mel.get("pca_dimension") != 20
        or log_mel.get("pca_dimension_selection")
        != "FIXED_20_BEFORE_EXECUTION_NOT_VARIANCE_OR_RESULT_SELECTED"
        or log_mel.get("pca_fit_partition") != "train"
        or log_mel.get("pca_solver") != "NUMPY_FLOAT64_THIN_SVD"
        or log_mel.get("component_order") != "DESCENDING_SINGULAR_VALUE"
        or log_mel.get("component_sign")
        != "LARGEST_ABSOLUTE_LOADING_POSITIVE_TIE_LOWEST_FEATURE_INDEX"
        or log_mel.get("score_standardization")
        != "TRAIN_ONLY_PER_COMPONENT_ZSCORE"
        or log_mel.get("minimum_train_samples") != 20
        or log_mel.get("minimum_numeric_rank") != 20
        or log_mel.get("singular_value_rank_tolerance")
        != "MAX_N_SAMPLES_N_FEATURES_TIMES_FLOAT64_EPSILON_TIMES_LARGEST_SINGULAR_VALUE"
        or log_mel.get("first_20_singular_values_must_be_finite") is not True
        or log_mel.get("twentieth_singular_value_must_exceed_rank_tolerance")
        is not True
        or log_mel.get("automatic_dimension_reduction_allowed") is not False
        or log_mel.get("rank_failure_action")
        != "LOG_MEL_VARIANT_NOT_ESTIMABLE_NO_AUTO_REDUCTION"
        or not _has_exact_keys(
            wav2vec2,
            {
                "source",
                "layer_keys",
                "input_dimension_per_layer",
                "standardization",
                "input_preprocessing",
                "pca_allowed",
                "test_layer_selection_allowed",
                "test_lag_selection_allowed",
                "test_alpha_selection_allowed",
            },
        )
        or wav2vec2.get("source")
        != "FROZEN_PROJECTED_PLUS_12_TRANSFORMER_LAYERS"
        or wav2vec2.get("layer_keys") != list(EXPECTED_LAYER_KEYS)
        or wav2vec2.get("input_dimension_per_layer") != 768
        or wav2vec2.get("standardization")
        != "TRAIN_ONLY_PER_LAYER_PER_DIMENSION_ZSCORE"
        or wav2vec2.get("input_preprocessing")
        != WAV2VEC2_INPUT_PREPROCESSING_CONTRACT
        or wav2vec2.get("pca_allowed") is not False
        or wav2vec2.get("test_layer_selection_allowed") is not False
        or wav2vec2.get("test_lag_selection_allowed") is not False
        or wav2vec2.get("test_alpha_selection_allowed") is not False
    ):
        errors.append("G4 feature definitions or train-only standardization drifted")

    ridge = _mapping(config.get("ridge"))
    if (
        ridge.get("solver") != "FLOAT64_THIN_SVD_RIDGE_CLOSED_FORM"
        or ridge.get("formula")
        != "BETA_EQUALS_V_TIMES_DIAG_S_OVER_S_SQUARED_PLUS_ALPHA_TIMES_U_TRANSPOSE_Y"
        or ridge.get("fit_intercept") is not False
        or ridge.get("numeric_dtype") != "float64"
        or ridge.get("feature_variant_count") != 15
        or ridge.get("decomposition_unit")
        != "ONE_FLOAT64_THIN_SVD_PER_FEATURE_VARIANT_PER_FIT_PARTITION"
        or ridge.get("train_decomposition_count_max") != 15
        or ridge.get("train_plus_validation_decomposition_count_max") != 15
        or ridge.get("total_decomposition_count_max") != 30
        or ridge.get("decomposition_reused_across")
        != "11_LAGS_BY_36_ELECTRODES_BY_6_ALPHAS_AS_MULTI_TARGET_RIDGE_PATH"
        or ridge.get("per_cell_decomposition_allowed") is not False
        or ridge.get("reuse_equivalence_required")
        != "SYNTHETIC_REUSED_SVD_COEFFICIENTS_AND_PREDICTIONS_MATCH_DIRECT_RIDGE_WITH_ABS_RTOL_1E_10"
        or ridge.get("alpha_grid") != EXPECTED_ALPHA_GRID
        or ridge.get("fit_partition") != "train"
        or ridge.get("alpha_selection_partition") != "validation"
        or ridge.get("alpha_selection_metric") != "PEARSON_R"
        or ridge.get("alpha_selection_unit") != "FEATURE_VARIANT_BY_LAG_BY_ELECTRODE"
        or not _same_number(ridge.get("alpha_tie_tolerance"), 1e-12)
        or ridge.get("alpha_tie_break") != "SMALLEST_ALPHA"
        or ridge.get("all_validation_scores_not_estimable_action")
        != "CELL_NOT_ESTIMABLE_NO_TEST_METRIC"
        or ridge.get("final_refit_partition_after_alpha_lock")
        != ["train", "validation"]
        or ridge.get("final_refit_uses_locked_alpha") is not True
        or ridge.get("transform_parameters_remain_train_only_during_final_refit")
        is not True
        or ridge.get("transform_refit_after_validation_allowed") is not False
        or ridge.get("test_evaluation_count") != 1
        or ridge.get("test_access_before_alpha_lock_allowed") is not False
        or ridge.get("layer_selection_allowed") is not False
        or ridge.get("lag_selection_allowed") is not False
    ):
        errors.append("ridge solver, alpha selection, or test-once policy drifted")

    nulls = _mapping(config.get("nulls"))
    matrix = _mapping(nulls.get("applicability_matrix"))
    wav_nulls = _mapping(matrix.get("wav2vec2"))
    envelope_nulls = _mapping(matrix.get("amplitude_envelope"))
    log_mel_nulls = _mapping(matrix.get("log_mel_pca"))
    derangement = _mapping(nulls.get("stimulus_derangement"))
    circular = _mapping(nulls.get("acoustic_circular_shift"))
    if (
        nulls.get("smoke_permutations") != EXPECTED_SMOKE_PERMUTATIONS
        or nulls.get("scientific_significance_claim_allowed") is not False
        or nulls.get("primary_null_source_all_families")
        != "STIMULUS_DERANGEMENT"
        or nulls.get("shared_mapping_scope")
        != "SAME_PASSAGE_DERANGEMENT_MAPPING_FOR_ALL_FEATURE_LAYER_LAG_ELECTRODE_CELLS_WITHIN_EACH_PERMUTATION"
        or nulls.get("post_result_primary_null_substitution_allowed") is not False
        or not _has_exact_keys(
            matrix, {"wav2vec2", "amplitude_envelope", "log_mel_pca"}
        )
        or not all(
            _has_exact_keys(
                _mapping(matrix.get(name)),
                {"stimulus_derangement", "within_passage_circular_shift"},
            )
            for name in ("wav2vec2", "amplitude_envelope", "log_mel_pca")
        )
        or wav_nulls.get("stimulus_derangement") != "PRIMARY_SMOKE_NULL"
        or wav_nulls.get("within_passage_circular_shift")
        != "NOT_APPLICABLE_GLOBAL_WITHIN_PASSAGE_TRANSFORMER_CONTEXT"
        or envelope_nulls
        != {
            "stimulus_derangement": "APPLICABLE",
            "within_passage_circular_shift": "SECONDARY_MECHANICAL_NULL",
        }
        or log_mel_nulls
        != {
            "stimulus_derangement": "APPLICABLE",
            "within_passage_circular_shift": "SECONDARY_MECHANICAL_NULL",
        }
        or derangement.get("partition") != "test"
        or derangement.get("expected_passages") != 8
        or derangement.get("identity_mapping_allowed") is not False
        or derangement.get("within_split_only") is not True
        or derangement.get("bijection_required") is not True
        or derangement.get("duration_stratum")
        != "ALL_8_PREDECLARED_TEST_PASSAGES_ONE_STRATUM"
        or derangement.get("enumerated_derangement_count")
        != EXPECTED_TEST_DERANGEMENT_COUNT
        or derangement.get("algorithm")
        != "ENUMERATE_ALL_14833_FIXED_POINT_FREE_BIJECTIONS_THEN_UNIFORMLY_SAMPLE_20_WITHOUT_REPLACEMENT_USING_NUMPY_PCG64_SEED_20260813"
        or derangement.get("random_generator") != "NUMPY_PCG64"
        or derangement.get("random_seed") != 20260813
        or derangement.get("sample_without_replacement") is not True
        or derangement.get("minimum_unique_derangements") != 20
        or derangement.get("mapping_direction")
        != "DONOR_LOCKED_TEST_PREDICTION_TO_TARGET_PASSAGE_NEURAL_TARGET"
        or derangement.get("mapping_object")
        != "DONOR_LOCKED_TEST_PREDICTION_SEQUENCE"
        or derangement.get("donor_to_target_time_mapping")
        != "WITHIN_PAIRED_PASSAGES_LINEAR_NORMALIZED_COMPLETE_SUPPORT_PHASE_TO_TARGET_COMMON_FRAMES_NO_EXTRAPOLATION"
        or derangement.get("cross_passage_concatenation_before_mapping_allowed")
        is not False
        or derangement.get("model_rerun_allowed") is not False
        or derangement.get("transform_refit_allowed") is not False
        or derangement.get("pca_refit_allowed") is not False
        or derangement.get("ridge_refit_allowed") is not False
        or derangement.get("alpha_reselection_allowed") is not False
        or derangement.get("test_neural_target_recomputed_or_refit_allowed")
        is not False
        or derangement.get("target_passage_contribution")
        != "ONE_PASSAGE_LEVEL_METRIC_PER_TARGET_PASSAGE_EQUAL_WEIGHT_IN_CELL_AGGREGATION"
        or derangement.get("observed_and_null_metric_aggregation_identical") is not True
        or derangement.get("duration_mismatch_role")
        != "REPORT_ONLY_NOT_USED_FOR_SELECTION"
        or derangement.get("singleton_or_insufficient_derangements_action") != "FAIL"
        or derangement.get("neural_or_model_result_used_for_assignment") is not False
        or not _has_exact_keys(
            derangement,
            {
                "partition",
                "expected_passages",
                "identity_mapping_allowed",
                "within_split_only",
                "bijection_required",
                "duration_stratum",
                "enumerated_derangement_count",
                "algorithm",
                "random_generator",
                "random_seed",
                "sample_without_replacement",
                "minimum_unique_derangements",
                "mapping_direction",
                "mapping_object",
                "donor_to_target_time_mapping",
                "cross_passage_concatenation_before_mapping_allowed",
                "model_rerun_allowed",
                "transform_refit_allowed",
                "pca_refit_allowed",
                "ridge_refit_allowed",
                "alpha_reselection_allowed",
                "test_neural_target_recomputed_or_refit_allowed",
                "target_passage_contribution",
                "observed_and_null_metric_aggregation_identical",
                "duration_mismatch_role",
                "singleton_or_insufficient_derangements_action",
                "neural_or_model_result_used_for_assignment",
            },
        )
        or circular.get("applicable_features")
        != ["amplitude_envelope", "log_mel_pca"]
        or circular.get("wav2vec2_allowed") is not False
        or circular.get("grid")
        != "ALIGNED_50_HZ_FEATURE_FRAMES_WITHIN_EACH_PASSAGE"
        or not _same_number(circular.get("minimum_shift_seconds"), 2.0)
        or circular.get("minimum_shift_frames") != 100
        or circular.get("allowed_offset_frames")
        != "INTEGER_OFFSETS_FROM_100_THROUGH_N_MINUS_100_INCLUSIVE"
        or circular.get("generator")
        != "NUMPY_PCG64_SEED_20260813_SAMPLE_WITHOUT_REPLACEMENT_PER_PASSAGE"
        or circular.get("insufficient_offsets_action") != "FAIL"
        or circular.get("cross_passage_or_split_allowed") is not False
        or circular.get("receptive_field_exceedance_claimed") is not False
        or circular.get("role")
        != "SECONDARY_MECHANICAL_DIAGNOSTIC_NOT_PRIMARY_AND_CANNOT_REPLACE_STIMULUS_DERANGEMENT"
        or not _has_exact_keys(
            circular,
            {
                "applicable_features",
                "wav2vec2_allowed",
                "grid",
                "minimum_shift_seconds",
                "minimum_shift_frames",
                "allowed_offset_frames",
                "generator",
                "insufficient_offsets_action",
                "cross_passage_or_split_allowed",
                "receptive_field_exceedance_claimed",
                "role",
            },
        )
    ):
        errors.append("null applicability, derangement, or acoustic shift policy drifted")

    correction = _mapping(config.get("multiple_comparison"))
    permutation_p = _mapping(correction.get("permutation_p"))
    if (
        correction.get("method") != "ONE_SIDED_MAX_STATISTIC"
        or correction.get("primary_metric") != "PEARSON_R"
        or correction.get("wav2vec2_family")
        != "36_ELECTRODES_BY_13_LAYERS_BY_11_LAGS"
        or correction.get("wav2vec2_family_cell_count") != 36 * 13 * 11
        or correction.get("acoustic_family")
        != "36_ELECTRODES_BY_2_FIXED_ACOUSTIC_CONTROLS_BY_11_LAGS"
        or correction.get("acoustic_family_cell_count") != 36 * 2 * 11
        or correction.get("permutation_statistic")
        != "MAXIMUM_PEARSON_R_ACROSS_ALL_CELLS_IN_THE_FIXED_FAMILY"
        or correction.get("shared_derangement_mapping_across_family_cells") is not True
        or correction.get("all_declared_cells_must_be_attempted_and_status_recorded")
        is not True
        or correction.get("effective_family_rule")
        != "FIXED_INTERSECTION_OF_CELLS_ESTIMABLE_IN_OBSERVED_AND_ALL_20_NULL_PERMUTATIONS"
        or correction.get("effective_family_fixed_across_permutations") is not True
        or correction.get("required_family_audit_fields")
        != [
            "declared_cell_ids",
            "effective_cell_ids",
            "excluded_cell_ids_with_reasons",
        ]
        or correction.get("empty_effective_family_action") != "FAIL"
        or correction.get("families_fixed_before_execution") is not True
        or correction.get("favorable_family_selection_allowed") is not False
        or correction.get("fdr_allowed") is not False
        or permutation_p
        != {
            "formula": "(1 + count(max_null >= observed)) / (20 + 1)",
            "numerator_pseudocount": 1,
            "null_comparison": "MAX_NULL_GREATER_THAN_OR_EQUAL_TO_OBSERVED",
            "permutation_count": 20,
            "denominator": 21,
        }
        or not _same_number(correction.get("minimum_attainable_p"), 1.0 / 21.0)
        or correction.get("smoke_interpretation")
        != "MECHANICAL_NULL_AND_CORRECTION_VALIDATION_ONLY_NOT_STABLE_SIGNIFICANCE"
    ):
        errors.append("max-statistic family or 20-permutation interpretation drifted")

    metrics = _mapping(config.get("metrics"))
    if (
        metrics.get("primary") != "TEST_ELECTRODE_LEVEL_PEARSON_R"
        or metrics.get("secondary") != "TEST_ELECTRODE_LEVEL_R2"
        or metrics.get("metric_computation_unit")
        != "TARGET_PASSAGE_BY_ELECTRODE_BY_FEATURE_LAYER_BY_LAG"
        or metrics.get("pearson_not_estimable_if")
        != "N_LT_3_OR_TARGET_SD_LE_1E_12_OR_PREDICTION_SD_LE_1E_12_OR_NONFINITE"
        or metrics.get("r2_not_estimable_if")
        != "N_LT_2_OR_TARGET_SST_LE_1E_12_OR_NONFINITE"
        or metrics.get("not_estimable_representation")
        != "NULL_VALUE_WITH_STATUS_NOT_ESTIMABLE_AND_REASON"
        or metrics.get("cell_pearson_aggregation")
        != "TANH_EQUAL_WEIGHT_MEAN_ATANH_OF_8_TARGET_PASSAGE_PEARSON_R_VALUES_CLIPPED_TO_PLUS_MINUS_1_MINUS_1E_12"
        or metrics.get("cell_r2_aggregation")
        != "EQUAL_WEIGHT_MEAN_OF_8_TARGET_PASSAGE_R2_VALUES"
        or metrics.get("cell_estimable_rule")
        != "ALL_8_TARGET_PASSAGE_METRICS_MUST_BE_ESTIMABLE_ELSE_CELL_NOT_ESTIMABLE_WITH_REASON"
        or metrics.get("observed_and_null_use_identical_passage_and_cell_aggregation")
        is not True
        or metrics.get("electrode_aggregation")
        != "MEDIAN_AND_IQR_OVER_ESTIMABLE_ELECTRODE_CELL_METRICS"
        or metrics.get("subject_summary")
        != "ONE_SUBJECT_ONE_RECORDING_DESCRIPTIVE_MEDIAN_IQR_AND_ESTIMABLE_ELECTRODE_COUNT_PER_FEATURE_LAYER_LAG"
        or metrics.get("generalization_claim")
        != "WITHIN_SUBJECT_WITHIN_SESSION_UNSEEN_STIMULUS_AND_BLOCK_ONLY"
        or metrics.get("model_vs_acoustic")
        != "PAIRED_ELECTRODE_DELTA_PEARSON_R_FOR_EACH_MODEL_LAYER_LAG_VERSUS_EACH_FIXED_ACOUSTIC_CONTROL_AT_SAME_LAG"
        or metrics.get("model_vs_acoustic_aggregation")
        != "DESCRIPTIVE_MEDIAN_IQR_NO_FAVORABLE_COMPARATOR_SELECTION"
        or metrics.get("model_vs_acoustic_inferential_p_value") != "NOT_COMPUTED"
        or metrics.get("model_vs_acoustic_correction_family")
        != "NONE_DESCRIPTIVE_ONLY"
        or metrics.get("region_summary") != "NOT_ESTIMABLE"
    ):
        errors.append("metric, aggregation, comparison, or anatomy boundary drifted")

    resources = _mapping(config.get("resources"))
    preflight = _mapping(resources.get("preflight_checks"))
    if (
        resources.get("remote_project_root") != "/home/fanyu/auditory_simulation_m6a"
        or resources.get("remote_only_outputs") is not True
        or resources.get("estimated_passages") != 40
        or not _same_number(
            resources.get("estimated_audio_seconds_from_manifest"),
            EXPECTED_MANIFEST_SECONDS,
        )
        or resources.get("estimated_50hz_frames_upper_bound") != 76_623
        or resources.get("estimated_core_tensor_bytes") != EXPECTED_CORE_TENSOR_BYTES
        or resources.get("estimated_new_remote_bytes_upper_bound") != 20_000_000_000
        or not _same_number(resources.get("estimated_total_gpu_hours_upper_bound"), 4.0)
        or not _same_number(resources.get("single_invocation_gpu_hours_hard_limit"), 2.0)
        or resources.get("gpu_execution_partitioning")
        != "RECOVERABLE_STAGES_EACH_STRICTLY_BELOW_2_HOURS_WITH_CHECKPOINT_BEFORE_LIMIT"
        or not _same_number(
            resources.get("checkpoint_and_stop_before_single_invocation_hours"), 2.0
        )
        or resources.get("resource_estimate_status")
        != "PROTOCOL_UPPER_BOUND_REQUIRES_EXECUTION_PREFLIGHT"
        or resources.get("execution_preflight_required") is not True
        or resources.get("execution_preflight_status")
        != "G4_RESOURCE_AND_RUNTIME_PREFLIGHT_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
        or resources.get("execution_requires_preflight_status") != "PASS"
        or resources.get("static_protocol_estimate_is_execution_evidence") is not False
        or preflight
        != {
            "actual_free_bytes_minimum": 500_000_000_000,
            "estimated_new_remote_bytes_maximum": 20_000_000_000,
            "project_data_plus_cache_plus_estimated_new_bytes_strictly_below": 500_000_000_000,
            "estimated_single_invocation_gpu_hours_strictly_below": 2.0,
            "estimated_total_gpu_hours_maximum": 4.0,
            "continuous_formal_gpu_hours_report_threshold": 24.0,
        }
        or resources.get(
            "preflight_fail_if_estimated_gpu_hours_or_remote_bytes_exceed_protocol_bound"
        )
        is not True
        or resources.get("ridge_train_decomposition_count_max") != 15
        or resources.get("ridge_train_plus_validation_decomposition_count_max") != 15
        or resources.get("ridge_total_decomposition_count_max") != 30
        or not _same_number(
            resources.get("stop_before_execution_if_continuous_gpu_hours_greater_than"),
            24.0,
        )
        or resources.get("stop_before_execution_if_projected_cache_data_bytes_at_or_above")
        != 500_000_000_000
        or resources.get("minimum_free_bytes") != 500_000_000_000
        or resources.get("large_arrays_in_git_allowed") is not False
    ):
        errors.append("G4 resource estimate or stop threshold drifted")

    execution = _mapping(config.get("execution"))
    expected_false = {
        "g4_execution_authorized",
        "new_real_edf_read",
        "new_real_audio_read",
        "new_feature_extraction_run",
        "ridge_run",
        "null_run",
        "metric_run",
        "formal_baseline_run",
        "scientific_result_claimed",
        "exchange_candidate_created",
    }
    if (
        execution.get("protocol_only") is not True
        or any(execution.get(key) is not False for key in expected_false)
        or execution.get("allowed_output_status") != G4_STATUS
    ):
        errors.append("G4 protocol candidate must keep every execution and claim gate closed")
    if (
        execution.get("g4_execution_authorized") is True
        and resources.get("execution_preflight_status") != "PASS"
    ):
        errors.append("G4 execution cannot be authorized before a real resource preflight PASS")
    return errors


def _load_scope_rows(path: str | Path) -> list[dict[str, str]]:
    required = {
        "sample_id",
        "participant_id",
        "session_id",
        "recording_id",
        "stimulus_id",
        "block_id",
        "analysis_eligible",
        "language",
        "audio_file",
        "audio_source_status",
        "audio_duration_seconds",
        "split",
    }
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("split CSV is missing G4 protocol columns")
        return [row for row in reader if row.get("recording_id") == EXPECTED_RECORDING]


def _rank_test_derangements(
    test_rows: list[dict[str, Any]], required_count: int = EXPECTED_SMOKE_PERMUTATIONS
) -> tuple[int, list[dict[str, Any]]]:
    if len(test_rows) < 2:
        raise ValueError("stimulus derangement requires at least two test passages")
    ordered = sorted(test_rows, key=lambda row: str(row.get("sample_id", "")))
    sample_ids = [str(row.get("sample_id", "")) for row in ordered]
    if any(not value for value in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("test sample IDs must be nonempty and unique")
    durations: list[float] = []
    for row in ordered:
        value = row.get("audio_duration_seconds")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("test duration is not numeric")
        try:
            duration = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError("test duration is not numeric") from error
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("test duration must be finite and positive")
        durations.append(duration)

    candidates: list[tuple[int, ...]] = []
    indices = tuple(range(len(ordered)))
    for permutation in itertools.permutations(indices):
        if any(target == donor for target, donor in enumerate(permutation)):
            continue
        candidates.append(permutation)
    if len(candidates) < required_count:
        raise ValueError("insufficient unique within-test stimulus derangements")
    rng = np.random.Generator(np.random.PCG64(20260813))
    selected_indices = rng.choice(len(candidates), size=required_count, replace=False)
    selected = [
        {
            "draw_index": draw_index,
            "enumerated_space_index": int(candidate_index),
            "duration_mismatch_sum_seconds": math.fsum(
                abs(durations[target] - durations[donor])
                for target, donor in enumerate(candidates[int(candidate_index)])
            ),
            "target_sample_ids": sample_ids,
            "donor_sample_ids": [
                sample_ids[index] for index in candidates[int(candidate_index)]
            ],
            "identity_pair_count": sum(
                target == donor
                for target, donor in enumerate(candidates[int(candidate_index)])
            ),
            "mapping_shared_across_all_features_layers_lags_electrodes_and_families": True,
        }
        for draw_index, candidate_index in enumerate(
            selected_indices.tolist(), start=1
        )
    ]
    return len(candidates), selected


def audit_g4_scope(split_csv: str | Path) -> dict[str, Any]:
    rows = _load_scope_rows(split_csv)
    issues: list[str] = []
    counts = Counter(row.get("split", "") for row in rows)
    block_assignments: dict[str, str] = {}
    for row in rows:
        block = row.get("block_id", "")
        split = row.get("split", "")
        if block in block_assignments and block_assignments[block] != split:
            issues.append(f"block spans splits: {block}")
        block_assignments[block] = split
    if len(rows) != 40 or dict(counts) != EXPECTED_SPLIT_COUNTS:
        issues.append("selected recording split counts are not 24/8/8")
    if block_assignments != EXPECTED_BLOCK_ASSIGNMENTS:
        issues.append("selected recording block assignments drifted")
    if any(row.get("participant_id") != EXPECTED_PARTICIPANT for row in rows):
        issues.append("selected scope includes another participant")
    if any(row.get("session_id") != EXPECTED_SESSION for row in rows):
        issues.append("selected scope includes another session")
    if any(row.get("recording_id") != EXPECTED_RECORDING for row in rows):
        issues.append("selected scope includes another recording")
    if any(row.get("analysis_eligible") != "True" for row in rows):
        issues.append("selected scope includes an ineligible passage")
    if any(row.get("language") != "en" for row in rows):
        issues.append("selected scope includes a non-English passage")
    if any(row.get("audio_source_status") != "BUNDLED_BLOCK_AUDIO" for row in rows):
        issues.append("selected scope includes an unexpected audio source")
    for field in ("sample_id", "stimulus_id", "audio_file"):
        values = [row.get(field, "") for row in rows]
        if any(not value for value in values) or len(set(values)) != len(values):
            issues.append(f"{field} must be nonempty and unique across the 40 passages")
    durations: list[float] = []
    for row in rows:
        try:
            duration = float(row.get("audio_duration_seconds", ""))
        except ValueError:
            duration = math.nan
        if not math.isfinite(duration) or duration <= 0:
            issues.append("audio duration is not finite and positive")
        durations.append(duration)

    test_rows = [row for row in rows if row.get("split") == "test"]
    try:
        derangement_count, selected_derangements = _rank_test_derangements(test_rows)
    except ValueError as error:
        issues.append(str(error))
        derangement_count = 0
        selected_derangements = []
    total_seconds = math.fsum(durations) if all(math.isfinite(value) for value in durations) else math.nan
    if not _same_number(total_seconds, EXPECTED_MANIFEST_SECONDS, tolerance=1e-9):
        issues.append("selected recording manifest duration total drifted")
    return {
        "split_csv": Path(split_csv).as_posix(),
        "recording_id": EXPECTED_RECORDING,
        "participant_id": EXPECTED_PARTICIPANT,
        "session_id": EXPECTED_SESSION,
        "excluded_sessions_present_count": sum(
            row.get("session_id") == "ses-01" for row in rows
        ),
        "passage_count": len(rows),
        "split_counts": dict(counts),
        "block_assignments": block_assignments,
        "unique_sample_id_count": len({row.get("sample_id") for row in rows}),
        "unique_stimulus_id_count": len({row.get("stimulus_id") for row in rows}),
        "unique_audio_file_count": len({row.get("audio_file") for row in rows}),
        "manifest_audio_seconds": total_seconds,
        "test_derangement_count": derangement_count,
        "selected_smoke_derangements": selected_derangements,
        "result_or_signal_input_used": False,
        "new_real_edf_read": False,
        "new_real_audio_read": False,
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }


def finalize_g4_protocol_report(
    protocol: dict[str, Any],
    schema: dict[str, Any],
    task_config: dict[str, Any],
    scope_audit: dict[str, Any],
) -> dict[str, Any]:
    protocol_errors = validate_g4_protocol(protocol, schema)
    task_g3 = _mapping(task_config.get("g3_single_recording"))
    task_split = _mapping(task_config.get("split"))
    task_target = _mapping(task_config.get("neural_target"))
    task_anatomy = _mapping(task_config.get("anatomy_mapping"))
    task_resources = _mapping(task_config.get("resources"))
    selected_derangements = scope_audit.get("selected_smoke_derangements", [])
    selected_space_indices = [
        item.get("enumerated_space_index")
        for item in selected_derangements
        if isinstance(item, dict)
    ] if isinstance(selected_derangements, list) else []
    checks = {
        "protocol_schema_and_semantics": protocol_errors == [],
        "g3_coordinator_acceptance_recorded": (
            task_g3.get("status") == G3_ACCEPTED_STATUS
            and task_g3.get("coordinator_review") == "ACCEPT"
            and task_g3.get("reviewed_on") == "2026-08-13"
            and task_g3.get("scientific_result_claimed") is False
        ),
        "accepted_split_and_method_dependencies": (
            task_split.get("split_status") == "BASELINE_FINAL_COORDINATOR_ACCEPTED"
            and task_split.get("final_embargo_status")
            == "FINAL_EMBARGO_COORDINATOR_ACCEPTED"
            and task_target.get("resolution_status") == "METHOD_FROZEN"
            and task_target.get("neural_extraction_allowed") is False
        ),
        "single_recording_24_8_8_scope": (
            scope_audit.get("status") == "PASS"
            and scope_audit.get("recording_id") == EXPECTED_RECORDING
            and scope_audit.get("session_id") == EXPECTED_SESSION
            and scope_audit.get("excluded_sessions_present_count") == 0
            and scope_audit.get("passage_count") == 40
            and scope_audit.get("split_counts") == EXPECTED_SPLIT_COUNTS
            and scope_audit.get("block_assignments") == EXPECTED_BLOCK_ASSIGNMENTS
            and scope_audit.get("unique_sample_id_count") == 40
            and scope_audit.get("unique_stimulus_id_count") == 40
            and scope_audit.get("unique_audio_file_count") == 40
        ),
        "uniform_test_derangement_is_feasible": (
            scope_audit.get("test_derangement_count")
            == EXPECTED_TEST_DERANGEMENT_COUNT
            and isinstance(selected_derangements, list)
            and len(selected_derangements)
            == EXPECTED_SMOKE_PERMUTATIONS
            and len(set(selected_space_indices)) == EXPECTED_SMOKE_PERMUTATIONS
            and all(
                item.get("identity_pair_count") == 0
                and item.get(
                    "mapping_shared_across_all_features_layers_lags_electrodes_and_families"
                )
                is True
                for item in selected_derangements
                if isinstance(item, dict)
            )
        ),
        "region_summary_remains_not_estimable": (
            task_anatomy.get("status") == "ANATOMY_MAPPING_NOT_READY"
            and task_anatomy.get("region_summary_status") == "NOT_ESTIMABLE"
            and _mapping(protocol.get("metrics")).get("region_summary")
            == "NOT_ESTIMABLE"
        ),
        "protocol_only_execution_blocked": (
            scope_audit.get("new_real_edf_read") is False
            and scope_audit.get("new_real_audio_read") is False
            and _mapping(protocol.get("execution")).get("protocol_only") is True
            and _mapping(protocol.get("execution")).get("g4_execution_authorized")
            is False
            and _mapping(protocol.get("execution")).get("ridge_run") is False
            and _mapping(protocol.get("execution")).get("null_run") is False
            and _mapping(protocol.get("execution")).get("metric_run") is False
            and _mapping(protocol.get("execution")).get("scientific_result_claimed")
            is False
        ),
        "resource_preflight_is_candidate_not_execution_acceptance": (
            _mapping(protocol.get("resources")).get("execution_preflight_status")
            == "G4_RESOURCE_AND_RUNTIME_PREFLIGHT_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
            and _mapping(protocol.get("resources")).get(
                "static_protocol_estimate_is_execution_evidence"
            )
            is False
        ),
        "task_resource_bounds_match_protocol": (
            task_resources.get("host_alias") == "server2203"
            and task_resources.get("remote_project_root")
            == "/home/fanyu/auditory_simulation_m6a"
            and task_resources.get("conda_environment")
            == "auditory_m6a_public_001"
            and task_resources.get("smoke_gpu_hours_limit") == 2
            and task_resources.get("continuous_gpu_hours_report_threshold") == 24
            and task_resources.get("storage_bytes_report_threshold")
            == 500_000_000_000
            and task_resources.get("minimum_free_bytes") == 500_000_000_000
        ),
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    return {
        "report_schema_version": G4_REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "status": G4_STATUS if not failed else "FAIL",
        "current_candidate": not failed,
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "protocol_config": "configs/m6a_g4_protocol_candidate.json",
        "protocol_schema": "schemas/m6a_g4_protocol_candidate.schema.json",
        "scope_audit": scope_audit,
        "required_checks": checks,
        "failed_checks": failed,
        "protocol_errors": protocol_errors,
        "prior_protocol_acceptance": protocol.get("prior_protocol_acceptance"),
        "protocol_amendment": protocol.get("amendment"),
        "wav2vec2_input_preprocessing": _mapping(
            _mapping(protocol.get("features")).get("wav2vec2")
        ).get("input_preprocessing"),
        "g4_execution_authorized": False,
        "new_real_edf_read": False,
        "new_real_audio_read": False,
        "new_feature_extraction_run": False,
        "ridge_run": False,
        "null_run": False,
        "metric_run": False,
        "formal_baseline_run": False,
        "scientific_result_claimed": False,
        "exchange_candidate_created": False,
    }


__all__ = [
    "G3_ACCEPTED_STATUS",
    "G4_REPORT_SCHEMA_VERSION",
    "G4_SCHEMA_VERSION",
    "G4_STATUS",
    "audit_pca20_train_matrix",
    "audit_g4_scope",
    "evaluate_g4_resource_preflight",
    "finalize_g4_protocol_report",
    "load_strict_json_object",
    "reused_svd_ridge_path",
    "validate_g4_protocol",
]
