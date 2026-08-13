from __future__ import annotations

import copy
import csv
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, TypedDict, cast
from unittest.mock import patch

import numpy as np

from m6a_public.g4_protocol_gate import (
    EXPECTED_LAGS_SECONDS,
    EXPECTED_TEST_DERANGEMENT_COUNT,
    G4_STATUS,
    _rank_test_derangements,
    audit_pca20_train_matrix,
    audit_g4_scope,
    evaluate_g4_resource_preflight,
    finalize_g4_protocol_report,
    load_strict_json_object,
    reused_svd_ridge_path,
    validate_g4_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


class ResourcePreflightArgs(TypedDict):
    actual_free_bytes: int
    estimated_new_bytes: int
    project_data_and_cache_bytes: int
    estimated_single_invocation_gpu_hours: float
    estimated_total_gpu_hours: float
    continuous_formal_gpu_hours: float


class G4ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_strict_json_object(
            ROOT / "configs" / "m6a_g4_protocol_candidate.json"
        )
        self.schema = load_strict_json_object(
            ROOT / "schemas" / "m6a_g4_protocol_candidate.schema.json"
        )
        self.task_config = load_strict_json_object(
            ROOT / "configs" / "m6a_public_001.json"
        )
        self.split_csv = ROOT / "reports" / "ds004703_primary_split.csv"

    def test_repository_protocol_and_scope_yield_candidate_only(self) -> None:
        self.assertEqual(validate_g4_protocol(self.protocol, self.schema), [])
        scope = audit_g4_scope(self.split_csv)
        report = finalize_g4_protocol_report(
            self.protocol, self.schema, self.task_config, scope
        )
        self.assertEqual(report["status"], G4_STATUS)
        self.assertEqual(report["failed_checks"], [])
        self.assertTrue(all(report["required_checks"].values()))
        self.assertFalse(report["g4_execution_authorized"])
        self.assertFalse(report["new_real_edf_read"])
        self.assertFalse(report["new_real_audio_read"])
        self.assertFalse(report["ridge_run"])
        self.assertFalse(report["scientific_result_claimed"])
        self.assertEqual(
            report["protocol_amendment"]["scope"],
            "WAV2VEC2_PREPROCESSING_INPUT_CONTRACT_ONLY",
        )
        self.assertTrue(report["wav2vec2_input_preprocessing"]["do_normalize"])
        self.assertFalse(
            report["wav2vec2_input_preprocessing"]["return_attention_mask"]
        )

    def test_scope_is_exact_single_recording_24_8_8(self) -> None:
        scope = audit_g4_scope(self.split_csv)
        self.assertEqual(scope["passage_count"], 40)
        self.assertEqual(
            scope["split_counts"], {"train": 24, "validation": 8, "test": 8}
        )
        self.assertEqual(scope["excluded_sessions_present_count"], 0)
        self.assertEqual(scope["unique_sample_id_count"], 40)
        self.assertEqual(scope["unique_stimulus_id_count"], 40)
        self.assertEqual(scope["unique_audio_file_count"], 40)
        self.assertFalse(scope["result_or_signal_input_used"])

    def test_exact_lag_sampling_is_not_rounded_to_50hz_frames(self) -> None:
        self.assertEqual(self.protocol["time_alignment"]["lags_seconds"][3], 0.15)
        self.assertTrue(
            all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    self.protocol["time_alignment"]["lags_seconds"],
                    EXPECTED_LAGS_SECONDS,
                    strict=True,
                )
            )
        )
        self.assertEqual(0.05 * 50, 2.5)
        self.assertFalse(float(0.05 * 50).is_integer())
        changed = copy.deepcopy(self.protocol)
        changed["time_alignment"]["fractional_grid_lag_policy"] = (
            "ROUND_TO_NEAREST_50HZ_FRAME"
        )
        self.assertTrue(validate_g4_protocol(changed, self.schema))

    def test_target_transform_is_population_std_and_one_fit_across_lags(self) -> None:
        target = self.protocol["target_transform"]
        self.assertEqual(
            target["scale"],
            "TRAIN_POPULATION_STANDARD_DEVIATION_DDOF_0_PER_CHANNEL_SUBBAND",
        )
        self.assertTrue(target["parameters_shared_across_all_lags"])
        self.assertFalse(target["lag_specific_refit_allowed"])
        self.assertIn("10MS_INTEGER_TICK", target["fit_population"])
        cases = {
            "sample_std": lambda item: item["target_transform"].__setitem__(
                "scale", "TRAIN_SAMPLE_STANDARD_DEVIATION_DDOF_1_PER_CHANNEL_SUBBAND"
            ),
            "lag_refit": lambda item: item["target_transform"].__setitem__(
                "lag_specific_refit_allowed", True
            ),
            "not_shared": lambda item: item["target_transform"].__setitem__(
                "parameters_shared_across_all_lags", False
            ),
            "validation_fit": lambda item: item["target_transform"].__setitem__(
                "validation_contributes_to_fit", True
            ),
            "test_fit": lambda item: item["target_transform"].__setitem__(
                "test_contributes_to_fit", True
            ),
            "epsilon": lambda item: item["target_transform"].__setitem__(
                "epsilon_formula", "FIXED_1E_30"
            ),
            "weights": lambda item: item["target_transform"]["subband_weights"].__setitem__(
                0, 0.2
            ),
        }
        self._assert_protocol_mutations_fail(cases)

    def test_features_ridge_refit_and_test_once_are_fail_closed(self) -> None:
        cases = {
            "silent_accept": lambda item: item.__setitem__(
                "status", "G4_PROTOCOL_COORDINATOR_ACCEPTED"
            ),
            "g3_raw_reuse": lambda item: item["amendment"].__setitem__(
                "g3_raw_input_representation_reuse_for_g4_scientific_baseline_allowed",
                True,
            ),
            "pca_dim": lambda item: item["features"]["log_mel"].__setitem__(
                "pca_dimension", 32
            ),
            "pca_variance": lambda item: item["features"]["log_mel"].__setitem__(
                "pca_dimension_selection", "VALIDATION_VARIANCE"
            ),
            "pca_auto_reduce": lambda item: item["features"]["log_mel"].__setitem__(
                "automatic_dimension_reduction_allowed", True
            ),
            "pca_rank_tolerance": lambda item: item["features"]["log_mel"].__setitem__(
                "singular_value_rank_tolerance", "AUTO_LIBRARY_DEFAULT"
            ),
            "wav_test_layer": lambda item: item["features"]["wav2vec2"].__setitem__(
                "test_layer_selection_allowed", True
            ),
            "wav_preprocessor_normalize": lambda item: item["features"]["wav2vec2"][
                "input_preprocessing"
            ].__setitem__("do_normalize", False),
            "wav_preprocessor_attention_mask": lambda item: item["features"]["wav2vec2"][
                "input_preprocessing"
            ].__setitem__("return_attention_mask", True),
            "wav_preprocessor_cross_passage": lambda item: item["features"]["wav2vec2"][
                "input_preprocessing"
            ].__setitem__("cross_passage_statistics_allowed", True),
            "feature_validation_fit": lambda item: item["features"].__setitem__(
                "validation_contributes_to_fit", True
            ),
            "solver": lambda item: item["ridge"].__setitem__("solver", "AUTO"),
            "alpha_test": lambda item: item["ridge"].__setitem__(
                "alpha_selection_partition", "test"
            ),
            "alpha_unit": lambda item: item["ridge"].__setitem__(
                "alpha_selection_unit", "GLOBAL_BEST_LAYER"
            ),
            "tie": lambda item: item["ridge"].__setitem__(
                "alpha_tie_break", "LARGEST_ALPHA"
            ),
            "no_refit": lambda item: item["ridge"].__setitem__(
                "final_refit_partition_after_alpha_lock", ["train"]
            ),
            "transform_refit": lambda item: item["ridge"].__setitem__(
                "transform_refit_after_validation_allowed", True
            ),
            "test_twice": lambda item: item["ridge"].__setitem__(
                "test_evaluation_count", 2
            ),
            "per_cell_svd": lambda item: item["ridge"].__setitem__(
                "per_cell_decomposition_allowed", True
            ),
            "too_many_svd": lambda item: item["ridge"].__setitem__(
                "total_decomposition_count_max", 5940
            ),
        }
        self._assert_protocol_mutations_fail(cases)

    def test_log_mel_pca20_rank_and_sample_gate_fails_closed(self) -> None:
        rng = np.random.Generator(np.random.PCG64(20260813))
        valid = audit_pca20_train_matrix(rng.normal(size=(64, 80)))
        self.assertEqual(valid["status"], "PASS")
        self.assertGreaterEqual(valid["numeric_rank"], 20)
        self.assertFalse(valid["automatic_dimension_reduction_used"])

        too_few = audit_pca20_train_matrix(rng.normal(size=(19, 80)))
        self.assertEqual(too_few["status"], "NOT_ESTIMABLE")
        rank_deficient = audit_pca20_train_matrix(
            rng.normal(size=(64, 1)) @ rng.normal(size=(1, 80))
        )
        self.assertEqual(rank_deficient["status"], "NOT_ESTIMABLE")
        nonfinite = rng.normal(size=(64, 80))
        nonfinite[0, 0] = math.inf
        self.assertEqual(
            audit_pca20_train_matrix(nonfinite)["status"], "NOT_ESTIMABLE"
        )

    def test_reused_svd_matches_direct_multi_target_ridge_and_runs_once(self) -> None:
        rng = np.random.Generator(np.random.PCG64(11))
        x = rng.normal(size=(64, 8)).astype(np.float64)
        y = rng.normal(size=(64, 11 * 36)).astype(np.float64)
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        original_svd = np.linalg.svd
        with patch("numpy.linalg.svd", wraps=original_svd) as mocked_svd:
            reused = reused_svd_ridge_path(x, y, alphas)
        self.assertEqual(mocked_svd.call_count, 1)
        xtx = x.T @ x
        xty = x.T @ y
        identity = np.eye(x.shape[1], dtype=np.float64)
        for alpha in alphas:
            direct = np.linalg.solve(xtx + alpha * identity, xty)
            np.testing.assert_allclose(reused[alpha], direct, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                x @ reused[alpha], x @ direct, rtol=1e-10, atol=1e-10
            )
        with self.assertRaises(ValueError):
            reused_svd_ridge_path(x, y, [0.0])
        x[0, 0] = math.nan
        with self.assertRaises(ValueError):
            reused_svd_ridge_path(x, y, alphas)

    def test_uniform_derangement_space_seed_without_replacement_and_shared_mapping(self) -> None:
        first = audit_g4_scope(self.split_csv)
        second = audit_g4_scope(self.split_csv)
        self.assertEqual(first["test_derangement_count"], EXPECTED_TEST_DERANGEMENT_COUNT)
        self.assertEqual(first["selected_smoke_derangements"], second["selected_smoke_derangements"])
        selected = first["selected_smoke_derangements"]
        indices = [item["enumerated_space_index"] for item in selected]
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(set(indices)), 20)
        self.assertTrue(all(item["identity_pair_count"] == 0 for item in selected))
        self.assertTrue(
            all(
                item[
                    "mapping_shared_across_all_features_layers_lags_electrodes_and_families"
                ]
                for item in selected
            )
        )
        self.assertTrue(
            all(len(set(item["donor_sample_ids"])) == 8 for item in selected)
        )

    def test_derangement_and_maxstat_drift_fail_closed(self) -> None:
        cases = {
            "seed": lambda item: item["nulls"]["stimulus_derangement"].__setitem__(
                "random_seed", 1
            ),
            "replacement": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__("sample_without_replacement", False),
            "cost_selection": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__("duration_mismatch_role", "SELECT_LOWEST_COST"),
            "null_transform_refit": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__("transform_refit_allowed", True),
            "null_ridge_refit": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__("ridge_refit_allowed", True),
            "null_model_rerun": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__("model_rerun_allowed", True),
            "mapping_direction": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__(
                "mapping_direction", "TARGET_NEURAL_TO_DONOR_PREDICTION"
            ),
            "cross_passage_warp": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__("cross_passage_concatenation_before_mapping_allowed", True),
            "null_aggregation_drift": lambda item: item["nulls"][
                "stimulus_derangement"
            ].__setitem__("observed_and_null_metric_aggregation_identical", False),
            "wav_circular": lambda item: item["nulls"][
                "acoustic_circular_shift"
            ].__setitem__("wav2vec2_allowed", True),
            "replace_primary": lambda item: item["nulls"].__setitem__(
                "post_result_primary_null_substitution_allowed", True
            ),
            "mapping_not_shared": lambda item: item["nulls"].__setitem__(
                "shared_mapping_scope", "PER_CELL_MAPPING"
            ),
            "fdr": lambda item: item["multiple_comparison"].__setitem__(
                "fdr_allowed", True
            ),
            "dynamic_family": lambda item: item["multiple_comparison"].__setitem__(
                "effective_family_fixed_across_permutations", False
            ),
            "empty_family_pass": lambda item: item["multiple_comparison"].__setitem__(
                "empty_effective_family_action", "PASS"
            ),
            "p_formula": lambda item: item["multiple_comparison"][
                "permutation_p"
            ].__setitem__("denominator", 20),
            "stable_claim": lambda item: item["multiple_comparison"].__setitem__(
                "smoke_interpretation", "STABLE_SIGNIFICANCE"
            ),
        }
        self._assert_protocol_mutations_fail(cases)

    def test_singleton_derangement_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            _rank_test_derangements(
                [{"sample_id": "only", "audio_duration_seconds": "10.0"}]
            )

    def test_metrics_claims_resources_and_execution_drift_fail_closed(self) -> None:
        cases = {
            "constant_pearson": lambda item: item["metrics"].__setitem__(
                "pearson_not_estimable_if", "RETURN_ZERO"
            ),
            "region": lambda item: item["metrics"].__setitem__(
                "region_summary", "ESTIMABLE"
            ),
            "paired_p": lambda item: item["metrics"].__setitem__(
                "model_vs_acoustic_inferential_p_value", "COMPUTED"
            ),
            "gpu_total": lambda item: item["resources"].__setitem__(
                "estimated_total_gpu_hours_upper_bound", 4.1
            ),
            "gpu_single": lambda item: item["resources"].__setitem__(
                "single_invocation_gpu_hours_hard_limit", 3.0
            ),
            "gpu_partition": lambda item: item["resources"].__setitem__(
                "gpu_execution_partitioning", "ONE_FOUR_HOUR_INVOCATION"
            ),
            "gpu_checkpoint": lambda item: item["resources"].__setitem__(
                "checkpoint_and_stop_before_single_invocation_hours", 2.5
            ),
            "storage": lambda item: item["resources"].__setitem__(
                "estimated_new_remote_bytes_upper_bound", 600_000_000_000
            ),
            "preflight": lambda item: item["resources"].__setitem__(
                "execution_preflight_required", False
            ),
            "execution_without_real_preflight": lambda item: item["execution"].__setitem__(
                "g4_execution_authorized", True
            ),
            "edf": lambda item: item["execution"].__setitem__(
                "new_real_edf_read", True
            ),
            "ridge": lambda item: item["execution"].__setitem__("ridge_run", True),
            "science": lambda item: item["execution"].__setitem__(
                "scientific_result_claimed", True
            ),
        }
        self._assert_protocol_mutations_fail(cases)

    def test_resource_preflight_enforces_single_total_and_report_thresholds(self) -> None:
        valid: ResourcePreflightArgs = {
            "actual_free_bytes": 600_000_000_000,
            "estimated_new_bytes": 20_000_000_000,
            "project_data_and_cache_bytes": 470_000_000_000,
            "estimated_single_invocation_gpu_hours": 1.99,
            "estimated_total_gpu_hours": 4.0,
            "continuous_formal_gpu_hours": 4.0,
        }
        self.assertEqual(evaluate_g4_resource_preflight(**valid)["status"], "PASS")
        failures = {
            "single_at_limit": ("estimated_single_invocation_gpu_hours", 2.0),
            "total_over_limit": ("estimated_total_gpu_hours", 4.01),
            "continuous_over_report_threshold": ("continuous_formal_gpu_hours", 24.01),
            "free_space_low": ("actual_free_bytes", 499_999_999_999),
            "project_total_at_limit": ("project_data_and_cache_bytes", 480_000_000_000),
        }
        for name, (field, value) in failures.items():
            with self.subTest(name=name):
                changed_values = cast(dict[str, int | float], copy.deepcopy(valid))
                changed_values[field] = value
                changed = cast(ResourcePreflightArgs, changed_values)
                self.assertEqual(
                    evaluate_g4_resource_preflight(**changed)["status"], "FAIL"
                )

    def test_task_resource_drift_blocks_machine_candidate(self) -> None:
        changed = copy.deepcopy(self.task_config)
        changed["resources"]["smoke_gpu_hours_limit"] = 3
        report = finalize_g4_protocol_report(
            self.protocol,
            self.schema,
            changed,
            audit_g4_scope(self.split_csv),
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(
            report["required_checks"]["task_resource_bounds_match_protocol"]
        )

    def test_schema_extra_missing_nonfinite_and_malformed_fail_without_exception(self) -> None:
        cases: list[dict[str, Any]] = []
        extra = copy.deepcopy(self.protocol)
        extra["ridge"]["unregistered"] = True
        cases.append(extra)
        missing = copy.deepcopy(self.protocol)
        missing["execution"].pop("metric_run")
        cases.append(missing)
        nonfinite = copy.deepcopy(self.protocol)
        nonfinite["ridge"]["alpha_tie_tolerance"] = math.nan
        cases.append(nonfinite)
        malformed = copy.deepcopy(self.protocol)
        malformed["features"] = []
        cases.append(malformed)
        for changed in cases:
            with self.subTest(case=len(changed)):
                self.assertTrue(validate_g4_protocol(changed, self.schema))

    def test_scope_fixture_drift_and_singleton_fail_report(self) -> None:
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        selected = [row for row in rows if row["recording_id"] == self.protocol["scope"]["recording_id"]]
        selected[0]["session_id"] = "ses-01"
        selected[1]["stimulus_id"] = selected[0]["stimulus_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(selected)
            scope = audit_g4_scope(path)
            self.assertEqual(scope["status"], "FAIL")
            report = finalize_g4_protocol_report(
                self.protocol, self.schema, self.task_config, scope
            )
            self.assertEqual(report["status"], "FAIL")

    def _assert_protocol_mutations_fail(
        self, cases: dict[str, Any]
    ) -> None:
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(self.protocol)
                mutate(changed)
                self.assertTrue(validate_g4_protocol(changed, self.schema))


if __name__ == "__main__":
    unittest.main()
