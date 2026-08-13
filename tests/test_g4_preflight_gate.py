from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from m6a_public.audio_context_gate import (
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
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
    audit_preprocessor_config,
    normalize_passage_waveform,
)


ROOT = Path(__file__).resolve().parents[1]


def valid_evidence() -> dict[str, Any]:
    category_specs = {
        "data": (["/home/fanyu/auditory_simulation_m6a/data"], 14_200_000_000),
        "cache": (["/home/fanyu/auditory_simulation_m6a/cache"], 400_000_000),
        "outputs": (["/home/fanyu/auditory_simulation_m6a/outputs"], 100_000_000),
        "log": (
            [
                "/home/fanyu/auditory_simulation_m6a/log",
                "/home/fanyu/auditory_simulation_m6a/logs",
            ],
            10_000_000,
        ),
        "code": (["/home/fanyu/auditory_simulation_m6a/code_snapshot"], 5_000_000),
    }
    categories: dict[str, Any] = {}
    for name, (paths, total) in category_specs.items():
        divided = [total // len(paths)] * len(paths)
        divided[-1] += total - sum(divided)
        items = [
            {
                "path": path,
                "exists": True,
                "bytes": size,
                "regular_file_count": 1,
                "symlink_entry_count_not_followed": 0,
                "modified_at_utc": "2026-08-13T10:00:00+00:00",
            }
            for path, size in zip(paths, divided, strict=True)
        ]
        categories[name] = {"paths": items, "bytes": total}
    selected_total = sum(value["bytes"] for value in categories.values())
    estimate = conservative_runtime_estimate(2.0, 3.0)
    estimate["duration_linear_scaling_assumed"] = False
    def preprocessing_equivalence(label: str, sample_count: int) -> dict[str, Any]:
        return {
            "label": label,
            "status": "PASS",
            "sample_count": sample_count,
            "feature_extractor_output_shape": [sample_count],
            "feature_extractor_output_dtype": "float32",
            "attention_mask_returned": False,
            "attention_mask_argument": "OMITTED",
            "padding_used": False,
            "absolute_tolerance": 1e-7,
            "relative_tolerance": 1e-7,
            "max_absolute_difference": 0.0,
            "equivalent": True,
            "normalization": {
                "sample_count": sample_count,
                "input_dtype": "float32",
                "normalization_epsilon": 1e-7,
                "pre_all_finite": True,
                "pre_mean": 0.0,
                "pre_population_variance": 0.00625,
                "pre_population_std": 0.07905694,
                "post_all_finite": True,
                "post_mean": -1e-8,
                "post_population_variance": 0.999984,
                "post_population_std": 0.999992,
            },
        }
    return {
        "report_schema_version": PREFLIGHT_REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "audited_at_utc": "2026-08-13T10:05:00+00:00",
        "config_path": "configs/m6a_g4_resource_runtime_preflight_candidate.json",
        "schema_path": "schemas/m6a_g4_resource_runtime_preflight_candidate.schema.json",
        "config_errors": [],
        "longest_passage_manifest_audit": {
            "status": "PASS",
            "manifest_path": "reports/ds004703_primary_split.csv",
            "recording_id": "sub-SD012_ses-02_task-PassiveListen",
            "selection_rule": "MAX_AUDIO_DURATION_SECONDS_THEN_SAMPLE_ID_LEXICOGRAPHIC",
            "passage_count": 40,
            "longest": {
                "sample_id": "sub-SD012_ses-02_task-PassiveListen__seg-028",
                "stimulus_id": "s3201a-ex01",
                "split": "validation",
                "audio_file_metadata_only": "stimuli/excerpts/Block 3/s3201a-ex01_normed.wav",
                "audio_duration_seconds": 77.08981859410432,
                "synthetic_sample_count_formula": "ceil(audio_duration_seconds * 16000)",
                "synthetic_sample_count": EXPECTED_SYNTHETIC_SAMPLE_COUNT,
                "analysis_eligible": "True",
                "audio_source_status": "BUNDLED_BLOCK_AUDIO",
            },
            "real_audio_read": False,
            "issues": [],
        },
        "storage_audit": {
            "audited_at_utc": "2026-08-13T10:01:00+00:00",
            "project_root": "/home/fanyu/auditory_simulation_m6a",
            "measurement": "REGULAR_FILE_BYTES_WITHOUT_FOLLOWING_SYMLINKS",
            "categories": categories,
            "selected_category_total_bytes": selected_total,
            "project_root_total_bytes": selected_total + 1_000_000,
            "project_root_regular_file_count": 7,
            "unclassified_other_bytes": 1_000_000,
            "actual_free_bytes": 600_000_000_000,
            "estimated_new_bytes_upper_bound": 20_000_000_000,
            "data_cache_plus_estimated_new_bytes": 34_600_000_000,
        },
        "runtime_canary": {
            "status": "PASS",
            "model_id": "facebook/wav2vec2-base",
            "revision_label": "main",
            "device": "cuda:0",
            "local_files_only": True,
            "trust_remote_code": False,
            "repository_custom_code_executed": False,
            "weights_only": True,
            "tensor_only": True,
            "download_attempted": False,
            "model_eval": True,
            "inference_mode": True,
            "parameter_requires_grad_count": 0,
            "loading_info": {
                "missing_keys": [],
                "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
                "mismatched_keys": [],
                "error_msgs": [],
            },
            "model_load_wall_seconds": 2.0,
            "transformers_version": "5.14.1",
            "preprocessor_remote_semantic_audit": {
                "status": "PASS",
                "audited_at_utc": "2026-08-13T10:02:00+00:00",
                "source_endpoint": "https://hf-mirror.com",
                "filename": "preprocessor_config.json",
                "prior_failed_audit": {
                    "path": "/home/fanyu/auditory_simulation_m6a/logs/wav2vec2_preprocessor_mirror_audit_20260813.json",
                    "status": "FAIL",
                    "fallback_http_status": 403,
                    "preserved": True,
                },
                "probes": [
                    {
                        "endpoint": "https://mirrors.tuna.tsinghua.edu.cn",
                        "http_status": 404,
                        "body_bytes": -1,
                        "semantic_fields": {},
                        "semantic_match": False,
                        "proxy_used": False,
                        "error": {"type": "HTTPError", "message": "HTTP 404"},
                    },
                    {
                        "endpoint": "https://hf-mirror.com",
                        "http_status": 200,
                        "body_bytes": 159,
                        "semantic_fields": dict(PREPROCESSOR_SEMANTICS),
                        "semantic_match": True,
                        "proxy_used": False,
                        "error": None,
                    },
                ],
                "http_status": 200,
                "mirror_body_bytes": 159,
                "mirror_semantic_fields": dict(PREPROCESSOR_SEMANTICS),
                "cache_bytes": 159,
                "cache_modified_at_utc": "2026-08-13T07:15:07+00:00",
                "remote_only": True,
                "cache_write_performed": False,
                "network_body_persisted": False,
                "proxy_used": False,
            },
            "preprocessor_config_audit": {
                "status": "PASS",
                "path": "/home/fanyu/auditory_simulation_m6a/cache/huggingface/facebook_wav2vec2_base_main_20260813/preprocessor_config.json",
                "filename": "preprocessor_config.json",
                "bytes": 159,
                "modified_at_utc": "2026-08-13T07:15:07+00:00",
                "semantic_fields": dict(PREPROCESSOR_SEMANTICS),
                "missing_fields": [],
                "unexpected_fields": [],
                "remote_only": True,
                "issues": [],
            },
            "feature_extractor_semantics": dict(PREPROCESSOR_SEMANTICS),
            "feature_extractor_equivalence": {
                "status": "PASS",
                "warmup": preprocessing_equivalence("ONE_SECOND_WARMUP", 16_000),
                "longest": preprocessing_equivalence(
                    "LONGEST_G4_PASSAGE", EXPECTED_SYNTHETIC_SAMPLE_COUNT
                ),
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
            "synthetic_input_all_finite": True,
            "preprocessed_input_all_finite": True,
            "synthetic_input_channels": 1,
            "warmup": {"status": "PASS", "sample_count": 16_000, "wall_seconds": 0.1},
            "longest_forward": {
                "status": "PASS",
                "batch_size": 1,
                "sample_count": EXPECTED_SYNTHETIC_SAMPLE_COUNT,
                "frame_count": EXPECTED_FRAME_COUNT,
                "layer_keys": list(EXPECTED_LAYER_KEYS),
                "layer_shapes": [[1, EXPECTED_FRAME_COUNT, 768]] * 13,
                "all_finite": True,
                "attention_scope": "GLOBAL_WITHIN_ONE_SYNTHETIC_PASSAGE",
                "chunked_or_windowed_approximation_used": False,
                "oom": False,
                "wall_seconds": 3.0,
                "cuda_peak_allocated_bytes": 1_000_000,
                "cuda_peak_reserved_bytes": 2_000_000,
            },
        },
        "runtime_error": None,
        "execution_estimate": estimate,
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


class G4PreflightGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_strict_json_object(
            ROOT / "configs" / "m6a_g4_resource_runtime_preflight_candidate.json"
        )
        self.schema = load_strict_json_object(
            ROOT / "schemas" / "m6a_g4_resource_runtime_preflight_candidate.schema.json"
        )
        self.main = load_strict_json_object(ROOT / "configs" / "m6a_public_001.json")
        self.protocol = load_strict_json_object(
            ROOT / "configs" / "m6a_g4_protocol_candidate.json"
        )
        self.split_csv = ROOT / "reports" / "ds004703_primary_split.csv"

    def test_repository_config_manifest_and_synthetic_evidence_pass_candidate_only(self) -> None:
        self.assertEqual(
            validate_preflight_config(self.config, self.schema, self.main, self.protocol),
            [],
        )
        manifest = audit_longest_passage_manifest(self.split_csv)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["longest"]["synthetic_sample_count"], 1_233_438)
        report = finalize_preflight_report(valid_evidence(), self.config)
        self.assertEqual(report["status"], PREFLIGHT_STATUS)
        self.assertEqual(report["failed_checks"], [])
        self.assertTrue(all(report["required_checks"].values()))

    def test_prior_failed_mirror_audit_is_optional_provenance(self) -> None:
        without_prior = valid_evidence()
        without_prior["runtime_canary"]["preprocessor_remote_semantic_audit"].pop(
            "prior_failed_audit"
        )
        report = finalize_preflight_report(without_prior, self.config)
        self.assertEqual(report["status"], PREFLIGHT_STATUS)
        self.assertTrue(
            report["required_checks"]["remote_preprocessor_config_semantic_audit"]
        )

        different_prior = valid_evidence()
        different_prior["runtime_canary"]["preprocessor_remote_semantic_audit"][
            "prior_failed_audit"
        ] = {
            "path": "/project/logs/transient_failure.json",
            "status": "FAIL",
            "fallback_http_status": 429,
            "preserved": True,
        }
        report = finalize_preflight_report(different_prior, self.config)
        self.assertEqual(report["status"], PREFLIGHT_STATUS)

    def test_current_mirror_semantics_remain_fail_closed(self) -> None:
        cases = {
            "tuna_not_404": lambda item: item["probes"][0].__setitem__(
                "http_status", 200
            ),
            "fallback_not_200": lambda item: item["probes"][1].__setitem__(
                "http_status", 404
            ),
            "body_bytes": lambda item: item.__setitem__("mirror_body_bytes", 158),
            "semantic_drift": lambda item: item["mirror_semantic_fields"].__setitem__(
                "do_normalize", False
            ),
            "cache_write": lambda item: item.__setitem__(
                "cache_write_performed", True
            ),
            "proxy": lambda item: item.__setitem__("proxy_used", True),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = valid_evidence()
                audit = changed["runtime_canary"][
                    "preprocessor_remote_semantic_audit"
                ]
                mutate(audit)
                report = finalize_preflight_report(changed, self.config)
                self.assertEqual(report["status"], "FAIL")
                self.assertFalse(
                    report["required_checks"][
                        "remote_preprocessor_config_semantic_audit"
                    ]
                )

    def test_config_is_fail_closed_for_runtime_and_execution_drift(self) -> None:
        cases = {
            "download": lambda item: item["model"].__setitem__("download_allowed", True),
            "remote_code": lambda item: item["model"].__setitem__("trust_remote_code", True),
            "real_path": lambda item: item["synthetic_input"].__setitem__(
                "input_path", "/data/real.wav"
            ),
            "length": lambda item: item["longest_passage"].__setitem__(
                "expected_synthetic_sample_count", 1_233_437
            ),
            "checkpoint": lambda item: item["checkpoint_design"].__setitem__(
                "successful_passage_checkpoint_required", False
            ),
            "execution": lambda item: item["execution"].__setitem__(
                "g4_execution_authorized", True
            ),
            "single_limit": lambda item: item["execution_estimate"].__setitem__(
                "single_invocation_gpu_hours_strictly_below", 3.0
            ),
            "do_normalize": lambda item: item["model"]["input_preprocessing"].__setitem__(
                "do_normalize", False
            ),
            "attention_mask": lambda item: item["model"]["input_preprocessing"].__setitem__(
                "return_attention_mask", True
            ),
            "missing_preprocessor": lambda item: item["model"].pop(
                "input_preprocessing"
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(self.config)
                mutate(changed)
                self.assertTrue(
                    validate_preflight_config(changed, self.schema, self.main, self.protocol)
                )

    def test_evidence_fail_closed_examples(self) -> None:
        cases = {
            "free": lambda item: item["storage_audit"].__setitem__(
                "actual_free_bytes", 499_999_999_999
            ),
            "data_cache_total": lambda item: item["storage_audit"].__setitem__(
                "data_cache_plus_estimated_new_bytes", 500_000_000_000
            ),
            "length": lambda item: item["runtime_canary"]["longest_forward"].__setitem__(
                "sample_count", 1_233_437
            ),
            "real_path": lambda item: item["runtime_canary"].__setitem__(
                "input_path", "/data/real.wav"
            ),
            "download": lambda item: item["runtime_canary"].__setitem__(
                "download_attempted", True
            ),
            "custom_code": lambda item: item["runtime_canary"].__setitem__(
                "repository_custom_code_executed", True
            ),
            "layers": lambda item: item["runtime_canary"]["longest_forward"].__setitem__(
                "layer_shapes", [[1, EXPECTED_FRAME_COUNT, 768]] * 12
            ),
            "nonfinite": lambda item: item["runtime_canary"]["longest_forward"].__setitem__(
                "wall_seconds", math.nan
            ),
            "oom": lambda item: item["runtime_canary"]["longest_forward"].__setitem__(
                "oom", True
            ),
            "single_two_hours": lambda item: item["execution_estimate"].__setitem__(
                "single_invocation_upper_bound_gpu_hours", 2.0
            ),
            "total_over_four": lambda item: item["execution_estimate"].__setitem__(
                "total_40_passage_upper_bound_gpu_hours", 4.01
            ),
            "checkpoint": lambda item: item["checkpoint_design_audit"].__setitem__(
                "atomic_final_rename_required", False
            ),
            "execution": lambda item: item.__setitem__("g4_execution_authorized", True),
            "preprocessor_missing": lambda item: item["runtime_canary"].__setitem__(
                "preprocessor_config_audit", {}
            ),
            "preprocessor_do_normalize": lambda item: item["runtime_canary"][
                "feature_extractor_semantics"
            ].__setitem__("do_normalize", False),
            "attention_mask_returned": lambda item: item["runtime_canary"][
                "feature_extractor_equivalence"
            ]["longest"].__setitem__("attention_mask_returned", True),
            "attention_mask_argument": lambda item: item["runtime_canary"].__setitem__(
                "attention_mask_argument", "ALL_ONES"
            ),
            "cross_passage_statistics": lambda item: item["runtime_canary"].__setitem__(
                "cross_passage_statistics_used", True
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = valid_evidence()
                mutate(changed)
                self.assertEqual(
                    finalize_preflight_report(changed, self.config)["status"], "FAIL"
                )

    def test_manifest_duration_drift_fails(self) -> None:
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        for row in rows:
            if row["sample_id"] == "sub-SD012_ses-02_task-PassiveListen__seg-028":
                row["audio_duration_seconds"] = "1.0"
                break
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(audit_longest_passage_manifest(path)["status"], "FAIL")

    def test_runtime_estimator_never_underestimates_measurements(self) -> None:
        estimate = conservative_runtime_estimate(31.0, 45.0)
        self.assertGreaterEqual(estimate["model_load_upper_bound_seconds"], 31.0)
        self.assertGreaterEqual(estimate["longest_forward_upper_bound_seconds"], 45.0)
        self.assertEqual(
            estimate["total_40_passage_upper_bound_gpu_hours"],
            40 * estimate["single_invocation_upper_bound_gpu_hours"],
        )
        with self.assertRaises(ValueError):
            conservative_runtime_estimate(math.inf, 1.0)

    def test_passage_normalization_matches_frozen_formula_and_rejects_bad_inputs(self) -> None:
        source = np.asarray([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
        normalized, audit = normalize_passage_waveform(source)
        expected = (source - source.mean()) / np.sqrt(source.var() + 1e-7)
        np.testing.assert_allclose(normalized, expected, rtol=1e-7, atol=1e-7)
        self.assertTrue(audit["pre_all_finite"])
        self.assertTrue(audit["post_all_finite"])
        for bad in (
            np.ones(8, dtype=np.float32),
            np.asarray([0.0, math.nan], dtype=np.float32),
            np.asarray([0.0, math.inf], dtype=np.float32),
        ):
            with self.subTest(bad=bad.tolist()):
                with self.assertRaises(ValueError):
                    normalize_passage_waveform(bad)

    def test_preprocessor_config_audit_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "preprocessor_config.json"
            path.write_text(json.dumps(PREPROCESSOR_SEMANTICS), encoding="utf-8")
            self.assertEqual(
                audit_preprocessor_config(path, expected_cache_root=root)["status"],
                "PASS",
            )
            changed = dict(PREPROCESSOR_SEMANTICS)
            changed["do_normalize"] = False
            path.write_text(json.dumps(changed), encoding="utf-8")
            self.assertEqual(
                audit_preprocessor_config(path, expected_cache_root=root)["status"],
                "FAIL",
            )


if __name__ == "__main__":
    unittest.main()
