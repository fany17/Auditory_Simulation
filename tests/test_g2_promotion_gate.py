from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from m6a_public.config_gate import load_json
from m6a_public.g2_promotion_gate import (
    CANDIDATE_STATUS,
    load_strict_json,
    main,
    validate_g2_promotion,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "scalp electrode, not included with data"


def valid_inputs() -> dict[str, dict[str, Any]]:
    dataset_root = "/home/fanyu/auditory_simulation_m6a/data/ds004703/v1.1.0"
    eligible_counts = [123, 123, 123, 123, 122, 122, 122, 122, 122, 122, 122]
    c_prefix_counts = [67, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66]
    dataset_headers: list[dict[str, Any]] = []
    recordings: list[dict[str, Any]] = []
    for index, (eligible_count, c_prefix_count) in enumerate(
        zip(eligible_counts, c_prefix_counts, strict=True)
    ):
        relative_path = f"sub-{index:02d}/recording.edf"
        eligible_names = [f"E{index:02d}_{channel:03d}" for channel in range(eligible_count)]
        c_prefix_names = [f"C{index:02d}_{channel:03d}" for channel in range(c_prefix_count)]
        channel_names = eligible_names + c_prefix_names
        sampling_rate = 512.0 if index % 2 == 0 else 1024.0
        dataset_headers.append(
            {
                "path": relative_path,
                "sampling_rate_hz": sampling_rate,
                "channels": len(channel_names),
                "channel_names": channel_names,
                "duration_seconds": 20.0,
            }
        )
        recordings.append(
            {
                "recording_id": f"recording-{index:02d}",
                "edf_file": relative_path,
                "sampling_rate_hz": sampling_rate,
                "power_line_frequency_hz": 60.0,
                "events_within_recording": True,
                "events_within_edf_timeline": True,
                "edf_header": {
                    "path": f"{dataset_root}/{relative_path}",
                    "channels": len(channel_names),
                    "channel_names": channel_names,
                },
                "edf_header_sampling_rate_matches_sidecar": True,
                "analysis_eligible_neural_channels_missing_from_edf": [],
                "audio_offset_seconds": 0.0,
                "channels": {
                    "analysis_eligible_neural_channel_count": eligible_count,
                    "analysis_eligible_neural_names": eligible_names,
                    "c_prefix_exclusion_count": c_prefix_count,
                    "c_prefix_names": c_prefix_names,
                },
            }
        )
    dataset_report: dict[str, Any] = {
        "report_schema_version": "m6a-dataset-audit-v2",
        "task_id": "M6A-PUBLIC-001",
        "dataset_id": "ds004703",
        "expected_version": "1.1.0",
        "audit_scope": "FULL_DATASET_G2_CANDIDATE",
        "audited_at_utc": "2026-08-11T14:00:00+00:00",
        "dataset_root": dataset_root,
        "integrity_policy": "NON_HASH_AUDIT",
        "file_count": 377,
        "total_bytes": 14_173_350_514,
        "active_partial_count": 0,
        "active_partial_files": [],
        "expected_inventory_reconciliation": {
            "expected_file_count": 377,
            "expected_total_bytes": 14_173_350_514,
            "missing_paths": [],
            "unexpected_paths": [],
            "byte_mismatches": [],
            "status": "PASS",
        },
        "expected_inventory_provenance": {
            "source": "https://s3.amazonaws.com/openneuro.org",
            "listed_at_utc": "2026-08-11T11:14:08+00:00",
            "acquisition_method": "PUBLIC_S3_LIST_OBJECTS_V2",
            "acquisition_script": "scripts/public_s3_inventory.py",
            "object_count": 377,
            "total_bytes": 14_173_350_514,
        },
        "dataset_description": {
            "DatasetDOI": "doi:10.18112/openneuro.ds004703.v1.1.0",
            "License": "CC0",
        },
        "dataset_boundary_checks": {
            "dataset_doi_matches": True,
            "declared_license_matches": True,
            "readme_noncommercial_restriction_found": True,
            "readme_no_reidentification_restriction_found": True,
            "status": "PASS",
        },
        "license_boundary": "CC0_PLUS_README_NONCOMMERCIAL_AND_NO_REIDENTIFICATION",
        "ieeg_samples": dataset_headers,
        "errors": [],
        "status": "PASS",
    }

    reference_values = [
        {
            "recording_id": f"recording-{index:02d}",
            "iEEGReference": REFERENCE,
            "matches_frozen_value": True,
        }
        for index in range(11)
    ]
    neural_report: dict[str, Any] = {
        "report_schema_version": "m6a-neural-metadata-audit-v2",
        "task_id": "M6A-PUBLIC-001",
        "dataset_id": "ds004703",
        "dataset_version": "1.1.0",
        "audited_at_utc": "2026-08-11T14:01:00+00:00",
        "dataset_root": dataset_root,
        "integrity_policy": "NON_HASH_AUDIT",
        "recording_count": 11,
        "necessary_metadata_readability": {
            "ieeg_sidecar_json_read": 11,
            "channels_tsv_read": 11,
            "events_tsv_read": 11,
            "audio_offset_json_read": 11,
            "expected_each": 11,
            "status": "PASS",
        },
        "sampling_rate_hz_values": [512.0, 1024.0],
        "power_line_frequency_hz_values": [60.0],
        "ieeg_reference_audit": {
            "expected_value": REFERENCE,
            "expected_recording_count": 11,
            "observed_recording_count": 11,
            "observed_unique_values": [REFERENCE],
            "recording_values": reference_values,
            "mismatch_recording_ids": [],
            "primary_policy": "AS_RECORDED_SCALP_REFERENCE",
            "status": "PASS",
        },
        "bids_layout": {
            "status": "PASS",
            "pybids_validate": True,
            "indexed_file_count": 377,
        },
        "analysis_eligible_neural_channel_count": 1346,
        "c_prefix_exclusion_count": 727,
        "spatial_metadata": {
            "standard_electrodes_tsv_count": 0,
            "standard_coordsystem_json_count": 0,
            "contact_ras_csv_count": 9,
            "status": "NONSTANDARD_COORDINATES_WITHOUT_ANATOMICAL_REGION_LABELS",
            "limitation": "No auditable atlas or anatomical region labels are available.",
        },
        "neural_target_gate": {
            "candidate": "LINE_HARMONIC_EXCLUDED_MULTIBAND_HIGH_GAMMA_LOG_POWER",
            "status": "METHOD_FROZEN_AWAITING_EXECUTION_GATES",
            "primary_reference_policy": "AS_RECORDED_SCALP_REFERENCE",
            "neural_extraction_allowed": False,
        },
        "recordings": recordings,
        "errors": [],
        "status": "PASS_WITH_METHOD_FROZEN_EXECUTION_BLOCKED",
    }
    split_report: dict[str, Any] = {
        "report_schema_version": "m6a-split-guard-v2",
        "task_id": "M6A-PUBLIC-001",
        "dataset_id": "ds004703",
        "dataset_version": "1.1.0",
        "status": "PASS",
        "rows": 319,
        "issues": [],
        "embargo_status": "PRELIMINARY_MINIMUM_ONLY",
        "preliminary_minimum_embargo_seconds": 2.0,
        "final_embargo_status": "PENDING_AUDIO_CONTEXT_MEASUREMENT_AND_GUARD_RERUN",
        "baseline_final": False,
        "split_counts": {"train": 223, "validation": 48, "test": 48},
        "block_assignments": {
            "block-01": "train",
            "block-02": "train",
            "block-03": "validation",
            "block-04": "test",
            "block-05": "train",
            "block-06": "train",
        },
        "language_counts": {"en": 319},
        "catalan_rows": 0,
    }
    return {
        "dataset_report": dataset_report,
        "neural_report": neural_report,
        "split_report": split_report,
        "config": load_json(ROOT / "configs" / "m6a_public_001.json"),
    }


def evaluate(bundle: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return validate_g2_promotion(
        bundle["dataset_report"],
        bundle["neural_report"],
        bundle["split_report"],
        bundle["config"],
    )


class G2PromotionGateTests(unittest.TestCase):
    def test_synthetic_complete_bundle_becomes_candidate_not_g2_pass(self) -> None:
        report = evaluate(valid_inputs())
        self.assertEqual(report["status"], CANDIDATE_STATUS)
        self.assertFalse(report["g2_pass_claimed"])
        self.assertFalse(report["candidate_contains_raw_data"])
        self.assertTrue(report["required_checks"])
        self.assertTrue(all(value is True for value in report["required_checks"].values()))

    def test_each_scientific_and_operational_gate_fails_closed(self) -> None:
        mutations: list[tuple[str, Callable[[dict[str, dict[str, Any]]], None], str]] = [
            ("dataset count", lambda b: b["dataset_report"].__setitem__("file_count", 376), "dataset_exact_inventory_totals"),
            ("dataset bytes", lambda b: b["dataset_report"].__setitem__("total_bytes", 1), "dataset_exact_inventory_totals"),
            ("active partial", lambda b: b["dataset_report"].__setitem__("active_partial_count", 1), "dataset_no_active_partials"),
            ("missing path", lambda b: b["dataset_report"]["expected_inventory_reconciliation"].__setitem__("missing_paths", ["missing.edf"]), "dataset_inventory_path_and_byte_reconciliation"),
            ("unexpected path", lambda b: b["dataset_report"]["expected_inventory_reconciliation"].__setitem__("unexpected_paths", ["extra.edf"]), "dataset_inventory_path_and_byte_reconciliation"),
            ("byte mismatch", lambda b: b["dataset_report"]["expected_inventory_reconciliation"].__setitem__("byte_mismatches", [{"path": "x"}]), "dataset_inventory_path_and_byte_reconciliation"),
            ("inventory provenance", lambda b: b["dataset_report"]["expected_inventory_provenance"].__setitem__("object_count", 376), "dataset_inventory_provenance"),
            ("DOI", lambda b: b["dataset_report"]["dataset_description"].__setitem__("DatasetDOI", "wrong"), "dataset_doi_license_and_readme_boundary"),
            ("README boundary", lambda b: b["dataset_report"]["dataset_boundary_checks"].__setitem__("readme_noncommercial_restriction_found", False), "dataset_doi_license_and_readme_boundary"),
            ("dataset EDF header", lambda b: b["dataset_report"]["ieeg_samples"].pop(), "dataset_11_of_11_edf_headers_read"),
            ("metadata count", lambda b: b["neural_report"]["necessary_metadata_readability"].__setitem__("events_tsv_read", 10), "neural_11_sidecars_channels_events_audio_offsets_read"),
            ("neural EDF header", lambda b: b["neural_report"]["recordings"][0].__setitem__("edf_header", None), "neural_11_of_11_edf_headers_read"),
            ("sampling", lambda b: b["neural_report"]["recordings"][0].__setitem__("edf_header_sampling_rate_matches_sidecar", False), "neural_sampling_512_or_1024_and_sidecar_match"),
            ("timeline", lambda b: b["neural_report"]["recordings"][0].__setitem__("events_within_edf_timeline", False), "neural_events_within_all_edf_timelines"),
            ("eligible name", lambda b: b["neural_report"]["recordings"][0].__setitem__("analysis_eligible_neural_channels_missing_from_edf", ["A1"]), "neural_all_analysis_eligible_names_exist_in_edf"),
            ("line frequency", lambda b: b["neural_report"].__setitem__("power_line_frequency_hz_values", [50.0]), "neural_sampling_512_or_1024_and_sidecar_match"),
            ("reference", lambda b: b["neural_report"]["ieeg_reference_audit"].__setitem__("status", "FAIL"), "neural_reference_11_of_11_as_recorded_scalp"),
            ("PyBIDS", lambda b: b["neural_report"]["bids_layout"].__setitem__("status", "FAIL"), "neural_pybids_validated_layout"),
            ("spatial limitation", lambda b: b["neural_report"]["spatial_metadata"].__setitem__("contact_ras_csv_count", 8), "neural_spatial_metadata_preserves_anatomy_limitation"),
            ("method state", lambda b: b["neural_report"]["neural_target_gate"].__setitem__("status", "CANDIDATE"), "neural_method_and_extraction_state_matches_config"),
            ("split rows", lambda b: b["split_report"].__setitem__("rows", 318), "split_319_rows_guard_pass"),
            ("anatomy", lambda b: b["config"]["anatomy_mapping"].__setitem__("status", "READY"), "config_anatomy_not_ready_and_region_not_estimable"),
            ("region", lambda b: b["config"]["anatomy_mapping"].__setitem__("region_summary_status", "PASS"), "config_anatomy_not_ready_and_region_not_estimable"),
            ("method", lambda b: b["config"]["neural_target"].__setitem__("status", "CANDIDATE"), "config_method_frozen_but_neural_extraction_blocked"),
            ("extraction", lambda b: b["config"]["neural_target"].__setitem__("neural_extraction_allowed", True), "config_method_frozen_but_neural_extraction_blocked"),
            (
                "baseline acceptance drift",
                lambda b: b["config"]["split"].__setitem__("baseline_final", False),
                "config_final_embargo_state_is_accepted_not_whole_m6a",
            ),
        ]
        for label, mutate, expected_check in mutations:
            with self.subTest(label=label):
                bundle = copy.deepcopy(valid_inputs())
                mutate(bundle)
                report = evaluate(bundle)
                self.assertEqual(report["status"], "FAIL")
                self.assertIn(expected_check, report["failed_checks"])

    def test_cross_report_path_channel_split_and_time_mismatches_fail_closed(self) -> None:
        cases: list[tuple[str, Callable[[dict[str, dict[str, Any]]], None], str]] = [
            ("EDF set mismatch", lambda b: b["neural_report"]["recordings"][0].__setitem__("edf_file", "sub-99/wrong.edf"), "dataset_and_neural_edf_paths_match"),
            ("EDF header path mismatch", lambda b: b["neural_report"]["recordings"][0]["edf_header"].__setitem__("path", "/wrong/root.edf"), "dataset_and_neural_edf_paths_match"),
            ("empty eligible names", lambda b: b["neural_report"]["recordings"][0]["channels"].__setitem__("analysis_eligible_neural_names", []), "neural_channel_selection_identity"),
            ("wrong eligible aggregate", lambda b: b["neural_report"].__setitem__("analysis_eligible_neural_channel_count", 1345), "neural_channel_selection_identity"),
            ("wrong c-prefix aggregate", lambda b: b["neural_report"].__setitem__("c_prefix_exclusion_count", 726), "neural_channel_selection_identity"),
            ("wrong split count", lambda b: b["split_report"]["split_counts"].__setitem__("train", 222), "split_exact_counts_blocks_and_language"),
            ("wrong block", lambda b: b["split_report"]["block_assignments"].__setitem__("block-03", "train"), "split_exact_counts_blocks_and_language"),
            ("wrong language", lambda b: b["split_report"].__setitem__("language_counts", {"ca": 319}), "split_exact_counts_blocks_and_language"),
            ("different dataset root", lambda b: b["neural_report"].__setitem__("dataset_root", "/different/root"), "dataset_and_neural_roots_match_config"),
            ("audit time too far apart", lambda b: b["neural_report"].__setitem__("audited_at_utc", "2026-08-11T14:31:00+00:00"), "dataset_and_neural_audits_within_30_minutes"),
        ]
        for label, mutate, expected_check in cases:
            with self.subTest(label=label):
                bundle = copy.deepcopy(valid_inputs())
                mutate(bundle)
                report = evaluate(bundle)
                self.assertEqual(report["status"], "FAIL")
                self.assertIn(expected_check, report["failed_checks"])

    def test_stale_status_missing_field_and_nonfinite_value_fail_closed(self) -> None:
        cases: list[tuple[str, Callable[[dict[str, dict[str, Any]]], None], str]] = [
            ("stale status", lambda b: b["neural_report"].__setitem__("status", "PASS_WITH_METHOD_FREEZE_CANDIDATE_NOT_AUTHORIZED"), "neural_report_pass_with_method_frozen_and_execution_blocked"),
            ("missing field", lambda b: b["split_report"].pop("rows"), "split_319_rows_guard_pass"),
            ("nonfinite", lambda b: b["neural_report"]["recordings"][0].__setitem__("audio_offset_seconds", float("inf")), "neural_report_all_numeric_values_finite"),
            ("tuple nonfinite", lambda b: b["neural_report"].__setitem__("tuple_values", (0.0, float("-inf"))), "neural_report_all_numeric_values_finite"),
        ]
        for label, mutate, expected_check in cases:
            with self.subTest(label=label):
                bundle = copy.deepcopy(valid_inputs())
                mutate(bundle)
                report = evaluate(bundle)
                self.assertEqual(report["status"], "FAIL")
                self.assertIn(expected_check, report["failed_checks"])

    def test_malformed_config_section_fails_instead_of_raising(self) -> None:
        bundle = valid_inputs()
        bundle["config"]["dataset"] = "not-an-object"
        report = evaluate(bundle)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("main_config_identity_and_semantic_gate", report["failed_checks"])

    def test_strict_json_loader_rejects_nonstandard_numeric_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-standard JSON numeric constant"):
                load_strict_json(path)

    def test_cli_missing_candidate_reports_is_pending_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "gate.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--dataset-report",
                        str(root / "dataset.json"),
                        "--neural-report",
                        str(root / "neural.json"),
                        "--split-report",
                        str(root / "split.json"),
                        "--config",
                        str(root / "config.json"),
                        "--output",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotEqual(exit_code, 0)
            self.assertEqual(report["status"], "PENDING")
            self.assertFalse(report["g2_pass_claimed"])


if __name__ == "__main__":
    unittest.main()
