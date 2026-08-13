from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from m6a_public.config_gate import (
    find_forbidden_fields,
    load_json,
    validate_candidate_report_governance,
    validate_task_config,
)


ROOT = Path(__file__).resolve().parents[1]


class ConfigGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_json(ROOT / "configs" / "m6a_public_001.json")

    def test_repository_config_passes(self) -> None:
        self.assertEqual(validate_task_config(self.config), [])
        self.assertEqual(validate_candidate_report_governance(ROOT), [])

    def test_download_requires_strict_license_boundary(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["dataset"]["license_status"] = "PENDING"
        self.assertTrue(any("download requires" in item for item in validate_task_config(changed)))

    def test_model_must_remain_frozen(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["model"]["trainable"] = True
        self.assertTrue(any("remain frozen" in item for item in validate_task_config(changed)))

    def test_model_cache_is_validated_remote_only_and_download_closed(self) -> None:
        model = self.config["model"]
        self.assertIs(model["download_allowed"], False)
        self.assertEqual(model["cache_state"], "SEMANTICALLY_VALIDATED_REMOTE_ONLY")

    def test_wav2vec2_preprocessing_contract_is_fail_closed(self) -> None:
        for field, value in (
            ("do_normalize", False),
            ("return_attention_mask", True),
            ("attention_mask_argument", "ALL_ONES"),
            ("cross_passage_statistics_allowed", True),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.config)
                changed["model"]["inference_input"]["preprocessing"][field] = value
                self.assertTrue(validate_task_config(changed))

    def test_neural_target_method_is_frozen_but_execution_remains_blocked(self) -> None:
        target = self.config["neural_target"]
        self.assertEqual(target["status"], "METHOD_FROZEN_AWAITING_EXECUTION_GATES")
        self.assertEqual(target["resolution_status"], "METHOD_FROZEN")
        self.assertEqual(target["method_coordinator_review"], "ACCEPT")
        self.assertEqual(target["primary_reference_policy"], "AS_RECORDED_SCALP_REFERENCE")
        self.assertEqual(target["sidecar_reference_recording_count"], 11)
        self.assertEqual(target["observed_power_line_frequency_hz"], 60)
        self.assertIn(120, target["line_harmonics_inside_candidate_band_hz"])
        self.assertIs(target["neural_extraction_allowed"], False)

    def test_anatomy_mapping_blocks_region_summary_without_blocking_electrode_smoke(self) -> None:
        anatomy = self.config["anatomy_mapping"]
        self.assertEqual(anatomy["status"], "ANATOMY_MAPPING_NOT_READY")
        self.assertEqual(anatomy["region_summary_status"], "NOT_ESTIMABLE")
        self.assertFalse(anatomy["contact_name_inference_allowed"])
        self.assertTrue(anatomy["electrode_level_smoke_allowed_after_other_gates"])
        self.assertNotIn("region_summary", self.config["baseline"]["secondary_metrics"])

    def test_baseline_final_split_records_original_no_go_and_block_assignments(self) -> None:
        split = self.config["split"]
        self.assertEqual(set(split["required_group_keys"]), {"stimulus_id", "block_id"})
        self.assertEqual(split["original_recording_grouping_status"], "INFEASIBLE_SINGLE_CONNECTED_COMPONENT")
        self.assertEqual(
            split["block_assignments"],
            {
                "block-01": "train",
                "block-02": "train",
                "block-03": "validation",
                "block-04": "test",
                "block-05": "train",
                "block-06": "train",
            },
        )
        self.assertEqual(split["split_status"], "BASELINE_FINAL_COORDINATOR_ACCEPTED")
        self.assertEqual(split["final_embargo_seconds"], 2.0)
        self.assertFalse(split["subject_heldout_claim_allowed"])
        self.assertFalse(split["speaker_heldout_claim_allowed"])
        self.assertFalse(split["cross_language_claim_allowed"])

    def test_g2_is_accepted_only_for_audio_context_gate(self) -> None:
        self.assertEqual(
            self.config["g2"]["status"],
            "G2_COORDINATOR_ACCEPTED_FOR_AUDIO_CONTEXT_GATE",
        )
        self.assertEqual(self.config["g2"]["coordinator_review"], "ACCEPT")
        self.assertIs(self.config["g2"]["whole_m6a_pass_claimed"], False)

    def test_final_embargo_and_split_are_coordinator_accepted(self) -> None:
        split = self.config["split"]
        self.assertEqual(split["split_status"], "BASELINE_FINAL_COORDINATOR_ACCEPTED")
        self.assertEqual(
            split["final_embargo_status"],
            "FINAL_EMBARGO_COORDINATOR_ACCEPTED",
        )
        self.assertEqual(split["final_embargo_candidate_seconds"], 2.0)
        self.assertEqual(split["final_embargo_seconds"], 2.0)
        self.assertIs(split["baseline_final"], True)

    def test_g3_authorization_is_single_recording_and_global_extraction_stays_closed(self) -> None:
        g3 = self.config["g3_single_recording"]
        self.assertEqual(
            g3["status"],
            "G3_SINGLE_RECORDING_COORDINATOR_ACCEPTED_ENGINEERING_ONLY",
        )
        self.assertEqual(g3["coordinator_review"], "ACCEPT")
        self.assertFalse(g3["scientific_result_claimed"])
        self.assertEqual(
            g3["representation_evidence_scope"], "ENGINEERING_SHAPE_TIME_ONLY"
        )
        self.assertFalse(
            g3["representation_reuse_for_g4_scientific_baseline_allowed"]
        )
        self.assertEqual(g3["eligible_channel_count"], 36)
        self.assertFalse(g3["other_recordings_allowed"])
        self.assertFalse(g3["other_segments_allowed"])
        self.assertFalse(g3["whole_dataset_neural_extraction_allowed"])
        self.assertFalse(self.config["neural_target"]["neural_extraction_allowed"])

    def test_g4_resource_bounds_are_frozen_but_execution_is_not_authorized(self) -> None:
        resources = self.config["resources"]
        self.assertEqual(resources["smoke_gpu_hours_limit"], 2)
        self.assertEqual(resources["minimum_free_bytes"], 500_000_000_000)
        self.assertFalse(self.config["features"]["g4_execution_authorized"])
        changed = copy.deepcopy(self.config)
        changed["resources"]["smoke_gpu_hours_limit"] = 3
        self.assertTrue(validate_task_config(changed))

    def test_g4_protocol_amendment_preserves_prior_acceptance_and_execution_stays_closed(self) -> None:
        protocol = self.config["g4_protocol"]
        self.assertEqual(
            protocol["status"],
            "G4_PROTOCOL_AMENDMENT_CANDIDATE_AWAITING_COORDINATOR_REVIEW",
        )
        self.assertEqual(protocol["coordinator_review"], "PENDING")
        self.assertIsNone(protocol["reviewed_on"])
        self.assertEqual(
            protocol["prior_accepted_status"], "G4_PROTOCOL_COORDINATOR_ACCEPTED"
        )
        self.assertFalse(
            protocol[
                "g3_raw_input_representation_reuse_for_g4_scientific_baseline_allowed"
            ]
        )
        self.assertFalse(protocol["scientific_result_claimed"])
        self.assertFalse(protocol["g4_execution_authorized"])
        self.assertEqual(
            protocol["candidate_report"],
            "reports/g4_protocol_amendment_candidate_20260813_v2.json",
        )
        self.assertEqual(
            protocol["preflight_report"],
            "reports/g4_resource_runtime_preflight_candidate_20260813_v3.json",
        )
        self.assertEqual(
            protocol["preflight_status"],
            "G4_RESOURCE_AND_RUNTIME_PREFLIGHT_CANDIDATE_AWAITING_COORDINATOR_REVIEW",
        )

    def test_only_v2_and_v3_are_current_candidate_reports(self) -> None:
        protocol_old = load_json(
            ROOT / "reports" / "g4_protocol_amendment_candidate_20260813.json"
        )
        protocol_current = load_json(
            ROOT / "reports" / "g4_protocol_amendment_candidate_20260813_v2.json"
        )
        preflight_v2 = load_json(
            ROOT / "reports" / "g4_resource_runtime_preflight_candidate_20260813_v2.json"
        )
        preflight_current = load_json(
            ROOT / "reports" / "g4_resource_runtime_preflight_candidate_20260813_v3.json"
        )
        self.assertEqual(
            protocol_old["status"], "SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE"
        )
        self.assertIs(protocol_old["current_candidate"], False)
        self.assertIs(protocol_current["current_candidate"], True)
        self.assertEqual(
            preflight_v2["status"], "SUPERSEDED_PROVENANCE_NOT_CURRENT_CANDIDATE"
        )
        self.assertIs(preflight_v2["current_candidate"], False)
        self.assertIs(preflight_current["current_candidate"], True)

    def test_governance_rejects_superseded_report_promoted_as_current(self) -> None:
        required = [
            "configs/m6a_public_001.json",
            "reports/g4_protocol_amendment_candidate_20260813.json",
            "reports/g4_protocol_amendment_candidate_20260813_v2.json",
            "reports/g4_resource_runtime_preflight_candidate_20260813.json",
            "reports/g4_resource_runtime_preflight_candidate_20260813_v2.json",
            "reports/g4_resource_runtime_preflight_candidate_20260813_v3.json",
            "reports/wav2vec2_preprocessor_mirror_audit_20260813.json",
            "reports/g4_protocol_candidate_20260813_v3.json",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in required:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    (ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
                )
            old = root / "reports/g4_protocol_amendment_candidate_20260813.json"
            payload = load_json(old)
            payload["status"] = (
                "G4_PROTOCOL_AMENDMENT_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
            )
            payload["current_candidate"] = True
            old.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(validate_candidate_report_governance(root))

    def test_forbidden_integrity_field_is_detected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["dataset"]["checksum"] = "not-allowed"
        self.assertEqual(find_forbidden_fields(changed), ["$.dataset.checksum"])

    def test_artifact_schema_contains_no_forbidden_fields(self) -> None:
        internal_schema = json.loads(
            (ROOT / "schemas" / "m6a_public_internal_manifest.schema.json").read_text(encoding="utf-8")
        )
        exchange_schema = json.loads(
            (ROOT / "schemas" / "m6a_to_m6b_exchange_manifest_v1.schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(find_forbidden_fields(internal_schema), [])
        self.assertEqual(find_forbidden_fields(exchange_schema), [])
        payload_formats = exchange_schema["properties"]["reference_feature_payload"]["properties"]["format"]["enum"]
        self.assertNotIn("TSV", payload_formats)
        canary_formats = exchange_schema["properties"]["canary_fixture"]["properties"]["expected_output_format"]["enum"]
        self.assertIn("TINY_TSV", canary_formats)

    def test_exchange_contract_is_not_frozen(self) -> None:
        artifact = self.config["artifact"]
        self.assertEqual(
            artifact["exchange_contract_status"],
            "REVISED_DRAFT_ACCEPTED_FOR_CANDIDATE_PREPARATION",
        )
        self.assertEqual(artifact["exchange_consumer_status"], "READY_WAITING_M6A_CANDIDATE")
        self.assertEqual(artifact["consumer_cross_test_status"], "NOT_RUN")
        self.assertIs(artifact["exchange_candidate_exists"], False)
        self.assertIs(artifact["frozen_m6a_artifact_exists"], False)


if __name__ == "__main__":
    unittest.main()
