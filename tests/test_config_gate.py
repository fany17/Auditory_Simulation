from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from m6a_public.config_gate import find_forbidden_fields, load_json, validate_task_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_json(ROOT / "configs" / "m6a_public_001.json")

    def test_repository_config_passes(self) -> None:
        self.assertEqual(validate_task_config(self.config), [])

    def test_download_requires_strict_license_boundary(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["dataset"]["license_status"] = "PENDING"
        self.assertTrue(any("download requires" in item for item in validate_task_config(changed)))

    def test_model_must_remain_frozen(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["model"]["trainable"] = True
        self.assertTrue(any("remain frozen" in item for item in validate_task_config(changed)))

    def test_refrozen_split_records_original_no_go_and_block_assignments(self) -> None:
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
        self.assertEqual(split["split_status"], "PRELIMINARY_NOT_BASELINE_FINAL")
        self.assertIsNone(split["final_embargo_seconds"])
        self.assertFalse(split["subject_heldout_claim_allowed"])
        self.assertFalse(split["speaker_heldout_claim_allowed"])
        self.assertFalse(split["cross_language_claim_allowed"])

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
        self.assertEqual(artifact["exchange_contract_status"], "REVISED_DRAFT_AWAITING_CONSUMER_REVIEW")
        self.assertIs(artifact["frozen_m6a_artifact_exists"], False)


if __name__ == "__main__":
    unittest.main()
