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

    def test_forbidden_integrity_field_is_detected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["dataset"]["checksum"] = "not-allowed"
        self.assertEqual(find_forbidden_fields(changed), ["$.dataset.checksum"])

    def test_artifact_schema_contains_no_forbidden_fields(self) -> None:
        schema = json.loads((ROOT / "schemas" / "m6a_artifact_manifest.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(find_forbidden_fields(schema), [])


if __name__ == "__main__":
    unittest.main()
