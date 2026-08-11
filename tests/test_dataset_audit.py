from __future__ import annotations

import unittest

from scripts.dataset_audit import reconcile_expected_inventory
from scripts.neural_metadata_audit import channel_inventory_difference, summarize_channels


class DatasetAuditTests(unittest.TestCase):
    def test_exact_path_and_byte_reconciliation_passes(self) -> None:
        expected = [{"path": "a.edf", "bytes": "10"}, {"path": "b.tsv", "bytes": "2"}]
        actual = [{"path": "a.edf", "bytes": 10}, {"path": "b.tsv", "bytes": 2}]
        report = reconcile_expected_inventory(expected, actual)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["expected_total_bytes"], 12)

    def test_missing_unexpected_and_wrong_size_fail_closed(self) -> None:
        expected = [{"path": "a.edf", "bytes": "10"}, {"path": "b.tsv", "bytes": "2"}]
        actual = [{"path": "a.edf", "bytes": 9}, {"path": "partial.tmp", "bytes": 1}]
        report = reconcile_expected_inventory(expected, actual)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["missing_paths"], ["b.tsv"])
        self.assertEqual(report["unexpected_paths"], ["partial.tmp"])
        self.assertEqual(report["byte_mismatches"][0]["path"], "a.edf")

    def test_channel_summary_applies_readme_c_prefix_exclusion(self) -> None:
        rows = [
            {"name": "LA1", "type": "SEEG", "units": "uV", "status": "good", "status_description": "n/a"},
            {"name": "C1", "type": "SEEG", "units": "uV", "status": "good", "status_description": "unused"},
            {"name": "DC1", "type": "MISC", "units": "uV", "status": "good", "status_description": "room audio"},
        ]
        report = summarize_channels(rows)
        self.assertEqual(report["good_neural_channel_count"], 2)
        self.assertEqual(report["analysis_eligible_neural_channel_count"], 1)
        self.assertEqual(report["analysis_eligible_neural_names"], ["LA1"])
        self.assertEqual(report["c_prefix_names"], ["C1"])
        self.assertEqual(report["dc1_channels"][0]["description"], "room audio")

    def test_channel_inventory_difference_preserves_names_and_types(self) -> None:
        rows = [
            {"name": "LA1", "type": "SEEG", "status": "good"},
            {"name": "Trigger Event", "type": "MISC", "status": "bad"},
        ]
        report = channel_inventory_difference(rows, ["LA1", "EDF annotation"])
        self.assertEqual(
            report["tsv_only"],
            [
                {
                    "name": "Trigger Event",
                    "type": "MISC",
                    "status": "bad",
                    "difference": "CHANNELS_TSV_ONLY",
                }
            ],
        )
        self.assertEqual(report["edf_only"][0]["name"], "EDF annotation")
        self.assertEqual(report["edf_only"][0]["type"], "UNDECLARED_IN_CHANNELS_TSV")


if __name__ == "__main__":
    unittest.main()
