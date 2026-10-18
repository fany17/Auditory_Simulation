"""Technical checks only; fixtures remain on 2203 and are not science results."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pub_01_s03_strong_preflight import array_smoke, contained, run, split_coverage


class AuditSafetyTests(unittest.TestCase):
    def setUp(self):
        # Deliberately retained, never permanently deleted.
        self.root = Path(tempfile.mkdtemp(prefix="strong_preflight_fixture_"))

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            contained(self.root, "../outside")
        self.assertEqual(contained(self.root, "a/b"), self.root / "a/b")

    def test_windows_cannot_read_signals(self):
        with patch("scripts.pub_01_s03_strong_preflight.platform.system", return_value="Windows"):
            with self.assertRaises(RuntimeError):
                run({"patient_data": "FORBIDDEN"})

    def test_outside_signal_paths_do_not_get_loaded(self):
        row = {"subject": "sub-001", "source_recording": "r", "story": "story",
               "sample_rate_hz": "64", "rate_basis": "fixture", "physical_sync": "UNKNOWN",
               "eeg_path": str(self.root.parent / "outside.npy"), "envelope_path": "unused"}
        result = array_smoke([row], self.root)
        self.assertEqual(result[0]["status"], "HOLD_READ")
        self.assertIn("outside public project", result[0]["error"])

    def test_partial_split_coverage_does_not_claim_readiness(self):
        folder = self.root / "source_inspection_only/HDPS/derivatives/SparrKULee"
        folder.mkdir(parents=True)
        (folder / "split.json").write_text(json.dumps({"sub-001": {"train": ["a"], "test": ["b"]}}))
        # An existing file is coverage metadata only; never loaded as an array.
        eeg = self.root / "a.npy"
        eeg.write_text("fixture only")
        result = split_coverage(self.root, [{"subject": "sub-001", "story": "a", "eeg_path": str(eeg)}])
        self.assertEqual(result["HDPS"]["requested_record_phase_pairs"], 2)
        self.assertEqual(result["HDPS"]["existing_64hz_matches"], 1)
        self.assertIn("not official 100Hz readiness", result["HDPS"]["warning"])


if __name__ == "__main__":
    unittest.main()
