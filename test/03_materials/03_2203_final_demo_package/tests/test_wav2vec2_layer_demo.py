from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from route_b_temporal_models.wav2vec2_layer_demo.core import downsample, levenshtein, verify_delivery


class MetricTests(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("KITTEN", "SITTING"), 3)
        self.assertEqual(levenshtein("", "ABC"), 3)
        self.assertEqual(levenshtein("ABC", "ABC"), 0)

    def test_downsample_is_bounded_and_finite(self):
        values = downsample(range(100), target=12)
        self.assertEqual(len(values), 12)
        self.assertTrue(all(isinstance(value, float) for value in values))

    def test_final_delivery_when_results_exist(self):
        pointer = ROOT / "outputs" / "TB001-DEMO001" / "current_run_group.json"
        if not pointer.exists():
            self.skipTest("final results not generated yet")
        result = verify_delivery(ROOT)
        self.assertEqual(result["status"], "PASS", json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
