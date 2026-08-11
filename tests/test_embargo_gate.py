from __future__ import annotations

import math
import unittest

from m6a_public.embargo_gate import evaluate_final_embargo


class EmbargoGateTests(unittest.TestCase):
    def test_pending_measurements_cannot_be_baseline_final(self) -> None:
        report = evaluate_final_embargo(
            {
                "preliminary_minimum_embargo_seconds": 2.0,
                "maximum_encoding_lag_seconds": 0.5,
                "filter_or_padding_edge_seconds": None,
                "audio_model_receptive_field_or_context_overlap_seconds": None,
            }
        )
        self.assertEqual(report["status"], "PENDING_MEASUREMENT")
        self.assertFalse(report["baseline_final"])
        self.assertIsNone(report["final_embargo_seconds"])

    def test_final_embargo_is_maximum_component(self) -> None:
        report = evaluate_final_embargo(
            {
                "preliminary_minimum_embargo_seconds": 2.0,
                "maximum_encoding_lag_seconds": 0.5,
                "filter_or_padding_edge_seconds": 3.25,
                "audio_model_receptive_field_or_context_overlap_seconds": 0.0,
            }
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["final_embargo_seconds"], 3.25)
        self.assertTrue(report["baseline_final"])

    def test_negative_and_nonfinite_components_fail_closed(self) -> None:
        for invalid in (-1.0, math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                report = evaluate_final_embargo(
                    {
                        "preliminary_minimum_embargo_seconds": 2.0,
                        "maximum_encoding_lag_seconds": 0.5,
                        "filter_or_padding_edge_seconds": invalid,
                        "audio_model_receptive_field_or_context_overlap_seconds": 0.0,
                    }
                )
                self.assertEqual(report["status"], "FAIL")
                self.assertFalse(report["baseline_final"])


if __name__ == "__main__":
    unittest.main()
