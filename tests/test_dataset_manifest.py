from __future__ import annotations

import unittest

from m6a_public.dataset_manifest import (
    classify_stimulus,
    connected_component_summary,
    contiguous_groups,
    deterministic_block_split,
    split_checks,
    speaker_id_for,
)


class DatasetManifestTests(unittest.TestCase):
    def test_language_and_role_classification(self) -> None:
        self.assertEqual(classify_stimulus("catalan-m01")[:2], ("PASSAGE", "ca"))
        self.assertEqual(classify_stimulus("s1303a-ex01")[:2], ("PASSAGE", "en"))
        self.assertEqual(classify_stimulus("welcome")[:2], ("CONTROL", "en"))
        self.assertEqual(classify_stimulus("unmapped")[:2], ("UNKNOWN", "UNKNOWN"))
        self.assertEqual(speaker_id_for("s1303a-ex02", "PASSAGE"), "s1303a")

    def test_block_split_has_four_one_one_profile(self) -> None:
        mapping = deterministic_block_split([f"block-{index:02d}" for index in range(1, 7)], 20260811)
        counts = {name: list(mapping.values()).count(name) for name in ("train", "validation", "test")}
        self.assertEqual(counts, {"train": 4, "validation": 1, "test": 1})
        self.assertEqual(mapping, deterministic_block_split(list(reversed(mapping)), 20260811))

    def test_contiguous_groups_preserve_repeated_control_segments(self) -> None:
        rows = [{"ex_name": "cue"}, {"ex_name": "cue"}, {"ex_name": "passage"}, {"ex_name": "cue"}]
        groups = contiguous_groups(rows)
        self.assertEqual([group[0]["ex_name"] for group in groups], ["cue", "passage", "cue"])
        self.assertEqual([len(group) for group in groups], [2, 1, 1])

    def test_original_recording_and_stimulus_grouping_forms_one_component(self) -> None:
        rows = [
            {"recording_id": "r1", "stimulus_id": "s1"},
            {"recording_id": "r1", "stimulus_id": "s2"},
            {"recording_id": "r2", "stimulus_id": "s2"},
        ]
        report = connected_component_summary(rows)
        self.assertEqual(report["component_count"], 1)
        self.assertFalse(report["feasible_for_three_splits"])

    def test_split_checks_detect_group_and_temporal_leakage(self) -> None:
        rows = [
            {"segment_id": "a", "recording_id": "r1", "stimulus_id": "s1", "block_id": "b1", "language": "en", "split": "train", "audio_onset_seconds": 0.0, "audio_offset_seconds": 5.0},
            {"segment_id": "b", "recording_id": "r1", "stimulus_id": "s1", "block_id": "b1", "language": "en", "split": "test", "audio_onset_seconds": 6.0, "audio_offset_seconds": 10.0},
        ]
        checks = split_checks(rows, 2.0)
        self.assertEqual(checks["stimulus_split_conflicts"], ["s1"])
        self.assertEqual(checks["block_split_conflicts"], ["b1"])
        self.assertEqual(len(checks["temporal_cross_split_violations"]), 1)


if __name__ == "__main__":
    unittest.main()
