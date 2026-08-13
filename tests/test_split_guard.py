from __future__ import annotations

import unittest

from m6a_public.split_guard import Assignment, summarize_assignments, validate_assignments


def row(
    sample_id: str,
    split: str,
    recording_id: str,
    stimulus_id: str,
    start_sec: float = 0.0,
    end_sec: float = 1.0,
    speaker_id: str = "",
    block_id: str = "block-a",
    language: str = "en",
) -> Assignment:
    return Assignment(
        sample_id=sample_id,
        split=split,
        subject_id="sub-01",
        session_id="ses-01",
        recording_id=recording_id,
        stimulus_id=stimulus_id,
        block_id=block_id,
        language=language,
        speaker_id=speaker_id,
        start_sec=start_sec,
        end_sec=end_sec,
    )


class SplitGuardTests(unittest.TestCase):
    def test_summary_records_actual_split_block_and_language_counts(self) -> None:
        rows = [
            row("a", "train", "rec-a", "stim-a", block_id="block-01", language="en"),
            row("b", "validation", "rec-b", "stim-b", block_id="block-03", language="en"),
            row("c", "test", "rec-c", "stim-c", block_id="block-04", language="en"),
        ]
        self.assertEqual(
            summarize_assignments(rows),
            {
                "split_counts": {"test": 1, "train": 1, "validation": 1},
                "block_assignments": {
                    "block-01": "train",
                    "block-03": "validation",
                    "block-04": "test",
                },
                "language_counts": {"en": 3},
                "catalan_rows": 0,
            },
        )

    def test_clean_grouped_split_passes(self) -> None:
        rows = [
            row("a", "train", "rec-a", "stim-a"),
            row("b", "validation", "rec-b", "stim-b"),
            row("c", "test", "rec-c", "stim-c"),
        ]
        self.assertEqual(
            validate_assignments(rows, ["stimulus_id", "recording_id"]),
            [],
        )

    def test_stimulus_leakage_fails(self) -> None:
        rows = [
            row("a", "train", "rec-a", "stim-shared"),
            row("b", "validation", "rec-b", "stim-shared"),
            row("c", "test", "rec-c", "stim-c"),
        ]
        issues = validate_assignments(rows, ["stimulus_id"])
        self.assertTrue(any("group leakage stimulus_id=stim-shared" in item for item in issues))

    def test_recording_leakage_fails(self) -> None:
        rows = [
            row("a", "train", "rec-shared", "stim-a"),
            row("b", "validation", "rec-shared", "stim-b", start_sec=10, end_sec=11),
            row("c", "test", "rec-c", "stim-c"),
        ]
        issues = validate_assignments(rows, ["recording_id"])
        self.assertTrue(any("group leakage recording_id=rec-shared" in item for item in issues))

    def test_temporal_embargo_fails_cross_split_neighbours(self) -> None:
        rows = [
            row("a", "train", "rec-shared", "stim-a", start_sec=0, end_sec=1),
            row("b", "validation", "rec-shared", "stim-b", start_sec=2.5, end_sec=3.5),
            row("c", "test", "rec-c", "stim-c"),
        ]
        issues = validate_assignments(
            rows,
            required_group_keys=[],
            temporal_embargo_seconds=2.0,
        )
        self.assertTrue(any("temporal leakage" in item for item in issues))

    def test_optional_speaker_leakage_is_visible(self) -> None:
        rows = [
            row("a", "train", "rec-a", "stim-a", speaker_id="speaker-x"),
            row("b", "validation", "rec-b", "stim-b", speaker_id="speaker-x"),
            row("c", "test", "rec-c", "stim-c", speaker_id="speaker-y"),
        ]
        issues = validate_assignments(rows, [], optional_group_keys=["speaker_id"])
        self.assertTrue(any("optional group leakage speaker_id=speaker-x" in item for item in issues))

    def test_language_stratification_requires_all_splits(self) -> None:
        rows = [
            row("a", "train", "rec-a", "stim-a", language="en"),
            row("b", "validation", "rec-b", "stim-b", language="en"),
            row("c", "test", "rec-c", "stim-c", language="en"),
            row("d", "train", "rec-d", "stim-d", language="ca"),
        ]
        issues = validate_assignments(rows, [], stratification_keys=["language"])
        self.assertTrue(any("stratification coverage missing language=ca" in item for item in issues))


if __name__ == "__main__":
    unittest.main()
