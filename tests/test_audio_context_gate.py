from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from m6a_public.audio_context_gate import (
    CANDIDATE_STATUS,
    EXPECTED_CONFIG_PROFILE,
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
    EXPECTED_WEIGHT_SUFFIX_SHAPES,
    MODEL_FILES,
    REPORT_SCHEMA_VERSION,
    SOURCE_SAMPLE_RATE_HZ,
    audit_split_audio_identity,
    crop_mono_passage,
    derive_convolution_timing,
    expected_frame_count,
    finalize_candidate_report,
    frame_center_seconds,
    frozen_resampling_spec,
    load_isolated_passage,
    load_strict_json_object,
    normalize_relative_audio_path,
    resample_independent_passage,
)
from m6a_public.config_gate import load_json
from m6a_public.embargo_gate import evaluate_final_embargo_candidate
from scripts.download_wav2vec2_model import validate_download_authorization


ROOT = Path(__file__).resolve().parents[1]


def valid_evidence(config: dict[str, Any]) -> dict[str, Any]:
    timing = derive_convolution_timing((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2))
    input_samples = 16_000
    frames = expected_frame_count(input_samples, timing)
    centers = frame_center_seconds(frames, timing)
    inventory = [
        {
            "path": name,
            "bytes": index + 100,
            "modified_at_utc": "2026-08-13T00:00:00+00:00",
            "sample_readability": True,
        }
        for index, name in enumerate(MODEL_FILES)
    ]
    components = config["split"]["final_embargo_components_seconds"]  # type: ignore[index]
    audio_assignments = [
        {
            "audio_file": f"stimuli/excerpts/Block 1/stimulus-{index:02d}.wav",
            "stimulus_ids": [f"stimulus-{index:02d}"],
            "block_ids": ["block-01"],
            "splits": ["train"],
        }
        for index in range(48)
    ]
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "created_at_utc": "2026-08-13T00:00:00+00:00",
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "model": {
            "model_id": "facebook/wav2vec2-base",
            "revision_label": "main",
            "resolved_at": "2026-08-13",
            "revision_limitation": "MUTABLE_MAIN_LABEL_NON_CRYPTOGRAPHIC_REPRODUCIBILITY_ONLY",
            "source_endpoint": "https://hf-mirror.com",
            "source_endpoint_role": "PUBLIC_HUGGING_FACE_ENDPOINT_MIRROR",
            "source_endpoint_limitation": (
                "THIRD_PARTY_MIRROR_PLUS_MUTABLE_MAIN_AND_NO_HASH_POLICY_DO_NOT_PROVIDE_"
                "CRYPTOGRAPHIC_INTEGRITY_OR_IMMUTABLE_PROVENANCE"
            ),
            "declared_license": "Apache-2.0",
            "trainable": False,
            "cache_state": "SEMANTICALLY_VALIDATED_REMOTE_ONLY",
            "config_semantic_audit": {
                "status": "PASS",
                "observed": EXPECTED_CONFIG_PROFILE,
                "errors": [],
            },
            "weight_semantic_audit": {
                "status": "PASS",
                "weights_only": True,
                "tensor_only": True,
                "tensor_count": 213,
                "key_parameter_shapes": {
                    suffix: shape for suffix, shape in EXPECTED_WEIGHT_SUFFIX_SHAPES.items()
                },
                "errors": [],
            },
        },
        "runtime": {
            "python": "3.13.5",
            "torch": "2.11.0",
            "transformers": "5.14.1",
            "numpy": "2.4.6",
            "scipy": "1.17.0",
            "soundfile": "0.13.1",
            "conda_environment": "auditory_m6a_public_001",
        },
        "model_cache_path": "/home/fanyu/auditory_simulation_m6a/cache/huggingface/facebook_wav2vec2_base_main_20260813",
        "model_cache_remote_only": True,
        "model_inventory": inventory,
        "model_inventory_total_bytes": sum(item["bytes"] for item in inventory),
        "input_semantics": {
            "source_sample_rate_hz": 44100,
            "model_sample_rate_hz": 16000,
            "channels": 1,
            "mono_policy": "REQUIRE_MONO_NO_IMPLICIT_DOWNMIX",
            "passage_policy": "ONE_ELIGIBLE_PASSAGE_PER_INFERENCE_CALL",
            "crop_interval": "HALF_OPEN_INTEGER_SAMPLE_BOUNDS_WITHIN_STANDALONE_PASSAGE_FILE",
            "neighbor_audio_read_allowed": False,
            "batch_padding": "FORBIDDEN_PRIMARY_INFERENCE",
        },
        "resampling": asdict(frozen_resampling_spec()),
        "split_sentinel": {
            "status": "PASS",
            "read_paths": ["train/passage.wav"],
            "forbidden_read_count": 0,
            "heldout_sentinel_observed": False,
            "output_samples": 100,
        },
        "model_canary": {
            "local_files_only": True,
            "trust_remote_code": False,
            "repository_custom_code_executed": False,
            "loading_info": {
                "missing_keys": [],
                "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
                "mismatched_keys": [],
                "error_msgs": [],
            },
            "unexpected_key_semantics": (
                "PREDECLARED_PRETRAINING_HEADS_EXCLUDED_FROM_FROZEN_BASE_ENCODER"
            ),
            "model_eval": True,
            "parameter_requires_grad_count": 0,
            "crop_half_open_interval_verified": True,
            "model_input_padding_samples": 0,
            "attention_mask_all_ones": True,
            "passage_local_resample_padding": True,
            "layer_keys": list(EXPECTED_LAYER_KEYS),
            "layer_shapes": [[1, frames, 768] for _ in range(13)],
            "all_outputs_finite": True,
        },
        "frame_timing": {
            "kernels": list(timing.kernels),
            "strides": list(timing.strides),
            "cumulative_stride_samples": timing.cumulative_stride_samples,
            "receptive_field_samples": timing.receptive_field_samples,
            "expected_frames": frames,
            "observed_frames": frames,
            "first_frame_center_seconds": float(centers[0]),
            "last_frame_center_seconds": float(centers[-1]),
            "frame_step_seconds": timing.cumulative_stride_samples / 16000,
        },
        "context": {
            "transformer_attention_scope": "GLOBAL_WITHIN_SINGLE_PASSAGE",
            "transformer_local_receptive_field_claimed": False,
            "cross_split_input_overlap_measured": True,
            "audio_cross_split_context_overlap_seconds": 0.0,
            "basis": "REAL_319_ROW_AUDIO_IDENTITY_GATE_PLUS_SYNTHETIC_PATH_SENTINEL",
        },
        "embargo_components_seconds": components,
        "embargo_evaluation": evaluate_final_embargo_candidate(components),  # type: ignore[arg-type]
        "split_guard": {
            "report_schema_version": "m6a-split-guard-final-embargo-candidate-v1",
            "status": "PASS",
            "rows": 319,
            "issues": [],
            "final_embargo_candidate_seconds": 2.0,
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
            "audio_identity": {
                "status": "PASS",
                "row_count": 319,
                "audio_file_nonempty": True,
                "empty_audio_file_rows": 0,
                "unique_audio_file_count": 48,
                "one_stimulus_per_audio_file": True,
                "one_block_per_audio_file": True,
                "one_split_per_audio_file": True,
                "audio_file_cross_split_count": 0,
                "audio_files_crossing_splits": [],
                "sample_rate_hz_values": [44100],
                "channel_values": [1],
                "audio_source_status_values": ["BUNDLED_BLOCK_AUDIO"],
                "audio_file_assignments": audio_assignments,
                "issues": [],
            },
        },
        "formal_feature_extraction_run": False,
        "real_neural_waveform_read": False,
        "baseline_run": False,
        "exchange_candidate_created": False,
        "scientific_result_claimed": False,
    }


class PassageInputTests(unittest.TestCase):
    def test_crop_is_half_open_and_resampling_has_frozen_length(self) -> None:
        source = np.linspace(-1.0, 1.0, SOURCE_SAMPLE_RATE_HZ)
        cropped = crop_mono_passage(source, 100, SOURCE_SAMPLE_RATE_HZ - 100)
        self.assertEqual(cropped.size, SOURCE_SAMPLE_RATE_HZ - 200)
        output = resample_independent_passage(cropped)
        self.assertEqual(output.size, math.ceil(cropped.size * 160 / 441))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_multichannel_short_and_nonfinite_audio_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            resample_independent_passage(np.zeros((100, 2)))
        with self.assertRaises(ValueError):
            resample_independent_passage(np.zeros(56))
        for invalid in (math.nan, math.inf, -math.inf):
            samples = np.zeros(100)
            samples[0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resample_independent_passage(samples)

    def test_isolated_loader_never_reads_a_different_passage(self) -> None:
        reads: list[str] = []

        def reader(path: str) -> tuple[np.ndarray, int]:
            reads.append(path)
            return np.zeros(100), SOURCE_SAMPLE_RATE_HZ

        samples, rate = load_isolated_passage(
            "train/a.wav", "train/a.wav", reader, 1, 99
        )
        self.assertEqual(samples.size, 98)
        self.assertEqual(rate, SOURCE_SAMPLE_RATE_HZ)
        with self.assertRaises(ValueError):
            load_isolated_passage("test/b.wav", "train/a.wav", reader, 1, 99)
        self.assertEqual(reads, ["train/a.wav"])

    def test_path_traversal_and_windows_paths_are_rejected(self) -> None:
        for value in ("", "/absolute.wav", "../escape.wav", "a\\b.wav"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_relative_audio_path(value)


class RealManifestAudioIdentityTests(unittest.TestCase):
    def test_repository_319_row_audio_identity_passes(self) -> None:
        report = audit_split_audio_identity(ROOT / "reports" / "ds004703_primary_split.csv")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["row_count"], 319)
        self.assertEqual(report["unique_audio_file_count"], 48)
        self.assertEqual(report["audio_file_cross_split_count"], 0)
        self.assertEqual(report["audio_files_crossing_splits"], [])
        self.assertEqual(report["sample_rate_hz_values"], [44100])
        self.assertEqual(report["channel_values"], [1])
        self.assertEqual(report["audio_source_status_values"], ["BUNDLED_BLOCK_AUDIO"])

    def test_audio_identity_drift_fails_closed(self) -> None:
        source = ROOT / "reports" / "ds004703_primary_split.csv"
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        cases = {
            "empty_file": ("audio_file", ""),
            "cross_split": ("split", "test"),
            "stimulus_drift": ("stimulus_id", "different-stimulus"),
            "block_drift": ("block_id", "block-99"),
            "sample_rate": ("audio_sample_rate_hz", "16000"),
            "channels": ("audio_channels", "2"),
            "source_status": ("audio_source_status", "OTHER"),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                changed = [dict(row) for row in rows]
                changed[0][field] = value
                path = Path(directory) / "split.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(changed)
                self.assertEqual(audit_split_audio_identity(path)["status"], "FAIL")


class FrameTimingTests(unittest.TestCase):
    def test_base_convolution_frame_count_and_centers(self) -> None:
        timing = derive_convolution_timing(
            (10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)
        )
        self.assertEqual(timing.cumulative_stride_samples, 320)
        self.assertEqual(timing.receptive_field_samples, 400)
        self.assertEqual(expected_frame_count(16_000, timing), 49)
        centers = frame_center_seconds(49, timing)
        self.assertAlmostEqual(centers[0], 199.5 / 16_000)
        self.assertAlmostEqual(centers[1] - centers[0], 0.02)


class CandidateGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_json(ROOT / "configs" / "m6a_public_001.json")
        self.evidence = valid_evidence(self.config)

    def test_complete_evidence_yields_candidate_only(self) -> None:
        report = finalize_candidate_report(self.evidence, self.config)
        self.assertEqual(report["status"], CANDIDATE_STATUS)
        self.assertEqual(report["failed_checks"], [])
        self.assertTrue(all(report["required_checks"].values()))
        self.assertIs(self.config["split"]["baseline_final"], False)
        self.assertIs(self.config["neural_target"]["neural_extraction_allowed"], False)

    def test_each_context_layer_padding_inventory_and_split_drift_fails(self) -> None:
        cases = {
            "context": lambda item: item["context"].__setitem__(
                "audio_cross_split_context_overlap_seconds", 0.1
            ),
            "local_transformer_claim": lambda item: item["context"].__setitem__(
                "transformer_local_receptive_field_claimed", True
            ),
            "layer_count": lambda item: item["model_canary"].__setitem__(
                "layer_shapes", item["model_canary"]["layer_shapes"][:-1]
            ),
            "padding": lambda item: item["model_canary"].__setitem__(
                "model_input_padding_samples", 1
            ),
            "inventory": lambda item: item.__setitem__(
                "model_inventory", item["model_inventory"][:-1]
            ),
            "unsafe_weight_load": lambda item: item["model"]["weight_semantic_audit"].__setitem__(
                "weights_only", False
            ),
            "custom_code": lambda item: item["model_canary"].__setitem__(
                "repository_custom_code_executed", True
            ),
            "missing_encoder_key": lambda item: item["model_canary"]["loading_info"].__setitem__(
                "missing_keys", ["encoder.layers.0.attention.q_proj.weight"]
            ),
            "split": lambda item: item["split_guard"].__setitem__(
                "split_counts", {"train": 222, "validation": 49, "test": 48}
            ),
            "audio_unique_count": lambda item: item["split_guard"]["audio_identity"].__setitem__(
                "unique_audio_file_count", 47
            ),
            "audio_cross_split": lambda item: item["split_guard"]["audio_identity"].update(
                {
                    "one_split_per_audio_file": False,
                    "audio_file_cross_split_count": 1,
                    "audio_files_crossing_splits": ["stimuli/drift.wav"],
                }
            ),
            "audio_sample_rate": lambda item: item["split_guard"]["audio_identity"].__setitem__(
                "sample_rate_hz_values", [16000]
            ),
            "audio_channels": lambda item: item["split_guard"]["audio_identity"].__setitem__(
                "channel_values", [2]
            ),
            "audio_source": lambda item: item["split_guard"]["audio_identity"].__setitem__(
                "audio_source_status_values", ["OTHER"]
            ),
            "audio_assignment_drift": lambda item: item["split_guard"]["audio_identity"][
                "audio_file_assignments"
            ][0].__setitem__("splits", ["train", "test"]),
            "scientific_claim": lambda item: item.__setitem__(
                "scientific_result_claimed", True
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(self.evidence)
                mutate(changed)
                self.assertEqual(finalize_candidate_report(changed, self.config)["status"], "FAIL")

    def test_nonfinite_tuple_list_missing_fields_and_malformed_config_fail_without_exception(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                changed = copy.deepcopy(self.evidence)
                changed["model_canary"]["layer_shapes"] = ([1, invalid, 768],)
                self.assertEqual(finalize_candidate_report(changed, self.config)["status"], "FAIL")
        changed = copy.deepcopy(self.evidence)
        changed["model_inventory"] = [[]]
        self.assertEqual(finalize_candidate_report(changed, self.config)["status"], "FAIL")
        self.assertEqual(finalize_candidate_report(self.evidence, {})["status"], "FAIL")

    def test_validated_cache_closes_the_download_tool(self) -> None:
        with self.assertRaises(ValueError):
            validate_download_authorization(self.config)

    def test_strict_json_loader_rejects_nonstandard_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for invalid in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(invalid=invalid):
                    path.write_text('{"value": ' + invalid + "}", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_strict_json_object(path)

    def test_report_round_trip_contains_no_nonstandard_numeric_values(self) -> None:
        report = finalize_candidate_report(self.evidence, self.config)
        rendered = json.dumps(report, allow_nan=False)
        self.assertEqual(json.loads(rendered)["status"], CANDIDATE_STATUS)


if __name__ == "__main__":
    unittest.main()
