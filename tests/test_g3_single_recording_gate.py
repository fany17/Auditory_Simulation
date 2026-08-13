from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import m6a_public.g3_single_recording_gate as g3_gate
from scripts.g3_single_recording_alignment_candidate import (
    _revalidate_existing_evidence,
)
from m6a_public.audio_context_gate import (
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
)
from m6a_public.g3_single_recording_gate import (
    EXPECTED_ALIGNED_FRAME_COUNT,
    EXPECTED_AUDIO_FILE,
    EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS,
    EXPECTED_AUDIO_NATIVE_FRAME_COUNT,
    EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS,
    EXPECTED_AUDIO_NATIVE_STEP_SECONDS,
    EXPECTED_AUDIO_RESAMPLED_FRAMES,
    EXPECTED_AUDIO_SOURCE_FRAMES,
    EXPECTED_CHANNEL_COUNT,
    EXPECTED_COMMON_FIRST_SECONDS,
    EXPECTED_COMMON_LAST_SECONDS,
    EXPECTED_COMMON_VALID_FRAME_COUNT,
    EXPECTED_EDF_FILE,
    EXPECTED_END_SECONDS,
    EXPECTED_FORMAT_SELECTION_CRITERIA,
    EXPECTED_FORMAT_SELECTION_REASON,
    EXPECTED_GRID_FIRST_SECONDS,
    EXPECTED_GRID_LAST_SECONDS,
    EXPECTED_NEURAL_NATIVE_FIRST_SECONDS,
    EXPECTED_NEURAL_NATIVE_LAST_SECONDS,
    EXPECTED_NEURAL_NATIVE_STEP_SECONDS,
    EXPECTED_PASSAGE_END_SAMPLE_CEIL,
    EXPECTED_PASSAGE_START_SAMPLE_FLOOR,
    EXPECTED_READ_END_SAMPLE_EXCLUSIVE,
    EXPECTED_READ_SAMPLE_COUNT,
    EXPECTED_READ_START_SAMPLE,
    EXPECTED_RECORDING_ID,
    EXPECTED_RECORDING_TOTAL_SAMPLES,
    EXPECTED_REMOTE_OUTPUTS_ROOT,
    EXPECTED_SAMPLE_ID,
    EXPECTED_START_SECONDS,
    EXPECTED_STIMULUS_ID,
    EXPECTED_SUPPORT_EDGE_SAMPLES,
    EXPECTED_TENSOR_SPECS,
    G3_STATUS,
    REPORT_SCHEMA_VERSION,
    SUPERSEDED_G3_STATUS,
    amplitude_envelope_native,
    audit_remote_tensor_outputs,
    finalize_g3_report,
    frozen_mel_filterbank,
    linear_align_no_extrapolation,
    load_strict_json_object,
    log_mel_native,
    native_audio_frame_centers,
    passage_grid_seconds,
    select_g3_scope,
    validate_g3_config,
)


ROOT = Path(__file__).resolve().parents[1]
def valid_evidence() -> dict[str, Any]:
    channels = [f"channel-{index:02d}" for index in range(EXPECTED_CHANNEL_COUNT)]
    inventory = [
        {
            "name": name,
            "relative_path": relative_path,
            "bytes": 128 + index,
            "modified_at_utc": "2026-08-13T08:14:12+00:00",
            "dtype": dtype,
            "shape": shape,
            "object_dtype": False,
            "remote_only": True,
        }
        for index, (name, (relative_path, dtype, shape)) in enumerate(
            EXPECTED_TENSOR_SPECS.items()
        )
    ]
    readback_checks = [
        {
            "name": item["name"],
            "relative_path": item["relative_path"],
            "bytes": item["bytes"],
            "dtype": item["dtype"],
            "shape": item["shape"],
            "object_dtype": False,
            "all_finite": True,
        }
        for item in inventory
    ]
    remote_output_root = (
        "/home/fanyu/auditory_simulation_m6a/outputs/g3_single_recording_test"
    )
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "integrity_policy": "NON_HASH_AUDIT",
        "cryptographic_integrity_claim": False,
        "selection": {
            "recording_id": EXPECTED_RECORDING_ID,
            "sample_id": EXPECTED_SAMPLE_ID,
            "stimulus_id": EXPECTED_STIMULUS_ID,
            "audio_file": EXPECTED_AUDIO_FILE,
            "edf_file": EXPECTED_EDF_FILE,
            "split": "train",
            "start_seconds": EXPECTED_START_SECONDS,
            "end_seconds": EXPECTED_END_SECONDS,
            "sampling_rate_hz": 512,
            "eligible_channel_count": EXPECTED_CHANNEL_COUNT,
            "eligible_channel_names": channels,
            "reference": "scalp electrode, not included with data",
            "g2_declared_recording_duration_seconds": 3552.75,
            "g2_edf_header_last_sample_time_seconds": 2599.748046875,
            "selection_candidate_recording_count": 6,
            "result_based_selection_used": False,
        },
        "real_neural_read": {
            "edf_file": EXPECTED_EDF_FILE,
            "preload_entire_recording": False,
            "segment_read_only": True,
            "requested_recording_count": 1,
            "requested_channel_count": EXPECTED_CHANNEL_COUNT,
            "returned_channel_count": EXPECTED_CHANNEL_COUNT,
            "returned_channel_names": channels,
            "sampling_rate_hz": 512,
            "recording_total_samples": EXPECTED_RECORDING_TOTAL_SAMPLES,
            "recording_sample_span_seconds": 2599.75,
            "passage_start_sample_floor": EXPECTED_PASSAGE_START_SAMPLE_FLOOR,
            "passage_end_sample_ceil": EXPECTED_PASSAGE_END_SAMPLE_CEIL,
            "support_edge_samples": EXPECTED_SUPPORT_EDGE_SAMPLES,
            "read_start_sample": EXPECTED_READ_START_SAMPLE,
            "read_end_sample_exclusive": EXPECTED_READ_END_SAMPLE_EXCLUSIVE,
            "read_start_seconds": 125.427734375,
            "read_end_seconds_exclusive": 162.244140625,
            "read_sample_count": EXPECTED_READ_SAMPLE_COUNT,
            "other_recording_or_segment_read_count": 0,
            "raw_waveform_saved": False,
            "mne_preload": False,
        },
        "audio_read": {
            "audio_file": EXPECTED_AUDIO_FILE,
            "requested_audio_file_count": 1,
            "other_audio_read_count": 0,
            "sample_rate_hz": 44100,
            "channels": 1,
            "source_frames": EXPECTED_AUDIO_SOURCE_FRAMES,
            "source_duration_seconds": EXPECTED_AUDIO_SOURCE_FRAMES / 44100,
            "resampled_frames": EXPECTED_AUDIO_RESAMPLED_FRAMES,
            "resampled_duration_seconds": EXPECTED_AUDIO_RESAMPLED_FRAMES / 16000,
            "neighbor_passage_read_allowed": False,
        },
        "model_runtime": {
            "local_files_only": True,
            "trust_remote_code": False,
            "repository_custom_code_executed": False,
            "weights_only": True,
            "tensor_only": True,
            "download_attempted": False,
            "model_eval": True,
            "parameter_requires_grad_count": 0,
            "loading_info": {
                "missing_keys": [],
                "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
                "mismatched_keys": [],
                "error_msgs": [],
            },
        },
        "native_grids": {
            "amplitude_envelope": {
                "frame_count": EXPECTED_AUDIO_NATIVE_FRAME_COUNT,
                "feature_dim": 1,
                "first_frame_center_seconds": EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS,
                "last_frame_center_seconds": EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS,
                "frame_step_seconds": EXPECTED_AUDIO_NATIVE_STEP_SECONDS,
                "shape": [EXPECTED_AUDIO_NATIVE_FRAME_COUNT, 1],
                "all_finite": True,
            },
            "log_mel": {
                "frame_count": EXPECTED_AUDIO_NATIVE_FRAME_COUNT,
                "feature_dim": 80,
                "first_frame_center_seconds": EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS,
                "last_frame_center_seconds": EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS,
                "frame_step_seconds": EXPECTED_AUDIO_NATIVE_STEP_SECONDS,
                "shape": [EXPECTED_AUDIO_NATIVE_FRAME_COUNT, 80],
                "all_finite": True,
            },
            "wav2vec2": {
                "layer_keys": list(EXPECTED_LAYER_KEYS),
                "layer_count": 13,
                "hidden_size": 768,
                "native_shape_frames_layers_hidden": [
                    EXPECTED_AUDIO_NATIVE_FRAME_COUNT,
                    13,
                    768,
                ],
                "native_frame_count_formula": EXPECTED_AUDIO_NATIVE_FRAME_COUNT,
                "native_first_frame_center_seconds": EXPECTED_AUDIO_NATIVE_FIRST_CENTER_SECONDS,
                "native_last_frame_center_seconds": EXPECTED_AUDIO_NATIVE_LAST_CENTER_SECONDS,
                "native_frame_step_seconds": EXPECTED_AUDIO_NATIVE_STEP_SECONDS,
                "aligned_shape_layers_frames_hidden": [
                    13,
                    EXPECTED_ALIGNED_FRAME_COUNT,
                    768,
                ],
                "valid_frame_count": 1730,
                "all_finite": True,
            },
            "neural": {
                "native_sample_count": EXPECTED_READ_SAMPLE_COUNT,
                "native_first_sample_time_seconds": EXPECTED_NEURAL_NATIVE_FIRST_SECONDS,
                "native_last_sample_time_seconds": EXPECTED_NEURAL_NATIVE_LAST_SECONDS,
                "native_sample_step_seconds": EXPECTED_NEURAL_NATIVE_STEP_SECONDS,
                "channel_count": EXPECTED_CHANNEL_COUNT,
                "subband_count": 6,
                "subbands_hz": [
                    [70.0, 80.0],
                    [80.0, 90.0],
                    [90.0, 100.0],
                    [100.0, 110.0],
                    [130.0, 140.0],
                    [140.0, 150.0],
                ],
                "raw_power_shape": [EXPECTED_ALIGNED_FRAME_COUNT, 36, 6],
                "pretransform_log_power_shape": [EXPECTED_ALIGNED_FRAME_COUNT, 36, 6],
                "valid_frame_count": EXPECTED_COMMON_VALID_FRAME_COUNT,
                "support_edge_samples": EXPECTED_SUPPORT_EDGE_SAMPLES,
                "support_edge_seconds": 1.091796875,
                "formal_train_only_transform_fitted": False,
                "smoke_statistics_reusable_for_baseline": False,
                "smoke_log_formula": "natural_log(max(raw_power, 0) + 1e-30)",
                "negative_raw_power_value_count_before_smoke_clip": 160,
                "negative_raw_power_abs_max_before_smoke_clip": 1.4e-14,
                "negative_power_clip_absolute_tolerance": 1e-12,
                "all_finite": True,
            },
        },
        "aligned_grid": {
            "grid": "RECORDING_ORIGIN_K_OVER_50_SECONDS",
            "frame_rate_hz": 50,
            "frame_count": EXPECTED_ALIGNED_FRAME_COUNT,
            "first_frame_seconds": EXPECTED_GRID_FIRST_SECONDS,
            "last_frame_seconds": EXPECTED_GRID_LAST_SECONDS,
            "common_valid_frame_count": EXPECTED_COMMON_VALID_FRAME_COUNT,
            "first_common_valid_frame_seconds": EXPECTED_COMMON_FIRST_SECONDS,
            "last_common_valid_frame_seconds": EXPECTED_COMMON_LAST_SECONDS,
            "timestamps_strictly_increasing": True,
            "all_tensor_timestamps_identical": True,
            "interpolation": "LINEAR_TWO_NEAREST_NATIVE_FRAMES_NO_EXTRAPOLATION",
            "common_mask_is_intersection": True,
            "individual_valid_frame_counts": {
                "amplitude_envelope": 1730,
                "log_mel": 1730,
                "wav2vec2": 1730,
                "neural": EXPECTED_COMMON_VALID_FRAME_COUNT,
            },
            "all_tensors_finite": True,
        },
        "tensor_inventory": inventory,
        "remote_tensor_readback": {
            "output_root": remote_output_root,
            "dedicated_outputs_root": str(EXPECTED_REMOTE_OUTPUTS_ROOT),
            "output_root_within_dedicated_outputs": True,
            "active_partial_count": 0,
            "npy_file_count": 7,
            "unexpected_npy_files": [],
            "all_files_present": True,
            "all_headers_match_inventory": True,
            "all_arrays_finite": True,
            "allow_pickle": False,
            "common_valid_true_count": EXPECTED_COMMON_VALID_FRAME_COUNT,
            "frame_times_strictly_increasing": True,
            "first_frame_seconds": EXPECTED_GRID_FIRST_SECONDS,
            "last_frame_seconds": EXPECTED_GRID_LAST_SECONDS,
            "tensor_checks": readback_checks,
        },
        "format_benchmark": {
            "formats_tested": ["NPY_PER_TENSOR", "NPZ_COMPRESSED"],
            "selected_format": "NPY_PER_TENSOR",
            "selection_status": "PROVISIONAL_INTERNAL_FORMAT_SELECTION",
            "selection_reason": EXPECTED_FORMAT_SELECTION_REASON,
            "selection_criteria": EXPECTED_FORMAT_SELECTION_CRITERIA,
            "new_dependency_used": False,
            "npy_per_tensor": {
                "bytes": sum(item["bytes"] for item in inventory),
                "write_seconds": 0.5,
                "full_read_seconds": 0.05,
                "single_wav2vec2_layer_seconds": 0.002,
                "single_neural_electrode_seconds": 0.0003,
                "mmap_supported": True,
                "direct_slice_supported": True,
                "atomic_per_tensor": True,
                "allow_pickle": False,
            },
            "npz_compressed": {
                "bytes": 800,
                "write_seconds": 0.6,
                "full_read_seconds": 0.1,
                "single_wav2vec2_layer_seconds": 0.08,
                "single_neural_electrode_seconds": 0.01,
                "mmap_supported": False,
                "direct_slice_supported": False,
                "atomic_archive": True,
                "allow_pickle": False,
                "relative_path": "npz_compressed/aligned_tensors.npz",
            },
        },
        "remote_output_root": remote_output_root,
        "high_dimensional_arrays_in_git": False,
        "real_neural_waveform_read_scope": (
            "ONE_SELECTED_RECORDING_ONE_PASSAGE_36_ELIGIBLE_CHANNELS_"
            "PLUS_FROZEN_FINITE_SUPPORT_ONLY"
        ),
        "formal_baseline_run": False,
        "scientific_result_claimed": False,
        "exchange_candidate_created": False,
        "other_recordings_or_segments_processed": False,
        "formal_train_only_transform_fitted": False,
    }


class G3ScopedConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_strict_json_object(
            ROOT / "configs" / "m6a_g3_single_recording_candidate.json"
        )

    def test_repository_scoped_config_passes(self) -> None:
        self.assertEqual(validate_g3_config(self.config), [])

    def test_mel_read_scope_model_and_claim_drift_fail_closed(self) -> None:
        cases = {
            "mel_bins": lambda item: item["audio_features"]["log_mel"].__setitem__(
                "mel_bins", 64
            ),
            "mel_padding": lambda item: item["audio_features"]["log_mel"].__setitem__(
                "padding", "REFLECT"
            ),
            "other_recording": lambda item: item["read_scope"].__setitem__(
                "other_recordings_allowed", True
            ),
            "whole_data": lambda item: item["read_scope"].__setitem__(
                "allowed_recording_count", 11
            ),
            "download": lambda item: item["audio_features"]["wav2vec2"].__setitem__(
                "download_allowed", True
            ),
            "baseline": lambda item: item["execution"].__setitem__(
                "formal_baseline_run", True
            ),
            "train_transform": lambda item: item["neural_smoke"].__setitem__(
                "formal_train_only_transform_fitted", True
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(self.config)
                mutate(changed)
                self.assertTrue(validate_g3_config(changed))


class G3SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = load_strict_json_object(
            ROOT / "reports" / "ds004703_neural_metadata_g2_candidate.json"
        )
        self.split_csv = ROOT / "reports" / "ds004703_primary_split.csv"

    def test_frozen_rule_reselects_expected_recording_and_passage(self) -> None:
        selected = select_g3_scope(self.report, self.split_csv)
        self.assertEqual(selected["recording_id"], EXPECTED_RECORDING_ID)
        self.assertEqual(selected["sample_id"], EXPECTED_SAMPLE_ID)
        self.assertEqual(selected["stimulus_id"], EXPECTED_STIMULUS_ID)
        self.assertEqual(selected["audio_file"], EXPECTED_AUDIO_FILE)
        self.assertEqual(selected["eligible_channel_count"], EXPECTED_CHANNEL_COUNT)

    def test_header_reference_or_count_drift_cannot_silently_change_selection(self) -> None:
        cases = {
            "header": lambda item: item.__setitem__(
                "edf_header_channel_count_matches_tsv", False
            ),
            "reference": lambda item: item.__setitem__("iEEGReference", "CAR"),
            "missing_name": lambda item: item.__setitem__(
                "analysis_eligible_neural_channels_missing_from_edf", ["LA1"]
            ),
            "count": lambda item: item["channels"].__setitem__(
                "analysis_eligible_neural_channel_count", 35
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(self.report)
                target = next(
                    item for item in changed["recordings"] if item["recording_id"] == EXPECTED_RECORDING_ID
                )
                mutate(target)
                with self.assertRaises(ValueError):
                    select_g3_scope(changed, self.split_csv)


class G3AcousticAndAlignmentTests(unittest.TestCase):
    def test_frozen_envelope_log_mel_shapes_and_native_centers(self) -> None:
        samples = np.sin(2 * np.pi * 440 * np.arange(16000, dtype=np.float64) / 16000)
        envelope, window = amplitude_envelope_native(samples)
        log_mel = log_mel_native(samples)
        expected_frames = 1 + (samples.size - 400) // 320
        centers = native_audio_frame_centers(expected_frames, 10.0)
        self.assertEqual(envelope.shape, (expected_frames, 1))
        self.assertEqual(log_mel.shape, (expected_frames, 80))
        self.assertEqual(window.shape, (400,))
        self.assertAlmostEqual(centers[0], 10.0 + 199.5 / 16000)
        self.assertAlmostEqual(centers[1] - centers[0], 0.02)
        self.assertTrue(np.all(np.isfinite(envelope)))
        self.assertTrue(np.all(np.isfinite(log_mel)))
        self.assertEqual(frozen_mel_filterbank().shape, (80, 257))

    def test_recording_origin_grid_and_no_extrapolation(self) -> None:
        grid = passage_grid_seconds(EXPECTED_START_SECONDS, EXPECTED_END_SECONDS)
        self.assertTrue(np.all(np.diff(grid) > 0))
        self.assertTrue(np.allclose(grid * 50, np.round(grid * 50), atol=1e-10, rtol=0))
        native_times = np.asarray([1.0, 2.0, 3.0])
        values = np.asarray([[10.0], [20.0], [30.0]])
        aligned, valid = linear_align_no_extrapolation(
            native_times, values, np.asarray([0.5, 1.5, 2.5, 3.5])
        )
        self.assertEqual(valid.tolist(), [False, True, True, False])
        self.assertEqual(aligned[:, 0].tolist(), [0.0, 15.0, 25.0, 0.0])

    def test_short_nonfinite_and_nonmonotonic_inputs_fail(self) -> None:
        with self.assertRaises(ValueError):
            amplitude_envelope_native(np.zeros(399))
        values = np.zeros(400)
        values[0] = math.nan
        with self.assertRaises(ValueError):
            log_mel_native(values)
        with self.assertRaises(ValueError):
            linear_align_no_extrapolation(
                np.asarray([0.0, 0.0]), np.zeros((2, 1)), np.asarray([0.0])
            )


class G3CandidateEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_config = load_strict_json_object(ROOT / "configs" / "m6a_public_001.json")
        self.g3_config = load_strict_json_object(
            ROOT / "configs" / "m6a_g3_single_recording_candidate.json"
        )
        self.evidence = valid_evidence()

    def test_complete_synthetic_evidence_yields_candidate_only(self) -> None:
        report = finalize_g3_report(self.evidence, self.task_config, self.g3_config)
        self.assertEqual(report["status"], G3_STATUS)
        self.assertEqual(report["failed_checks"], [])
        self.assertTrue(all(report["required_checks"].values()))

    def test_superseded_repository_report_cannot_be_promoted(self) -> None:
        path = ROOT / "reports" / "g3_single_recording_candidate_pre_numeric_tolerance_v1.json"
        superseded = load_strict_json_object(path)
        self.assertEqual(superseded["status"], SUPERSEDED_G3_STATUS)
        self.assertEqual(
            finalize_g3_report(superseded, self.task_config, self.g3_config)["status"],
            "FAIL",
        )
        with self.assertRaisesRegex(ValueError, "superseded G3 provenance"):
            _revalidate_existing_evidence(
                path,
                Path("/unused/output"),
                Path("/unused"),
                self.task_config,
                self.g3_config,
            )

    def test_scope_time_shape_format_and_claim_drift_fail_closed(self) -> None:
        cases = {
            "recording": lambda item: item["selection"].__setitem__(
                "recording_id", "sub-other"
            ),
            "channel": lambda item: item["real_neural_read"].__setitem__(
                "requested_channel_count", 37
            ),
            "whole_preload": lambda item: item["real_neural_read"].__setitem__(
                "preload_entire_recording", True
            ),
            "model_custom_code": lambda item: item["model_runtime"].__setitem__(
                "repository_custom_code_executed", True
            ),
            "model_not_tensor_only": lambda item: item["model_runtime"].__setitem__(
                "tensor_only", False
            ),
            "model_not_eval": lambda item: item["model_runtime"].__setitem__(
                "model_eval", False
            ),
            "model_trainable_parameter": lambda item: item["model_runtime"].__setitem__(
                "parameter_requires_grad_count", 1
            ),
            "model_loading_missing": lambda item: item["model_runtime"][
                "loading_info"
            ]["missing_keys"].append("encoder.weight"),
            "model_loading_unexpected": lambda item: item["model_runtime"][
                "loading_info"
            ].__setitem__("unexpected_keys", []),
            "audio_identity": lambda item: item["audio_read"].__setitem__(
                "audio_file", "stimuli/other.wav"
            ),
            "audio_other_read": lambda item: item["audio_read"].__setitem__(
                "other_audio_read_count", 1
            ),
            "audio_sample_rate": lambda item: item["audio_read"].__setitem__(
                "sample_rate_hz", 48000
            ),
            "audio_source_frames": lambda item: item["audio_read"].__setitem__(
                "source_frames", EXPECTED_AUDIO_SOURCE_FRAMES - 1
            ),
            "audio_resample_formula": lambda item: item["audio_read"].__setitem__(
                "resampled_frames", EXPECTED_AUDIO_RESAMPLED_FRAMES - 1
            ),
            "audio_neighbor": lambda item: item["audio_read"].__setitem__(
                "neighbor_passage_read_allowed", True
            ),
            "edf_identity": lambda item: item["real_neural_read"].__setitem__(
                "edf_file", "sub-other.edf"
            ),
            "edf_sampling": lambda item: item["real_neural_read"].__setitem__(
                "sampling_rate_hz", 1024
            ),
            "edf_mne_preload": lambda item: item["real_neural_read"].__setitem__(
                "mne_preload", True
            ),
            "passage_floor": lambda item: item["real_neural_read"].__setitem__(
                "passage_start_sample_floor", EXPECTED_PASSAGE_START_SAMPLE_FLOOR + 1
            ),
            "bounded_read_start": lambda item: item["real_neural_read"].__setitem__(
                "read_start_sample", EXPECTED_READ_START_SAMPLE + 1
            ),
            "bounded_read_end": lambda item: item["real_neural_read"].__setitem__(
                "read_end_sample_exclusive", EXPECTED_READ_END_SAMPLE_EXCLUSIVE - 1
            ),
            "bounded_read_count": lambda item: item["real_neural_read"].__setitem__(
                "read_sample_count", EXPECTED_READ_SAMPLE_COUNT - 1
            ),
            "layer": lambda item: item["native_grids"]["wav2vec2"].__setitem__(
                "layer_count", 12
            ),
            "audio_native_count": lambda item: item["native_grids"][
                "amplitude_envelope"
            ].__setitem__("frame_count", 1730),
            "audio_native_first": lambda item: item["native_grids"][
                "amplitude_envelope"
            ].__setitem__("first_frame_center_seconds", 126.53),
            "audio_native_step": lambda item: item["native_grids"][
                "log_mel"
            ].__setitem__("frame_step_seconds", 0.04),
            "audio_native_nonfinite_flag": lambda item: item["native_grids"][
                "log_mel"
            ].__setitem__("all_finite", False),
            "wav_formula": lambda item: item["native_grids"]["wav2vec2"].__setitem__(
                "native_frame_count_formula", 1730
            ),
            "wav_aligned_shape": lambda item: item["native_grids"][
                "wav2vec2"
            ].__setitem__("aligned_shape_layers_frames_hidden", [13, 1731, 768]),
            "neural_native_step": lambda item: item["native_grids"][
                "neural"
            ].__setitem__("native_sample_step_seconds", 0.01),
            "neural_shape": lambda item: item["native_grids"]["neural"].__setitem__(
                "raw_power_shape", [1732, 35, 6]
            ),
            "negative_power_beyond_tolerance": lambda item: item["native_grids"][
                "neural"
            ].__setitem__("negative_raw_power_abs_max_before_smoke_clip", 2e-12),
            "timestamps": lambda item: item["aligned_grid"].__setitem__(
                "all_tensor_timestamps_identical", False
            ),
            "empty_common": lambda item: item["aligned_grid"].__setitem__(
                "common_valid_frame_count", 0
            ),
            "individual_valid": lambda item: item["aligned_grid"][
                "individual_valid_frame_counts"
            ].__setitem__("wav2vec2", 1729),
            "tensor_path": lambda item: item["tensor_inventory"][0].__setitem__(
                "relative_path", "npy_per_tensor/other.npy"
            ),
            "tensor_dtype": lambda item: item["tensor_inventory"][1].__setitem__(
                "dtype", "uint8"
            ),
            "tensor_shape": lambda item: item["tensor_inventory"][2].__setitem__(
                "shape", [1731, 1]
            ),
            "tensor_empty": lambda item: item["tensor_inventory"][3].__setitem__(
                "bytes", 0
            ),
            "tensor_object": lambda item: item["tensor_inventory"][4].__setitem__(
                "object_dtype", True
            ),
            "tensor_local": lambda item: item["tensor_inventory"][5].__setitem__(
                "remote_only", False
            ),
            "readback_partial": lambda item: item["remote_tensor_readback"].__setitem__(
                "active_partial_count", 1
            ),
            "readback_nonfinite": lambda item: item["remote_tensor_readback"][
                "tensor_checks"
            ][0].__setitem__("all_finite", False),
            "readback_shape": lambda item: item["remote_tensor_readback"][
                "tensor_checks"
            ][4].__setitem__("shape", [13, 1731, 768]),
            "readback_inventory_byte_mismatch": lambda item: item[
                "remote_tensor_readback"
            ]["tensor_checks"][4].__setitem__(
                "bytes", item["tensor_inventory"][4]["bytes"] + 1
            ),
            "format": lambda item: item["format_benchmark"].__setitem__(
                "selected_format", "NPZ_COMPRESSED"
            ),
            "format_reason": lambda item: item["format_benchmark"].__setitem__(
                "selection_reason", "POST_HOC_REASON"
            ),
            "format_bytes": lambda item: item["format_benchmark"][
                "npy_per_tensor"
            ].__setitem__("bytes", 0),
            "format_negative_time": lambda item: item["format_benchmark"][
                "npz_compressed"
            ].__setitem__("full_read_seconds", -1.0),
            "format_npy_pickle": lambda item: item["format_benchmark"][
                "npy_per_tensor"
            ].__setitem__("allow_pickle", True),
            "format_npz_direct": lambda item: item["format_benchmark"][
                "npz_compressed"
            ].__setitem__("direct_slice_supported", True),
            "scientific": lambda item: item.__setitem__("scientific_result_claimed", True),
            "baseline": lambda item: item.__setitem__("formal_baseline_run", True),
            "exchange": lambda item: item.__setitem__("exchange_candidate_created", True),
            "other_segment": lambda item: item.__setitem__(
                "other_recordings_or_segments_processed", True
            ),
            "formal_transform": lambda item: item.__setitem__(
                "formal_train_only_transform_fitted", True
            ),
            "superseded_provenance": lambda item: item.__setitem__(
                "status", SUPERSEDED_G3_STATUS
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(self.evidence)
                mutate(changed)
                self.assertEqual(
                    finalize_g3_report(changed, self.task_config, self.g3_config)["status"],
                    "FAIL",
                )

    def test_nonfinite_missing_tensor_and_malformed_config_fail_without_exception(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                changed = copy.deepcopy(self.evidence)
                changed["aligned_grid"]["first_frame_seconds"] = invalid
                self.assertEqual(
                    finalize_g3_report(changed, self.task_config, self.g3_config)["status"],
                    "FAIL",
                )
        changed = copy.deepcopy(self.evidence)
        changed["tensor_inventory"] = changed["tensor_inventory"][:-1]
        self.assertEqual(
            finalize_g3_report(changed, self.task_config, self.g3_config)["status"], "FAIL"
        )
        malformed = copy.deepcopy(self.g3_config)
        malformed_values: tuple[Any, ...] = ([], (), {"nested": [math.nan]})
        for malformed_value in malformed_values:
            with self.subTest(malformed_value=repr(malformed_value)):
                malformed["audio_features"] = malformed_value
                self.assertEqual(
                    finalize_g3_report(self.evidence, self.task_config, malformed)["status"],
                    "FAIL",
                )

    def test_temp_remote_tensor_readback_is_exact_and_detects_partial(self) -> None:
        tiny_specs = {
            "frame_times_seconds": (
                "npy_per_tensor/frame_times_seconds.npy",
                "float64",
                [2],
            ),
            "common_valid_mask": (
                "npy_per_tensor/common_valid_mask.npy",
                "bool",
                [2],
            ),
        }
        arrays = {
            "frame_times_seconds": np.asarray([126.52, 126.54], dtype=np.float64),
            "common_valid_mask": np.asarray([True, False], dtype=np.bool_),
        }
        with tempfile.TemporaryDirectory() as directory:
            dedicated = Path(directory) / "outputs"
            output = dedicated / "candidate"
            npy_root = output / "npy_per_tensor"
            npy_root.mkdir(parents=True)
            inventory = []
            for name, array in arrays.items():
                path = npy_root / f"{name}.npy"
                np.save(path, array, allow_pickle=False)
                relative_path, dtype, shape = tiny_specs[name]
                inventory.append(
                    {
                        "name": name,
                        "relative_path": relative_path,
                        "bytes": path.stat().st_size,
                        "modified_at_utc": "2026-08-13T08:14:12+00:00",
                        "dtype": dtype,
                        "shape": shape,
                        "object_dtype": False,
                        "remote_only": True,
                    }
                )
            with patch.dict(g3_gate.EXPECTED_TENSOR_SPECS, tiny_specs, clear=True):
                result = audit_remote_tensor_outputs(output, dedicated, inventory)
                self.assertTrue(result["all_headers_match_inventory"])
                self.assertTrue(result["all_arrays_finite"])
                self.assertEqual(result["active_partial_count"], 0)
                partial = npy_root / "interrupted.npy.partial"
                partial.write_bytes(b"incomplete")
                result = audit_remote_tensor_outputs(output, dedicated, inventory)
                self.assertEqual(result["active_partial_count"], 1)

    def test_strict_json_rejects_nonstandard_constants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for invalid in ("NaN", "Infinity", "-Infinity"):
                with self.subTest(invalid=invalid):
                    path.write_text('{"value": ' + invalid + "}", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_strict_json_object(path)


if __name__ == "__main__":
    unittest.main()
