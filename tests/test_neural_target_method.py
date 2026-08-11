from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import freqz

from m6a_public.neural_target_gate import (
    load_strict_json_object,
    validate_neural_target_method,
)
from m6a_public.neural_target_method import (
    PRIMARY_BANDS_HZ,
    TrainOnlyPowerTransform,
    apply_train_only_power_transform,
    design_bandpass_fir,
    design_power_smoothing_fir,
    finite_support_power,
    fit_train_only_power_transform,
    fully_supported_frame_mask,
    overlap_add_fir_same,
    support_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


class NeuralTargetCandidateGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidate = load_strict_json_object(
            ROOT / "configs" / "m6a_neural_target_method_candidate.json"
        )
        self.schema = load_strict_json_object(
            ROOT / "schemas" / "m6a_neural_target_method_candidate.schema.json"
        )

    def test_repository_candidate_passes_schema_and_semantic_gate(self) -> None:
        self.assertEqual(validate_neural_target_method(self.candidate, self.schema), [])

    def test_schema_freezes_six_bands_and_two_sampling_profiles(self) -> None:
        for section_name in (
            "primary_target",
            "reference",
            "bandpass_fir",
            "power_estimator",
            "train_only_transform",
            "time_grid",
            "lag",
            "sensitivity",
            "final_embargo",
            "synthetic_test_thresholds",
            "anatomy",
            "execution",
        ):
            with self.subTest(section_name=section_name):
                self.assertIs(
                    self.schema["properties"][section_name]["additionalProperties"], False
                )
        subbands = self.schema["properties"]["primary_target"]["properties"]["subbands"]
        self.assertEqual(len(subbands["prefixItems"]), 6)
        self.assertIs(subbands["items"], False)
        for item in subbands["prefixItems"]:
            definition = self.schema["$defs"][item["$ref"].rsplit("/", 1)[1]]
            self.assertIs(definition["additionalProperties"], False)
            self.assertIn("const", definition["properties"]["weight"])
        profiles = self.schema["properties"]["sampling_rate_profiles"]
        self.assertEqual(len(profiles["prefixItems"]), 2)
        self.assertIs(profiles["items"], False)
        self.assertEqual(
            [
                self.schema["$defs"][item["$ref"].rsplit("/", 1)[1]]["properties"][
                    "sampling_rate_hz"
                ]["const"]
                for item in profiles["prefixItems"]
            ],
            [512, 1024],
        )

    def test_mutated_band_weight_reference_and_filter_fail_closed(self) -> None:
        mutations = (
            ("band", ("primary_target", "subbands", 0, "low_hz"), 71),
            ("weight", ("primary_target", "subbands", 0, "weight"), 0.2),
            ("reference", ("reference", "primary_policy"), "CAR"),
            ("filter", ("bandpass_fir", "transition_width_hz"), 3),
            ("backend", ("bandpass_fir", "execution_backend"), "numpy.convolve"),
        )
        for name, path, value in mutations:
            with self.subTest(name=name):
                changed = copy.deepcopy(self.candidate)
                cursor: object = changed
                for key in path[:-1]:
                    cursor = cursor[key]  # type: ignore[index]
                cursor[path[-1]] = value  # type: ignore[index]
                self.assertTrue(validate_neural_target_method(changed, self.schema))

    def test_additional_property_and_nonfinite_threshold_fail_closed(self) -> None:
        changed = copy.deepcopy(self.candidate)
        changed["reference"]["unreviewed"] = True
        self.assertTrue(validate_neural_target_method(changed, self.schema))

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.candidate)
                changed["synthetic_test_thresholds"][
                    "harmonic_120_hz_maximum_gain_db"
                ] = value
                errors = validate_neural_target_method(changed, self.schema)
                self.assertTrue(any("non-finite" in item for item in errors))


class NeuralTargetSyntheticFilterTests(unittest.TestCase):
    thresholds: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        candidate = load_strict_json_object(
            ROOT / "configs" / "m6a_neural_target_method_candidate.json"
        )
        cls.thresholds = candidate["synthetic_test_thresholds"]

    def test_firs_are_odd_symmetric_and_profiles_match_actual_design(self) -> None:
        for sampling_rate_hz in (512, 1024):
            with self.subTest(sampling_rate_hz=sampling_rate_hz):
                profile = support_metadata(sampling_rate_hz)
                smoothing = design_power_smoothing_fir(sampling_rate_hz)
                self.assertEqual(smoothing.size, profile["power_smoothing_numtaps"])
                self.assertTrue(
                    np.allclose(
                        smoothing,
                        smoothing[::-1],
                        rtol=0,
                        atol=self.thresholds["coefficient_symmetry_absolute_tolerance"],
                    )
                )
                for band_hz in PRIMARY_BANDS_HZ:
                    bandpass = design_bandpass_fir(sampling_rate_hz, band_hz)
                    self.assertEqual(bandpass.size, profile["bandpass_numtaps"])
                    self.assertEqual(bandpass.size % 2, 1)
                    self.assertTrue(
                        np.allclose(
                            bandpass,
                            bandpass[::-1],
                            rtol=0,
                            atol=self.thresholds["coefficient_symmetry_absolute_tolerance"],
                        )
                    )

    def test_overlap_add_matches_direct_convolution_on_bounded_input(self) -> None:
        rng = np.random.default_rng(20260811)
        for sampling_rate_hz in (512, 1024):
            with self.subTest(sampling_rate_hz=sampling_rate_hz):
                kernel = design_bandpass_fir(sampling_rate_hz, (90.0, 100.0))
                signal = rng.normal(size=kernel.size + 257)
                actual = overlap_add_fir_same(signal, kernel)
                expected = np.convolve(signal, kernel, mode="same")
                self.assertEqual(actual.shape, signal.shape)
                self.assertTrue(
                    np.allclose(
                        actual,
                        expected,
                        rtol=self.thresholds["overlap_add_vs_direct_relative_tolerance"],
                        atol=self.thresholds["overlap_add_vs_direct_absolute_tolerance"],
                    )
                )

    def test_frequency_response_meets_frozen_passband_and_harmonic_thresholds(self) -> None:
        for sampling_rate_hz in (512, 1024):
            for band_hz in PRIMARY_BANDS_HZ:
                with self.subTest(sampling_rate_hz=sampling_rate_hz, band_hz=band_hz):
                    center_hz = sum(band_hz) / 2.0
                    _, response = freqz(
                        design_bandpass_fir(sampling_rate_hz, band_hz),
                        worN=np.asarray([60.0, 120.0, center_hz]),
                        fs=sampling_rate_hz,
                    )
                    gain_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-300))
                    self.assertLessEqual(
                        gain_db[0], self.thresholds["line_60_hz_maximum_gain_db"]
                    )
                    self.assertLessEqual(
                        gain_db[1], self.thresholds["harmonic_120_hz_maximum_gain_db"]
                    )
                    self.assertGreaterEqual(
                        gain_db[2], self.thresholds["passband_center_minimum_gain_db"]
                    )
                    self.assertLessEqual(
                        gain_db[2], self.thresholds["passband_center_maximum_gain_db"]
                    )

    def test_impulse_step_and_sine_outputs_obey_finite_support(self) -> None:
        for sampling_rate_hz in (512, 1024):
            with self.subTest(sampling_rate_hz=sampling_rate_hz):
                profile = support_metadata(sampling_rate_hz)
                length = max(8192, 8 * int(profile["bandpass_numtaps"]))
                center = length // 2
                impulse = np.zeros(length)
                impulse[center] = 1.0
                impulse_power = finite_support_power(
                    impulse, sampling_rate_hz, (90.0, 100.0)
                )
                radius = int(profile["finite_fir_chain_radius_samples"])
                outside = np.ones(length, dtype=bool)
                outside[center - radius : center + radius + 1] = False
                self.assertLessEqual(
                    float(np.max(np.abs(impulse_power[outside]))),
                    self.thresholds[
                        "impulse_outside_finite_fir_support_absolute_tolerance"
                    ],
                )

                step_power = finite_support_power(
                    np.ones(length), sampling_rate_hz, (90.0, 100.0)
                )
                self.assertEqual(step_power.shape, (length,))
                self.assertTrue(np.all(np.isfinite(step_power)))

                times = np.arange(length, dtype=np.float64) / sampling_rate_hz
                interior = slice(
                    int(profile["total_filter_resampling_edge_samples"]),
                    -int(profile["total_filter_resampling_edge_samples"]),
                )
                passband_power = finite_support_power(
                    np.sin(2 * np.pi * 95.0 * times), sampling_rate_hz, (90.0, 100.0)
                )
                passband_rms = float(np.sqrt(np.mean(passband_power[interior] ** 2)))
                for harmonic_hz in (60.0, 120.0):
                    harmonic_power = finite_support_power(
                        np.sin(2 * np.pi * harmonic_hz * times),
                        sampling_rate_hz,
                        (90.0, 100.0),
                    )
                    ratio = float(
                        np.sqrt(np.mean(harmonic_power[interior] ** 2)) / passband_rms
                    )
                    self.assertLessEqual(
                        ratio,
                        self.thresholds[
                            "time_domain_harmonic_to_passband_rms_maximum_ratio"
                        ],
                    )

    def test_short_and_nonfinite_inputs_fail_closed(self) -> None:
        for sampling_rate_hz in (512, 1024):
            kernel_size = support_metadata(sampling_rate_hz)["bandpass_numtaps"]
            with self.subTest(sampling_rate_hz=sampling_rate_hz, case="short"):
                with self.assertRaises(ValueError):
                    finite_support_power(
                        np.zeros(int(kernel_size) - 1),
                        sampling_rate_hz,
                        (90.0, 100.0),
                    )
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(sampling_rate_hz=sampling_rate_hz, value=value):
                    signal = np.zeros(int(kernel_size))
                    signal[0] = value
                    with self.assertRaises(ValueError):
                        finite_support_power(signal, sampling_rate_hz, (90.0, 100.0))

    def test_support_mask_includes_exact_boundaries_and_rejects_nonfinite(self) -> None:
        mask = fully_supported_frame_mask(
            np.asarray([0.999, 1.0, 5.0, 9.0, 9.001]),
            allowed_start_seconds=0.0,
            allowed_end_seconds=10.0,
            support_edge_seconds=1.0,
        )
        np.testing.assert_array_equal(mask, [False, True, True, True, False])
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    fully_supported_frame_mask(
                        np.asarray([value]), 0.0, 10.0, support_edge_seconds=1.0
                    )


class TrainOnlyTransformTests(unittest.TestCase):
    def test_fit_uses_train_valid_frames_only_and_aggregation_is_equal_weight(self) -> None:
        rng = np.random.default_rng(17)
        power = np.exp(rng.normal(size=(12, 6)))
        mask = np.zeros(12, dtype=bool)
        mask[:8] = True
        first = fit_train_only_power_transform(power, mask)
        changed = power.copy()
        changed[~mask] = math.nan
        second = fit_train_only_power_transform(changed, mask)
        np.testing.assert_array_equal(first.epsilon_by_band, second.epsilon_by_band)
        np.testing.assert_array_equal(first.center_by_band, second.center_by_band)
        np.testing.assert_array_equal(first.scale_by_band, second.scale_by_band)

        standardized, target = apply_train_only_power_transform(power, first)
        np.testing.assert_allclose(target, np.mean(standardized, axis=1), rtol=0, atol=0)

    def test_degenerate_fit_and_invalid_frozen_parameters_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            fit_train_only_power_transform(np.ones((8, 6)), np.ones(8, dtype=bool))

        valid_power = np.ones((4, 6))
        invalid_transforms = (
            TrainOnlyPowerTransform(np.ones(5), np.zeros(6), np.ones(6)),
            TrainOnlyPowerTransform(np.zeros(6), np.zeros(6), np.ones(6)),
            TrainOnlyPowerTransform(np.ones(6), np.full(6, math.nan), np.ones(6)),
            TrainOnlyPowerTransform(np.ones(6), np.zeros(6), np.full(6, math.inf)),
            TrainOnlyPowerTransform(np.ones(6), np.zeros(6), np.zeros(6)),
        )
        for transform in invalid_transforms:
            with self.subTest(transform=transform):
                with self.assertRaises(ValueError):
                    apply_train_only_power_transform(valid_power, transform)


if __name__ == "__main__":
    unittest.main()
