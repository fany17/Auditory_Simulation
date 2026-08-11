from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, NoReturn

from jsonschema import Draft202012Validator

from m6a_public.config_gate import find_forbidden_fields
from m6a_public.neural_target_method import PRIMARY_BANDS_HZ, support_metadata


EXPECTED_REFERENCE = "scalp electrode, not included with data"
EXPECTED_PRIMARY_POLICY = "AS_RECORDED_SCALP_REFERENCE"
EXPECTED_LAYER_KEYS = (
    "hg_70_80",
    "hg_80_90",
    "hg_90_100",
    "hg_100_110",
    "hg_130_140",
    "hg_140_150",
)
EXPECTED_THRESHOLDS = {
    "coefficient_symmetry_absolute_tolerance": 1e-12,
    "overlap_add_vs_direct_absolute_tolerance": 1e-12,
    "overlap_add_vs_direct_relative_tolerance": 1e-12,
    "impulse_outside_finite_fir_support_absolute_tolerance": 1e-12,
    "passband_center_minimum_gain_db": -0.1,
    "passband_center_maximum_gain_db": 0.1,
    "line_60_hz_maximum_gain_db": -55,
    "harmonic_120_hz_maximum_gain_db": -55,
    "time_domain_harmonic_to_passband_rms_maximum_ratio": 0.01,
    "nonfinite_output_allowed": False,
}


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_strict_json_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_nonstandard_json_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _nonfinite_numeric_errors(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"non-finite numeric value: {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_nonfinite_numeric_errors(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_nonfinite_numeric_errors(child, f"{path}[{index}]"))
    return errors


def _schema_errors(candidate: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors: list[str] = []
    for error in sorted(validator.iter_errors(candidate), key=lambda item: list(item.absolute_path)):
        location = "$"
        for part in error.absolute_path:
            location += f"[{part}]" if isinstance(part, int) else f".{part}"
        errors.append(f"schema {location}: {error.message}")
    return errors


def _same_number(actual: Any, expected: int | float) -> bool:
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    )


def validate_neural_target_method(
    candidate: dict[str, Any],
    schema: dict[str, Any],
) -> list[str]:
    errors = _nonfinite_numeric_errors(candidate)
    errors.extend(_schema_errors(candidate, schema))
    if errors:
        return errors

    forbidden = find_forbidden_fields(candidate)
    if forbidden:
        errors.append("forbidden integrity fields: " + ", ".join(forbidden))

    primary = candidate["primary_target"]
    expected_subbands = [
        {
            "layer_key": layer_key,
            "low_hz": low_hz,
            "high_hz": high_hz,
            "weight": 1.0 / len(PRIMARY_BANDS_HZ),
        }
        for layer_key, (low_hz, high_hz) in zip(EXPECTED_LAYER_KEYS, PRIMARY_BANDS_HZ)
    ]
    if primary["subbands"] != expected_subbands:
        errors.append("primary subbands, layer order and equal weights must match the frozen six-band inventory")
    if primary["excluded_harmonic_guard_hz"] != [110, 130]:
        errors.append("110-130 Hz harmonic guard must remain excluded")
    if primary["selection_rule"] != "PREDECLARED_PRIMARY_NEVER_SELECTED_BY_ENCODING_PERFORMANCE":
        errors.append("primary target cannot be selected or replaced using encoding performance")

    reference = candidate["reference"]
    if reference["sidecar_value"] != EXPECTED_REFERENCE:
        errors.append("sidecar iEEGReference value is not frozen")
    if reference["primary_policy"] != EXPECTED_PRIMARY_POLICY:
        errors.append("primary reference must remain AS_RECORDED_SCALP_REFERENCE")
    if reference["missing_reference_reconstruction"] != "FORBIDDEN":
        errors.append("missing scalp reference reconstruction must remain forbidden")
    if reference["contact_name_bipolar_pairing"] != "FORBIDDEN":
        errors.append("contact-name bipolar construction must remain forbidden")

    bandpass = candidate["bandpass_fir"]
    if bandpass["execution_backend"] != "scipy.signal.oaconvolve":
        errors.append("formal FIR backend must use audited overlap-add convolution")
    if bandpass["transition_width_hz"] != 2 or bandpass["stopband_attenuation_db"] != 60:
        errors.append("bandpass transition and attenuation parameters are not frozen")
    if bandpass["minimum_input_samples_rule"] != "INPUT_LENGTH_MUST_BE_AT_LEAST_BANDPASS_NUMTAPS":
        errors.append("short input rejection rule is not frozen")

    power = candidate["power_estimator"]
    if power["method"] != "SQUARE_THEN_FINITE_FIR_LOWPASS":
        errors.append("power estimator must remain finite square-then-lowpass")
    if power["hilbert_transform"] != "FORBIDDEN":
        errors.append("whole-recording FFT Hilbert processing is forbidden")
    if (
        power["lowpass_passband_edge_hz"] != 10
        or power["lowpass_stopband_edge_hz"] != 20
        or power["stopband_attenuation_db"] != 60
    ):
        errors.append("power smoothing FIR parameters are not frozen")

    profiles = candidate["sampling_rate_profiles"]
    if [item["sampling_rate_hz"] for item in profiles] != [512, 1024]:
        errors.append("sampling profiles must be ordered 512 then 1024 Hz")
    for profile in profiles:
        expected_profile = support_metadata(int(profile["sampling_rate_hz"]))
        for key, expected_value in expected_profile.items():
            if not _same_number(profile.get(key), expected_value):
                errors.append(
                    f"sampling profile {profile['sampling_rate_hz']} has non-frozen {key}"
                )

    transform = candidate["train_only_transform"]
    if transform["fit_scope"] != "TRAIN_VALID_FRAMES_ONLY":
        errors.append("epsilon, center and scale must be fit on train valid frames only")
    if transform["application_scope"] != "FROZEN_APPLY_TO_VALIDATION_AND_TEST":
        errors.append("train-only transform must be frozen before validation/test application")
    if transform["degenerate_scale_action"] != "FAIL_CHANNEL_SUBBAND":
        errors.append("degenerate train scale must fail closed")

    time_grid = candidate["time_grid"]
    if time_grid["target_frame_rate_hz"] != 50:
        errors.append("neural target grid must remain 50 Hz")
    if time_grid["frame_timestamp"] != "RECORDING_TIME_SECONDS_AT_OUTPUT_FRAME_CENTER":
        errors.append("frame timestamp semantics are not frozen")
    if "COMPLETE_FILTER_AND_INTERPOLATION_SUPPORT" not in time_grid["output_validity"]:
        errors.append("output frames must require full finite support inside the allowed interval")

    lag = candidate["lag"]
    if lag["positive_lag_semantics"] != "AUDIO_AT_T_PREDICTS_NEURAL_AT_T_PLUS_LAG":
        errors.append("positive lag semantics are not frozen")
    if lag["maximum_lag_seconds"] != 0.5:
        errors.append("maximum encoding lag must remain 0.5 seconds")

    sensitivity = candidate["sensitivity"]
    if sensitivity["status"] != "PREDECLARED_SENSITIVITY_NOT_PRIMARY":
        errors.append("legacy 70-150 Hz method must remain sensitivity-only")
    if sensitivity["may_replace_primary"] or sensitivity["selection_by_result_allowed"]:
        errors.append("sensitivity cannot replace the primary or be selected by results")
    if sensitivity["execution_requires_separate_approval"] is not True:
        errors.append("sensitivity execution requires separate approval")

    if candidate["synthetic_test_thresholds"] != EXPECTED_THRESHOLDS:
        errors.append("synthetic filter thresholds are not frozen")

    edge = candidate["final_embargo"]
    computed_max = max(
        float(profile["total_filter_resampling_edge_seconds"]) for profile in profiles
    )
    if not _same_number(edge["computed_filter_resampling_edge_seconds_max"], computed_max):
        errors.append("final embargo filter/resampling edge disagrees with sampling profiles")
    if edge["audio_cross_split_context_overlap_seconds"] is not None:
        errors.append("audio context overlap must remain pending until the frozen model is measured")
    if edge["baseline_final"] is not False:
        errors.append("baseline cannot be final before audio context measurement and split guard rerun")

    execution = candidate["execution"]
    if execution != {
        "synthetic_tests_allowed": True,
        "real_neural_waveform_target_extraction_allowed": False,
        "model_download_allowed": False,
        "baseline_allowed": False,
    }:
        errors.append("candidate execution gate permits only synthetic tests")

    anatomy = candidate["anatomy"]
    if anatomy["region_summary"] != "NOT_ESTIMABLE":
        errors.append("region summary must remain NOT_ESTIMABLE")
    if anatomy["contact_name_region_inference"] != "FORBIDDEN":
        errors.append("contact-name region inference must remain forbidden")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the neural target method candidate.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("schema", type=Path)
    args = parser.parse_args()
    errors = validate_neural_target_method(
        load_strict_json_object(args.candidate),
        load_strict_json_object(args.schema),
    )
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
