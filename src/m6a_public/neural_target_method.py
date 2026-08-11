from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import firwin, kaiserord, oaconvolve


PRIMARY_BANDS_HZ = (
    (70.0, 80.0),
    (80.0, 90.0),
    (90.0, 100.0),
    (100.0, 110.0),
    (130.0, 140.0),
    (140.0, 150.0),
)
SUPPORTED_SAMPLING_RATES_HZ = (512, 1024)
BANDPASS_TRANSITION_HZ = 2.0
STOPBAND_ATTENUATION_DB = 60.0
POWER_PASSBAND_EDGE_HZ = 10.0
POWER_STOPBAND_EDGE_HZ = 20.0
POWER_CUTOFF_HZ = 15.0
POWER_TRANSITION_HZ = POWER_STOPBAND_EDGE_HZ - POWER_PASSBAND_EDGE_HZ
TARGET_FRAME_RATE_HZ = 50.0
LOG_EPSILON_MULTIPLIER = 1e-6
LOG_EPSILON_ABSOLUTE_FLOOR = 1e-30
TRANSFORM_BAND_COUNT = len(PRIMARY_BANDS_HZ)


def odd_kaiser_numtaps(sampling_rate_hz: int, transition_hz: float) -> tuple[int, float]:
    if sampling_rate_hz not in SUPPORTED_SAMPLING_RATES_HZ:
        raise ValueError(f"unsupported sampling rate: {sampling_rate_hz}")
    numtaps, beta = kaiserord(
        STOPBAND_ATTENUATION_DB,
        transition_hz / (sampling_rate_hz / 2.0),
    )
    if numtaps % 2 == 0:
        numtaps += 1
    return int(numtaps), float(beta)


def design_bandpass_fir(
    sampling_rate_hz: int,
    band_hz: tuple[float, float],
) -> np.ndarray:
    if band_hz not in PRIMARY_BANDS_HZ:
        raise ValueError(f"band is not in the frozen primary inventory: {band_hz}")
    numtaps, beta = odd_kaiser_numtaps(sampling_rate_hz, BANDPASS_TRANSITION_HZ)
    low_hz, high_hz = band_hz
    cutoffs = (
        low_hz - BANDPASS_TRANSITION_HZ / 2.0,
        high_hz + BANDPASS_TRANSITION_HZ / 2.0,
    )
    return np.asarray(
        firwin(
            numtaps,
            cutoffs,
            window=("kaiser", beta),
            pass_zero=False,
            scale=True,
            fs=sampling_rate_hz,
        ),
        dtype=np.float64,
    )


def design_power_smoothing_fir(sampling_rate_hz: int) -> np.ndarray:
    numtaps, beta = odd_kaiser_numtaps(sampling_rate_hz, POWER_TRANSITION_HZ)
    return np.asarray(
        firwin(
            numtaps,
            POWER_CUTOFF_HZ,
            window=("kaiser", beta),
            pass_zero="lowpass",
            scale=True,
            fs=sampling_rate_hz,
        ),
        dtype=np.float64,
    )


def support_metadata(sampling_rate_hz: int) -> dict[str, Any]:
    bandpass_numtaps, bandpass_beta = odd_kaiser_numtaps(
        sampling_rate_hz, BANDPASS_TRANSITION_HZ
    )
    power_numtaps, power_beta = odd_kaiser_numtaps(sampling_rate_hz, POWER_TRANSITION_HZ)
    bandpass_radius_samples = (bandpass_numtaps - 1) // 2
    power_radius_samples = (power_numtaps - 1) // 2
    interpolation_radius_samples = 1
    total_radius_samples = (
        bandpass_radius_samples + power_radius_samples + interpolation_radius_samples
    )
    return {
        "sampling_rate_hz": sampling_rate_hz,
        "bandpass_numtaps": bandpass_numtaps,
        "bandpass_kaiser_beta": bandpass_beta,
        "bandpass_radius_samples": bandpass_radius_samples,
        "power_smoothing_numtaps": power_numtaps,
        "power_smoothing_kaiser_beta": power_beta,
        "power_smoothing_radius_samples": power_radius_samples,
        "finite_fir_chain_radius_samples": bandpass_radius_samples
        + power_radius_samples,
        "linear_interpolation_radius_samples": interpolation_radius_samples,
        "total_filter_resampling_edge_samples": total_radius_samples,
        "total_filter_resampling_edge_seconds": total_radius_samples / sampling_rate_hz,
    }


def overlap_add_fir_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply an odd, symmetric FIR with overlap-add and preserve input length.

    Short inputs are rejected instead of relying on ambiguous ``same``-mode
    behavior when the FIR is longer than the recording.
    """

    values = np.asarray(signal, dtype=np.float64)
    coefficients = np.asarray(kernel, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a finite one-dimensional array")
    if coefficients.ndim != 1 or coefficients.size % 2 != 1:
        raise ValueError("FIR kernel must be a one-dimensional odd-length array")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("FIR kernel must be finite")
    if values.size < coefficients.size:
        raise ValueError("signal length must be at least the FIR kernel length")
    result = np.asarray(oaconvolve(values, coefficients, mode="same"), dtype=np.float64)
    if result.shape != values.shape:
        raise RuntimeError("overlap-add same-mode convolution changed the signal length")
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite FIR output")
    return result


def finite_support_power(
    signal: np.ndarray,
    sampling_rate_hz: int,
    band_hz: tuple[float, float],
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("signal must be a finite one-dimensional array")
    bandpass_fir = design_bandpass_fir(sampling_rate_hz, band_hz)
    power_smoothing_fir = design_power_smoothing_fir(sampling_rate_hz)
    if values.size < bandpass_fir.size:
        raise ValueError("signal is shorter than the frozen bandpass FIR")
    bandpassed = overlap_add_fir_same(values, bandpass_fir)
    squared = bandpassed * bandpassed
    power = overlap_add_fir_same(squared, power_smoothing_fir)
    if not np.all(np.isfinite(power)):
        raise ValueError("non-finite power output")
    return power


def fully_supported_frame_mask(
    frame_times_seconds: np.ndarray,
    allowed_start_seconds: float,
    allowed_end_seconds: float,
    support_edge_seconds: float,
) -> np.ndarray:
    frame_times = np.asarray(frame_times_seconds, dtype=np.float64)
    if frame_times.ndim != 1 or not np.all(np.isfinite(frame_times)):
        raise ValueError("frame times must be a finite one-dimensional array")
    values = (allowed_start_seconds, allowed_end_seconds, support_edge_seconds)
    if not all(math.isfinite(value) for value in values) or support_edge_seconds < 0:
        raise ValueError("support mask bounds must be finite and edge must be non-negative")
    if allowed_end_seconds <= allowed_start_seconds:
        raise ValueError("allowed interval must have positive duration")
    return (frame_times - support_edge_seconds >= allowed_start_seconds) & (
        frame_times + support_edge_seconds <= allowed_end_seconds
    )


@dataclass(frozen=True)
class TrainOnlyPowerTransform:
    epsilon_by_band: np.ndarray
    center_by_band: np.ndarray
    scale_by_band: np.ndarray


def validate_train_only_power_transform(transform: TrainOnlyPowerTransform) -> None:
    fields = {
        "epsilon_by_band": np.asarray(transform.epsilon_by_band, dtype=np.float64),
        "center_by_band": np.asarray(transform.center_by_band, dtype=np.float64),
        "scale_by_band": np.asarray(transform.scale_by_band, dtype=np.float64),
    }
    expected_shape = (TRANSFORM_BAND_COUNT,)
    for name, values in fields.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    if np.any(fields["epsilon_by_band"] <= 0):
        raise ValueError("epsilon_by_band must be strictly positive")
    if np.any(fields["scale_by_band"] <= 0):
        raise ValueError("scale_by_band must be strictly positive")


def fit_train_only_power_transform(
    power_by_frame_and_band: np.ndarray,
    train_valid_mask: np.ndarray,
) -> TrainOnlyPowerTransform:
    power = np.asarray(power_by_frame_and_band, dtype=np.float64)
    mask = np.asarray(train_valid_mask, dtype=bool)
    if power.ndim != 2 or power.shape[1] != len(PRIMARY_BANDS_HZ):
        raise ValueError("power must have shape [frames, six frozen bands]")
    if mask.shape != (power.shape[0],) or not np.any(mask):
        raise ValueError("train_valid_mask must select at least one frame")
    train = power[mask]
    if not np.all(np.isfinite(train)) or np.any(train < 0):
        raise ValueError("train power must be finite and non-negative")

    epsilon_values: list[float] = []
    for band_index in range(train.shape[1]):
        positive = train[:, band_index][train[:, band_index] > 0]
        if positive.size == 0:
            raise ValueError(f"band {band_index} has no positive train power")
        epsilon_values.append(
            max(
                LOG_EPSILON_ABSOLUTE_FLOOR,
                LOG_EPSILON_MULTIPLIER * float(np.median(positive)),
            )
        )
    epsilon = np.asarray(epsilon_values, dtype=np.float64)
    logged_train = np.log(train + epsilon)
    center = np.mean(logged_train, axis=0)
    scale = np.std(logged_train, axis=0, ddof=0)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("train-only center/scale is non-finite or degenerate")
    transform = TrainOnlyPowerTransform(epsilon, center, scale)
    validate_train_only_power_transform(transform)
    return transform


def apply_train_only_power_transform(
    power_by_frame_and_band: np.ndarray,
    transform: TrainOnlyPowerTransform,
) -> tuple[np.ndarray, np.ndarray]:
    power = np.asarray(power_by_frame_and_band, dtype=np.float64)
    if power.ndim != 2 or power.shape[1] != len(PRIMARY_BANDS_HZ):
        raise ValueError("power must have shape [frames, six frozen bands]")
    if not np.all(np.isfinite(power)) or np.any(power < 0):
        raise ValueError("power must be finite and non-negative")
    validate_train_only_power_transform(transform)
    epsilon = np.asarray(transform.epsilon_by_band, dtype=np.float64)
    center = np.asarray(transform.center_by_band, dtype=np.float64)
    scale = np.asarray(transform.scale_by_band, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logged = np.log(power + epsilon)
        standardized = (logged - center) / scale
    target = np.mean(standardized, axis=1)
    if not np.all(np.isfinite(standardized)) or not np.all(np.isfinite(target)):
        raise ValueError("non-finite standardized target")
    return standardized, target
