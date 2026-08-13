from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from m6a_public.audio_context_gate import (
    EXPECTED_LAYER_KEYS,
    EXPECTED_PRETRAINING_HEAD_KEYS,
    audit_pytorch_weight_file,
    derive_convolution_timing,
    expected_frame_count,
    frame_center_seconds,
    resample_independent_passage,
)
from m6a_public.g3_single_recording_gate import (
    amplitude_envelope_native,
    linear_align_no_extrapolation,
    log_mel_native,
    native_audio_frame_centers,
    passage_grid_seconds,
)
from m6a_public.g4_protocol_gate import _rank_test_derangements
from m6a_public.neural_target_method import (
    PRIMARY_BANDS_HZ,
    finite_support_power,
    support_metadata,
)
from m6a_public.wav2vec2_preprocessing import PREPROCESSOR_SEMANTICS


RECORDING_ID = "sub-SD012_ses-02_task-PassiveListen"
PARTICIPANT_ID = "sub-SD012"
SESSION_ID = "ses-02"
EXPECTED_SPLITS = {"train": 24, "validation": 8, "test": 8}
ALPHAS = np.asarray([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float64)
LAGS = np.asarray([index * 0.05 for index in range(11)], dtype=np.float64)
MINIMUM_SCALE = 1e-12
RANK_TOLERANCE_DTYPE = np.float64


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant: {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial-{os.getpid()}")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".partial-{os.getpid()}")
    with partial.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _load_scope(split_csv: Path) -> list[dict[str, Any]]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("recording_id") == RECORDING_ID
        ]
    counts = {name: sum(row.get("split") == name for row in rows) for name in EXPECTED_SPLITS}
    problems: list[str] = []
    if len(rows) != 40 or counts != EXPECTED_SPLITS:
        problems.append(f"expected exact 40 passages and 24/8/8, observed {len(rows)} and {counts}")
    if any(
        row.get("participant_id") != PARTICIPANT_ID
        or row.get("session_id") != SESSION_ID
        or row.get("analysis_eligible") != "True"
        or row.get("language") != "en"
        or row.get("audio_source_status") != "BUNDLED_BLOCK_AUDIO"
        or row.get("audio_sample_rate_hz") != "44100"
        or row.get("audio_channels") != "1"
        for row in rows
    ):
        problems.append("scope or frozen input metadata drifted")
    for field in ("sample_id", "stimulus_id", "audio_file"):
        values = [str(row.get(field, "")) for row in rows]
        if any(not value for value in values) or len(set(values)) != 40:
            problems.append(f"{field} must be nonempty and unique across the 40 passages")
    stimulus_splits: dict[str, set[str]] = {}
    for row in rows:
        stimulus_splits.setdefault(str(row["stimulus_id"]), set()).add(str(row["split"]))
    leaked = sorted(key for key, values in stimulus_splits.items() if len(values) != 1)
    if leaked:
        problems.append(f"stimulus leakage across split: {leaked}")
    if problems:
        raise ValueError("; ".join(problems))
    normalized: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        item: dict[str, Any] = dict(row)
        item["start_seconds"] = float(row["start_sec"])
        item["end_seconds"] = float(row["end_sec"])
        item["audio_frames"] = int(row["audio_frames"])
        item["audio_duration_seconds"] = float(row["audio_duration_seconds"])
        normalized.append(item)
    return normalized


def _select_recording(neural_report: dict[str, Any]) -> dict[str, Any]:
    recordings = neural_report.get("recordings")
    if not isinstance(recordings, list):
        raise ValueError("neural report recordings are missing")
    matches = [item for item in recordings if item.get("recording_id") == RECORDING_ID]
    if len(matches) != 1:
        raise ValueError("expected one frozen neural recording")
    recording = cast(dict[str, Any], matches[0])
    channels = cast(dict[str, Any], recording.get("channels", {}))
    names = channels.get("analysis_eligible_neural_names")
    header = cast(dict[str, Any], recording.get("edf_header", {}))
    if (
        recording.get("participant_id") != PARTICIPANT_ID
        or recording.get("session_id") != SESSION_ID
        or float(recording.get("sampling_rate_hz", 0)) != 512.0
        or recording.get("iEEGReference") != "scalp electrode, not included with data"
        or channels.get("analysis_eligible_neural_channel_count") != 36
        or not isinstance(names, list)
        or len(names) != 36
        or len(set(names)) != 36
        or recording.get("events_within_edf_timeline") is not True
        or recording.get("edf_header_channel_count_matches_tsv") is not True
        or recording.get("analysis_eligible_neural_channels_missing_from_edf") != []
        or float(header.get("sampling_rate_hz", 0)) != 512.0
    ):
        raise ValueError("frozen neural recording input gate failed")
    return {
        "edf_file": str(recording["edf_file"]),
        "sampling_rate_hz": 512,
        "eligible_channel_names": [str(value) for value in names],
        "recording_total_samples": int(round(float(header["duration_seconds"]) * 512)) + 1,
    }


class BoundedNeuralReader:
    def __init__(self, edf_path: Path, recording: dict[str, Any]) -> None:
        import mne  # type: ignore[import-untyped]

        self.raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        names = list(recording["eligible_channel_names"])
        if self.raw.preload or not set(names).issubset(set(self.raw.ch_names)):
            raise ValueError("bounded EDF input gate failed")
        self.raw.pick(names)
        if list(self.raw.ch_names) != names or int(round(self.raw.info["sfreq"])) != 512:
            raise ValueError("EDF channel identity or sampling rate drifted")
        self.names = names
        self.fs = 512
        self.edge = int(support_metadata(self.fs)["total_filter_resampling_edge_samples"])

    def close(self) -> None:
        self.raw.close()

    def power(self, row: dict[str, Any]) -> tuple[np.ndarray, int, dict[str, Any]]:
        start_floor = math.floor(float(row["start_seconds"]) * self.fs)
        end_ceil = math.ceil(float(row["end_seconds"]) * self.fs)
        read_start = max(0, start_floor - self.edge)
        read_end = min(int(self.raw.n_times), end_ceil + self.edge)
        data = self.raw.get_data(start=read_start, stop=read_end)
        if data.shape != (36, read_end - read_start) or not np.all(np.isfinite(data)):
            raise ValueError("bounded EDF read returned invalid data")
        power = np.empty((data.shape[1], 36, 6), dtype=np.float32)
        valid_start = max(
            0,
            math.ceil((float(row["start_seconds"]) + self.edge / self.fs) * self.fs)
            - read_start,
        )
        valid_end = min(
            data.shape[1],
            math.floor((float(row["end_seconds"]) - self.edge / self.fs) * self.fs)
            - read_start
            + 1,
        )
        if valid_end <= valid_start:
            raise ValueError("neural complete-support interval is empty")
        complete_support_negative_minimum = 0.0
        for channel in range(36):
            for band, band_hz in enumerate(PRIMARY_BANDS_HZ):
                values = finite_support_power(data[channel], self.fs, band_hz)
                minimum = float(np.min(values[valid_start:valid_end]))
                complete_support_negative_minimum = min(
                    complete_support_negative_minimum, minimum
                )
                if minimum < -1e-12 or not np.all(np.isfinite(values)):
                    raise ValueError("neural finite-support power gate failed")
                power[:, channel, band] = np.maximum(values, 0.0).astype(np.float32)
        return power, read_start, {
            "read_start_sample": read_start,
            "read_end_sample_exclusive": read_end,
            "read_sample_count": read_end - read_start,
            "support_edge_samples": self.edge,
            "complete_support_start_offset": valid_start,
            "complete_support_end_offset_exclusive": valid_end,
            "complete_support_negative_minimum_before_clip": (
                complete_support_negative_minimum
            ),
            "negative_power_absolute_tolerance": 1e-12,
            "channel_count": 36,
            "mne_preload": False,
        }


class FrozenWav2Vec2:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        weight_audit = audit_pytorch_weight_file(model_dir / "pytorch_model.bin")
        if weight_audit.get("status") != "PASS" or weight_audit.get("weights_only") is not True:
            raise ValueError("weights-only audit failed")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_dir, local_files_only=True
        )
        observed = {
            "feature_size": int(self.extractor.feature_size),
            "sampling_rate": int(self.extractor.sampling_rate),
            "padding_value": float(self.extractor.padding_value),
            "do_normalize": bool(self.extractor.do_normalize),
            "return_attention_mask": bool(self.extractor.return_attention_mask),
            "padding_side": str(self.extractor.padding_side),
        }
        if observed != PREPROCESSOR_SEMANTICS:
            raise ValueError(f"preprocessor semantics drifted: {observed}")
        loaded = Wav2Vec2Model.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            output_loading_info=True,
        )
        self.model, loading_info = cast(tuple[Any, dict[str, Any]], loaded)
        normalized_loading = {
            "missing_keys": sorted(str(key) for key in loading_info.get("missing_keys", [])),
            "unexpected_keys": sorted(str(key) for key in loading_info.get("unexpected_keys", [])),
            "mismatched_keys": sorted(str(key) for key in loading_info.get("mismatched_keys", [])),
            "error_msgs": [str(value) for value in loading_info.get("error_msgs", [])],
        }
        if normalized_loading != {
            "missing_keys": [],
            "unexpected_keys": list(EXPECTED_PRETRAINING_HEAD_KEYS),
            "mismatched_keys": [],
            "error_msgs": [],
        }:
            raise ValueError("model loading information drifted")
        self.model.requires_grad_(False)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("G4 wav2vec2 extraction requires the authorized 2203 GPU")
        self.model.to(self.device)
        self.torch = torch
        self.timing = derive_convolution_timing(
            tuple(int(value) for value in self.model.config.conv_kernel),
            tuple(int(value) for value in self.model.config.conv_stride),
        )
        self.runtime = {
            "device": str(self.device),
            "local_files_only": True,
            "trust_remote_code": False,
            "weights_only": True,
            "download_attempted": False,
            "attention_mask_argument": "OMITTED",
            "model_eval": not self.model.training,
            "parameter_requires_grad_count": sum(
                int(parameter.requires_grad) for parameter in self.model.parameters()
            ),
        }

    def extract(
        self, waveform: np.ndarray, passage_start: float, grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if waveform.ndim != 1 or not np.all(np.isfinite(waveform)) or float(np.var(waveform)) <= 0:
            raise ValueError("wav2vec2 waveform must be finite, mono and nonconstant")
        batch = self.extractor(
            waveform.astype(np.float32, copy=False),
            sampling_rate=16000,
            return_tensors="np",
            padding=False,
        )
        if "attention_mask" in batch:
            raise ValueError("attention mask must be omitted")
        values = np.asarray(batch["input_values"][0], dtype=np.float32)
        if values.shape != waveform.shape or not np.all(np.isfinite(values)):
            raise ValueError("preprocessor output gate failed")
        input_values = self.torch.from_numpy(values).unsqueeze(0).to(self.device)
        projected: list[Any] = []

        def capture_projection(_module: Any, _inputs: Any, output: Any) -> None:
            projected.append(output[0].detach())

        hook = self.model.feature_projection.register_forward_hook(capture_projection)
        try:
            with self.torch.inference_mode():
                output = self.model(
                    input_values=input_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook.remove()
        if len(projected) != 1 or output.hidden_states is None or len(output.hidden_states) != 13:
            raise ValueError("wav2vec2 layer inventory drifted")
        layers = [projected[0], *output.hidden_states[1:]]
        if not all(bool(self.torch.isfinite(layer).all().item()) for layer in layers):
            raise ValueError("wav2vec2 hidden states are non-finite")
        native = np.stack(
            [layer[0].detach().cpu().numpy().astype(np.float32) for layer in layers],
            axis=1,
        )
        frames = expected_frame_count(int(values.size), self.timing)
        if native.shape != (frames, 13, 768):
            raise ValueError("wav2vec2 frame shape drifted")
        native_times = passage_start + frame_center_seconds(frames, self.timing)
        aligned, valid = linear_align_no_extrapolation(native_times, native, grid)
        return np.transpose(aligned, (1, 0, 2)).astype(np.float32), valid


def _interpolate_power(
    power: np.ndarray, read_start: int, sampling_rate: int, times: np.ndarray
) -> np.ndarray:
    positions = times * sampling_rate - read_start
    lower = np.floor(positions).astype(np.int64)
    upper = lower + 1
    if np.any(lower < 0) or np.any(upper >= power.shape[0]):
        raise ValueError("neural target interpolation would extrapolate")
    weight = (positions - lower).reshape(-1, 1, 1)
    return (
        power[lower].astype(np.float64) * (1.0 - weight)
        + power[upper].astype(np.float64) * weight
    )


def _checkpoint_passage(
    row: dict[str, Any],
    output_root: Path,
    dataset_root: Path,
    reader: BoundedNeuralReader,
    wav: FrozenWav2Vec2,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    final_root = output_root / "passages" / sample_id
    metadata_path = final_root / "metadata.json"
    required = [
        "times.npy",
        "envelope.npy",
        "logmel.npy",
        "wav2vec2.npy",
        "neural_power_native.npy",
        "metadata.json",
    ]
    if final_root.is_dir() and all((final_root / name).is_file() for name in required):
        return _load_json(metadata_path)
    if final_root.exists():
        raise RuntimeError(f"incomplete final passage checkpoint exists: {final_root}")
    partial_root = final_root.with_name(final_root.name + ".partial")
    if partial_root.exists():
        raise RuntimeError(f"interrupted passage checkpoint requires review: {partial_root}")
    partial_root.mkdir(parents=True, exist_ok=False)
    import soundfile as sf  # type: ignore[import-untyped]

    audio_path = (dataset_root / str(row["audio_file"])).resolve()
    audio_path.relative_to(dataset_root.resolve())
    audio, rate = sf.read(audio_path, dtype="float64", always_2d=True)
    if (
        rate != 44100
        or audio.shape != (int(row["audio_frames"]), 1)
        or not np.all(np.isfinite(audio))
    ):
        raise ValueError(f"audio input gate failed: {sample_id}")
    model_audio = resample_independent_passage(audio[:, 0])
    grid = passage_grid_seconds(float(row["start_seconds"]), float(row["end_seconds"]))
    envelope_native, _ = amplitude_envelope_native(model_audio)
    logmel_native_values = log_mel_native(model_audio)
    native_times = native_audio_frame_centers(
        envelope_native.shape[0], float(row["start_seconds"])
    )
    envelope, envelope_valid = linear_align_no_extrapolation(
        native_times, envelope_native, grid
    )
    logmel, logmel_valid = linear_align_no_extrapolation(
        native_times, logmel_native_values, grid
    )
    wav_values, wav_valid = wav.extract(model_audio, float(row["start_seconds"]), grid)
    neural_power, read_start, read_meta = reader.power(row)
    edge_seconds = reader.edge / reader.fs
    neural_all_lag_valid = (
        (grid >= float(row["start_seconds"]) + edge_seconds)
        & (grid + float(LAGS[-1]) <= float(row["end_seconds"]) - edge_seconds)
    )
    common = envelope_valid & logmel_valid & wav_valid & neural_all_lag_valid
    if np.count_nonzero(common) < 3:
        raise ValueError(f"insufficient common frames: {sample_id}")
    times = grid[common].astype(np.float64)
    arrays = {
        "times.npy": times,
        "envelope.npy": envelope[common].astype(np.float32),
        "logmel.npy": logmel[common].astype(np.float32),
        "wav2vec2.npy": wav_values[:, common, :].astype(np.float32),
        "neural_power_native.npy": neural_power,
    }
    expected_shapes = {
        "envelope.npy": (times.size, 1),
        "logmel.npy": (times.size, 80),
        "wav2vec2.npy": (13, times.size, 768),
        "neural_power_native.npy": (read_meta["read_sample_count"], 36, 6),
    }
    for name, array in arrays.items():
        if array.dtype.hasobject or not np.all(np.isfinite(array)):
            raise ValueError(f"invalid tensor {sample_id}/{name}")
        if name in expected_shapes and array.shape != expected_shapes[name]:
            raise ValueError(f"shape drift {sample_id}/{name}: {array.shape}")
        _atomic_npy(partial_root / name, array)
    metadata = {
        "sample_id": sample_id,
        "stimulus_id": row["stimulus_id"],
        "block_id": row["block_id"],
        "split": row["split"],
        "audio_file": row["audio_file"],
        "start_seconds": row["start_seconds"],
        "end_seconds": row["end_seconds"],
        "source_audio_frames": audio.shape[0],
        "resampled_audio_frames": int(model_audio.size),
        "common_frame_count": int(times.size),
        "first_common_time": float(times[0]),
        "last_common_time": float(times[-1]),
        "read_start_sample": read_start,
        "sampling_rate_hz": reader.fs,
        "read": read_meta,
        "tensor_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "all_finite": True,
    }
    _atomic_json(partial_root / "metadata.json", metadata)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_root, final_root)
    return metadata


@dataclass(frozen=True)
class TargetTransform:
    epsilon: np.ndarray
    center: np.ndarray
    scale: np.ndarray


def _fit_target_transform(rows: list[dict[str, Any]], output_root: Path) -> TargetTransform:
    populations: list[np.ndarray] = []
    for row in rows:
        if row["split"] != "train":
            continue
        root = output_root / "passages" / str(row["sample_id"])
        meta = _load_json(root / "metadata.json")
        times = np.load(root / "times.npy", allow_pickle=False)
        ticks = np.unique(
            np.rint((times[:, None] + LAGS[None, :]) * 100.0).astype(np.int64)
        )
        exact_times = ticks.astype(np.float64) / 100.0
        power = np.load(root / "neural_power_native.npy", mmap_mode="r", allow_pickle=False)
        populations.append(
            _interpolate_power(
                power,
                int(meta["read_start_sample"]),
                int(meta["sampling_rate_hz"]),
                exact_times,
            )
        )
    population = np.concatenate(populations, axis=0)
    if population.shape[1:] != (36, 6) or not np.all(np.isfinite(population)):
        raise ValueError("target fit population gate failed")
    positive = population > 0
    epsilon = np.empty((36, 6), dtype=np.float64)
    for channel in range(36):
        for band in range(6):
            values = population[:, channel, band][positive[:, channel, band]]
            if values.size == 0:
                raise ValueError("target transform has no positive train power")
            epsilon[channel, band] = max(1e-30, 1e-6 * float(np.median(values)))
    logged = np.log(np.maximum(population, 0.0) + epsilon)
    center = np.mean(logged, axis=0)
    scale = np.std(logged, axis=0, ddof=0)
    if (
        not np.all(np.isfinite(epsilon))
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= MINIMUM_SCALE)
    ):
        raise ValueError("target transform parameters are invalid")
    return TargetTransform(epsilon=epsilon, center=center, scale=scale)


def _write_targets(
    rows: list[dict[str, Any]], output_root: Path, transform: TargetTransform
) -> None:
    transform_path = output_root / "target_transform.npz"
    if not transform_path.exists():
        partial = transform_path.with_name(transform_path.name + f".partial-{os.getpid()}")
        with partial.open("xb") as handle:
            np.savez(
                handle,
                epsilon=transform.epsilon,
                center=transform.center,
                scale=transform.scale,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, transform_path)
    for row in rows:
        root = output_root / "passages" / str(row["sample_id"])
        target_path = root / "targets.npy"
        if target_path.exists():
            continue
        meta = _load_json(root / "metadata.json")
        times = np.load(root / "times.npy", allow_pickle=False)
        power = np.load(root / "neural_power_native.npy", mmap_mode="r", allow_pickle=False)
        targets = np.empty((times.size, 11, 36), dtype=np.float32)
        for lag_index, lag in enumerate(LAGS):
            raw = _interpolate_power(
                power,
                int(meta["read_start_sample"]),
                int(meta["sampling_rate_hz"]),
                times + lag,
            )
            transformed = (
                np.log(np.maximum(raw, 0.0) + transform.epsilon) - transform.center
            ) / transform.scale
            targets[:, lag_index, :] = np.mean(transformed, axis=2).astype(np.float32)
        if targets.shape != (times.size, 11, 36) or not np.all(np.isfinite(targets)):
            raise ValueError(f"target tensor gate failed: {row['sample_id']}")
        _atomic_npy(target_path, targets)


def _standardize_train(
    train: np.ndarray, values: list[np.ndarray]
) -> tuple[list[np.ndarray], dict[str, Any]]:
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0, ddof=1)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= MINIMUM_SCALE):
        raise ValueError("feature standardization gate failed")
    transformed = [((value - center) / scale).astype(np.float64) for value in values]
    return transformed, {"center": center, "scale": scale}


def _feature_matrices(
    variant: str, rows: list[dict[str, Any]], output_root: Path
) -> tuple[list[np.ndarray], dict[str, Any]]:
    raw: list[np.ndarray] = []
    for row in rows:
        root = output_root / "passages" / str(row["sample_id"])
        if variant == "amplitude_envelope":
            raw.append(np.asarray(np.load(root / "envelope.npy", allow_pickle=False), dtype=np.float64))
        elif variant == "log_mel_pca20":
            raw.append(np.asarray(np.load(root / "logmel.npy", allow_pickle=False), dtype=np.float64))
        else:
            layer = int(variant.split("_")[-1])
            array = np.load(root / "wav2vec2.npy", mmap_mode="r", allow_pickle=False)
            raw.append(np.asarray(array[layer], dtype=np.float64))
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    train_raw = np.concatenate([raw[index] for index in train_indices], axis=0)
    if variant != "log_mel_pca20":
        transformed, parameters = _standardize_train(train_raw, raw)
        return transformed, {"standardization": parameters}

    pre_pca, pre_parameters = _standardize_train(train_raw, raw)
    train_pre = np.concatenate([pre_pca[index] for index in train_indices], axis=0)
    _, singular, vt = np.linalg.svd(train_pre, full_matrices=False)
    tolerance = max(train_pre.shape) * np.finfo(RANK_TOLERANCE_DTYPE).eps * singular[0]
    if singular.size < 20 or not np.all(np.isfinite(singular[:20])) or singular[19] <= tolerance:
        raise ValueError("log-mel fixed PCA20 rank gate failed")
    components = vt[:20].copy()
    for index in range(20):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1
    scores = [value @ components.T for value in pre_pca]
    train_scores = np.concatenate([scores[index] for index in train_indices], axis=0)
    transformed, score_parameters = _standardize_train(train_scores, scores)
    return transformed, {
        "pre_pca_standardization": pre_parameters,
        "components": components,
        "singular_values_first20": singular[:20],
        "rank_tolerance": float(tolerance),
        "score_standardization": score_parameters,
    }


def _targets(rows: list[dict[str, Any]], output_root: Path) -> list[np.ndarray]:
    values = [
        np.asarray(
            np.load(
                output_root / "passages" / str(row["sample_id"]) / "targets.npy",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        for row in rows
    ]
    if any(value.ndim != 3 or value.shape[1:] != (11, 36) for value in values):
        raise ValueError("target shape gate failed")
    return values


def _passage_pearson(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] < 3:
        return np.full((11, 36), np.nan)
    px = prediction - np.mean(prediction, axis=0)
    ty = target - np.mean(target, axis=0)
    numerator = np.sum(px * ty, axis=0)
    denominator = np.sqrt(np.sum(px * px, axis=0) * np.sum(ty * ty, axis=0))
    result = np.full((11, 36), np.nan)
    valid = denominator > 1e-12
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _passage_r2(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = np.sum((target - prediction) ** 2, axis=0)
    centered = target - np.mean(target, axis=0)
    total = np.sum(centered**2, axis=0)
    result = np.full((11, 36), np.nan)
    valid = total > 1e-12
    result[valid] = 1.0 - residual[valid] / total[valid]
    return result


def _aggregate_pearson(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values)
    valid = np.all(np.isfinite(stack), axis=0)
    output = np.full(stack.shape[1:], np.nan)
    clipped = np.clip(stack[:, valid], -1.0 + 1e-12, 1.0 - 1e-12)
    output[valid] = np.tanh(np.mean(np.arctanh(clipped), axis=0))
    return output


def _aggregate_mean(values: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(values)
    valid = np.all(np.isfinite(stack), axis=0)
    output = np.full(stack.shape[1:], np.nan)
    output[valid] = np.mean(stack[:, valid], axis=0)
    return output


def _svd_parts(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("ridge input contains non-finite values")
    u, singular, vt = np.linalg.svd(x, full_matrices=False)
    return singular, vt, u.T @ y


def _ridge_predict(
    x: np.ndarray,
    singular: np.ndarray,
    vt: np.ndarray,
    uty: np.ndarray,
    alpha: float,
) -> np.ndarray:
    beta = vt.T @ ((singular / (singular * singular + alpha))[:, None] * uty)
    return x @ beta


def _normalized_phase_map(donor: np.ndarray, target_length: int) -> np.ndarray:
    if donor.shape[0] < 2 or target_length < 2:
        raise ValueError("normalized-time mapping requires at least two frames")
    position = np.linspace(0.0, donor.shape[0] - 1.0, target_length)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, donor.shape[0] - 1)
    weight = (position - lower).reshape(-1, 1, 1)
    return donor[lower] * (1.0 - weight) + donor[upper] * weight


def _run_variant(
    variant: str,
    rows: list[dict[str, Any]],
    features: list[np.ndarray],
    targets: list[np.ndarray],
    derangements: list[dict[str, Any]],
) -> dict[str, Any]:
    partitions = {
        name: [index for index, row in enumerate(rows) if row["split"] == name]
        for name in EXPECTED_SPLITS
    }
    train_idx = partitions["train"]
    validation_idx = partitions["validation"]
    test_idx = partitions["test"]
    x_train = np.concatenate([features[index] for index in train_idx], axis=0)
    y_train = np.concatenate([targets[index] for index in train_idx], axis=0).reshape(-1, 396)
    x_validation = np.concatenate([features[index] for index in validation_idx], axis=0)
    validation_targets = [targets[index] for index in validation_idx]
    validation_bounds = np.cumsum([0] + [value.shape[0] for value in validation_targets])
    singular, vt, uty = _svd_parts(x_train, y_train)
    validation_scores = np.full((ALPHAS.size, 11, 36), np.nan)
    for alpha_index, alpha in enumerate(ALPHAS):
        prediction = _ridge_predict(x_validation, singular, vt, uty, float(alpha)).reshape(-1, 11, 36)
        passage_scores = [
            _passage_pearson(
                prediction[validation_bounds[index] : validation_bounds[index + 1]],
                validation_targets[index],
            )
            for index in range(len(validation_targets))
        ]
        validation_scores[alpha_index] = _aggregate_pearson(passage_scores)
    if np.any(~np.any(np.isfinite(validation_scores), axis=0)):
        raise ValueError(f"all validation alphas not estimable: {variant}")
    safe_scores = np.where(np.isfinite(validation_scores), validation_scores, -np.inf)
    best_alpha_index = np.argmax(safe_scores, axis=0)

    tv_idx = train_idx + validation_idx
    x_tv = np.concatenate([features[index] for index in tv_idx], axis=0)
    y_tv = np.concatenate([targets[index] for index in tv_idx], axis=0).reshape(-1, 396)
    x_test = np.concatenate([features[index] for index in test_idx], axis=0)
    test_targets = [targets[index] for index in test_idx]
    test_bounds = np.cumsum([0] + [value.shape[0] for value in test_targets])
    singular_tv, vt_tv, uty_tv = _svd_parts(x_tv, y_tv)
    prediction_flat = np.empty((x_test.shape[0], 396), dtype=np.float64)
    selected_flat = best_alpha_index.reshape(-1)
    for alpha_index, alpha in enumerate(ALPHAS):
        columns = np.flatnonzero(selected_flat == alpha_index)
        if columns.size == 0:
            continue
        prediction_flat[:, columns] = _ridge_predict(
            x_test, singular_tv, vt_tv, uty_tv[:, columns], float(alpha)
        )
    prediction = prediction_flat.reshape(-1, 11, 36)
    predictions = [
        prediction[test_bounds[index] : test_bounds[index + 1]]
        for index in range(len(test_targets))
    ]
    observed_pearson = _aggregate_pearson(
        [_passage_pearson(pred, target) for pred, target in zip(predictions, test_targets, strict=True)]
    )
    observed_r2 = _aggregate_mean(
        [_passage_r2(pred, target) for pred, target in zip(predictions, test_targets, strict=True)]
    )
    test_rows = [rows[index] for index in test_idx]
    ordered_ids = [str(row["sample_id"]) for row in test_rows]
    null_pearson = np.empty((len(derangements), 11, 36), dtype=np.float64)
    for permutation_index, item in enumerate(derangements):
        if item["target_sample_ids"] != ordered_ids:
            raise ValueError("derangement target order differs from test passage order")
        donor_lookup = {sample_id: index for index, sample_id in enumerate(ordered_ids)}
        mapped_predictions = [
            _normalized_phase_map(
                predictions[donor_lookup[str(donor_id)]], test_targets[target_index].shape[0]
            )
            for target_index, donor_id in enumerate(item["donor_sample_ids"])
        ]
        null_pearson[permutation_index] = _aggregate_pearson(
            [
                _passage_pearson(pred, target)
                for pred, target in zip(mapped_predictions, test_targets, strict=True)
            ]
        )
    return {
        "variant": variant,
        "observed_pearson": observed_pearson,
        "observed_r2": observed_r2,
        "null_pearson": null_pearson,
        "selected_alpha": ALPHAS[best_alpha_index],
        "validation_pearson": np.take_along_axis(
            validation_scores, best_alpha_index[None, :, :], axis=0
        )[0],
        "test_predictions_shape": list(prediction.shape),
    }


def _summary(
    variant_results: list[dict[str, Any]], derangements: list[dict[str, Any]]
) -> dict[str, Any]:
    layer_names = list(EXPECTED_LAYER_KEYS)
    records: list[dict[str, Any]] = []
    for result in variant_results:
        pearson = cast(np.ndarray, result["observed_pearson"])
        r2 = cast(np.ndarray, result["observed_r2"])
        median_by_lag = np.nanmedian(pearson, axis=1)
        best_lag_index = int(np.nanargmax(median_by_lag))
        electrode_values = pearson[best_lag_index]
        r2_values = r2[best_lag_index]
        records.append(
            {
                "variant": result["variant"],
                "best_lag_seconds_descriptive_test_only": float(LAGS[best_lag_index]),
                "median_test_pearson_r": float(np.nanmedian(electrode_values)),
                "test_pearson_r_iqr": [
                    float(np.nanpercentile(electrode_values, 25)),
                    float(np.nanpercentile(electrode_values, 75)),
                ],
                "median_test_r2": float(np.nanmedian(r2_values)),
                "estimable_electrodes": int(np.count_nonzero(np.isfinite(electrode_values))),
            }
        )
    wav_results = variant_results[2:]
    acoustic_results = variant_results[:2]
    wav_observed = np.stack([cast(np.ndarray, item["observed_pearson"]) for item in wav_results])
    wav_null = np.stack([cast(np.ndarray, item["null_pearson"]) for item in wav_results], axis=1)
    acoustic_observed = np.stack(
        [cast(np.ndarray, item["observed_pearson"]) for item in acoustic_results]
    )
    acoustic_null = np.stack(
        [cast(np.ndarray, item["null_pearson"]) for item in acoustic_results], axis=1
    )

    def family(observed: np.ndarray, null: np.ndarray) -> dict[str, Any]:
        effective = np.isfinite(observed) & np.all(np.isfinite(null), axis=0)
        if not np.any(effective):
            raise ValueError("null family effective cell intersection is empty")
        observed_max = float(np.max(observed[effective]))
        null_max = np.asarray(
            [np.max(null[index][effective]) for index in range(null.shape[0])],
            dtype=np.float64,
        )
        exceedance = int(np.count_nonzero(null_max >= observed_max))
        return {
            "declared_cell_count": int(observed.size),
            "effective_cell_count": int(np.count_nonzero(effective)),
            "excluded_cell_count": int(observed.size - np.count_nonzero(effective)),
            "observed_max_pearson_r": observed_max,
            "null_max_pearson_r": null_max.tolist(),
            "null_max_percentile_of_observed": float(
                100.0 * np.count_nonzero(null_max < observed_max) / null_max.size
            ),
            "exceedance_count_max_null_ge_observed": exceedance,
            "mechanical_smoke_p": float((1 + exceedance) / 21),
            "stable_significance_claimed": False,
        }

    best_wav = max(records[2:], key=lambda item: item["median_test_pearson_r"])
    return {
        "variant_summaries": records,
        "wav2vec2_layer_names": layer_names,
        "best_wav2vec2_layer_lag_descriptive_test_only": best_wav,
        "wav2vec2_family_null": family(wav_observed, wav_null),
        "acoustic_family_null": family(acoustic_observed, acoustic_null),
        "derangements": derangements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the minimal authorized SD012 ses-02 G4.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--neural-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--synthetic-smoke-only", action="store_true")
    parser.add_argument("--authorized-minimal-g4", action="store_true")
    args = parser.parse_args()
    if not args.authorized_minimal_g4:
        raise PermissionError("explicit --authorized-minimal-g4 is required")
    output_root = args.output_root.resolve()
    project_outputs = Path("/home/fanyu/auditory_simulation_m6a/outputs").resolve()
    output_root.relative_to(project_outputs)
    if args.synthetic_smoke_only:
        rng = np.random.Generator(np.random.PCG64(20260813))
        x = rng.normal(size=(128, 8))
        y = rng.normal(size=(128, 396))
        singular, vt, uty = _svd_parts(x, y)
        prediction = _ridge_predict(x, singular, vt, uty, 1.0)
        if prediction.shape != y.shape or not np.all(np.isfinite(prediction)):
            raise RuntimeError("synthetic ridge smoke failed")
        print(json.dumps({"status": "PASS", "shape": list(prediction.shape)}))
        return 0

    started = time.perf_counter()
    rows = _load_scope(args.split_csv)
    recording = _select_recording(_load_json(args.neural_report))
    dataset_root = args.dataset_root.resolve()
    model_dir = args.model_dir.resolve()
    if output_root.exists() and not (output_root / "run_identity.json").is_file():
        raise RuntimeError("output root exists without a valid run identity")
    output_root.mkdir(parents=True, exist_ok=True)
    identity_path = output_root / "run_identity.json"
    if not identity_path.exists():
        _atomic_json(
            identity_path,
            {
                "task_id": "M6A-PUBLIC-001",
                "run": "G4_MINIMAL_SD012_SES02_PRELIMINARY",
                "recording_id": RECORDING_ID,
                "passage_counts": EXPECTED_SPLITS,
                "integrity_policy": "NON_HASH_AUDIT",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    wav = FrozenWav2Vec2(model_dir)
    reader = BoundedNeuralReader(dataset_root / recording["edf_file"], recording)
    passage_metadata: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows, start=1):
            passage_metadata.append(
                _checkpoint_passage(row, output_root, dataset_root, reader, wav)
            )
            print(
                json.dumps(
                    {
                        "stage": "passage_checkpoint",
                        "complete": index,
                        "total": 40,
                        "sample_id": row["sample_id"],
                    }
                ),
                flush=True,
            )
    finally:
        reader.close()
    transform = _fit_target_transform(rows, output_root)
    _write_targets(rows, output_root, transform)
    target_values = _targets(rows, output_root)
    test_rows = [row for row in rows if row["split"] == "test"]
    _, derangements = _rank_test_derangements(test_rows, required_count=20)
    variants = ["amplitude_envelope", "log_mel_pca20"] + [
        f"wav2vec2_{index:02d}" for index in range(13)
    ]
    results: list[dict[str, Any]] = []
    result_root = output_root / "variant_results"
    result_root.mkdir(exist_ok=True)
    for index, variant in enumerate(variants, start=1):
        feature_values, parameters = _feature_matrices(variant, rows, output_root)
        result = _run_variant(
            variant, rows, feature_values, target_values, derangements
        )
        results.append(result)
        path = result_root / f"{variant}.npz"
        if not path.exists():
            partial = path.with_name(path.name + f".partial-{os.getpid()}")
            with partial.open("xb") as handle:
                np.savez(
                    handle,
                    observed_pearson=result["observed_pearson"],
                    observed_r2=result["observed_r2"],
                    null_pearson=result["null_pearson"],
                    selected_alpha=result["selected_alpha"],
                    validation_pearson=result["validation_pearson"],
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
        print(
            json.dumps(
                {"stage": "ridge_variant", "complete": index, "total": 15, "variant": variant}
            ),
            flush=True,
        )
        del feature_values, parameters
    summary = _summary(results, derangements)
    report = {
        "status": "G4_MINIMAL_SD012_SES02_PRELIMINARY_COMPLETE",
        "task_id": "M6A-PUBLIC-001",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_policy": "NON_HASH_AUDIT",
        "scope": {
            "participant_id": PARTICIPANT_ID,
            "session_id": SESSION_ID,
            "recording_id": RECORDING_ID,
            "passage_counts": EXPECTED_SPLITS,
            "channel_count": 36,
            "lags_seconds": LAGS.tolist(),
            "stimulus_cross_split_count": 0,
        },
        "model_runtime": wav.runtime,
        "passage_metadata": passage_metadata,
        "necessary_gates": {
            "input_readable_and_scope_exact": True,
            "tensor_shapes_and_finite": True,
            "split_and_stimulus_no_leakage": True,
        },
        "summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
        "scientific_boundary": {
            "single_subject_single_recording_preliminary": True,
            "stable_significance_claimed": False,
            "region_summary": "NOT_ESTIMABLE",
            "subject_heldout_generalization_claimed": False,
            "speaker_heldout_generalization_claimed": False,
            "cross_language_generalization_claimed": False,
        },
    }
    _atomic_json(output_root / "g4_preliminary_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_root": str(output_root),
                "elapsed_seconds": report["elapsed_seconds"],
                "best_wav2vec2": summary["best_wav2vec2_layer_lag_descriptive_test_only"],
                "wav_null": summary["wav2vec2_family_null"],
                "acoustic_null": summary["acoustic_family_null"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
