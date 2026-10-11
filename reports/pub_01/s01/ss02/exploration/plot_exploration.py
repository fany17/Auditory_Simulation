#!/usr/bin/env python3
"""Server-side descriptive exploration for PUB-01-S01-SS02.

Only existing lightweight indexes and a deterministic small sample are read.
The output contains static figures and text, never raw signal/audio arrays.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

SPARR_FS = 64.0
PUBLIC_PYTHON = "/home/fanyu/.conda/envs/auditory_m6a_public_001/bin/python"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "eligible"}


def first(mapping: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = str(mapping.get(key, "")).strip()
        if value:
            return value
    return default


def parse_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
        if isinstance(decoded, list):
            return [str(item) for item in decoded]
    except (json.JSONDecodeError, TypeError):
        pass
    return [item.strip(" '\"") for item in text.strip("[]").split(",") if item.strip(" '\"")]


def access(row: dict[str, str], provenance: dict[str, dict[str, str]]) -> str:
    value = first(row, ["source_access", "raw_access"])
    if value:
        return value.upper()
    source = row.get("source_recording", "")
    if source in provenance:
        return first(provenance[source], ["source_access", "raw_access"], "UNKNOWN").upper()
    for key, item in provenance.items():
        if Path(key).name == Path(source).name:
            return first(item, ["source_access", "raw_access"], "UNKNOWN").upper()
    return "UNKNOWN"


def labels(row: dict[str, Any], n: int) -> list[str]:
    value = first(
        row,
        ["channel_names", "ordered_channel_names", "publisher_channel_names", "eeg_channel_names"],
    )
    result = parse_list(value)
    result.extend([f"Ch{i + 1:02d}" for i in range(len(result), n)])
    return result[:n]


def orient(array: np.ndarray, hint: int = 64) -> np.ndarray:
    if array.ndim == 1:
        return array[None, :]
    if array.ndim != 2:
        raise ValueError(f"unexpected EEG shape {array.shape}")
    if array.shape[0] <= hint and array.shape[0] <= array.shape[1]:
        return array
    if array.shape[1] <= hint:
        return array.T
    raise ValueError(f"cannot infer EEG channel axis from {array.shape}")


def load_sparr(row: dict[str, Any]) -> tuple[np.ndarray, float, list[str], str]:
    array = np.load(Path(row["eeg_path"]), mmap_mode="r", allow_pickle=False)
    data = orient(np.asarray(array))
    return data, SPARR_FS, labels(row.get("_provenance", {}), data.shape[0]), "µV (publisher preprocessed NPY contract)"


def load_envelope(path: str | Path) -> tuple[np.ndarray, float]:
    array = np.asarray(np.load(Path(path), mmap_mode="r", allow_pickle=False))
    if array.ndim == 1:
        values = array.astype(float)
    elif array.ndim == 2:
        values = np.nanmean(array.astype(float), axis=1 if array.shape[0] >= array.shape[1] else 0)
    else:
        values = array.reshape(-1).astype(float)
    return values, SPARR_FS


def load_sparr_audio(root: Path, story: str) -> tuple[np.ndarray, float, str] | None:
    paths = sorted(root.rglob(f"{story}.npz.gz")) or sorted(root.rglob(f"{story}.npz"))
    if not paths:
        return None
    path = paths[0]
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            with np.load(handle, allow_pickle=False) as archive:
                keys = list(archive.files)
                data_key = next(
                    (key for key in ("stimulus_data", "audio", "waveform", "wav", "data") if key in keys),
                    None,
                )
                if data_key is None:
                    numeric = []
                    for key in keys:
                        value = np.asarray(archive[key])
                        if np.issubdtype(value.dtype, np.number) and value.ndim <= 2:
                            numeric.append((value.size, key))
                    if not numeric:
                        return None
                    data_key = max(numeric)[1]
                audio = np.asarray(archive[data_key]).squeeze()
                if audio.ndim == 2:
                    audio = audio[:, 0] if audio.shape[0] >= audio.shape[1] else audio[0, :]
                rate = None
                for key in ("fs", "sampling_rate", "sample_rate", "audio_rate_hz", "stimulus_sampling_rate_hz"):
                    if key in keys and np.asarray(archive[key]).size == 1:
                        rate = fnum(np.asarray(archive[key]).squeeze().item())
                        break
                if rate is None or not np.isfinite(rate) or rate <= 0:
                    return None
                return np.asarray(audio, dtype=float).reshape(-1), float(rate), str(path)
    except (OSError, ValueError, KeyError, EOFError):
        return None


def read_wav(path: str | Path) -> tuple[np.ndarray, float]:
    rate, raw = wavfile.read(str(path))
    array = np.asarray(raw)
    if array.ndim == 2:
        array = array.mean(axis=1)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        array = array.astype(float) / max(abs(info.min), info.max)
    return array.astype(float).reshape(-1), float(rate)


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values)
    scale = np.nanpercentile(np.abs(values - center), 99.5)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(values)
    return (values - center) / (scale if np.isfinite(scale) and scale > 0 else 1.0)


def load_ds(
    segment: dict[str, str],
    recording: dict[str, str],
    channel_rows: list[dict[str, str]],
    pre_s: float = 0.5,
    duration_s: float = 12.0,
) -> dict[str, Any]:
    import mne

    raw = mne.io.read_raw_edf(recording["server_raw_path"], preload=False, verbose="ERROR")
    try:
        eligible = [
            item["channel"]
            for item in channel_rows
            if truthy(item.get("eligible", "")) and item.get("channel", "") in raw.ch_names
        ][:8]
        if not eligible:
            raise ValueError(f"no eligible channels for {recording['recording']}")
        fs = float(raw.info["sfreq"])
        event_s = fnum(segment.get("offset_s"))
        event_sample = int(round(event_s * fs))
        start = max(0, event_sample - int(round(pre_s * fs)))
        stop = min(int(raw.n_times), start + int(round((pre_s + duration_s) * fs)))
        data = raw.get_data(picks=eligible, start=start, stop=stop)
        original_units = getattr(raw, "_orig_units", {})
        unit_labels = [str(original_units.get(name, "")) for name in eligible]
        verified = bool(unit_labels) and all(
            any(token in unit.lower() for token in ("uv", "µv", "microvolt")) for unit in unit_labels
        )
        if verified:
            data = data * 1e6
            unit = "µV (EDF original µV; MNE SI V converted ×1e6)"
        else:
            unit = "MNE data unit (conversion to µV not verified)"
        time = (np.arange(data.shape[1]) + start - event_sample) / fs
        return {
            "data": np.asarray(data, dtype=float),
            "time": time,
            "fs": fs,
            "labels": eligible,
            "unit": unit,
            "unit_verified": verified,
            "recording": recording["recording"],
            "raw_path": recording["server_raw_path"],
        }
    finally:
        try:
            raw.close()
        except Exception:
            pass


def stacked(axis: plt.Axes, time: np.ndarray, data: np.ndarray, names: list[str], title: str, unit: str) -> None:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    values = values - np.nanmedian(values, axis=1, keepdims=True)
    scale = np.nanmedian(np.nanpercentile(np.abs(values), 98, axis=1)) * 3
    scale = float(scale) if np.isfinite(scale) and scale > 0 else 1.0
    offsets = np.arange(values.shape[0] - 1, -1, -1) * scale
    for row, offset in zip(values, offsets):
        axis.plot(time, row + offset, linewidth=0.65)
    axis.set_yticks(offsets)
    axis.set_yticklabels(names[: len(offsets)], fontsize=7)
    axis.set_xlabel("time (s)")
    axis.set_ylabel(f"{unit} + display offset")
    axis.set_title(title, fontsize=9, loc="left")


def psd(axis: plt.Axes, data: np.ndarray, fs: float, title: str) -> None:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    nperseg = min(2048, values.shape[1])
    freq, power = signal.welch(
        values - np.nanmean(values, axis=1, keepdims=True),
        fs=fs,
        nperseg=nperseg,
        axis=1,
        scaling="density",
    )
    mask = freq <= fs / 2 + 1e-9
    mean_power = np.nanmean(power, axis=0)
    axis.plot(freq[mask], 10 * np.log10(np.maximum(mean_power[mask], 1e-20)))
    axis.set_xlim(0, fs / 2)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("mean PSD (dB/Hz)")
    axis.set_title(f"{title}; Nyquist={fs / 2:g} Hz", fontsize=9, loc="left")


def display_envelope(audio: np.ndarray, fs: float) -> np.ndarray:
    envelope = np.abs(signal.hilbert(audio))
    cutoff = min(20.0, fs * 0.2)
    if cutoff > 1:
        envelope = signal.sosfiltfilt(signal.butter(4, cutoff, btype="lowpass", fs=fs, output="sos"), envelope)
    return envelope


def audio_axis(axis: plt.Axes, time: np.ndarray, values: np.ndarray, ylabel: str) -> None:
    axis.plot(time, normalize(values), color="#5b6770", linewidth=0.65)
    axis.set_ylabel(ylabel)


def stats_for(
    sparr: list[dict[str, str]],
    recordings: list[dict[str, str]],
    segments: list[dict[str, str]],
    audio: list[dict[str, str]],
) -> dict[str, Any]:
    sparr_durations = np.array([inum(row.get("samples")) / SPARR_FS for row in sparr], dtype=float)
    edf_durations = np.array([fnum(row.get("duration_s")) for row in recordings], dtype=float)
    return {
        "sparr_rows": len(sparr),
        "sparr_subjects": len({row.get("subject", "") for row in sparr}),
        "sparr_sessions": len({(row.get("subject", ""), row.get("session", "")) for row in sparr}),
        "sparr_stories": len({row.get("story", "") for row in sparr}),
        "sparr_total_s": float(np.nansum(sparr_durations)),
        "sparr_min_s": float(np.nanmin(sparr_durations)),
        "sparr_max_s": float(np.nanmax(sparr_durations)),
        "ds_recordings": len(recordings),
        "ds_subjects": len({row.get("recording", "").split("_")[0] for row in recordings}),
        "ds_stories": len({row.get("story", "") for row in segments if row.get("story", "")}),
        "ds_segments": len(segments),
        "ds_audio_files": len({row.get("path", "") for row in audio}),
        "ds_total_s": float(np.nansum(edf_durations)),
        "ds_min_s": float(np.nanmin(edf_durations)),
        "ds_max_s": float(np.nanmax(edf_durations)),
        "ds_rates": sorted({fnum(row.get("sample_rate_hz")) for row in recordings}),
        "ds_channels": [(row.get("recording", ""), inum(row.get("channels")), inum(row.get("eligible_channels"))) for row in recordings],
    }


def overview_plot(stats: dict[str, Any], sparr: list[dict[str, str]], recordings: list[dict[str, str]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].bar(
        ["Sparr\nEEG rows", "ds\nEDF recordings", "ds\neligible segments", "ds\naudio files"],
        [stats["sparr_rows"], stats["ds_recordings"], stats["ds_segments"], stats["ds_audio_files"]],
        color=["#2a6fbb", "#e07a5f", "#81b29a", "#f2cc8f"],
    )
    axes[0].set_title("Objects by native index unit")
    axes[0].set_ylabel("count")
    axes[1].bar(
        ["Sparr\nsubjects", "ds\nsubjects", "Sparr\nstories", "ds\nstories"],
        [stats["sparr_subjects"], stats["ds_subjects"], stats["sparr_stories"], stats["ds_stories"]],
        color=["#2a6fbb", "#e07a5f", "#2a6fbb", "#e07a5f"],
    )
    axes[1].set_title("Subject and story coverage")
    axes[1].set_ylabel("count")
    duration_sets = [
        [inum(row.get("samples")) / SPARR_FS for row in sparr],
        [fnum(row.get("duration_s")) for row in recordings],
    ]
    axes[2].boxplot(duration_sets, vert=False, tick_labels=["Sparr derivative", "ds EDF"], showfliers=True)
    axes[2].set_title("Indexed duration distributions; seconds")
    axes[2].set_xlabel("duration (s)")
    axes[2].text(0.98, 0.08, f"n={len(duration_sets[0])} / {len(duration_sets[1])}", transform=axes[2].transAxes, ha="right", fontsize=8)
    all_channels = [item[1] for item in stats["ds_channels"]]
    eligible_channels = [item[2] for item in stats["ds_channels"]]
    order = np.arange(1, len(all_channels) + 1)
    axes[3].plot(order, all_channels, "o-", color="#e07a5f", label="all EDF channels")
    axes[3].plot(order, eligible_channels, "s-", color="#81b29a", label="eligible EEG channels")
    axes[3].axhline(64, linestyle="--", color="#2a6fbb", label="Sparr derivative channels=64")
    axes[3].set_title("Channel counts from recording QC")
    axes[3].set_xlabel("recording order")
    axes[3].set_ylabel("channels")
    axes[3].legend(fontsize=8)
    figure.suptitle("PUB-01-S01-SS02 descriptive overview", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def signal_plot(selected: list[dict[str, Any]], dataset: str, output: Path) -> None:
    figure, axes = plt.subplots(len(selected), 2, figsize=(15, max(4, len(selected) * 3.2)))
    axes = np.atleast_2d(axes)
    for index, item in enumerate(selected):
        time = item["time"]
        mask = time <= 12
        if dataset == "ds004703":
            mask &= time >= 0
        data = item["data"][:8, mask]
        plot_time = time[mask]
        access_label = item.get("source_access", "PUBLIC")
        stacked(
            axes[index, 0],
            plot_time,
            data,
            item["labels"][:8],
            f"{item['subject']} | {item.get('story', '')} | {item['role']} | access={access_label}",
            item["unit"].split(" (")[0],
        )
        psd(axes[index, 1], data, item["fs"], f"{item['subject']} | {item['role']}")
    figure.suptitle(f"{dataset}: deterministic multi-channel traces and PSD", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def sparr_overlay_plot(selected: list[dict[str, Any]], root: Path, output: Path) -> list[str]:
    figure, axes = plt.subplots(len(selected), 3, figsize=(15, max(5, len(selected) * 4.5)))
    axes = np.atleast_2d(axes)
    notes: list[str] = []
    for index, item in enumerate(selected):
        envelope, env_fs = load_envelope(item["envelope_path"])
        audio_result = load_sparr_audio(root, item["story"])
        audio = audio_fs = audio_path = None
        if audio_result:
            audio, audio_fs, audio_path = audio_result
        duration = min(12, envelope.size / env_fs)
        if audio is not None:
            duration = min(duration, audio.size / audio_fs)
            audio = audio[: int(duration * audio_fs)]
            audio_axis(axes[index, 0], np.arange(audio.size) / audio_fs, audio, "raw audio\n(normalized)")
            notes.append(f"{item['sample_key']}: raw stimulus audio read from {audio_path}")
        else:
            axes[index, 0].text(0.5, 0.5, "raw stimulus audio unavailable", ha="center", va="center")
            notes.append(f"{item['sample_key']}: raw stimulus audio unavailable; envelope retained")
        envelope = envelope[: int(duration * env_fs)]
        audio_axis(axes[index, 1], np.arange(envelope.size) / env_fs, envelope, "published\n64-Hz envelope")
        neural_time = item["time"]
        neural_mask = (neural_time >= 0) & (neural_time <= duration)
        neural = np.nanmedian(item["data"][:8, neural_mask], axis=0)
        axes[index, 2].plot(neural_time[neural_mask], neural, color="#2a6fbb", linewidth=0.8)
        axes[index, 2].set_ylabel(item["unit"].split(" (")[0])
        axes[index, 0].set_title(f"{item['subject']} | {item['role']} | audio", fontsize=9, loc="left")
        axes[index, 1].set_title("publisher envelope", fontsize=9, loc="left")
        axes[index, 2].set_title("neural median over plotted channels", fontsize=9, loc="left")
        for axis in axes[index, :]:
            axis.axvline(0, color="#b23a48", linestyle="--", linewidth=0.8)
            axis.set_xlim(0, duration)
            axis.set_xlabel("local time (s)")
            axis.text(0.01, 0.94, "array common start; not a hardware event", transform=axis.transAxes, fontsize=7, color="#b23a48", va="top")
    figure.suptitle("SparrKULee: audio / publisher envelope / neural index", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return notes


def ds_overlay_plot(selected: list[dict[str, Any]], output: Path) -> list[str]:
    figure, axes = plt.subplots(len(selected), 3, figsize=(15, max(5, len(selected) * 4.5)))
    axes = np.atleast_2d(axes)
    notes: list[str] = []
    for index, item in enumerate(selected):
        try:
            audio, audio_fs = read_wav(item["audio_path"])
            duration = min(12, audio.size / audio_fs)
            audio = audio[: int(duration * audio_fs)]
            envelope = display_envelope(audio, audio_fs)
            notes.append(f"{item['sample_key']}: public WAV read; envelope=Hilbert+20-Hz low-pass")
        except (OSError, ValueError, TypeError) as exc:
            audio = np.zeros(1)
            audio_fs = 1
            envelope = audio.copy()
            duration = 1
            notes.append(f"{item['sample_key']}: audio overlay skipped: {exc}")
        audio_axis(axes[index, 0], np.arange(audio.size) / audio_fs, audio, "raw audio\n(normalized)")
        audio_axis(axes[index, 1], np.arange(envelope.size) / audio_fs, envelope, "derived\namplitude envelope")
        neural_mask = (item["time"] >= -0.5) & (item["time"] <= duration)
        neural = np.nanmedian(item["data"][:8, neural_mask], axis=0)
        axes[index, 2].plot(item["time"][neural_mask], neural, color="#2a6fbb", linewidth=0.8)
        axes[index, 2].set_ylabel(item["unit"].split(" (")[0])
        axes[index, 0].set_title(f"{item['subject']} | {item['story']} | raw EDF", fontsize=9, loc="left")
        axes[index, 1].set_title("audio-derived envelope", fontsize=9, loc="left")
        axes[index, 2].set_title("neural median over plotted channels", fontsize=9, loc="left")
        for axis in axes[index, :]:
            axis.axvline(0, color="#b23a48", linestyle="--", linewidth=0.8)
            axis.set_xlim(-0.5, duration)
            axis.set_xlabel("local time (s)")
            axis.text(0.01, 0.94, "metadata event onset; physical sync UNKNOWN", transform=axis.transAxes, fontsize=7, color="#b23a48", va="top")
    figure.suptitle("ds004703: raw audio / derived envelope / original EDF neural index", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return notes


def tail_plots(
    rows: list[dict[str, str]],
    provenance: dict[str, dict[str, str]],
    output_overview: Path,
    output_example: Path,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row.get("source_recording", ""))
    durations = np.array([inum(row.get("samples")) / SPARR_FS for row in rows])
    tails = np.array([inum(row.get("eeg_tail_samples")) / SPARR_FS for row in rows])
    colors = ["#e07a5f" if access(row, provenance) == "RESTRICTED" else "#2a6fbb" for row in rows]
    display_jitter = (np.arange(len(rows), dtype=float) - (len(rows) - 1) / 2.0) * 0.03
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(durations, tails + display_jitter, c=colors, s=42, edgecolor="white", linewidth=0.4)
    axis.axhline(3, color="#555555", linestyle="--", label="3 s advisory marker")
    axis.set_xlabel("indexed common duration (s)")
    axis.set_ylabel("EEG tail beyond common segment (s)")
    axis.set_title("All 26 LARGE_TAIL_REVIEW rows; 3 s is not an admission line")
    axis.legend()
    axis.text(0.02, 0.03, f"n={len(rows)}; blue=public source, orange=restricted source\nvertical jitter is display-only", transform=axis.transAxes, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_overview, bbox_inches="tight")
    plt.close(figure)

    representative = rows[0]
    eeg, fs, _, unit = load_sparr({"eeg_path": representative["eeg_path"], "_provenance": provenance.get(representative.get("source_recording", ""), {})})
    envelope, env_fs = load_envelope(representative["envelope_path"])
    common = min(eeg.shape[1], envelope.size)
    tail_samples = max(0, eeg.shape[1] - common)
    start = max(0, common - int(6 * fs))
    time = (np.arange(start, eeg.shape[1]) - common) / fs
    neural = np.nanmedian(eeg[: min(8, eeg.shape[0]), start:], axis=0)
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(time, neural, color="#2a6fbb", linewidth=0.75)
    axes[0].axvline(0, color="#b23a48", linestyle="--")
    axes[0].set_ylabel(unit.split(" (")[0])
    axes[0].set_title(f"Representative tail: {representative['source_recording']} | {tail_samples} samples = {tail_samples / fs:.2f} s", fontsize=10, loc="left")
    env_start = max(0, common - int(6 * env_fs))
    env_time = (np.arange(env_start, common) - common) / env_fs
    axes[1].plot(env_time, normalize(envelope[env_start:common]), color="#81b29a")
    axes[1].axvline(0, color="#b23a48", linestyle="--")
    axes[1].set_ylabel("published envelope\n(normalized)")
    axes[1].set_xlabel("time relative to common envelope end (s)")
    axes[1].text(0.01, 0.84, "0 s = common envelope end; positive time is EEG-only tail", transform=axes[1].transAxes, fontsize=8, color="#b23a48")
    figure.tight_layout()
    figure.savefig(output_example, bbox_inches="tight")
    plt.close(figure)
    return {
        "n": len(rows),
        "min_s": float(np.nanmin(tails)),
        "median_s": float(np.nanmedian(tails)),
        "max_s": float(np.nanmax(tails)),
        "representative": representative,
        "tail_samples": tail_samples,
    }


def make_manifest(sparr: list[dict[str, Any]], ds: list[dict[str, Any]], output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in sparr:
        rows.append(
            {
                "dataset": "SparrKULee",
                "role": item["role"],
                "sample_key": item["sample_key"],
                "subject": item["subject"],
                "session": item["session"],
                "recording": item["source_recording"],
                "story": item["story"],
                "raw_access": item["source_access"],
                "derivative_read": "YES",
                "restricted_raw_read": "NO" if item["source_access"] == "RESTRICTED" else "NOT_APPLICABLE",
                "sampling_rate_hz": item["fs"],
                "units": item["unit"],
                "window": "0-12 s or available duration",
                "audio_or_envelope": "published envelope; indexed raw stimulus attempted",
                "selection_rule": item["selection_rule"],
            }
        )
    for item in ds:
        rows.append(
            {
                "dataset": "ds004703",
                "role": item["role"],
                "sample_key": item["sample_key"],
                "subject": item["subject"],
                "session": item["session"],
                "recording": item["recording"],
                "story": item["story"],
                "raw_access": "PUBLIC_BUNDLE_WITH_README_LIMITATION",
                "derivative_read": "NO",
                "restricted_raw_read": "NO",
                "sampling_rate_hz": item["fs"],
                "units": item["unit"],
                "window": "-0.5 to 12 s around metadata event onset",
                "audio_or_envelope": "public WAV; derived Hilbert+20-Hz low-pass envelope",
                "selection_rule": item["selection_rule"],
            }
        )
    fields = [
        "dataset", "role", "sample_key", "subject", "session", "recording", "story",
        "raw_access", "derivative_read", "restricted_raw_read", "sampling_rate_hz",
        "units", "window", "audio_or_envelope", "selection_rule",
    ]
    write_csv(output / "sample_manifest.csv", rows, fields)


def write_run_notes(output: Path, project_root: Path, failures: list[str], notes: list[str]) -> None:
    command = f"{PUBLIC_PYTHON} {output / 'plot_exploration.py'} --project-root {project_root} --output {output}"
    lines = [
        "# SS02 exploration sample / parameter / command list",
        "",
        "本目录只包含静态图、轻量索引摘要、Markdown/HTML 和绘图脚本；没有导出原始信号、音频数组或交互式原始载荷。",
        "",
        "## Deterministic selection",
        "",
        "- SparrKULee：在现有 cleaning_index.csv 中按 subject, source_recording 排序，取前三个最先出现且 raw source 为 PUBLIC、tail flag 为 WITHIN_3S_ADVISORY 的不同被试；另取排序后第一个 raw source 为 RESTRICTED 且有公开 EEG derivative 的样本。受限 raw 不读取。",
        "- ds004703：在现有 clean_segment_index.csv 的 eligible=1 行中按 subject, recording, trial 排序，取前三个不同被试；读取原 EDF 的选定通道/小时间窗，不读取全库。",
        "- 尾差：全部 26 个 LARGE_TAIL_REVIEW 行画长度关系；代表尾端图取 source_recording 字典序第一行。3 秒只作为 advisory marker。",
        "",
        "## Parameters",
        "",
        "- SparrKULee 使用发布方 64 Hz 预处理 EEG；PSD 上限为 32 Hz Nyquist。",
        "- ds004703 使用原 EDF 记录采样率；PSD 上限为各样本的 fs/2。",
        "- 多通道图读取最多 8 个通道；时域窗口最多 12 s。ds004703 事件相对窗口为 -0.5 至 12 s。",
        "- ds004703 音频包络由公开 WAV 在服务器上即时计算（Hilbert amplitude + 20 Hz low-pass）；没有把该数组写入输出。",
        "- SparrKULee 的 publisher envelope 直接从公开 derivative 读取；物理同步仍标记 UNKNOWN。时间零点是数组/metadata 参考，不是硬件同步证明。",
        "",
        "## Server command",
        "",
        command,
        "",
        "## Run notes",
        "",
        "- 本批只复用既有 full index，并做确定性小样本读取；坏对象按同一排序规则跳过并记录。",
    ]
    if failures:
        lines.extend(["", "### Skipped or failed objects", ""])
        lines.extend(f"- {item}" for item in failures)
    if notes:
        lines.extend(["", "### Audio notes", ""])
        lines.extend(f"- {item}" for item in notes)
    (output / "SAMPLE_PARAMETERS_COMMANDS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_content(
    stats: dict[str, Any],
    tail: dict[str, Any],
    sparr: list[dict[str, Any]],
    ds: list[dict[str, Any]],
    failures: list[str],
) -> tuple[str, str]:
    access_counts = Counter(item["source_access"] for item in sparr)
    failure_text = "；".join(failures) if failures else "本批没有小样本读取失败。"
    sparr_subjects = ", ".join(item["subject"] for item in sparr)
    ds_subjects = ", ".join(item["subject"] for item in ds)
    md = f"""# PUB-01-S01-SS02：先探索数据的描述性报告

## 结论先行

本批已在 server2203 对既有索引和确定性小样本完成实际读取、时域图、PSD、音频/包络/事件时间轴和尾差图。图形足以开始描述性分析；正式 benchmark 的 split、物理同步、版本身份和因果/生物解释仍保留为后续 QC/实验设计问题。

- SparrKULee：{stats["sparr_rows"]} 条发布方预处理 EEG derivative 行，{stats["sparr_subjects"]} 个被试、{stats["sparr_sessions"]} 个 session、{stats["sparr_stories"]} 个 story 标签；索引时长总和 {stats["sparr_total_s"] / 3600:.2f} h，单行 {stats["sparr_min_s"]:.2f}–{stats["sparr_max_s"]:.2f} s，发布方采样率 64 Hz、PSD Nyquist 32 Hz。样本被试：{sparr_subjects}；本批 raw access 计数 {dict(access_counts)}。
- ds004703：{stats["ds_recordings"]} 条原 EDF recording、{stats["ds_segments"]} 个 eligible segment、{stats["ds_audio_files"]} 个索引 WAV，{stats["ds_subjects"]} 个被试、{stats["ds_stories"]} 个 eligible story 标签；EDF 时长总和 {stats["ds_total_s"] / 3600:.2f} h，单 recording {stats["ds_min_s"]:.2f}–{stats["ds_max_s"]:.2f} s，原生采样率为 {', '.join(f'{rate:g} Hz' for rate in stats["ds_rates"])}。样本被试：{ds_subjects}。
- Sparr 尾差：全部 {tail["n"]} 条 LARGE_TAIL_REVIEW 均绘制；尾差 {tail["min_s"]:.2f}–{tail["max_s"]:.2f} s，中位数 {tail["median_s"]:.2f} s。图中 3 s 只是旧提示的可视化参照，不是本轮描述性分析的准入线。

## 1. 规模、时长、通道和 story 覆盖

![overview_counts](overview_counts.png)

实际内容：柱状图区分 Sparr derivative 行、ds EDF recording、ds eligible segment 和 WAV 文件；时长直方图以秒为单位；通道图使用 recording_qc 的原索引字段，并把 Sparr 64 个发布方 derivative 通道单独标出。

观察：两个数据集的对象数不是同一计数单位，不能直接当作一个样本量比较。Sparr 的 64 Hz derivative 行适合短窗表征；ds 的原 EDF recording、segment 和音频分别对应不同层级。

可开展的分析：先按 recording、subject、story 做描述性时频和事件窗口汇总，再根据目标定义 segment-level 或 subject-level 汇总。不要把 666 行 derivative、11 条 EDF 和 319 个 segment 混成一个 n。

后续 QC：故事版本、物理同步、通道选择/参考和 split 需在正式 benchmark 前冻结。

## 2. SparrKULee 多通道时域和 PSD

![sparrkulee_signal_examples](sparrkulee_signal_examples.png)

实际内容：发布方 64 Hz 预处理 EEG，每行最多 8 个通道；PSD 显示到 32 Hz，未超过 Nyquist；含 normal public pair 和 raw restricted、public derivative 示例。

观察：可看到 derivative 通道振幅和低采样率可解析的频谱；受限 raw 示例仍可读公开 derivative，但不能由图推断受限 raw 的物理单位、同步或刺激波形。

可开展的分析：可继续做公开 derivative 的时域统计、低频 PSD 和包络相关描述；正式模型训练仍需冻结 subject/story 分组。

后续 QC：发布方 NPY 单位依据既有 provenance 合同标为 µV；正式分析保留 derivative 版本、通道映射和 physical sync UNKNOWN。

## 3. ds004703 原 EDF 多通道时域和 PSD

![ds004703_signal_examples](ds004703_signal_examples.png)

实际内容：原 EDF 事件附近小窗和 eligible 通道。只有在 EDF original unit 为 µV 且 MNE 返回 SI V 时才转换成 µV；PSD 每个 panel 的横轴上限是该样本 fs/2。

观察：native rate 差异直接影响 Nyquist；event/segment 是 metadata 时间参考，尚未证明与音频物理同步。

可开展的分析：可开展 recording-level、segment-level 的功率、通道稳定性和故事覆盖描述。

后续 QC：正式跨 recording 频带比较前冻结滤波、参考、通道排除、单位和采样率处理。

## 4. 音频、包络、神经索引和事件时间轴

![sparrkulee_audio_event_overlays](sparrkulee_audio_event_overlays.png)

Sparr 图使用公开 raw stimulus audio（story 可索引时）、公开 publisher envelope 和公开 EEG derivative。时间零点是 array common start，不是 hardware event。受限示例不请求 restricted raw。

![ds004703_audio_event_overlays](ds004703_audio_event_overlays.png)

ds 图使用公开 WAV，并在服务器即时计算 Hilbert + 20 Hz low-pass amplitude envelope；虚线 0 s 是 clean segment 的 metadata event onset，physical_sync=UNKNOWN。

观察：两数据集都能形成可读的音频、包络、神经时间轴，但参考时间 provenance 不同。图形相关性只能作为描述性线索，不能升级为硬件同步或生物机制证明。

可开展的分析：可先计算局部包络-神经相关、事件前后功率变化和故事/被试描述性差异，并分数据集报告。

后续 QC：核实音频/包络版本关系、事件定义、offset 残差和物理 loopback；这些问题影响正式解释，但不阻挡本轮探索。

## 5. 26 条尾差的总体关系和代表尾端

![tail_overview](tail_overview.png)

![tail_example](tail_example.png)

第一张图包含全部 26 条 LARGE_TAIL_REVIEW；横轴为索引 common duration，纵轴为 EEG-only tail duration。第二张取 source_recording 字典序第一行，显示 common envelope end 后的 EEG-only tail。3 s 虚线只是 advisory 参照。

观察：尾差是连续的长度关系，不应压缩成“超过 3 s 就坏、未超过就好”的二元标签。

可开展的分析：可把 tail duration 作为 alignment diagnostic、敏感性分层或后续裁剪策略的候选变量。

后续 QC：正式任务需明确 common overlap、tail 保留/裁剪及其与事件定义、split 的关系。

## 本轮范围和限制

- 只读取既有 full index 加确定性小样本；不重读全库，不下载，不请求受限载荷，不训练。
- {failure_text}
- 旧四个准入产物保留为历史 formal-admission 记录；本报告不把它们改写成探索阻碍。
- 本轮 acceptance 依据是真实图形、单位/Nyquist 检查和逐图解释；未知项标注为未知。

## 运行材料

- [sample_manifest.csv](sample_manifest.csv)
- [SAMPLE_PARAMETERS_COMMANDS.md](SAMPLE_PARAMETERS_COMMANDS.md)
- [plot_exploration.py](plot_exploration.py)
"""
    html_body = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PUB-01-S01-SS02 descriptive exploration</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; color: #202124; max-width: 1180px; margin: 0 auto; padding: 2rem; }}
h1, h2 {{ color: #1f3b5b; }}
figure {{ margin: 1.2rem 0 2.2rem; }}
figure img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; }}
figcaption {{ color: #4e5d6c; margin-top: .45rem; }}
.note {{ background: #f4f7fa; border-left: 4px solid #2a6fbb; padding: .8rem 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; }}
th, td {{ border: 1px solid #d9e2ec; padding: .45rem .6rem; text-align: left; vertical-align: top; }}
th {{ background: #eef3f8; }}
</style>
</head>
<body>
<h1>PUB-01-S01-SS02：先探索数据的描述性报告</h1>
<div class="note"><strong>结论先行：</strong>本批已在 server2203 完成既有索引与确定性小样本的实际读取和绘图。图形支持继续做描述性分析；正式 benchmark 的 split、物理同步和因果/生物解释仍需后续单独冻结。</div>
<h2>规模摘要</h2>
<table>
<tr><th>Dataset / unit</th><th>Scale</th><th>Duration</th><th>Sampling / channels</th><th>Story coverage</th></tr>
<tr><td>SparrKULee published derivative rows</td><td>{stats["sparr_rows"]} rows; {stats["sparr_subjects"]} subjects; {stats["sparr_sessions"]} sessions</td><td>{stats["sparr_total_s"] / 3600:.2f} h total; {stats["sparr_min_s"]:.2f}–{stats["sparr_max_s"]:.2f} s/row</td><td>64 Hz; 64 publisher derivative channels; Nyquist 32 Hz</td><td>{stats["sparr_stories"]} story labels</td></tr>
<tr><td>ds004703 original EDF</td><td>{stats["ds_recordings"]} recordings; {stats["ds_segments"]} eligible segments; {stats["ds_audio_files"]} WAV files</td><td>{stats["ds_total_s"] / 3600:.2f} h total; {stats["ds_min_s"]:.2f}–{stats["ds_max_s"]:.2f} s/recording</td><td>native {html.escape(', '.join(f'{rate:g} Hz' for rate in stats["ds_rates"]))}; channel counts from recording QC</td><td>{stats["ds_stories"]} eligible story labels</td></tr>
</table>
<h2>1. Overview</h2>
<figure><img src="overview_counts.png" alt="overview counts, duration, channel and story coverage"></figure>
<p><strong>实际内容：</strong>对象数、时长、通道和 story 覆盖来自已有 full indexes；不同对象单位没有混为一个 n。</p>
<p><strong>观察与可开展分析：</strong>Sparr derivative 行适合 64-Hz 短窗表征；ds 的 recording、segment 和 WAV 支持不同层级的描述性汇总。正式分析需按目标明确汇总层级。</p>
<p><strong>后续 QC：</strong>故事版本、物理同步、通道处理和 split 仍需冻结。</p>
<h2>2. SparrKULee signal examples</h2>
<figure><img src="sparrkulee_signal_examples.png" alt="SparrKULee multi-channel traces and PSD"></figure>
<p>发布方 64-Hz preprocessed EEG，PSD 不超过 32-Hz Nyquist；含 normal public pair 和 restricted raw、public derivative 示例。受限 raw 未读取，图形不能证明其原始单位或同步。</p>
<h2>3. ds004703 signal examples</h2>
<figure><img src="ds004703_signal_examples.png" alt="ds004703 original EDF multi-channel traces and PSD"></figure>
<p>原 EDF 事件附近小窗和 eligible 通道；单位只在可核验时转换为 µV；PSD 上限为各样本的 fs/2。event alignment 不是物理同步证明。</p>
<h2>4. Audio, envelope, neural index and event timeline</h2>
<figure><img src="sparrkulee_audio_event_overlays.png" alt="SparrKULee audio envelope neural overlays"></figure>
<p>Sparr 的 0 s 是 array common start，不是 hardware event；受限示例不请求 restricted raw。</p>
<figure><img src="ds004703_audio_event_overlays.png" alt="ds004703 audio envelope neural event overlays"></figure>
<p>ds 的 envelope 由公开 WAV 在服务器即时计算；0 s 是 metadata event onset，physical sync UNKNOWN。</p>
<h2>5. Tail diagnostics</h2>
<figure><img src="tail_overview.png" alt="all 26 tail length relations"><figcaption>All {tail["n"]} LARGE_TAIL_REVIEW rows; 3 s is an advisory marker only.</figcaption></figure>
<figure><img src="tail_example.png" alt="deterministic representative tail end example"></figure>
<p>尾差是连续长度关系，不应简化为 3 s 二元准入；可作为 alignment diagnostic 或敏感性分层候选变量。</p>
<h2>范围与运行材料</h2>
<ul>
<li>仅既有 full index + 确定性小样本；无全库重读、下载、受限载荷请求或训练。</li>
<li>{html.escape(failure_text)}</li>
<li>旧四个准入产物保留为历史 formal-admission 记录，本报告不把它们改写成探索阻碍。</li>
</ul>
<p><a href="sample_manifest.csv">sample_manifest.csv</a> · <a href="SAMPLE_PARAMETERS_COMMANDS.md">SAMPLE_PARAMETERS_COMMANDS.md</a> · <a href="plot_exploration.py">plot_exploration.py</a></p>
</body>
</html>
"""
    return md, html_body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/home/fanyu/auditory_simulation_m6a"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    sparr_index = project_root / "pub_01/s06/sparrkulee_virtual_20260919_v4/cleaning_index.csv"
    provenance_path = project_root / "pub_01/s06/sparrkulee_channel_provenance_20260920_v3/channel_provenance.csv"
    ds_root = project_root / "pub_01/s06/ds004703_qc_20260919_v2"
    sparr_rows = read_csv(sparr_index)
    provenance_rows = read_csv(provenance_path)
    provenance = {row.get("source_recording", ""): row for row in provenance_rows}
    for row in sparr_rows:
        row["_source_access"] = access(row, provenance)
        row["_provenance"] = provenance.get(row.get("source_recording", ""), {})
    recordings = read_csv(ds_root / "recording_qc.csv")
    segments = read_csv(ds_root / "clean_segment_index.csv")
    audio = read_csv(ds_root / "audio_qc.csv")
    channel_rows = read_csv(ds_root / "clean_channel_index.csv")
    recording_map = {row.get("recording", ""): row for row in recordings}
    channels_by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in channel_rows:
        channels_by_recording[row.get("recording", "")].append(row)

    failures: list[str] = []
    eligible_sparr = [
        row for row in sparr_rows
        if str(row.get("status", "")).upper().startswith("PASS")
        and str(row.get("tail_flag", "")).upper() == "WITHIN_3S_ADVISORY"
    ]
    eligible_sparr.sort(key=lambda row: (row.get("subject", ""), row.get("source_recording", "")))
    selected_sparr: list[dict[str, Any]] = []
    seen: set[str] = set()
    public_candidates = [row for row in eligible_sparr if row["_source_access"] == "PUBLIC"]
    for candidate in public_candidates:
        if len([item for item in selected_sparr if item["role"] == "public_signal"]) >= 3:
            break
        if candidate.get("subject", "") in seen:
            continue
        try:
            data, fs, channel_names, unit = load_sparr(
                {"eeg_path": candidate["eeg_path"], "_provenance": candidate["_provenance"]}
            )
            item = dict(candidate)
            item.update(
                {
                    "data": data,
                    "time": np.arange(data.shape[1]) / fs,
                    "fs": fs,
                    "labels": channel_names,
                    "unit": unit,
                    "subject": candidate.get("subject", ""),
                    "session": candidate.get("session", ""),
                    "story": candidate.get("story", ""),
                    "sample_key": candidate.get("source_recording", ""),
                    "source_recording": candidate.get("source_recording", ""),
                    "source_access": candidate["_source_access"],
                    "role": "public_signal",
                    "selection_rule": "first sorted public subject with WITHIN_3S_ADVISORY",
                }
            )
            selected_sparr.append(item)
            seen.add(item["subject"])
        except Exception as exc:
            failures.append(f"Sparr {candidate.get('source_recording', '')}: {type(exc).__name__}: {exc}")
    restricted_candidates = [row for row in eligible_sparr if row["_source_access"] == "RESTRICTED"]
    for candidate in restricted_candidates:
        try:
            data, fs, channel_names, unit = load_sparr(
                {"eeg_path": candidate["eeg_path"], "_provenance": candidate["_provenance"]}
            )
            item = dict(candidate)
            item.update(
                {
                    "data": data,
                    "time": np.arange(data.shape[1]) / fs,
                    "fs": fs,
                    "labels": channel_names,
                    "unit": unit,
                    "subject": candidate.get("subject", ""),
                    "session": candidate.get("session", ""),
                    "story": candidate.get("story", ""),
                    "sample_key": candidate.get("source_recording", ""),
                    "source_recording": candidate.get("source_recording", ""),
                    "source_access": candidate["_source_access"],
                    "role": "restricted_raw_public_derivative",
                    "selection_rule": "first sorted restricted source with public derivative",
                }
            )
            selected_sparr.append(item)
            break
        except Exception as exc:
            failures.append(f"Sparr restricted {candidate.get('source_recording', '')}: {type(exc).__name__}: {exc}")
    if len([item for item in selected_sparr if item["role"] == "public_signal"]) < 3:
        raise RuntimeError("fewer than three usable distinct public Sparr subjects")
    if not any(item["role"] == "restricted_raw_public_derivative" for item in selected_sparr):
        raise RuntimeError("no restricted raw/public derivative Sparr example")

    eligible_segments = [row for row in segments if truthy(row.get("eligible", ""))]
    eligible_segments.sort(key=lambda row: (row.get("subject", ""), row.get("recording", ""), inum(row.get("trial"))))
    selected_ds: list[dict[str, Any]] = []
    seen_ds: set[str] = set()
    for segment in eligible_segments:
        subject = segment.get("subject", "")
        if subject in seen_ds:
            continue
        recording = recording_map.get(segment.get("recording", ""))
        if recording is None:
            failures.append(f"ds004703 {segment.get('recording', '')}: no recording_qc row")
            continue
        try:
            loaded = load_ds(segment, recording, channels_by_recording.get(segment.get("recording", ""), []))
            item = dict(segment)
            item.update(loaded)
            item.update(
                {
                    "subject": subject,
                    "session": segment.get("session", ""),
                    "story": segment.get("story", ""),
                    "sample_key": f"{segment.get('recording', '')}_trial-{segment.get('trial', '')}",
                    "role": "eligible_segment_signal",
                    "selection_rule": "first sorted eligible segment per distinct subject",
                    "audio_path": segment.get("audio_path", ""),
                    "source_access": "PUBLIC_BUNDLE_WITH_README_LIMITATION",
                }
            )
            selected_ds.append(item)
            seen_ds.add(subject)
        except Exception as exc:
            failures.append(f"ds004703 {segment.get('recording', '')}/trial-{segment.get('trial', '')}: {type(exc).__name__}: {exc}")
        if len(selected_ds) >= 3:
            break
    if len(selected_ds) < 3:
        raise RuntimeError("fewer than three usable distinct ds004703 subjects")

    summary = stats_for(sparr_rows, recordings, segments, audio)
    overview_plot(summary, sparr_rows, recordings, output / "overview_counts.png")
    signal_plot(selected_sparr, "SparrKULee", output / "sparrkulee_signal_examples.png")
    signal_plot(selected_ds, "ds004703", output / "ds004703_signal_examples.png")
    raw_stimulus = project_root / "pub_01/s06/sparrkulee_v3.1/raw/stimuli/eeg"
    overlay_rows = selected_sparr[:1] + [item for item in selected_sparr if item["role"] == "restricted_raw_public_derivative"]
    audio_notes = sparr_overlay_plot(overlay_rows, raw_stimulus, output / "sparrkulee_audio_event_overlays.png")
    audio_notes.extend(ds_overlay_plot(selected_ds, output / "ds004703_audio_event_overlays.png"))

    tail_rows = [row for row in sparr_rows if str(row.get("tail_flag", "")).upper() == "LARGE_TAIL_REVIEW"]
    if len(tail_rows) != 26:
        failures.append(f"tail inventory expected 26 rows, found {len(tail_rows)}")
    tail = tail_plots(tail_rows, provenance, output / "tail_overview.png", output / "tail_example.png")

    make_manifest(selected_sparr, selected_ds, output)
    write_run_notes(output, project_root, failures, audio_notes)
    md, html_body = report_content(summary, tail, selected_sparr, selected_ds, failures)
    (output / "overview.md").write_text(md, encoding="utf-8")
    (output / "overview.html").write_text(html_body, encoding="utf-8")
    result = {
        "status": "DESCRIPTIVE_EXPLORATION_COMPLETE",
        "selected_sparr": [
            {
                "sample_key": item["sample_key"],
                "subject": item["subject"],
                "role": item["role"],
                "source_access": item["source_access"],
                "sampling_rate_hz": item["fs"],
            }
            for item in selected_sparr
        ],
        "selected_ds004703": [
            {
                "sample_key": item["sample_key"],
                "subject": item["subject"],
                "recording": item["recording"],
                "sampling_rate_hz": item["fs"],
                "units": item["unit"],
                "unit_verified": item["unit_verified"],
            }
            for item in selected_ds
        ],
        "tail": {
            "n": tail["n"],
            "min_s": tail["min_s"],
            "median_s": tail["median_s"],
            "max_s": tail["max_s"],
            "representative": tail["representative"].get("source_recording", ""),
        },
        "failures": failures,
        "audio_notes": audio_notes,
    }
    (output / "exploration_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
