from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
IEEG_SUFFIXES = {".edf", ".vhdr", ".set", ".fif", ".nwb"}


def utc_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def role_for(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name == "dataset_description.json":
        return "dataset_description"
    if name == "readme" or name.startswith("readme."):
        return "readme"
    if name == "participants.tsv":
        return "participants"
    if name.endswith("_events.tsv"):
        return "events"
    if name.endswith("_channels.tsv"):
        return "channels"
    if name.endswith("_electrodes.tsv"):
        return "electrodes"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if "_ieeg" in name and suffix in IEEG_SUFFIXES:
        return "ieeg"
    return "other"


def read_tsv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        return next(reader)


def read_audio_info(path: Path) -> dict[str, Any]:
    import soundfile

    info = soundfile.info(str(path))
    return {
        "path": path.as_posix(),
        "samplerate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "format": info.format,
    }


def read_ieeg_header(path: Path) -> dict[str, Any]:
    import mne

    suffix = path.suffix.lower()
    if suffix == ".edf":
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    elif suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose="ERROR")
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(path, preload=False, verbose="ERROR")
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    else:
        raise ValueError(f"unsupported iEEG header format: {suffix}")
    return {
        "path": path.as_posix(),
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "channels": int(len(raw.ch_names)),
        "duration_seconds": float(raw.times[-1]) if raw.n_times else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ds004703 without hashes.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--audio-samples", type=int, default=5)
    parser.add_argument("--ieeg-samples", type=int, default=1)
    parser.add_argument("--scope", default="FULL_DATASET")
    parser.add_argument("--expected-object-count", type=int)
    parser.add_argument("--expected-total-bytes", type=int)
    parser.add_argument("--require-neural-files", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"dataset root does not exist: {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    inventory: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    bytes_by_role: Counter[str] = Counter()
    for path in files:
        relative = path.relative_to(root).as_posix()
        role = role_for(path)
        size = path.stat().st_size
        counts[role] += 1
        bytes_by_role[role] += size
        inventory.append(
            {
                "path": relative,
                "bytes": size,
                "modified_at_utc": utc_mtime(path),
                "role": role,
                "sample_readability": "NOT_SAMPLED",
            }
        )

    by_relative = {item["path"]: item for item in inventory}
    errors: list[str] = []
    warnings: list[str] = []

    description_path = root / "dataset_description.json"
    description: dict[str, Any] = {}
    try:
        with description_path.open("r", encoding="utf-8") as handle:
            description = json.load(handle)
        by_relative["dataset_description.json"]["sample_readability"] = "JSON_PARSED"
    except Exception as exc:
        errors.append(f"dataset_description unreadable: {type(exc).__name__}: {exc}")

    expected_doi = "doi:10.18112/openneuro.ds004703.v1.1.0"
    if description.get("DatasetDOI") != expected_doi:
        errors.append(f"unexpected DatasetDOI: {description.get('DatasetDOI')!r}")
    if str(description.get("License", "")).upper() != "CC0":
        errors.append(f"unexpected declared license: {description.get('License')!r}")

    readme_candidates = [path for path in files if role_for(path) == "readme"]
    readme_text = ""
    if readme_candidates:
        try:
            readme_text = readme_candidates[0].read_text(encoding="utf-8")
            rel = readme_candidates[0].relative_to(root).as_posix()
            by_relative[rel]["sample_readability"] = "TEXT_READ"
        except Exception as exc:
            errors.append(f"README unreadable: {type(exc).__name__}: {exc}")
    else:
        errors.append("README missing")
    lower_readme = readme_text.lower()
    if "may not be used for commer" not in lower_readme:
        errors.append("README noncommercial restriction not found")
    if "disambiguates participant identity" not in lower_readme:
        errors.append("README reidentification restriction not found")

    participants_path = root / "participants.tsv"
    participant_count = 0
    if participants_path.is_file():
        try:
            header = read_tsv_header(participants_path)
            with participants_path.open("r", encoding="utf-8-sig") as handle:
                participant_count = max(sum(1 for _ in handle) - 1, 0)
            by_relative["participants.tsv"]["sample_readability"] = "TSV_HEADER_READ"
            if "participant_id" not in header:
                errors.append("participants.tsv lacks participant_id")
        except Exception as exc:
            errors.append(f"participants.tsv unreadable: {type(exc).__name__}: {exc}")
    else:
        errors.append("participants.tsv missing")

    tsv_samples = [path for path in files if path.suffix.lower() == ".tsv"][:100]
    for path in tsv_samples:
        rel = path.relative_to(root).as_posix()
        try:
            read_tsv_header(path)
            if by_relative[rel]["sample_readability"] == "NOT_SAMPLED":
                by_relative[rel]["sample_readability"] = "TSV_HEADER_READ"
        except Exception as exc:
            errors.append(f"TSV unreadable {rel}: {type(exc).__name__}: {exc}")

    audio_reports: list[dict[str, Any]] = []
    for path in [item for item in files if path_suffix_role(item) == "audio"][: args.audio_samples]:
        rel = path.relative_to(root).as_posix()
        try:
            info = read_audio_info(path)
            info["path"] = rel
            audio_reports.append(info)
            by_relative[rel]["sample_readability"] = "AUDIO_HEADER_READ"
        except Exception as exc:
            errors.append(f"audio unreadable {rel}: {type(exc).__name__}: {exc}")

    ieeg_reports: list[dict[str, Any]] = []
    ieeg_paths = [item for item in files if role_for(item) == "ieeg"]
    supported_ieeg = [item for item in ieeg_paths if item.suffix.lower() != ".nwb"]
    for path in supported_ieeg[: args.ieeg_samples]:
        rel = path.relative_to(root).as_posix()
        try:
            info = read_ieeg_header(path)
            info["path"] = rel
            ieeg_reports.append(info)
            by_relative[rel]["sample_readability"] = "IEEG_HEADER_READ"
        except Exception as exc:
            errors.append(f"iEEG unreadable {rel}: {type(exc).__name__}: {exc}")
    if ieeg_paths and not supported_ieeg:
        warnings.append("only NWB iEEG files found; header reader not implemented")

    total_bytes = sum(item["bytes"] for item in inventory)
    if args.expected_object_count is not None and len(files) != args.expected_object_count:
        errors.append(f"object count mismatch: expected {args.expected_object_count}, found {len(files)}")
    if args.expected_total_bytes is not None and total_bytes != args.expected_total_bytes:
        errors.append(f"total bytes mismatch: expected {args.expected_total_bytes}, found {total_bytes}")
    if args.require_neural_files:
        for required_role in ("ieeg", "events", "channels"):
            if counts[required_role] == 0:
                errors.append(f"required role missing: {required_role}")

    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "bytes", "modified_at_utc", "role", "sample_readability"],
        )
        writer.writeheader()
        writer.writerows(inventory)

    report = {
        "task_id": "M6A-PUBLIC-001",
        "dataset_id": "ds004703",
        "expected_version": "1.1.0",
        "audit_scope": args.scope,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "integrity_policy": "NON_HASH_AUDIT",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "counts_by_role": dict(sorted(counts.items())),
        "bytes_by_role": dict(sorted(bytes_by_role.items())),
        "participant_count": participant_count,
        "dataset_description": description,
        "license_boundary": "CC0_PLUS_README_NONCOMMERCIAL_AND_NO_REIDENTIFICATION",
        "audio_samples": audio_reports,
        "ieeg_samples": ieeg_reports,
        "warnings": warnings,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def path_suffix_role(path: Path) -> str:
    return role_for(path)


if __name__ == "__main__":
    raise SystemExit(main())
