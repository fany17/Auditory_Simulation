from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.dataset_audit import read_ieeg_header, utc_mtime


NEURAL_CHANNEL_TYPES = {"SEEG", "ECOG"}
EXPECTED_IEEG_REFERENCE = "scalp electrode, not included with data"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def summarize_channels(rows: list[dict[str, str]]) -> dict[str, Any]:
    type_counts = Counter((row.get("type") or "UNKNOWN").upper() for row in rows)
    status_counts = Counter((row.get("status") or "UNKNOWN").lower() for row in rows)
    unit_counts = Counter(row.get("units") or "UNKNOWN" for row in rows)
    good_neural = [
        row
        for row in rows
        if (row.get("type") or "").upper() in NEURAL_CHANNEL_TYPES
        and (row.get("status") or "").lower() == "good"
    ]
    c_prefix_names = sorted(
        row.get("name") or "" for row in rows if (row.get("name") or "").upper().startswith("C")
    )
    analysis_eligible_neural = [
        row for row in good_neural if not (row.get("name") or "").upper().startswith("C")
    ]
    analysis_eligible_neural_names = sorted(row.get("name") or "" for row in analysis_eligible_neural)
    dc1 = [
        {
            "name": row.get("name") or "",
            "status": row.get("status") or "",
            "description": row.get("status_description") or "",
        }
        for row in rows
        if (row.get("name") or "").upper() == "DC1"
    ]
    return {
        "channel_count": len(rows),
        "type_counts": dict(sorted(type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "unit_counts": dict(sorted(unit_counts.items())),
        "good_neural_channel_count": len(good_neural),
        "analysis_eligible_neural_channel_count": len(analysis_eligible_neural),
        "analysis_eligible_neural_names": analysis_eligible_neural_names,
        "c_prefix_exclusion_count": len(c_prefix_names),
        "c_prefix_names": c_prefix_names,
        "dc1_channels": dc1,
    }


def event_summary(path: Path) -> dict[str, Any]:
    row_count = 0
    maximum_offset = 0.0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            row_count += 1
            maximum_offset = max(maximum_offset, float(row["onset"]) + float(row["duration"]))
    return {"event_row_count": row_count, "maximum_event_offset_seconds": maximum_offset}


def channel_inventory_difference(
    channel_rows: list[dict[str, str]],
    edf_channel_names: list[str],
) -> dict[str, list[dict[str, str]]]:
    channels_by_name = {row.get("name") or "": row for row in channel_rows}
    tsv_names = set(channels_by_name)
    edf_names = set(edf_channel_names)
    return {
        "tsv_only": [
            {
                "name": name,
                "type": channels_by_name[name].get("type") or "UNKNOWN",
                "status": channels_by_name[name].get("status") or "UNKNOWN",
                "difference": "CHANNELS_TSV_ONLY",
            }
            for name in sorted(tsv_names - edf_names)
        ],
        "edf_only": [
            {
                "name": name,
                "type": "UNDECLARED_IN_CHANNELS_TSV",
                "status": "UNKNOWN",
                "difference": "EDF_ONLY",
            }
            for name in sorted(edf_names - tsv_names)
        ],
    }


def summarize_ieeg_references(
    recordings: list[dict[str, Any]],
    expected_count: int = 11,
) -> dict[str, Any]:
    recording_values = [
        {
            "recording_id": item["recording_id"],
            "iEEGReference": item.get("iEEGReference"),
            "matches_frozen_value": item.get("iEEGReference") == EXPECTED_IEEG_REFERENCE,
        }
        for item in recordings
    ]
    observed_values = sorted(
        {str(item["iEEGReference"]) for item in recording_values if item["iEEGReference"] is not None}
    )
    mismatch_ids = [
        str(item["recording_id"])
        for item in recording_values
        if item["matches_frozen_value"] is not True
    ]
    passed = (
        len(recording_values) == expected_count
        and observed_values == [EXPECTED_IEEG_REFERENCE]
        and not mismatch_ids
    )
    return {
        "expected_value": EXPECTED_IEEG_REFERENCE,
        "expected_recording_count": expected_count,
        "observed_recording_count": len(recording_values),
        "observed_unique_values": observed_values,
        "recording_values": recording_values,
        "mismatch_recording_ids": mismatch_ids,
        "primary_policy": "AS_RECORDED_SCALP_REFERENCE",
        "missing_reference_reconstruction": "FORBIDDEN",
        "contact_name_bipolar_pairing": "FORBIDDEN",
        "status": "PASS" if passed else "FAIL",
    }


def build_neural_metadata_audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    recordings: list[dict[str, Any]] = []
    sidecars = sorted(root.glob("sub-*/ses-*/ieeg/*_ieeg.json"))

    for sidecar_path in sidecars:
        recording_id = sidecar_path.name.removesuffix("_ieeg.json")
        participant_id = sidecar_path.relative_to(root).parts[0]
        session_id = sidecar_path.relative_to(root).parts[1]
        ieeg_dir = sidecar_path.parent
        session_dir = ieeg_dir.parent
        edf_path = ieeg_dir / f"{recording_id}_ieeg.edf"
        channels_path = ieeg_dir / f"{recording_id}_channels.tsv"
        events_path = session_dir / f"{recording_id}_events.tsv"
        offset_path = ieeg_dir / f"{recording_id}_ieeg_audio-offset.json"

        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            channel_rows = read_tsv(channels_path)
            channels = summarize_channels(channel_rows)
            events = event_summary(events_path)
            offset = json.loads(offset_path.read_text(encoding="utf-8"))["AudioOffset"]
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"metadata unreadable for {recording_id}: {type(exc).__name__}: {exc}")
            continue

        edf_header: dict[str, Any] | None = None
        if not edf_path.is_file():
            errors.append(f"EDF missing for {recording_id}")
        else:
            try:
                edf_header = read_ieeg_header(edf_path)
            except (OSError, RuntimeError, ValueError) as exc:
                errors.append(f"EDF unreadable for {recording_id}: {type(exc).__name__}: {exc}")

        sidecar_sampling = float(sidecar["SamplingFrequency"])
        sidecar_duration = float(sidecar["RecordingDuration"])
        declared_neural_channels = int(sidecar.get("SEEGChannelCount", 0)) + int(
            sidecar.get("ECOGChannelCount", 0)
        )
        edf_channel_names = set(edf_header["channel_names"]) if edf_header is not None else set()
        analysis_neural_names = set(channels["analysis_eligible_neural_names"])
        analysis_neural_missing_from_edf = (
            sorted(analysis_neural_names - edf_channel_names) if edf_header is not None else []
        )
        if analysis_neural_missing_from_edf:
            errors.append(
                f"analysis-eligible neural channels missing from EDF for {recording_id}: "
                + ", ".join(analysis_neural_missing_from_edf)
            )
        inventory_difference = (
            channel_inventory_difference(channel_rows, list(edf_header["channel_names"]))
            if edf_header is not None
            else {"tsv_only": [], "edf_only": []}
        )
        events_within_edf = (
            events["maximum_event_offset_seconds"]
            <= float(edf_header["duration_seconds"]) + 1.0 / sidecar_sampling
            if edf_header is not None
            else None
        )
        recording = {
            "participant_id": participant_id,
            "session_id": session_id,
            "recording_id": recording_id,
            "sampling_rate_hz": sidecar_sampling,
            "power_line_frequency_hz": float(sidecar["PowerLineFrequency"]),
            "recording_duration_seconds": sidecar_duration,
            "task_name": sidecar.get("TaskName"),
            "recording_type": sidecar.get("RecordingType"),
            "iEEGReference": sidecar.get("iEEGReference"),
            "declared_seeg_plus_ecog_channels": declared_neural_channels,
            "channels": channels,
            "events": events,
            "audio_offset_seconds": float(offset),
            "events_within_recording": events["maximum_event_offset_seconds"] <= sidecar_duration,
            "events_within_edf_timeline": events_within_edf,
            "edf_file": edf_path.relative_to(root).as_posix(),
            "edf_bytes": edf_path.stat().st_size if edf_path.is_file() else None,
            "edf_modified_at_utc": utc_mtime(edf_path) if edf_path.is_file() else None,
            "edf_header": edf_header,
            "edf_header_channel_count_matches_tsv": (
                edf_header is not None and int(edf_header["channels"]) == channels["channel_count"]
            ),
            "analysis_eligible_neural_channels_missing_from_edf": analysis_neural_missing_from_edf,
            "channel_inventory_difference": {
                "tsv_only": inventory_difference["tsv_only"],
                "edf_only": inventory_difference["edf_only"],
            },
            "edf_header_sampling_rate_matches_sidecar": (
                edf_header is not None
                and float(edf_header["sampling_rate_hz"]) == sidecar_sampling
            ),
        }
        if not recording["events_within_recording"]:
            errors.append(f"events extend beyond recording duration for {recording_id}")
        if edf_header is not None and events_within_edf is not True:
            errors.append(f"events extend beyond EDF timeline for {recording_id}")
        if edf_header is not None and not recording["edf_header_channel_count_matches_tsv"]:
            warnings.append(f"EDF/channels total row-count difference for {recording_id}")
        if edf_header is not None and (
            inventory_difference["tsv_only"] or inventory_difference["edf_only"]
        ):
            warnings.append(
                f"EDF/channels named inventory difference for {recording_id}: "
                f"tsv_only={len(inventory_difference['tsv_only'])}, "
                f"edf_only={len(inventory_difference['edf_only'])}"
            )
        if edf_header is not None and not recording["edf_header_sampling_rate_matches_sidecar"]:
            errors.append(f"EDF/sidecar sampling mismatch for {recording_id}")
        recordings.append(recording)

    contact_files = sorted(root.glob("sub-*/ses-*/anat/*contact_RAS.csv"))
    contact_participants = sorted({path.relative_to(root).parts[0] for path in contact_files})
    standard_electrodes = sorted(root.rglob("*_electrodes.tsv"))
    coordinate_systems = sorted(root.rglob("*_coordsystem.json"))

    bids_layout_status = "PASS"
    bids_indexed_files = 0
    bids_subjects: list[str] = []
    try:
        from bids import BIDSLayout  # type: ignore[import-untyped]

        layout = BIDSLayout(root, validate=True)
        bids_indexed_files = len(layout.get(return_type="filename"))
        bids_subjects = sorted(layout.get_subjects())
    except Exception as exc:
        bids_layout_status = "FAIL"
        errors.append(f"PyBIDS validated layout failed: {type(exc).__name__}: {exc}")

    line_frequencies = sorted({item["power_line_frequency_hz"] for item in recordings})
    sampling_rates = sorted({item["sampling_rate_hz"] for item in recordings})
    ieeg_reference_audit = summarize_ieeg_references(recordings)
    line_harmonics_in_candidate_band = sorted(
        {
            harmonic
            for line_frequency in line_frequencies
            for harmonic in (line_frequency, 2 * line_frequency, 3 * line_frequency)
            if 70 <= harmonic <= 150
        }
    )
    if len(sidecars) != 11:
        errors.append(f"expected 11 iEEG sidecars, found {len(sidecars)}")
    if ieeg_reference_audit["status"] != "PASS":
        errors.append(
            "iEEGReference must match the frozen as-recorded scalp reference for all 11 recordings"
        )

    return {
        "report_schema_version": "m6a-neural-metadata-audit-v2",
        "task_id": "M6A-PUBLIC-001",
        "dataset_id": "ds004703",
        "dataset_version": "1.1.0",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(root),
        "integrity_policy": "NON_HASH_AUDIT",
        "recording_count": len(recordings),
        "necessary_metadata_readability": {
            "ieeg_sidecar_json_read": len(recordings),
            "channels_tsv_read": len(recordings),
            "events_tsv_read": len(recordings),
            "audio_offset_json_read": len(recordings),
            "expected_each": 11,
            "status": "PASS" if len(recordings) == 11 else "FAIL",
        },
        "sampling_rate_hz_values": sampling_rates,
        "power_line_frequency_hz_values": line_frequencies,
        "ieeg_reference_audit": ieeg_reference_audit,
        "analysis_eligible_neural_channel_count": sum(
            item["channels"]["analysis_eligible_neural_channel_count"] for item in recordings
        ),
        "c_prefix_exclusion_count": sum(
            item["channels"]["c_prefix_exclusion_count"] for item in recordings
        ),
        "bids_layout": {
            "status": bids_layout_status,
            "pybids_validate": True,
            "indexed_file_count": bids_indexed_files,
            "subjects": bids_subjects,
        },
        "spatial_metadata": {
            "standard_electrodes_tsv_count": len(standard_electrodes),
            "standard_coordsystem_json_count": len(coordinate_systems),
            "contact_ras_csv_count": len(contact_files),
            "contact_ras_participants": contact_participants,
            "status": "NONSTANDARD_COORDINATES_WITHOUT_ANATOMICAL_REGION_LABELS",
            "limitation": "SD012 anatomical scans are unavailable per README; contact RAS files do not provide anatomical region labels.",
        },
        "neural_target_gate": {
            "candidate": "LINE_HARMONIC_EXCLUDED_MULTIBAND_HIGH_GAMMA_LOG_POWER",
            "status": "METHOD_FROZEN_AWAITING_EXECUTION_GATES",
            "line_harmonics_in_band_hz": line_harmonics_in_candidate_band,
            "excluded_harmonic_guard_hz": [110, 130],
            "primary_reference_policy": "AS_RECORDED_SCALP_REFERENCE",
            "neural_extraction_allowed": False,
            "required_action": "G2 full audit, audio context measurement, final embargo and baseline-final split guard are required before real neural extraction.",
        },
        "recordings": recordings,
        "warnings": warnings,
        "errors": errors,
        "status": "FAIL" if errors else "PASS_WITH_METHOD_FROZEN_EXECUTION_BLOCKED",
    }


def write_recording_csv(path: Path, recordings: list[dict[str, Any]]) -> None:
    rows = [
        {
            "participant_id": item["participant_id"],
            "session_id": item["session_id"],
            "recording_id": item["recording_id"],
            "sampling_rate_hz": item["sampling_rate_hz"],
            "power_line_frequency_hz": item["power_line_frequency_hz"],
            "iEEGReference": item["iEEGReference"],
            "recording_duration_seconds": item["recording_duration_seconds"],
            "channel_count": item["channels"]["channel_count"],
            "good_neural_channel_count": item["channels"]["good_neural_channel_count"],
            "analysis_eligible_neural_channel_count": item["channels"]["analysis_eligible_neural_channel_count"],
            "c_prefix_exclusion_count": item["channels"]["c_prefix_exclusion_count"],
            "events_within_recording": item["events_within_recording"],
            "events_within_edf_timeline": item["events_within_edf_timeline"],
            "edf_header_read": item["edf_header"] is not None,
            "edf_header_channel_count_matches_tsv": item["edf_header_channel_count_matches_tsv"],
            "analysis_eligible_neural_channels_missing_from_edf": len(
                item["analysis_eligible_neural_channels_missing_from_edf"]
            ),
            "channels_tsv_only_count": len(item["channel_inventory_difference"]["tsv_only"]),
            "channels_edf_only_count": len(item["channel_inventory_difference"]["edf_only"]),
            "edf_header_sampling_rate_matches_sidecar": item["edf_header_sampling_rate_matches_sidecar"],
        }
        for item in recordings
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ds004703 neural metadata without hashes.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, required=True)
    args = parser.parse_args()
    report = build_neural_metadata_audit(args.dataset_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["recordings"]:
        write_recording_csv(args.recordings, report["recordings"])
    print(json.dumps({key: value for key, value in report.items() if key != "recordings"}, ensure_ascii=False, indent=2))
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
