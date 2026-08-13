from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from m6a_public.config_gate import validate_task_config


REPORT_SCHEMA_VERSION = "m6a-g2-promotion-gate-v1"
CANDIDATE_STATUS = "G2_CANDIDATE_AWAITING_COORDINATOR_REVIEW"
EXPECTED_FILE_COUNT = 377
EXPECTED_TOTAL_BYTES = 14_173_350_514
EXPECTED_RECORDING_COUNT = 11
EXPECTED_SPLIT_ROWS = 319
EXPECTED_ELIGIBLE_CHANNEL_COUNT = 1346
EXPECTED_C_PREFIX_EXCLUSION_COUNT = 727
MAX_AUDIT_TIME_DIFFERENCE_SECONDS = 30 * 60
EXPECTED_SPLIT_COUNTS = {"train": 223, "validation": 48, "test": 48}
EXPECTED_BLOCK_ASSIGNMENTS = {
    "block-01": "train",
    "block-02": "train",
    "block-03": "validation",
    "block-04": "test",
    "block-05": "train",
    "block-06": "train",
}
EXPECTED_LANGUAGE_COUNTS = {"en": 319}
MISSING = object()


def _reject_parse_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_strict_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_parse_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def find_non_finite(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            findings.extend(find_non_finite(child, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            findings.extend(find_non_finite(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        findings.append(path)
    return findings


def _get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return MISSING
        current = current[key]
    return current


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _parse_aware_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _is_aware_iso_datetime(value: Any) -> bool:
    return _parse_aware_iso_datetime(value) is not None


def _is_list(value: Any) -> bool:
    return isinstance(value, list)


def _recordings(neural_report: Mapping[str, Any]) -> list[Any]:
    value = neural_report.get("recordings")
    return value if isinstance(value, list) else []


def _dataset_headers(dataset_report: Mapping[str, Any]) -> list[Any]:
    value = dataset_report.get("ieeg_samples")
    return value if isinstance(value, list) else []


def _config_gate_passes(config: dict[str, Any]) -> bool:
    try:
        return validate_task_config(config) == []
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _normalize_remote_root(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


def _string_set(items: list[Any], key: str) -> set[str]:
    return {
        value
        for item in items
        if isinstance(item, Mapping)
        for value in [item.get(key)]
        if isinstance(value, str) and value
    }


def validate_g2_promotion(
    dataset_report: dict[str, Any],
    neural_report: dict[str, Any],
    split_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}

    inputs = {
        "dataset_report": dataset_report,
        "neural_report": neural_report,
        "split_report": split_report,
        "config": config,
    }
    for name, payload in inputs.items():
        checks[f"{name}_all_numeric_values_finite"] = not find_non_finite(payload)

    checks["dataset_report_identity_and_version"] = (
        dataset_report.get("report_schema_version") == "m6a-dataset-audit-v2"
        and dataset_report.get("task_id") == "M6A-PUBLIC-001"
        and dataset_report.get("dataset_id") == "ds004703"
        and dataset_report.get("expected_version") == "1.1.0"
        and dataset_report.get("audit_scope") == "FULL_DATASET_G2_CANDIDATE"
        and _is_aware_iso_datetime(dataset_report.get("audited_at_utc"))
        and dataset_report.get("integrity_policy") == "NON_HASH_AUDIT"
    )
    checks["dataset_report_pass_without_errors"] = (
        dataset_report.get("status") == "PASS" and dataset_report.get("errors") == []
    )
    checks["dataset_exact_inventory_totals"] = (
        dataset_report.get("file_count") == EXPECTED_FILE_COUNT
        and dataset_report.get("total_bytes") == EXPECTED_TOTAL_BYTES
    )
    checks["dataset_no_active_partials"] = (
        dataset_report.get("active_partial_count") == 0
        and dataset_report.get("active_partial_files") == []
    )
    reconciliation = _get(dataset_report, "expected_inventory_reconciliation")
    checks["dataset_inventory_path_and_byte_reconciliation"] = (
        isinstance(reconciliation, Mapping)
        and reconciliation.get("status") == "PASS"
        and reconciliation.get("expected_file_count") == EXPECTED_FILE_COUNT
        and reconciliation.get("expected_total_bytes") == EXPECTED_TOTAL_BYTES
        and reconciliation.get("missing_paths") == []
        and reconciliation.get("unexpected_paths") == []
        and reconciliation.get("byte_mismatches") == []
    )
    provenance = _get(dataset_report, "expected_inventory_provenance")
    checks["dataset_inventory_provenance"] = (
        isinstance(provenance, Mapping)
        and provenance.get("source") == "https://s3.amazonaws.com/openneuro.org"
        and provenance.get("acquisition_method") == "PUBLIC_S3_LIST_OBJECTS_V2"
        and provenance.get("acquisition_script") == "scripts/public_s3_inventory.py"
        and provenance.get("object_count") == EXPECTED_FILE_COUNT
        and provenance.get("total_bytes") == EXPECTED_TOTAL_BYTES
        and _is_aware_iso_datetime(provenance.get("listed_at_utc"))
    )
    boundary = _get(dataset_report, "dataset_boundary_checks")
    description = _get(dataset_report, "dataset_description")
    checks["dataset_doi_license_and_readme_boundary"] = (
        isinstance(boundary, Mapping)
        and boundary.get("status") == "PASS"
        and boundary.get("dataset_doi_matches") is True
        and boundary.get("declared_license_matches") is True
        and boundary.get("readme_noncommercial_restriction_found") is True
        and boundary.get("readme_no_reidentification_restriction_found") is True
        and isinstance(description, Mapping)
        and description.get("DatasetDOI") == "doi:10.18112/openneuro.ds004703.v1.1.0"
        and str(description.get("License", "")).upper() == "CC0"
        and dataset_report.get("license_boundary")
        == "CC0_PLUS_README_NONCOMMERCIAL_AND_NO_REIDENTIFICATION"
    )
    headers = _dataset_headers(dataset_report)
    dataset_header_paths = _string_set(headers, "path")
    checks["dataset_11_of_11_edf_headers_read"] = (
        len(headers) == EXPECTED_RECORDING_COUNT
        and len(dataset_header_paths) == EXPECTED_RECORDING_COUNT
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
            and bool(item.get("path"))
            and item.get("sampling_rate_hz") in (512.0, 1024.0)
            and isinstance(item.get("channels"), int)
            and item.get("channels", 0) > 0
            and _is_finite_number(item.get("duration_seconds"))
            and item.get("duration_seconds", 0) > 0
            and _is_list(item.get("channel_names"))
            and bool(item.get("channel_names"))
            for item in headers
        )
    )

    checks["neural_report_identity_and_version"] = (
        neural_report.get("report_schema_version") == "m6a-neural-metadata-audit-v2"
        and neural_report.get("task_id") == "M6A-PUBLIC-001"
        and neural_report.get("dataset_id") == "ds004703"
        and neural_report.get("dataset_version") == "1.1.0"
        and _is_aware_iso_datetime(neural_report.get("audited_at_utc"))
        and neural_report.get("integrity_policy") == "NON_HASH_AUDIT"
    )
    checks["neural_report_pass_with_method_frozen_and_execution_blocked"] = (
        neural_report.get("status") == "PASS_WITH_METHOD_FROZEN_EXECUTION_BLOCKED"
        and neural_report.get("errors") == []
    )
    readability = _get(neural_report, "necessary_metadata_readability")
    checks["neural_11_sidecars_channels_events_audio_offsets_read"] = (
        neural_report.get("recording_count") == EXPECTED_RECORDING_COUNT
        and isinstance(readability, Mapping)
        and readability.get("expected_each") == EXPECTED_RECORDING_COUNT
        and readability.get("ieeg_sidecar_json_read") == EXPECTED_RECORDING_COUNT
        and readability.get("channels_tsv_read") == EXPECTED_RECORDING_COUNT
        and readability.get("events_tsv_read") == EXPECTED_RECORDING_COUNT
        and readability.get("audio_offset_json_read") == EXPECTED_RECORDING_COUNT
        and readability.get("status") == "PASS"
    )
    recordings = _recordings(neural_report)
    recording_ids = _string_set(recordings, "recording_id")
    neural_edf_paths = _string_set(recordings, "edf_file")
    checks["neural_11_of_11_edf_headers_read"] = (
        len(recordings) == EXPECTED_RECORDING_COUNT
        and len(recording_ids) == EXPECTED_RECORDING_COUNT
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("recording_id"), str)
            and bool(item.get("recording_id"))
            and isinstance(item.get("edf_file"), str)
            and bool(item.get("edf_file"))
            and isinstance(item.get("edf_header"), Mapping)
            for item in recordings
        )
    )
    dataset_root = _normalize_remote_root(dataset_report.get("dataset_root"))
    neural_root = _normalize_remote_root(neural_report.get("dataset_root"))
    config_root = _normalize_remote_root(_get(config, "dataset", "remote_root"))
    checks["dataset_and_neural_roots_match_config"] = (
        dataset_root is not None and dataset_root == neural_root == config_root
    )
    dataset_time = _parse_aware_iso_datetime(dataset_report.get("audited_at_utc"))
    neural_time = _parse_aware_iso_datetime(neural_report.get("audited_at_utc"))
    checks["dataset_and_neural_audits_within_30_minutes"] = (
        dataset_time is not None
        and neural_time is not None
        and abs((dataset_time - neural_time).total_seconds())
        <= MAX_AUDIT_TIME_DIFFERENCE_SECONDS
    )
    header_paths_match_edf_files = dataset_header_paths == neural_edf_paths
    header_paths_correspond = (
        neural_root is not None
        and len(recordings) == EXPECTED_RECORDING_COUNT
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("edf_file"), str)
            and isinstance(item.get("edf_header"), Mapping)
            and _normalize_remote_root(item["edf_header"].get("path"))
            == (PurePosixPath(neural_root) / item["edf_file"]).as_posix()
            for item in recordings
        )
    )
    checks["dataset_and_neural_edf_paths_match"] = (
        header_paths_match_edf_files and header_paths_correspond
    )
    checks["neural_sampling_512_or_1024_and_sidecar_match"] = (
        neural_report.get("sampling_rate_hz_values") == [512.0, 1024.0]
        and neural_report.get("power_line_frequency_hz_values") == [60.0]
        and all(
            isinstance(item, Mapping)
            and item.get("sampling_rate_hz") in (512.0, 1024.0)
            and item.get("power_line_frequency_hz") == 60.0
            and item.get("edf_header_sampling_rate_matches_sidecar") is True
            for item in recordings
        )
    )
    checks["neural_events_within_all_edf_timelines"] = (
        len(recordings) == EXPECTED_RECORDING_COUNT
        and all(
            isinstance(item, Mapping)
            and item.get("events_within_recording") is True
            and item.get("events_within_edf_timeline") is True
            for item in recordings
        )
    )
    checks["neural_all_analysis_eligible_names_exist_in_edf"] = (
        len(recordings) == EXPECTED_RECORDING_COUNT
        and all(
            isinstance(item, Mapping)
            and item.get("analysis_eligible_neural_channels_missing_from_edf") == []
            for item in recordings
        )
    )
    eligible_total = 0
    c_prefix_total = 0
    channel_identity_valid = len(recordings) == EXPECTED_RECORDING_COUNT
    for item in recordings:
        channels = item.get("channels") if isinstance(item, Mapping) else None
        header = item.get("edf_header") if isinstance(item, Mapping) else None
        if not isinstance(channels, Mapping) or not isinstance(header, Mapping):
            channel_identity_valid = False
            continue
        eligible_count = channels.get("analysis_eligible_neural_channel_count")
        eligible_names = channels.get("analysis_eligible_neural_names")
        c_prefix_count = channels.get("c_prefix_exclusion_count")
        c_prefix_names = channels.get("c_prefix_names")
        header_names = header.get("channel_names")
        if (
            not isinstance(eligible_count, int)
            or isinstance(eligible_count, bool)
            or eligible_count <= 0
            or not isinstance(eligible_names, list)
            or len(eligible_names) != eligible_count
            or any(not isinstance(name, str) or not name for name in eligible_names)
            or len(set(eligible_names)) != eligible_count
            or not isinstance(c_prefix_count, int)
            or isinstance(c_prefix_count, bool)
            or c_prefix_count < 0
            or not isinstance(c_prefix_names, list)
            or len(c_prefix_names) != c_prefix_count
            or any(not isinstance(name, str) or not name for name in c_prefix_names)
            or not isinstance(header_names, list)
            or any(not isinstance(name, str) or not name for name in header_names)
            or not set(eligible_names).issubset(set(header_names))
        ):
            channel_identity_valid = False
            continue
        eligible_total += eligible_count
        c_prefix_total += c_prefix_count
    checks["neural_channel_selection_identity"] = (
        channel_identity_valid
        and eligible_total == EXPECTED_ELIGIBLE_CHANNEL_COUNT
        and c_prefix_total == EXPECTED_C_PREFIX_EXCLUSION_COUNT
        and neural_report.get("analysis_eligible_neural_channel_count")
        == EXPECTED_ELIGIBLE_CHANNEL_COUNT
        and neural_report.get("c_prefix_exclusion_count")
        == EXPECTED_C_PREFIX_EXCLUSION_COUNT
    )
    reference = _get(neural_report, "ieeg_reference_audit")
    reference_values = (
        reference.get("recording_values", []) if isinstance(reference, Mapping) else []
    )
    if not isinstance(reference_values, list):
        reference_values = []
    checks["neural_reference_11_of_11_as_recorded_scalp"] = (
        isinstance(reference, Mapping)
        and reference.get("status") == "PASS"
        and reference.get("expected_value") == "scalp electrode, not included with data"
        and reference.get("expected_recording_count") == EXPECTED_RECORDING_COUNT
        and reference.get("observed_recording_count") == EXPECTED_RECORDING_COUNT
        and reference.get("observed_unique_values")
        == ["scalp electrode, not included with data"]
        and reference.get("mismatch_recording_ids") == []
        and reference.get("primary_policy") == "AS_RECORDED_SCALP_REFERENCE"
        and len(reference_values) == EXPECTED_RECORDING_COUNT
        and all(
            isinstance(item, Mapping) and item.get("matches_frozen_value") is True
            for item in reference_values
        )
    )
    bids = _get(neural_report, "bids_layout")
    checks["neural_pybids_validated_layout"] = (
        isinstance(bids, Mapping)
        and bids.get("status") == "PASS"
        and bids.get("pybids_validate") is True
        and isinstance(bids.get("indexed_file_count"), int)
        and bids.get("indexed_file_count", 0) > 0
    )
    spatial = _get(neural_report, "spatial_metadata")
    checks["neural_spatial_metadata_preserves_anatomy_limitation"] = (
        isinstance(spatial, Mapping)
        and spatial.get("standard_electrodes_tsv_count") == 0
        and spatial.get("standard_coordsystem_json_count") == 0
        and spatial.get("contact_ras_csv_count") == 9
        and spatial.get("status")
        == "NONSTANDARD_COORDINATES_WITHOUT_ANATOMICAL_REGION_LABELS"
        and isinstance(spatial.get("limitation"), str)
        and bool(spatial.get("limitation"))
    )
    target_gate = _get(neural_report, "neural_target_gate")
    checks["neural_method_and_extraction_state_matches_config"] = (
        isinstance(target_gate, Mapping)
        and target_gate.get("candidate")
        == "LINE_HARMONIC_EXCLUDED_MULTIBAND_HIGH_GAMMA_LOG_POWER"
        and target_gate.get("status") == "METHOD_FROZEN_AWAITING_EXECUTION_GATES"
        and target_gate.get("primary_reference_policy") == "AS_RECORDED_SCALP_REFERENCE"
        and target_gate.get("neural_extraction_allowed") is False
        and target_gate.get("status") == _get(config, "neural_target", "status")
        and target_gate.get("candidate") == _get(config, "neural_target", "name")
        and target_gate.get("primary_reference_policy")
        == _get(config, "neural_target", "primary_reference_policy")
        and target_gate.get("neural_extraction_allowed")
        is _get(config, "neural_target", "neural_extraction_allowed")
    )

    checks["split_report_identity_and_version"] = (
        split_report.get("report_schema_version") == "m6a-split-guard-v2"
        and split_report.get("task_id") == "M6A-PUBLIC-001"
        and split_report.get("dataset_id") == "ds004703"
        and split_report.get("dataset_version") == "1.1.0"
    )
    checks["split_319_rows_guard_pass"] = (
        split_report.get("status") == "PASS"
        and split_report.get("rows") == EXPECTED_SPLIT_ROWS
        and split_report.get("issues") == []
        and split_report.get("embargo_status") == "PRELIMINARY_MINIMUM_ONLY"
        and split_report.get("preliminary_minimum_embargo_seconds") == 2.0
        and split_report.get("final_embargo_status")
        == "PENDING_AUDIO_CONTEXT_MEASUREMENT_AND_GUARD_RERUN"
        and split_report.get("baseline_final") is False
    )
    checks["split_exact_counts_blocks_and_language"] = (
        split_report.get("split_counts") == EXPECTED_SPLIT_COUNTS
        and split_report.get("block_assignments") == EXPECTED_BLOCK_ASSIGNMENTS
        and split_report.get("language_counts") == EXPECTED_LANGUAGE_COUNTS
        and split_report.get("catalan_rows") == 0
    )

    checks["main_config_identity_and_semantic_gate"] = (
        config.get("schema_version") == "m6a-task-config-v1"
        and config.get("task_id") == "M6A-PUBLIC-001"
        and _config_gate_passes(config)
    )
    checks["config_anatomy_not_ready_and_region_not_estimable"] = (
        _get(config, "anatomy_mapping", "status") == "ANATOMY_MAPPING_NOT_READY"
        and _get(config, "anatomy_mapping", "region_summary_status") == "NOT_ESTIMABLE"
        and _get(config, "anatomy_mapping", "contact_name_inference_allowed") is False
        and _get(config, "anatomy_mapping", "standard_electrodes_tsv_count") == 0
        and _get(config, "anatomy_mapping", "standard_coordsystem_json_count") == 0
        and _get(config, "anatomy_mapping", "nonstandard_contact_ras_csv_count") == 9
        and _get(config, "anatomy_mapping", "contact_ras_has_region_labels") is False
        and isinstance(spatial, Mapping)
        and spatial.get("standard_electrodes_tsv_count")
        == _get(config, "anatomy_mapping", "standard_electrodes_tsv_count")
        and spatial.get("standard_coordsystem_json_count")
        == _get(config, "anatomy_mapping", "standard_coordsystem_json_count")
        and spatial.get("contact_ras_csv_count")
        == _get(config, "anatomy_mapping", "nonstandard_contact_ras_csv_count")
        and _get(config, "baseline", "gated_metrics")
        == [
            {
                "name": "region_summary",
                "status": "NOT_ESTIMABLE",
                "gate": "ANATOMY_MAPPING_NOT_READY",
            }
        ]
    )
    checks["config_method_frozen_but_neural_extraction_blocked"] = (
        _get(config, "neural_target", "status")
        == "METHOD_FROZEN_AWAITING_EXECUTION_GATES"
        and _get(config, "neural_target", "resolution_status") == "METHOD_FROZEN"
        and _get(config, "neural_target", "method_coordinator_review") == "ACCEPT"
        and _get(config, "neural_target", "neural_extraction_allowed") is False
    )
    checks["config_baseline_not_final"] = (
        _get(config, "split", "baseline_final") is False
        and _get(config, "split", "split_status")
        in {
            "PRELIMINARY_NOT_BASELINE_FINAL",
            "FINAL_EMBARGO_CANDIDATE_NOT_BASELINE_FINAL",
        }
        and _get(config, "split", "final_embargo_seconds") is None
    )

    failed_checks = [name for name, passed in checks.items() if passed is not True]
    status = CANDIDATE_STATUS if not failed_checks else "FAIL"
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "status": status,
        "g2_pass_claimed": False,
        "candidate_contains_raw_data": False,
        "required_checks": checks,
        "failed_checks": failed_checks,
    }


def pending_report(missing_inputs: list[str]) -> dict[str, Any]:
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "task_id": "M6A-PUBLIC-001",
        "status": "PENDING",
        "g2_pass_claimed": False,
        "candidate_contains_raw_data": False,
        "required_checks": {},
        "failed_checks": [f"missing_input:{name}" for name in missing_inputs],
    }


def _write_output(path: Path | None, report: dict[str, Any]) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed M6A-PUBLIC-001 G2 promotion gate.")
    parser.add_argument("--dataset-report", type=Path, required=True)
    parser.add_argument("--neural-report", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    paths = {
        "dataset_report": args.dataset_report,
        "neural_report": args.neural_report,
        "split_report": args.split_report,
        "config": args.config,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        report = pending_report(missing)
        _write_output(args.output, report)
        return 1

    try:
        payloads = {name: load_strict_json(path) for name, path in paths.items()}
        report = validate_g2_promotion(
            payloads["dataset_report"],
            payloads["neural_report"],
            payloads["split_report"],
            payloads["config"],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "report_schema_version": REPORT_SCHEMA_VERSION,
            "task_id": "M6A-PUBLIC-001",
            "status": "FAIL",
            "g2_pass_claimed": False,
            "candidate_contains_raw_data": False,
            "required_checks": {},
            "failed_checks": [f"input_parse_error:{type(exc).__name__}:{exc}"],
        }
    _write_output(args.output, report)
    return 0 if report["status"] == CANDIDATE_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
