from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FORBIDDEN_FIELD_NAMES = {
    "sha",
    "sha1",
    "sha256",
    "sha512",
    "md5",
    "checksum",
    "etag",
    "file_hash",
    "object_hash",
    "commit_hash",
    "commit_id",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def find_forbidden_fields(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if (
                normalized in FORBIDDEN_FIELD_NAMES
                or normalized.endswith("_hash")
                or normalized.startswith("hash_")
                or normalized.endswith("_checksum")
            ):
                findings.append(f"{path}.{key}")
            findings.extend(find_forbidden_fields(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(find_forbidden_fields(child, f"{path}[{index}]"))
    return findings


def validate_task_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("task_id") != "M6A-PUBLIC-001":
        errors.append("task_id must be M6A-PUBLIC-001")
    if config.get("status") != "ACTIVE_EXECUTION":
        errors.append("status must be ACTIVE_EXECUTION")

    integrity = config.get("integrity_policy", {})
    if integrity.get("mode") != "NON_HASH_AUDIT":
        errors.append("integrity_policy.mode must be NON_HASH_AUDIT")
    if integrity.get("cryptographic_integrity_claim") is not False:
        errors.append("cryptographic_integrity_claim must be false")

    dataset = config.get("dataset", {})
    if dataset.get("dataset_id") != "ds004703":
        errors.append("dataset_id must be ds004703")
    if dataset.get("version") != "1.1.0":
        errors.append("dataset version must be 1.1.0")
    license_status = dataset.get("license_status")
    if dataset.get("download_allowed") and license_status != "ACCEPTED_WITH_STRICTER_README_BOUNDARY":
        errors.append("dataset download requires the stricter README boundary")
    boundaries = set(dataset.get("strict_use_boundaries", []))
    required_boundaries = {
        "NONCOMMERCIAL_ACADEMIC_RESEARCH_ONLY",
        "NO_REIDENTIFICATION",
        "NO_RAW_DATA_EXPORT_FROM_2203",
    }
    if not required_boundaries.issubset(boundaries):
        errors.append("dataset strict_use_boundaries are incomplete")
    if dataset.get("interactive_terms_stop") is not True:
        errors.append("interactive_terms_stop must be true")

    model = config.get("model", {})
    if model.get("model_id") != "facebook/wav2vec2-base":
        errors.append("first model must be facebook/wav2vec2-base")
    if model.get("trainable") is not False:
        errors.append("first model must remain frozen")
    if model.get("sampling_rate_hz") != 16000:
        errors.append("wav2vec2 input sampling rate must be 16000 Hz")

    split = config.get("split", {})
    required_groups = set(split.get("required_group_keys", []))
    if required_groups != {"stimulus_id", "block_id"}:
        errors.append("split must guard stimulus_id and block_id")
    if "language" not in set(split.get("stratification_keys", [])):
        errors.append("split must carry language as an explicit stratification key")
    if split.get("language_policy") != "EXPLICIT_MANIFEST_AND_SPLIT_COVERAGE_AUDIT":
        errors.append("split language policy must audit explicit split coverage")
    if split.get("recording_policy") != "MAY_SPAN_SPLITS_WITH_NONOVERLAPPING_PASSAGE_WINDOWS_AND_TEMPORAL_EMBARGO":
        errors.append("split recording policy must preserve within-recording temporal isolation")
    if split.get("original_recording_grouping_status") != "INFEASIBLE_SINGLE_CONNECTED_COMPONENT":
        errors.append("original recording grouping no-go must remain recorded")
    expected_block_assignments = {
        "block-01": "train",
        "block-02": "train",
        "block-03": "validation",
        "block-04": "test",
        "block-05": "train",
        "block-06": "train",
    }
    if split.get("block_assignments") != expected_block_assignments:
        errors.append("block assignments must match the reviewed deterministic ratio optimum")
    if split.get("assignment_method") != "DETERMINISTIC_GROUP_SIZE_RATIO_OPTIMIZATION":
        errors.append("split assignment must use deterministic group-size ratio optimization")
    if split.get("preliminary_minimum_embargo_seconds", 0) != 2.0:
        errors.append("preliminary minimum embargo must remain 2 seconds")
    if split.get("split_status") != "PRELIMINARY_NOT_BASELINE_FINAL":
        errors.append("split must remain preliminary until final embargo is measured and guarded")
    if split.get("final_embargo_seconds") is not None:
        errors.append("final embargo must remain unset before G3 measurement")
    if split.get("final_embargo_status") != "PENDING_G3_MEASUREMENT_AND_GUARD_RERUN":
        errors.append("final embargo must require G3 measurement and guard rerun")
    if split.get("primary_generalization_scope") != "WITHIN_SUBJECT_UNSEEN_STIMULUS_AND_BLOCK_ONLY":
        errors.append("primary generalization scope must remain within-subject unseen stimulus/block only")
    for claim_key in (
        "subject_heldout_claim_allowed",
        "speaker_heldout_claim_allowed",
        "cross_language_claim_allowed",
    ):
        if split.get(claim_key) is not False:
            errors.append(f"{claim_key} must be false at the preliminary split gate")
    if split.get("secondary_subject_generalization") is not False:
        errors.append("secondary subject generalization is not supported by the current split")
    if set(split.get("allowed_splits", [])) != {"train", "validation", "test"}:
        errors.append("allowed_splits must be train/validation/test")

    neural_target = config.get("neural_target", {})
    if neural_target.get("status") != "REDESIGN_REQUIRED_BEFORE_G3":
        errors.append("neural target must record the G2 high-gamma redesign gate")
    if neural_target.get("observed_power_line_frequency_hz") != 60:
        errors.append("neural target must record the observed 60 Hz line frequency")
    if 120 not in neural_target.get("line_harmonics_inside_candidate_band_hz", []):
        errors.append("neural target must record the 120 Hz harmonic inside 70-150 Hz")
    if neural_target.get("neural_extraction_allowed") is not False:
        errors.append("neural extraction must remain blocked until target method is refrozen")
    if neural_target.get("resolution_status") != "PENDING_METHOD_FREEZE":
        errors.append("neural target resolution must remain pending method freeze")

    baseline = config.get("baseline", {})
    if baseline.get("primary") != "layerwise_ridge_encoding":
        errors.append("primary baseline must be layerwise_ridge_encoding")

    nulls = config.get("nulls", {})
    if nulls.get("smoke_permutations", 0) < 20:
        errors.append("smoke_permutations must be at least 20")
    if nulls.get("formal_permutations", 0) < 1000:
        errors.append("formal_permutations must be at least 1000")

    artifact = config.get("artifact", {})
    if artifact.get("internal_schema_path") != "schemas/m6a_public_internal_manifest.schema.json":
        errors.append("internal run manifest path is not frozen")
    if artifact.get("exchange_contract_status") != "REVISED_DRAFT_AWAITING_CONSUMER_REVIEW":
        errors.append("exchange contract must remain REVISED_DRAFT_AWAITING_CONSUMER_REVIEW")
    if artifact.get("frozen_m6a_artifact_exists") is not False:
        errors.append("no frozen M6A artifact exists at G0-G2")

    forbidden = find_forbidden_fields(config)
    if forbidden:
        errors.append("forbidden integrity fields: " + ", ".join(forbidden))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the M6A task gate without hashes.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    errors = validate_task_config(load_json(args.config))
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
