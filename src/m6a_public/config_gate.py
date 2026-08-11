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
    if not {"stimulus_id", "recording_id"}.issubset(required_groups):
        errors.append("split must guard stimulus_id and recording_id")
    if split.get("temporal_embargo_seconds", 0) <= 0:
        errors.append("temporal embargo must be positive")
    if set(split.get("allowed_splits", [])) != {"train", "validation", "test"}:
        errors.append("allowed_splits must be train/validation/test")

    baseline = config.get("baseline", {})
    if baseline.get("primary") != "layerwise_ridge_encoding":
        errors.append("primary baseline must be layerwise_ridge_encoding")

    nulls = config.get("nulls", {})
    if nulls.get("smoke_permutations", 0) < 20:
        errors.append("smoke_permutations must be at least 20")
    if nulls.get("formal_permutations", 0) < 1000:
        errors.append("formal_permutations must be at least 1000")

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
