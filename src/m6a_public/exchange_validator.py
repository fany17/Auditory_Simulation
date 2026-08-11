from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from m6a_public.config_gate import find_forbidden_fields, load_json


DRAFT_STATUS = "DRAFT_PROPOSED_BY_M6A"
REVISED_STATUS = "REVISED_DRAFT_AWAITING_CONSUMER_REVIEW"
CANDIDATE_STATUS = "CANDIDATE_FOR_CROSS_TEST"
ALLOWED_TRANSITIONS = {
    (DRAFT_STATUS, DRAFT_STATUS),
    (DRAFT_STATUS, REVISED_STATUS),
    (REVISED_STATUS, REVISED_STATUS),
    (REVISED_STATUS, CANDIDATE_STATUS),
    (CANDIDATE_STATUS, CANDIDATE_STATUS),
}
SINGLETON_CANDIDATE_ROLES = {
    "METHOD_ENTRYPOINT",
    "RUNTIME_SPEC",
    "EXTRACTION_CONFIG_SCHEMA",
    "EXTRACTION_OUTPUT_SCHEMA",
    "CANARY_INPUT",
    "CANARY_EXPECTED_OUTPUT",
}


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value or ":" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and value == path.as_posix() and ".." not in path.parts and "." not in path.parts


def _schema_errors(manifest: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors: list[str] = []
    for error in sorted(validator.iter_errors(manifest), key=lambda item: list(item.absolute_path)):
        location = "$"
        for part in error.absolute_path:
            location += f"[{part}]" if isinstance(part, int) else f".{part}"
        errors.append(f"schema {location}: {error.message}")
    return errors


def validate_exchange_manifest(
    manifest: dict[str, Any],
    schema: dict[str, Any],
    bundle_root: str | Path | None = None,
) -> list[str]:
    errors = _schema_errors(manifest, schema)
    if errors:
        return errors

    forbidden = find_forbidden_fields(manifest)
    if forbidden:
        errors.append("forbidden integrity fields: " + ", ".join(forbidden))

    identity = manifest["identity"]
    status = identity["release_status"]
    history = [item["status"] for item in identity["status_history"]]
    if history[0] != DRAFT_STATUS:
        errors.append("status history must start with DRAFT_PROPOSED_BY_M6A")
    if history[-1] != status:
        errors.append("status history must end with identity.release_status")
    for previous, current in zip(history, history[1:]):
        if (previous, current) not in ALLOWED_TRANSITIONS:
            errors.append(f"illegal release transition: {previous} -> {current}")

    inventory = manifest["file_inventory"]
    inventory_paths = [item["path"] for item in inventory]
    if len(inventory_paths) != len(set(inventory_paths)):
        errors.append("file_inventory paths must be unique")
    for path in inventory_paths:
        if not _safe_relative_path(path):
            errors.append(f"unsafe inventory path: {path}")
    inventory_by_path = {item["path"]: item for item in inventory}
    role_counts = Counter(item["role"] for item in inventory)

    model_role_counts = Counter(item["role"] for item in manifest["model"]["model_files"])
    for required_model_role in ("MODEL_CONFIG", "MODEL_WEIGHTS"):
        if model_role_counts[required_model_role] < 1:
            errors.append(f"model_files requires role {required_model_role}")
    if "main" in manifest["model"]["revision_label"].lower() and manifest["model"]["revision_immutable"]:
        errors.append("a main revision label cannot be declared immutable")
    model_inventory_roles = {
        "MODEL_CONFIG": "MODEL_CONFIG_SOURCE",
        "MODEL_PREPROCESSOR": "MODEL_CONFIG_SOURCE",
        "MODEL_WEIGHTS": "MODEL_WEIGHT_SOURCE",
    }
    for model_file in manifest["model"]["model_files"]:
        path = model_file["inventory_path"]
        if not _safe_relative_path(path):
            errors.append(f"unsafe model inventory path: {path}")
            continue
        item = inventory_by_path.get(path)
        if item is None:
            errors.append(f"model file is not listed in file_inventory: {path}")
            continue
        if item["role"] != model_inventory_roles[model_file["role"]]:
            errors.append(f"model file has wrong inventory role: {path}")
        if item["bytes"] != model_file["bytes"]:
            errors.append(f"model file byte size disagrees with inventory: {path}")
        if item["modified_at_utc"] != model_file["modified_at_utc"]:
            errors.append(f"model file timestamp disagrees with inventory: {path}")
        if item["availability"] != "REMOTE_REFERENCE_ONLY":
            errors.append(f"model cache file must remain REMOTE_REFERENCE_ONLY: {path}")

    path_fields = {
        "extraction_spec.entrypoint_file": manifest["extraction_spec"]["entrypoint_file"],
        "extraction_spec.config_schema_file": manifest["extraction_spec"]["config_schema_file"],
        "extraction_spec.output_schema_file": manifest["extraction_spec"]["output_schema_file"],
        "extraction_spec.runtime_spec_file": manifest["extraction_spec"]["runtime_spec_file"],
        "method_package.entrypoint_file": manifest["method_package"]["entrypoint_file"],
        "method_package.runtime_spec_file": manifest["method_package"]["runtime_spec_file"],
        "canary_fixture.input_file": manifest["canary_fixture"]["input_file"],
        "canary_fixture.expected_output_file": manifest["canary_fixture"]["expected_output_file"],
    }
    expected_roles = {
        "extraction_spec.entrypoint_file": "METHOD_ENTRYPOINT",
        "extraction_spec.config_schema_file": "EXTRACTION_CONFIG_SCHEMA",
        "extraction_spec.output_schema_file": "EXTRACTION_OUTPUT_SCHEMA",
        "extraction_spec.runtime_spec_file": "RUNTIME_SPEC",
        "method_package.entrypoint_file": "METHOD_ENTRYPOINT",
        "method_package.runtime_spec_file": "RUNTIME_SPEC",
        "canary_fixture.input_file": "CANARY_INPUT",
        "canary_fixture.expected_output_file": "CANARY_EXPECTED_OUTPUT",
    }
    for field, path in path_fields.items():
        if not _safe_relative_path(path):
            errors.append(f"unsafe linked path: {field}={path}")
            continue
        item = inventory_by_path.get(path)
        if item is None:
            errors.append(f"{field} is not listed in file_inventory: {path}")
        elif item["role"] != expected_roles[field]:
            errors.append(f"{field} must have inventory role {expected_roles[field]}")

    if manifest["method_package"]["entrypoint_file"] != manifest["extraction_spec"]["entrypoint_file"]:
        errors.append("method and extraction entrypoint files must match")
    if manifest["method_package"]["runtime_spec_file"] != manifest["extraction_spec"]["runtime_spec_file"]:
        errors.append("method and extraction runtime spec files must match")

    for path in manifest["method_package"]["files"]:
        if not _safe_relative_path(path):
            errors.append(f"unsafe method package path: {path}")
        elif path not in inventory_by_path:
            errors.append(f"method package file is not listed in file_inventory: {path}")

    layer_inventory = manifest["layer_inventory"]
    layer_keys = [item["layer_key"] for item in layer_inventory]
    ordinals = [item["ordinal"] for item in layer_inventory]
    if len(layer_keys) != len(set(layer_keys)):
        errors.append("layer_key values must be unique")
    if len(ordinals) != len(set(ordinals)):
        errors.append("layer ordinal values must be unique")
    if ordinals != list(range(len(ordinals))):
        errors.append("layer ordinals must be ordered and contiguous from zero")

    layer_by_key = {item["layer_key"]: item for item in layer_inventory}
    for transform in manifest["transferable_transforms"]:
        path = transform["artifact_file"]
        if transform["input_layer"] not in layer_by_key:
            errors.append(f"transform input layer is not declared: {transform['input_layer']}")
        item = inventory_by_path.get(path)
        if item is None:
            errors.append(f"transform artifact is not listed in file_inventory: {path}")
        elif item["role"] != "TRANSFORM_ARTIFACT":
            errors.append(f"transform artifact must have inventory role TRANSFORM_ARTIFACT: {path}")

    canary = manifest["canary_fixture"]
    if canary["input_sample_rate_hz"] != manifest["audio_preprocessing"]["input_rate_hz"]:
        errors.append("canary input sample rate must match audio preprocessing")
    expected_order = canary["expected_layer_order"]
    expected_layers = canary["expected_layers"]
    expected_keys = [item["layer_key"] for item in expected_layers]
    if expected_order != layer_keys:
        errors.append("canary expected layer order must exactly match layer_inventory order")
    if expected_keys != expected_order:
        errors.append("canary expected_layers must exactly follow expected_layer_order")
    if len(expected_keys) != len(set(expected_keys)):
        errors.append("canary expected layer keys must be unique")
    frame_counts: set[int] = set()
    for expected in expected_layers:
        layer = layer_by_key.get(expected["layer_key"])
        if layer is None:
            errors.append(f"canary layer is not declared: {expected['layer_key']}")
            continue
        if expected["shape"] != [expected["frame_count"], expected["feature_dim"]]:
            errors.append(f"canary shape fields disagree for layer {expected['layer_key']}")
        if expected["feature_dim"] != layer["feature_dim"]:
            errors.append(f"canary feature_dim disagrees for layer {expected['layer_key']}")
        if expected["dtype"] != layer["dtype"]:
            errors.append(f"canary dtype disagrees for layer {expected['layer_key']}")
        frame_counts.add(expected["frame_count"])
    if len(frame_counts) != 1:
        errors.append("canary layers must use one shared frame count")
    elif canary["frame_time"]["shape"] != [next(iter(frame_counts))]:
        errors.append("canary frame-time shape must match the shared frame count")

    if status == CANDIDATE_STATUS:
        for role in SINGLETON_CANDIDATE_ROLES:
            if role_counts[role] != 1:
                errors.append(f"candidate requires exactly one inventory item with role {role}")
        if manifest["validation"]["evidence_profile"] != "CANDIDATE_EVIDENCE":
            errors.append("candidate requires CANDIDATE_EVIDENCE")
        if bundle_root is None:
            errors.append("candidate validation requires bundle_root")
        else:
            root = Path(bundle_root).resolve()
            for item in inventory:
                if item["availability"] != "INCLUDED_LOCAL":
                    continue
                target = (root / Path(*PurePosixPath(item["path"]).parts)).resolve()
                try:
                    target.relative_to(root)
                except ValueError:
                    errors.append(f"inventory path escapes bundle root: {item['path']}")
                    continue
                if not target.is_file():
                    errors.append(f"candidate local file is missing: {item['path']}")
                elif target.stat().st_size != item["bytes"]:
                    errors.append(f"candidate local file byte size differs: {item['path']}")
                else:
                    try:
                        with target.open("rb") as handle:
                            handle.read(1)
                    except OSError as exc:
                        errors.append(f"candidate local file is not readable: {item['path']} ({type(exc).__name__})")
        for path in set(path_fields.values()) | set(manifest["method_package"]["files"]):
            item = inventory_by_path.get(path)
            if item is not None and item["availability"] != "INCLUDED_LOCAL":
                errors.append(f"candidate cross-test file must be INCLUDED_LOCAL: {path}")
        for transform in manifest["transferable_transforms"]:
            item = inventory_by_path.get(transform["artifact_file"])
            if item is not None and item["availability"] != "INCLUDED_LOCAL":
                errors.append(f"candidate transform must be INCLUDED_LOCAL: {transform['artifact_file']}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed validator for the M6A to M6B draft exchange manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()
    errors = validate_exchange_manifest(load_json(args.manifest), load_json(args.schema), args.bundle_root)
    print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
