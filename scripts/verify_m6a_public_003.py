"""Independent lightweight verifier for M6A-PUBLIC-003 outputs.

This verifier checks schemas, finite numeric values, RF invariants, seed coverage,
required reports and figure presence.  It intentionally performs no hashing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


REQUIRED_REPORTS = [
    "STRUCTURAL_MODIFICATION_REGISTRY.md",
    "receptive_field_by_layer.csv",
    "TEMPORAL_BENCHMARK_DESIGN.md",
    "DOWNSAMPLING_ABLATION.md",
    "RECEPTIVE_FIELD_ABLATION.md",
    "EVENT_BRANCH_ABLATION.md",
    "M6A-PUBLIC-003_SUMMARY.md",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def check_csv(path: Path, required: Iterable[str], errors: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        errors.append(f"missing CSV: {path.name}")
        return []
    rows = read_csv(path)
    missing = set(required) - set(rows[0] if rows else [])
    if missing:
        errors.append(f"{path.name} missing fields: {sorted(missing)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    reports = root / "reports"
    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED_REPORTS:
        if not (reports / name).exists():
            errors.append(f"missing required report: {name}")
    required_csvs = {
        "m6a_public_003_model_parameters.csv": ["variant", "parameter_count", "relative_parameter_difference", "parameter_match_status"],
        "receptive_field_by_layer.csv": ["model", "layer", "kernel", "stride", "dilation", "theoretical_RF_samples", "theoretical_RF_ms", "output_time_resolution_ms"],
        "m6a_public_003_run_status.csv": ["variant", "seed", "status", "failure"],
        "m6a_public_003_metrics_by_seed.csv": ["variant", "seed", "split", "metric", "value"],
        "m6a_public_003_metrics_summary.csv": ["variant", "split", "metric", "mean", "std", "n_seeds"],
        "m6a_public_003_representation_metrics.csv": ["variant", "seed", "metric", "value"],
    }
    csv_rows = {name: check_csv(reports / name, fields, errors) for name, fields in required_csvs.items()}

    parameters = csv_rows["m6a_public_003_model_parameters.csv"]
    statuses = csv_rows["m6a_public_003_run_status.csv"]
    metrics = csv_rows["m6a_public_003_metrics_by_seed.csv"]
    summaries = csv_rows["m6a_public_003_metrics_summary.csv"]
    rf_rows = csv_rows["receptive_field_by_layer.csv"]
    representations = csv_rows["m6a_public_003_representation_metrics.csv"]

    if len(parameters) != 15:
        errors.append(f"expected 15 parameter rows, observed {len(parameters)}")
    if len(statuses) != 45:
        errors.append(f"expected 45 seed status rows, observed {len(statuses)}")
    if len(set(row.get("seed", "") for row in statuses)) != 3:
        errors.append("seed coverage is not exactly three seeds")
    if statuses and any(row.get("status") != "PASS" for row in statuses):
        errors.append("formal status contains non-PASS row")
    if len(representations) != 135:
        errors.append(f"expected 135 representation rows, observed {len(representations)}")

    for row in parameters:
        if not finite(row.get("parameter_count", "")) or int(float(row["parameter_count"])) <= 0:
            errors.append(f"invalid parameter count for {row.get('variant')}")
        if row.get("parameter_match_status") not in {"PASS", "CONFOUNDED"}:
            errors.append(f"invalid parameter match status for {row.get('variant')}")
    for row in metrics:
        if not finite(row.get("value", "")):
            errors.append(f"non-finite metric: {row.get('variant')} {row.get('metric')}")
    for row in representations:
        if not finite(row.get("value", "")):
            errors.append(f"non-finite representation metric: {row.get('variant')} {row.get('metric')}")
    for row in rf_rows:
        for field in ["kernel", "stride", "dilation", "theoretical_RF_samples", "theoretical_RF_ms", "output_time_resolution_ms"]:
            if not finite(row.get(field, "")):
                errors.append(f"non-finite RF field {field} in {row.get('model')} {row.get('layer')}")

    parameter_map = {row.get("variant"): row for row in parameters}
    coupled = parameter_map.get("rf_stride_coupled", {})
    decoupled = parameter_map.get("rf_dilation_decoupled", {})
    if coupled.get("final_theoretical_RF_samples") != decoupled.get("final_theoretical_RF_samples"):
        errors.append("RF/downsampling matched pair has unequal final RF")
    if coupled.get("final_output_time_resolution_ms") != "4.0" or decoupled.get("final_output_time_resolution_ms") != "1.0":
        errors.append("RF/downsampling matched pair has unexpected final resolutions")
    if parameter_map.get("kernel_31", {}).get("parameter_match_status") != "CONFOUNDED":
        warnings.append("kernel_31 was not retained as an explicit confounded comparison")

    figure_dir = reports / "figures"
    figure_stems = [
        "m6a_public_003_rf_and_resolution",
        "m6a_public_003_performance_by_perturbation",
        "m6a_public_003_matched_architecture_comparison",
        "m6a_public_003_representation_checks",
    ]
    for stem in figure_stems:
        for suffix in [".png", ".svg", ".pdf"]:
            if not (figure_dir / f"{stem}{suffix}").exists():
                errors.append(f"missing figure: {stem}{suffix}")

    forbidden_large_extensions = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    forbidden_files = [path.name for path in root.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_large_extensions]
    if forbidden_files:
        errors.append(f"large model/checkpoint files present: {forbidden_files}")

    manifest_path = reports / "m6a_public_003_run_manifest.json"
    runtime_path = reports / "m6a_public_003_runtime.json"
    for path in [manifest_path, runtime_path]:
        if not path.exists():
            errors.append(f"missing JSON artifact: {path.name}")
    if runtime_path.exists():
        runtime: dict[str, Any] = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("patient_stn_data_read") is not False:
            errors.append("runtime does not explicitly record patient_stn_data_read=false")
        if runtime.get("no_pretrained_model") is not True:
            errors.append("runtime does not explicitly record no_pretrained_model=true")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "required_reports": len(REQUIRED_REPORTS),
            "parameter_rows": len(parameters),
            "rf_rows": len(rf_rows),
            "status_rows": len(statuses),
            "metric_rows": len(metrics),
            "summary_rows": len(summaries),
            "representation_rows": len(representations),
            "figure_stems": len(figure_stems),
            "large_checkpoint_files": len(forbidden_files),
        },
        "integrity_policy": "NON_HASH_AUDIT",
    }
    output = reports / "m6a_public_003_verify.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
