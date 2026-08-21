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


def check_perturbation_supplement(root: Path, errors: list[str], warnings: list[str]) -> dict[str, int]:
    """Validate the additive 10/20/50-ms package without hashing any file."""

    reports = root / "reports"
    smoke_path = reports / "m6a_public_003_perturbation_smoke.json"
    smoke_parameter_path = reports / "m6a_public_003_perturbation_parameter_smoke.csv"
    smoke_rf_path = reports / "m6a_public_003_perturbation_rf_smoke.csv"
    for path in [smoke_path, smoke_parameter_path, smoke_rf_path]:
        if not path.exists():
            errors.append(f"missing perturbation smoke artifact: {path.name}")
    if smoke_path.exists():
        smoke: dict[str, Any] = json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke.get("status") != "PASS":
            errors.append("perturbation parameter/RF smoke is not PASS")
        if smoke.get("parameter_rows") != 15 or smoke.get("rf_rows") != 99:
            errors.append("unexpected perturbation parameter/RF smoke row counts")
        known_rf = smoke.get("known_rf_checks", {})
        if known_rf.get("rf_stride_coupled_samples") != known_rf.get("rf_dilation_decoupled_samples"):
            errors.append("supplement RF smoke pair has unequal final RF")
        if known_rf.get("rf_stride_coupled_resolution_ms") != 4.0 or known_rf.get("rf_dilation_decoupled_resolution_ms") != 1.0:
            errors.append("supplement RF smoke pair has unexpected final resolutions")
    smoke_parameter_rows = read_csv(smoke_parameter_path) if smoke_parameter_path.exists() else []
    smoke_rf_rows = read_csv(smoke_rf_path) if smoke_rf_path.exists() else []
    if len(smoke_parameter_rows) != 15:
        errors.append(f"expected 15 perturbation parameter smoke rows, observed {len(smoke_parameter_rows)}")
    if len(smoke_rf_rows) != 99:
        errors.append(f"expected 99 perturbation RF smoke rows, observed {len(smoke_rf_rows)}")
    required_csvs = {
        "m6a_public_003_perturbation_metrics_by_seed.csv": ["variant", "seed", "magnitude_ms", "probe", "metric", "value"],
        "m6a_public_003_perturbation_summary.csv": ["variant", "magnitude_ms", "probe", "metric", "mean", "std", "n_seeds"],
        "m6a_public_003_perturbation_run_status.csv": ["variant", "family", "seed", "status", "failure", "wall_seconds"],
    }
    rows = {name: check_csv(reports / name, fields, errors) for name, fields in required_csvs.items()}
    metric_rows = rows["m6a_public_003_perturbation_metrics_by_seed.csv"]
    summary_rows = rows["m6a_public_003_perturbation_summary.csv"]
    status_rows = rows["m6a_public_003_perturbation_run_status.csv"]
    variants = {
        "early_downsample", "late_downsample", "uniform_local", "exponential_growth",
        "delayed_growth", "parallel_multiscale", "rf_stride_coupled", "rf_dilation_decoupled",
        "kernel_3", "kernel_7", "kernel_15", "kernel_31", "event_baseline", "explicit_change",
        "ordinary_second_branch",
    }
    seeds = {"11", "22", "33"}
    magnitudes = {10.0, 20.0, 50.0}
    metrics = {
        "localization_onset_mae_ms", "localization_shift_recovery_mae_ms",
        "discrimination_balanced_accuracy", "generalization_jitter_mae_ms",
        "generalization_relative_jitter_error",
    }
    probes = {"localization", "discrimination", "generalization"}
    expected_metric_keys = {
        (variant, seed, magnitude, metric)
        for variant in variants
        for seed in seeds
        for magnitude in magnitudes
        for metric in metrics
    }
    observed_metric_keys: set[tuple[str, str, float, str]] = set()
    if len(metric_rows) != 675:
        errors.append(f"expected 675 perturbation metric rows, observed {len(metric_rows)}")
    for row in metric_rows:
        variant = row.get("variant", "")
        seed = row.get("seed", "")
        metric = row.get("metric", "")
        try:
            magnitude = float(row.get("magnitude_ms", "nan"))
        except ValueError:
            magnitude = float("nan")
        observed_metric_keys.add((variant, seed, magnitude, metric))
        if variant not in variants or seed not in seeds or magnitude not in magnitudes or metric not in metrics:
            errors.append(f"unexpected perturbation metric key: {variant} {seed} {magnitude} {metric}")
        if row.get("probe") not in probes:
            errors.append(f"invalid perturbation probe: {row.get('probe')}")
        if not finite(row.get("value", "")):
            errors.append(f"non-finite perturbation metric: {variant} {magnitude} {metric}")
    if observed_metric_keys != expected_metric_keys:
        errors.append("perturbation metric coverage is not exactly 15 variants x 3 seeds x 3 magnitudes x 5 metrics")

    expected_summary_keys = {
        (variant, magnitude, metric)
        for variant in variants
        for magnitude in magnitudes
        for metric in metrics
    }
    observed_summary_keys: set[tuple[str, float, str]] = set()
    if len(summary_rows) != 225:
        errors.append(f"expected 225 perturbation summary rows, observed {len(summary_rows)}")
    for row in summary_rows:
        variant = row.get("variant", "")
        metric = row.get("metric", "")
        try:
            magnitude = float(row.get("magnitude_ms", "nan"))
        except ValueError:
            magnitude = float("nan")
        observed_summary_keys.add((variant, magnitude, metric))
        if row.get("probe") not in probes or variant not in variants or magnitude not in magnitudes or metric not in metrics:
            errors.append(f"unexpected perturbation summary key: {variant} {magnitude} {metric}")
        if row.get("n_seeds") != "3":
            errors.append(f"perturbation summary is not three-seed: {variant} {magnitude} {metric}")
        for field in ["mean", "std", "n_seeds"]:
            if not finite(row.get(field, "")):
                errors.append(f"non-finite perturbation summary field {field}: {variant} {magnitude} {metric}")
    if observed_summary_keys != expected_summary_keys:
        errors.append("perturbation summary coverage is not exactly 15 variants x 3 magnitudes x 5 metrics")

    if len(status_rows) != 45:
        errors.append(f"expected 45 perturbation status rows, observed {len(status_rows)}")
    if status_rows and any(row.get("status") != "PASS" for row in status_rows):
        errors.append("perturbation status contains non-PASS row; failure evidence must remain visible")
    for row in status_rows:
        if not finite(row.get("wall_seconds", "")):
            errors.append(f"non-finite perturbation wall time: {row.get('variant')} {row.get('seed')}")

    manifest_path = reports / "m6a_public_003_perturbation_manifest.json"
    runtime_path = reports / "m6a_public_003_perturbation_runtime.json"
    for path in [manifest_path, runtime_path]:
        if not path.exists():
            errors.append(f"missing perturbation JSON artifact: {path.name}")
    if manifest_path.exists():
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("magnitudes_ms") != [10.0, 20.0, 50.0]:
            errors.append("manifest magnitudes are not exactly [10.0, 20.0, 50.0]")
        if manifest.get("magnitude_axis_is_explicit") is not True:
            errors.append("manifest does not record an explicit magnitude axis")
        if manifest.get("supplement_is_unseen_extrapolation") is not True:
            errors.append("manifest does not record unseen/extrapolative relation")
        if manifest.get("formal_results_preserved") is not True:
            errors.append("manifest does not record formal result preservation")
        if manifest.get("no_pretrained_model") is not True or manifest.get("patient_stn_data_read") is not False:
            errors.append("perturbation manifest violates model/data boundary")
    if runtime_path.exists():
        runtime: dict[str, Any] = json.loads(runtime_path.read_text(encoding="utf-8"))
        if runtime.get("formal_results_preserved") is not True:
            errors.append("perturbation runtime does not record formal result preservation")
        if runtime.get("no_pretrained_model") is not True or runtime.get("patient_stn_data_read") is not False:
            errors.append("perturbation runtime violates model/data boundary")

    figure_dir = reports / "figures"
    stem = "m6a_public_003_performance_vs_perturbation_magnitude"
    for suffix in [".png", ".svg", ".pdf"]:
        if not (figure_dir / f"{stem}{suffix}").exists():
            errors.append(f"missing perturbation figure: {stem}{suffix}")
    for name in ["TEMPORAL_BENCHMARK_DESIGN.md", "DOWNSAMPLING_ABLATION.md", "RECEPTIVE_FIELD_ABLATION.md", "EVENT_BRANCH_ABLATION.md", "M6A-PUBLIC-003_SUMMARY.md"]:
        if not (reports / name).exists():
            errors.append(f"missing updated perturbation report: {name}")
    forbidden_large_extensions = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
    forbidden_files = [path.name for path in root.rglob("*") if path.is_file() and path.suffix.lower() in forbidden_large_extensions]
    if forbidden_files:
        errors.append(f"large model/checkpoint files present in perturbation package: {forbidden_files}")
    return {
        "metric_rows": len(metric_rows),
        "summary_rows": len(summary_rows),
        "status_rows": len(status_rows),
        "parameter_smoke_rows": len(smoke_parameter_rows),
        "rf_smoke_rows": len(smoke_rf_rows),
        "figure_stems": 1,
        "large_checkpoint_files": len(forbidden_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--perturbation-root", type=Path, default=None)
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
    perturbation_checks: dict[str, int] = {}
    if args.perturbation_root is not None:
        perturbation_checks = check_perturbation_supplement(args.perturbation_root, errors, warnings)
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
            "perturbation_supplement": perturbation_checks,
        },
        "integrity_policy": "NON_HASH_AUDIT",
    }
    output = reports / "m6a_public_003_verify.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
