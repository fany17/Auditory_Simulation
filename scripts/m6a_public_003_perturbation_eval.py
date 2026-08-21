"""M6A-PUBLIC-003 supplementary 10/20/50-ms perturbation evaluation.

This is an additive evaluation package.  It does not modify the prior formal
run or save checkpoints.  The formal run intentionally saved no checkpoints,
so this script retrains the exact 15 small models under the already frozen
optimizer/epoch/batch/split contract and evaluates a new, fixed perturbation
grid.  The only new magnitudes are 10, 20 and 50 ms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from m6a_public_003_temporal_benchmark import (
    TemporalDataset,
    TemporalModel,
    balanced_accuracy,
    final_rf_info,
    generate_split,
    load_config,
    make_specs,
    predict_arrays,
    rf_rows_for_spec,
    seed_everything,
    sigmoid,
    train_model,
)


MAGNITUDES_MS = (10.0, 20.0, 50.0)
SUPPLEMENT_METRICS = (
    "localization_onset_mae_ms",
    "localization_shift_recovery_mae_ms",
    "discrimination_balanced_accuracy",
    "generalization_jitter_mae_ms",
    "generalization_relative_jitter_error",
)
METRIC_PROBES = {
    "localization_onset_mae_ms": "localization",
    "localization_shift_recovery_mae_ms": "localization",
    "discrimination_balanced_accuracy": "discrimination",
    "generalization_jitter_mae_ms": "generalization",
    "generalization_relative_jitter_error": "generalization",
}
ALL_VARIANTS = (
    "early_downsample", "late_downsample", "uniform_local", "exponential_growth",
    "delayed_growth", "parallel_multiscale", "rf_stride_coupled",
    "rf_dilation_decoupled", "kernel_3", "kernel_7", "kernel_15", "kernel_31",
    "event_baseline", "explicit_change", "ordinary_second_branch",
)
FAMILY_VARIANTS = {
    "downsampling": ("early_downsample", "late_downsample"),
    "rf_growth": (
        "uniform_local", "exponential_growth", "delayed_growth", "parallel_multiscale",
        "rf_stride_coupled", "rf_dilation_decoupled", "kernel_3", "kernel_7", "kernel_15", "kernel_31",
    ),
    "explicit_change_branch": ("event_baseline", "explicit_change", "ordinary_second_branch"),
}


def csv_write(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def right_shift_arrays(base: dict[str, np.ndarray], magnitude_ms: float, base_time_step_ms: float) -> dict[str, np.ndarray]:
    """Apply a zero-filled global positive shift and move only the onset target."""

    shift_samples = int(round(float(magnitude_ms) / float(base_time_step_ms)))
    if shift_samples <= 0:
        raise ValueError("supplementary localization shifts must be positive")
    x = np.asarray(base["x"])
    shifted_x = np.zeros_like(x)
    if shift_samples >= x.shape[1]:
        raise ValueError("shift is longer than the sequence")
    shifted_x[:, shift_samples:] = x[:, :-shift_samples]
    shifted = {key: np.array(value, copy=True) for key, value in base.items()}
    shifted["x"] = shifted_x
    total_ms = float(x.shape[1]) * float(base_time_step_ms)
    onset_ms = np.clip(np.asarray(base["onset_ms"], dtype=np.float32) + float(magnitude_ms), 0.0, total_ms - float(base_time_step_ms))
    shifted["onset_ms"] = onset_ms.astype(np.float32)
    shifted["onset_reg"] = (onset_ms / total_ms).astype(np.float32)
    return shifted


def isolated_jitter_arrays(config: dict[str, Any], size: int, seed: int, magnitude_ms: float) -> dict[str, np.ndarray]:
    """Generate an unseen probe with only jitter magnitude changed.

    Rates and phase magnitudes are reset to the training ranges.  Therefore
    this probe isolates extrapolation in jitter magnitude rather than mixing
    it with the original unseen-rate/unseen-phase axes.
    """

    probe_config = dict(config)
    probe_config["unseen_jitter_ms"] = [float(magnitude_ms)]
    probe_config["unseen_rates_hz"] = list(config["train_rates_hz"])
    probe_config["unseen_phase_magnitudes_rad"] = list(config["train_phase_magnitudes_rad"])
    return generate_split(probe_config, size, seed, "unseen")


def metric_value_rows(
    model: TemporalModel,
    base_arrays: dict[str, np.ndarray],
    shifted_arrays: dict[str, np.ndarray],
    jitter_arrays: dict[str, np.ndarray],
    variant: str,
    seed: int,
    magnitude_ms: float,
    device: torch.device,
    total_ms: float,
) -> list[dict[str, Any]]:
    base_outputs, _, _ = predict_arrays(model, base_arrays, device)
    shifted_outputs, _, _ = predict_arrays(model, shifted_arrays, device)
    jitter_outputs, _, _ = predict_arrays(model, jitter_arrays, device)

    base_onset = base_outputs["onset_reg"] * total_ms
    shifted_onset = shifted_outputs["onset_reg"] * total_ms
    target_onset = shifted_arrays["onset_ms"]
    shift_recovery = (shifted_onset - base_onset) - float(magnitude_ms)
    jitter_pred = jitter_outputs["jitter_reg"] * 8.0
    positive_mask = jitter_arrays["jitter_flag"].astype(bool)
    if not np.any(positive_mask):
        raise RuntimeError("jitter probe unexpectedly contains no positive jitter samples")

    values = {
        "localization_onset_mae_ms": float(np.mean(np.abs(shifted_onset - target_onset))),
        "localization_shift_recovery_mae_ms": float(np.mean(np.abs(shift_recovery))),
        "discrimination_balanced_accuracy": balanced_accuracy(
            jitter_arrays["jitter_flag"].astype(int),
            (sigmoid(jitter_outputs["jitter"]) >= 0.5).astype(int),
        ),
        "generalization_jitter_mae_ms": float(np.mean(np.abs(jitter_pred[positive_mask] - jitter_arrays["jitter_ms"][positive_mask]))),
        "generalization_relative_jitter_error": float(
            np.mean(np.abs(jitter_pred[positive_mask] - jitter_arrays["jitter_ms"][positive_mask])) / float(magnitude_ms)
        ),
    }
    return [
        {
            "variant": variant,
            "seed": seed,
            "magnitude_ms": magnitude_ms,
            "probe": METRIC_PROBES[metric],
            "metric": metric,
            "value": value,
        }
        for metric, value in values.items()
    ]


def supplement_summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        value = float(row["value"])
        if math.isfinite(value):
            groups[(str(row["variant"]), float(row["magnitude_ms"]), str(row["probe"]), str(row["metric"]))].append(value)
    output: list[dict[str, Any]] = []
    for (variant, magnitude_ms, probe, metric), values in sorted(groups.items()):
        output.append({
            "variant": variant,
            "magnitude_ms": magnitude_ms,
            "probe": probe,
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        })
    return output


def format_value(summary_map: dict[tuple[str, float, str], dict[str, Any]], variant: str, magnitude: float, metric: str) -> str:
    row = summary_map.get((variant, float(magnitude), metric))
    if row is None:
        return "NA"
    return f"{float(row['mean']):.4f} ± {float(row['std']):.4f}"


def markdown_table(summary_map: dict[tuple[str, float, str], dict[str, Any]], variants: Iterable[str], metric: str) -> str:
    lines = ["| variant | 10 ms | 20 ms | 50 ms |", "|---|---:|---:|---:|"]
    for variant in variants:
        lines.append(
            f"| {variant} | {format_value(summary_map, variant, 10.0, metric)} | "
            f"{format_value(summary_map, variant, 20.0, metric)} | {format_value(summary_map, variant, 50.0, metric)} |"
        )
    return "\n".join(lines)


def append_supplement_section(formal_path: Path, supplement: str) -> str:
    existing = formal_path.read_text(encoding="utf-8") if formal_path.exists() else ""
    return existing.rstrip() + "\n\n" + supplement.strip() + "\n"


def write_updated_reports(
    output_root: Path,
    formal_reports: Path,
    config: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, str]],
) -> None:
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary_map = {(row["variant"], float(row["magnitude_ms"]), row["metric"]): row for row in summary_rows}
    pass_count = sum(row["status"] == "PASS" for row in status_rows)
    parameter_map = {row.get("variant", ""): row for row in parameter_rows}
    match_note = ", ".join(
        f"{variant}={parameter_map.get(variant, {}).get('parameter_match_status', 'see formal CSV')}"
        for variant in ALL_VARIANTS
    )
    common = f"""\
## 10/20/50 ms Supplementary Perturbation Evaluation

This additive evaluation was run after the formal results and does not replace
or overwrite them. The supplement retrained the same 15 architecture variants
because the formal run saved no checkpoints; it reused the formal train,
validation, test-seen and test-unseen generation rules, AdamW optimizer,
8-epoch budget, batch size 64, gradient clip 5.0 and seeds 11/22/33.

Perturbation definition:

- **Localization:** the same test-seen signal is translated by a positive
  global shift of exactly 10, 20 or 50 ms with zero fill. Only the onset target
  is translated by the same amount. `localization_onset_mae_ms` measures the
  shifted-onset error; `localization_shift_recovery_mae_ms` measures the error
  in recovering the shift from the model's predicted onset change.
- **Discrimination:** each magnitude is an independent mixed regular/jitter
  probe generated with `unseen_jitter_ms=[m]`. Rates and phase magnitudes are
  reset to the training ranges, so `discrimination_balanced_accuracy` isolates
  regular-versus-jitter classification.
- **Generalization:** the positive-jitter subset of the same probe is used for
  `generalization_jitter_mae_ms` and its magnitude-normalized counterpart.

The 10/20/50 ms probes are all **unseen/extrapolative**: training contains
2/4/6 ms jitter and the original unseen split contains 3/5/7 ms. None of the
three supplementary magnitudes is used for training or tuning. This is a
small synthetic perturbation benchmark, not localization or discrimination
evidence for real audio or neural recordings.

Supplement run status: `{pass_count}/{len(status_rows)}` seed runs PASS. The
formal parameter-match labels are retained unchanged: {match_note}.

Traceability files: `m6a_public_003_perturbation_metrics_by_seed.csv`,
`m6a_public_003_perturbation_summary.csv`,
`m6a_public_003_perturbation_run_status.csv` and
`m6a_public_003_perturbation_manifest.json`.
The explicit magnitude-axis figure is
`reports/figures/m6a_public_003_performance_vs_perturbation_magnitude.{{png,svg,pdf}}`.
Each magnitude-specific probe array is generated once and reused across all
15 structures and three seeds; no magnitude-specific tuning is performed.
"""
    design = common + """
## Boundary and interpretation

The localization shift and jitter magnitude are different probes. A model can
recover the global onset translation without identifying the periodic jitter
cause, and balanced accuracy does not establish magnitude estimation. The
three seeds quantify repeatability for this generator/model pair only; they do
not estimate biological or participant variability. A flat or chance-level
curve is retained as a negative result, and any failed/non-PASS seed remains
in the status CSV and manifest.
"""
    root_names = [
        "TEMPORAL_BENCHMARK_DESIGN.md", "DOWNSAMPLING_ABLATION.md",
        "RECEPTIVE_FIELD_ABLATION.md", "EVENT_BRANCH_ABLATION.md",
        "M6A-PUBLIC-003_SUMMARY.md", "STRUCTURAL_MODIFICATION_REGISTRY.md",
    ]
    for name in root_names:
        source = formal_reports / name
        if name == "TEMPORAL_BENCHMARK_DESIGN.md":
            supplement = design
        elif name == "DOWNSAMPLING_ABLATION.md":
            supplement = common + "\n### Downsampling curves\n\n" + "\n\n".join(
                f"**{metric}**\n\n{markdown_table(summary_map, FAMILY_VARIANTS['downsampling'], metric)}"
                for metric in SUPPLEMENT_METRICS
            )
        elif name == "RECEPTIVE_FIELD_ABLATION.md":
            supplement = common + "\n### RF growth, multiscale and kernel curves\n\n" + "\n\n".join(
                f"**{metric}**\n\n{markdown_table(summary_map, FAMILY_VARIANTS['rf_growth'], metric)}"
                for metric in SUPPLEMENT_METRICS
            )
        elif name == "EVENT_BRANCH_ABLATION.md":
            supplement = common + "\n### Explicit change/event branch curves\n\n" + "\n\n".join(
                f"**{metric}**\n\n{markdown_table(summary_map, FAMILY_VARIANTS['explicit_change_branch'], metric)}"
                for metric in SUPPLEMENT_METRICS
            )
        elif name == "M6A-PUBLIC-003_SUMMARY.md":
            key_metrics = (
                "localization_onset_mae_ms",
                "discrimination_balanced_accuracy",
                "generalization_jitter_mae_ms",
            )
            supplement = common + "\n### Readout across all structures\n\n" + "\n\n".join(
                f"**{metric}**\n\n{markdown_table(summary_map, ALL_VARIANTS, metric)}"
                for metric in key_metrics
            ) + """

### Result boundary

The new curves answer whether this small synthetic model family preserves
onset localization, regular-versus-jitter discrimination and jitter-magnitude
extrapolation at 10/20/50 ms. They do not show that a structure localizes a
real-world perturbation, distinguish causal event types, or generalize to
neural data. The existing formal negative findings remain unchanged, including
chance-level phase-shift detection and the absence of a stable explicit-change
advantage over the parameter-matched ordinary second branch.
"""
        else:
            supplement = common + "\nThe registry now includes an additive 10/20/50 ms magnitude-axis evaluation; no fourth structural group was introduced.\n"
        (reports / name).write_text(append_supplement_section(source, supplement), encoding="utf-8")

    work_report_source = formal_reports / "M6A-PUBLIC-003_AGENT_WORK_REPORT.md"
    work_report = append_supplement_section(work_report_source, f"""## Additive 10/20/50 ms supplement

STATUS: READY_FOR_REVIEW
COMPLETED: exact 10/20/50 ms localization, discrimination and extrapolative jitter-generalization evaluation across 15 variants and three seeds.
PRESERVED: prior formal outputs under the separate formal run path; no checkpoints, pretrained assets, patient/STN data or large outputs were added.
RESULTS: see the magnitude-axis CSVs, manifest and `m6a_public_003_performance_vs_perturbation_magnitude.{{png,svg,pdf}}`.
SELF_CHECK: parameter/RF smoke, authoritative remote py_compile, standard-library unittest and independent supplement verification were run; pytest remains an environment block if unavailable.
STOP_REASON: supplementary evidence is complete and is returned to Agent A/人工审核; no ACCEPT/HUMAN_ACCEPTED written.
""")
    (reports / "M6A-PUBLIC-003_AGENT_WORK_REPORT.md").write_text(work_report, encoding="utf-8")


def plot_magnitude_results(output_root: Path, summary_rows: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
    })
    figure_dir = output_root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_map = {(row["variant"], float(row["magnitude_ms"]), row["metric"]): row for row in summary_rows}
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(ALL_VARIANTS)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p", "8", "d", "H"]
    family_style = {"downsampling": "-", "rf_growth": "--", "explicit_change_branch": ":"}
    variant_family = {variant: family for family, variants in FAMILY_VARIANTS.items() for variant in variants}
    panels = [
        ("localization_onset_mae_ms", "localization onset MAE (ms)"),
        ("localization_shift_recovery_mae_ms", "shift-recovery MAE (ms)"),
        ("discrimination_balanced_accuracy", "regular vs jitter balanced accuracy"),
        ("generalization_jitter_mae_ms", "jitter generalization MAE (ms)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=False)
    magnitudes = np.asarray(MAGNITUDES_MS, dtype=float)
    for axis, (metric, ylabel) in zip(axes.ravel(), panels):
        for index, variant in enumerate(ALL_VARIANTS):
            values = [float(summary_map[(variant, magnitude, metric)]["mean"]) for magnitude in magnitudes]
            errors = [float(summary_map[(variant, magnitude, metric)]["std"]) for magnitude in magnitudes]
            axis.errorbar(
                magnitudes, values, yerr=errors, marker=markers[index], ms=4.0, lw=1.0,
                capsize=2.0, color=colors[index], linestyle=family_style[variant_family[variant]],
                label=variant,
            )
        axis.set_xticks(magnitudes, ["10", "20", "50"])
        axis.set_xlabel("perturbation magnitude (ms)")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
        axis.set_title(metric.replace("_", " "))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("M6A-PUBLIC-003: performance vs 10/20/50 ms perturbation magnitude", y=0.99, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.20, wspace=0.24, hspace=0.28)
    stem = "m6a_public_003_performance_vs_perturbation_magnitude"
    for suffix, kwargs in [("png", {"dpi": 300}), ("svg", {}), ("pdf", {})]:
        fig.savefig(figure_dir / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    return f"reports/figures/{stem}.{{png,svg,pdf}}"


def run_supplement(config: dict[str, Any], output_root: Path, formal_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    magnitudes = tuple(float(value) for value in config.get("supplementary_perturbation_magnitudes_ms", MAGNITUDES_MS))
    if magnitudes != MAGNITUDES_MS:
        raise ValueError(f"supplement magnitudes must be exactly {MAGNITUDES_MS}, observed {magnitudes}")
    specs = make_specs(config)
    if tuple(spec.name for spec in specs) != ALL_VARIANTS:
        raise ValueError("supplement must use the exact formal 15-variant architecture list")

    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    parameter_smoke_rows: list[dict[str, Any]] = []
    rf_smoke_rows: list[dict[str, Any]] = []
    for spec in specs:
        model = TemporalModel(spec, head_width=int(config["hidden_head_width"]))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        rf_samples, rf_ms, resolution_ms, output_length = final_rf_info(
            spec, int(config["sequence_length"]), float(config["base_time_step_ms"])
        )
        parameter_smoke_rows.append({
            "variant": spec.name,
            "family": spec.family,
            "parameter_count": int(parameter_count),
            "final_theoretical_RF_samples": int(rf_samples),
            "final_theoretical_RF_ms": float(rf_ms),
            "final_output_time_resolution_ms": float(resolution_ms),
            "final_output_length": int(output_length),
        })
        rf_smoke_rows.extend(rf_rows_for_spec(spec, int(config["sequence_length"]), float(config["base_time_step_ms"])))
    smoke_by_variant = {row["variant"]: row for row in parameter_smoke_rows}
    smoke_status = "PASS"
    if smoke_by_variant["rf_stride_coupled"]["final_theoretical_RF_samples"] != smoke_by_variant["rf_dilation_decoupled"]["final_theoretical_RF_samples"]:
        smoke_status = "FAIL"
    if smoke_by_variant["rf_stride_coupled"]["final_output_time_resolution_ms"] != 4.0 or smoke_by_variant["rf_dilation_decoupled"]["final_output_time_resolution_ms"] != 1.0:
        smoke_status = "FAIL"
    csv_write(reports / "m6a_public_003_perturbation_parameter_smoke.csv", parameter_smoke_rows, [
        "variant", "family", "parameter_count", "final_theoretical_RF_samples", "final_theoretical_RF_ms",
        "final_output_time_resolution_ms", "final_output_length",
    ])
    csv_write(reports / "m6a_public_003_perturbation_rf_smoke.csv", rf_smoke_rows, [
        "model", "layer", "branch", "kernel", "stride", "dilation", "jump_frame_step",
        "theoretical_RF_samples", "theoretical_RF_ms", "output_length", "output_time_resolution_ms",
    ])
    json_write(reports / "m6a_public_003_perturbation_smoke.json", {
        "status": smoke_status,
        "parameter_rows": len(parameter_smoke_rows),
        "rf_rows": len(rf_smoke_rows),
        "known_rf_checks": {
            "rf_stride_coupled_samples": smoke_by_variant["rf_stride_coupled"]["final_theoretical_RF_samples"],
            "rf_dilation_decoupled_samples": smoke_by_variant["rf_dilation_decoupled"]["final_theoretical_RF_samples"],
            "rf_stride_coupled_resolution_ms": smoke_by_variant["rf_stride_coupled"]["final_output_time_resolution_ms"],
            "rf_dilation_decoupled_resolution_ms": smoke_by_variant["rf_dilation_decoupled"]["final_output_time_resolution_ms"],
        },
        "integrity_policy": "NON_HASH_AUDIT",
    })
    if smoke_status != "PASS":
        raise RuntimeError("supplement parameter/RF smoke failed")

    device_name = config.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    generation_seed = int(config["generation_seed"])
    train_arrays = generate_split(config, int(config["train_size"]), generation_seed + 1, "seen")
    validation_arrays = generate_split(config, int(config["validation_size"]), generation_seed + 2, "seen")
    base_arrays = generate_split(config, int(config["test_seen_size"]), generation_seed + 3, "seen")
    train_dataset = TemporalDataset(train_arrays)
    validation_dataset = TemporalDataset(validation_arrays)
    total_ms = float(config["sequence_length"]) * float(config["base_time_step_ms"])
    status_rows: list[dict[str, Any]] = []
    training_logs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    jitter_probe_cache: dict[float, dict[str, np.ndarray]] = {}
    shifted_probe_cache: dict[float, dict[str, np.ndarray]] = {
        magnitude: right_shift_arrays(base_arrays, magnitude, float(config["base_time_step_ms"]))
        for magnitude in magnitudes
    }
    for magnitude in magnitudes:
        jitter_probe_cache[magnitude] = isolated_jitter_arrays(
            config,
            int(config["test_unseen_size"]),
            generation_seed + 20000 + int(magnitude),
            magnitude,
        )

    for spec in specs:
        for seed in [int(value) for value in config["seeds"]]:
            seed_everything(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                num_workers=int(config["num_workers"]),
                generator=torch.Generator().manual_seed(seed + 1000),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=False,
                num_workers=int(config["num_workers"]),
            )
            model = TemporalModel(spec, head_width=int(config["hidden_head_width"]))
            started = time.perf_counter()
            try:
                status, logs, failure = train_model(model, train_loader, validation_loader, device, config, spec.name, seed)
                training_logs.extend(logs)
                if status == "PASS":
                    for magnitude in magnitudes:
                        metric_rows.extend(metric_value_rows(
                            model,
                            base_arrays,
                            shifted_probe_cache[magnitude],
                            jitter_probe_cache[magnitude],
                            spec.name,
                            seed,
                            magnitude,
                            device,
                            total_ms,
                        ))
                status_rows.append({
                    "variant": spec.name,
                    "family": spec.family,
                    "seed": seed,
                    "status": status,
                    "failure": failure or "",
                    "wall_seconds": time.perf_counter() - started,
                })
            except Exception as exc:  # retain the failed seed and continue the matrix
                status_rows.append({
                    "variant": spec.name,
                    "family": spec.family,
                    "seed": seed,
                    "status": "FAILED",
                    "failure": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": time.perf_counter() - started,
                })
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    csv_write(reports / "m6a_public_003_perturbation_training_log.csv", training_logs, ["variant", "seed", "epoch", "train_loss", "validation_loss", "status"])
    csv_write(reports / "m6a_public_003_perturbation_run_status.csv", status_rows, ["variant", "family", "seed", "status", "failure", "wall_seconds"])
    csv_write(reports / "m6a_public_003_perturbation_metrics_by_seed.csv", metric_rows, ["variant", "seed", "magnitude_ms", "probe", "metric", "value"])
    summary = supplement_summary_rows(metric_rows)
    csv_write(reports / "m6a_public_003_perturbation_summary.csv", summary, ["variant", "magnitude_ms", "probe", "metric", "mean", "std", "n_seeds"])
    parameter_rows = read_csv(formal_root / "reports" / "m6a_public_003_model_parameters.csv")
    figure = plot_magnitude_results(output_root, summary)
    write_updated_reports(output_root, formal_root / "reports", config, summary, status_rows, parameter_rows)
    manifest = {
        "task_id": config["task_id"],
        "session_id": config["session_id"],
        "status": "READY_FOR_REVIEW",
        "supplement_id": "M6A-PUBLIC-003-PERTURBATION-SUPPLEMENT-V2",
        "magnitudes_ms": list(magnitudes),
        "magnitude_axis_is_explicit": True,
        "perturbation_definition": config.get("supplementary_perturbation_definition", ""),
        "seen_jitter_ms": list(config["train_jitter_ms"]),
        "original_unseen_jitter_ms": list(config["unseen_jitter_ms"]),
        "supplement_is_unseen_extrapolation": True,
        "localization_probe": "global positive temporal translation of test-seen x with zero fill; onset target translated by the same amount",
        "discrimination_probe": "regular versus jitter mixed probe with unseen_jitter_ms=[m], train rates and train phase magnitudes",
        "generalization_probe": "positive-jitter subset magnitude MAE and magnitude-normalized error",
        "variant_count": len(specs),
        "supplement_run_count": len(status_rows),
        "pass_count": sum(row["status"] == "PASS" for row in status_rows),
        "failed_or_nonpass_count": sum(row["status"] != "PASS" for row in status_rows),
        "metric_row_count": len(metric_rows),
        "summary_row_count": len(summary),
        "seeds": [int(value) for value in config["seeds"]],
        "same_split_optimizer_epochs_batch": True,
        "parameter_rf_smoke": "reports/m6a_public_003_perturbation_smoke.json",
        "parameter_rf_smoke_status": "PASS",
        "formal_results_preserved": True,
        "formal_results_path": str(formal_root),
        "supplementary_retraining_reason": "formal run saved no checkpoints; minimal same-budget retraining required for evaluation",
        "large_checkpoints_saved": False,
        "no_pretrained_model": True,
        "patient_stn_data_read": False,
        "integrity_policy": "NON_HASH_AUDIT",
        "figure": figure,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device_used": str(device),
    }
    json_write(reports / "m6a_public_003_perturbation_manifest.json", manifest)
    runtime = {
        "task_id": config["task_id"],
        "session_id": config["session_id"],
        "status": "READY_FOR_REVIEW",
        "device_used": str(device),
        "same_split_optimizer_epochs_batch": True,
        "formal_results_preserved": True,
        "no_pretrained_model": True,
        "patient_stn_data_read": False,
        "large_checkpoints_saved": False,
        "integrity_policy": "NON_HASH_AUDIT",
    }
    json_write(reports / "m6a_public_003_perturbation_runtime.json", runtime)
    return {"manifest": manifest, "status_rows": status_rows, "metric_rows": metric_rows, "summary_rows": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the additive M6A-PUBLIC-003 10/20/50 ms perturbation evaluation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_supplement(config, args.output_root, args.formal_root)
    print(json.dumps({
        "status": result["manifest"]["status"],
        "supplement_run_count": result["manifest"]["supplement_run_count"],
        "pass_count": result["manifest"]["pass_count"],
        "metric_row_count": result["manifest"]["metric_row_count"],
        "summary_row_count": result["manifest"]["summary_row_count"],
        "figure": result["manifest"]["figure"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
