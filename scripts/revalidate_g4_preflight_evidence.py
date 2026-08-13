from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m6a_public.audio_context_gate import load_strict_json_object
from m6a_public.g4_preflight_gate import (
    PREFLIGHT_STATUS,
    finalize_preflight_report,
    validate_preflight_config,
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite revalidated preflight report: {path}")
    partial = path.with_name(f".{path.name}.partial-revalidation-{os.getpid()}")
    if partial.exists():
        raise FileExistsError(f"revalidation partial already exists: {partial}")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Revalidate existing G4 synthetic preflight evidence without rerunning the model."
    )
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_logs = Path("/home/fanyu/auditory_simulation_m6a/logs").resolve()
    source = args.source_report.resolve()
    output = args.output.resolve()
    if source.parent != project_logs or output.parent != project_logs or source == output:
        raise ValueError("source and output must be distinct files inside project logs")
    evidence = load_strict_json_object(source)
    config = load_strict_json_object(args.config)
    schema = load_strict_json_object(args.schema)
    main_config = load_strict_json_object(args.main_config)
    protocol_config = load_strict_json_object(args.protocol_config)
    current_config_errors = validate_preflight_config(
        config, schema, main_config, protocol_config
    )
    if current_config_errors:
        raise ValueError(
            "current preflight configuration failed revalidation: "
            + "; ".join(current_config_errors)
        )
    runtime = _mapping(evidence.get("runtime_canary"))
    if not (
        evidence.get("status") == "FAIL"
        and evidence.get("failed_checks")
        == ["passage_wise_feature_extractor_equivalence"]
        and evidence.get("runtime_error") is None
        and runtime.get("status") == "PASS"
        and evidence.get("new_real_edf_read") is False
        and evidence.get("new_real_audio_read") is False
        and evidence.get("real_feature_extraction_run") is False
        and evidence.get("ridge_run") is False
        and evidence.get("null_run") is False
        and evidence.get("metric_run") is False
    ):
        raise ValueError("source report is not the exact preserved gate-only failure")

    evidence.pop("required_checks", None)
    evidence.pop("failed_checks", None)
    evidence.pop("status", None)
    evidence["config_errors"] = current_config_errors
    evidence["evidence_revalidation"] = {
        "revalidated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": source.as_posix(),
        "source_status": "FAIL",
        "source_failure": "GATE_REJECTED_NEGATIVE_FINITE_NORMALIZED_MEAN",
        "source_report_preserved": True,
        "model_computation_rerun": False,
        "real_audio_or_edf_reread": False,
    }
    report = finalize_preflight_report(evidence, config)
    _atomic_write(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "required_checks": report["required_checks"],
                "failed_checks": report["failed_checks"],
                "model_computation_rerun": False,
                "output": output.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == PREFLIGHT_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
