#!/usr/bin/env python3
"""Separate metadata-only review pass for the HOLD reproduction package."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def review(report, server_base):
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})

    mandatory = ["PROTOCOL_MATRIX.md", "REPRO_REPORT.md", "RESULTS.csv", "REVIEW.md", "COMMANDS.md"]
    check("mandatory_deliverables", all((report / p).is_file() and (report / p).stat().st_size for p in mandatory))
    with (report / "RESULTS.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    check("result_schema_no_extra_or_missing_columns", all(None not in r and None not in r.values() for r in rows))
    check("ten_explicit_not_run_rows", len(rows) == 10 and all(r["status"] == "NOT_RUN" and r["acceptance"] == "HOLD" for r in rows))
    metrics = ["candidate_n", "seed", "top1", "top5", "top10", "accuracy_2afc", "accuracy_5afc", "pearson_r", "r2"]
    check("no_fabricated_metrics_or_actual_candidate_counts", all(r["subjects"] == "0" and all(r[k] == "" for k in metrics) for r in rows))
    check("official_internal_separated", {r["protocol"] for r in rows} == {"OFFICIAL_PROTOCOL", "INTERNAL_PROTOCOL"})
    for run in ("run01", "run02"):
        folder = report / "evidence" / run
        summary = json.loads((folder / "run_summary.json").read_text())
        check(run + "_no_training_no_author_execution", summary["training"] == "NOT_RUN" and summary["evaluation"] == "NOT_RUN" and summary["no_author_code_execution"])
        check(run + "_no_payload_or_weight_download", summary["no_payload_download"] and summary["no_weight_download"])
    source = json.loads((report / "evidence/run02/source_manifest.json").read_text())
    check("37_pinned_sources_read_and_python_ast_pass", len(source) == 37 and all(r["status"] == "READ_OK" and r.get("syntax", "PASS_AST_ONLY") == "PASS_AST_ONLY" for r in source))
    check("no_payload_in_source_allowlist", all(Path(r["file"]).suffix.lower() in {".py", ".md", ".json", ".txt", ".yml", ""} for r in source))
    smoke = json.loads((report / "evidence/run02/array_smoke.json").read_text())
    check("fixed_sub001_ten_real_records_readable", len(smoke) == 10 and all(r["subject"] == "sub-001" and r["status"] == "PASS_READABILITY" and r["common_samples"] > 0 for r in smoke))
    check("unknown_physical_sync_not_upgraded", all(r["physical_sync"] == "UNKNOWN" for r in smoke))
    env = json.loads((report / "evidence/run02/environment.json").read_text())
    check("actual_gpu_runtime_smoke", env["gpu_smoke"]["available"] and env["gpu_smoke"]["finite_tensor_operation"])
    inventory = json.loads((report / "evidence/run02/asset_inventory.json").read_text())
    check("HDPS_647_of_647_file_coverage_only", inventory["split_coverage"]["HDPS"]["existing_64hz_matches"] == 647 and inventory["split_coverage"]["HDPS"]["subjects"] == 81)
    clip = inventory["split_coverage"]["Transformer"]
    check("Transformer_71_of_91_train_14_of_14_test_file_coverage", len(clip["train_subs_available"]) == 71 and len(clip["train_subs_missing"]) == 20 and len(clip["test_subs_available"]) == 14)
    logs = (report / "evidence/run02/technical_tests.log").read_text()
    check("four_technical_tests_passed", "Ran 4 tests" in logs and "\nOK\n" in logs)
    failed_sources = json.loads((report / "evidence/run01/source_manifest.json").read_text())
    check("failed_attempt_preserved", len(failed_sources) == 37 and all(r["status"] == "READ_FAILED" for r in failed_sources))
    findings = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "review_mode": "separate_post_execution_metadata_review",
                "technical": "PASS" if all(c["pass"] for c in checks) else "REWORK",
                "acceptance": "HOLD", "parent_consistency": "PASS_SCOPE_LIMITED_NO_ARTIFACT",
                "model_performance_validated": False, "checks": checks}
    write = report / "review_checks.json"
    if write.exists():
        raise FileExistsError("preserve previous review; use fresh report directory")
    write.write_text(json.dumps(findings, indent=2) + "\n")
    # File sizes and paths only: no checksum/hash audit. Do not include self or report copy.
    with (report / "SERVER_MANIFEST.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["host", "path", "bytes", "kind", "local_return"])
        writer.writeheader()
        for path in sorted(server_base.rglob("*")):
            if not path.is_file() or path.is_relative_to(report):
                continue
            author_source = "source_inspection_only" in path.parts or "source" in path.parts
            bytecode = "__pycache__" in path.parts
            kind = "python_runtime_bytecode" if bytecode else "author_source_text_inspection_only" if author_source else "own_code_config_or_summary_log"
            writer.writerow({"host": "server2203", "path": str(path), "bytes": path.stat().st_size,
                             "kind": kind, "local_return": "NO_GIT" if author_source or bytecode else "SUMMARY_OR_OWN_CODE_ALLOWED"})
    print(json.dumps(findings))
    if findings["technical"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--server-base", type=Path, required=True)
    args = parser.parse_args()
    review(args.report, args.server_base)
