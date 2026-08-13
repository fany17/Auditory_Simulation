from __future__ import annotations

import argparse
import json
from pathlib import Path

from m6a_public.g4_protocol_gate import (
    G4_STATUS,
    audit_g4_scope,
    finalize_g4_protocol_report,
    load_strict_json_object,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the M6A G4 protocol candidate without executing G4 data analysis."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_strict_json_object(args.protocol)
    schema = load_strict_json_object(args.schema)
    task_config = load_strict_json_object(args.task_config)
    scope_audit = audit_g4_scope(args.split_csv)
    report = finalize_g4_protocol_report(protocol, schema, task_config, scope_audit)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G4 protocol report: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "required_checks": report["required_checks"],
                "failed_checks": report["failed_checks"],
                "passage_count": report["scope_audit"]["passage_count"],
                "split_counts": report["scope_audit"]["split_counts"],
                "test_derangement_count": report["scope_audit"][
                    "test_derangement_count"
                ],
                "output": str(args.output),
                "g4_execution_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == G4_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
