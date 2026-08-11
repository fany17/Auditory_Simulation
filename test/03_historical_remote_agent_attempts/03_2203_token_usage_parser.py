#!/usr/bin/env python3
"""Create a redacted token summary from private Codex JSONL event logs."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

AUDIT_ROOT = Path("/home/fanyu/.codex-run-audit/tb001-demo001-20260806")
PROJECT_ROOT = Path("/home/fanyu/auditory_simulation_tb001_demo001_20260806")

ATTEMPTS = [
    ("attempt1_workspace_write", AUDIT_ROOT / "codex_exec.jsonl", AUDIT_ROOT / "run_meta.txt"),
    ("attempt2_inherit_path", AUDIT_ROOT / "codex_exec_attempt2.jsonl", AUDIT_ROOT / "run_meta_attempt2.txt"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def parse_attempt(name: str, events_path: Path, meta_path: Path) -> dict:
    turns = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
                input_tokens = int(usage.get("input_tokens", 0))
                cached_input_tokens = int(usage.get("cached_input_tokens", 0))
                output_tokens = int(usage.get("output_tokens", 0))
                reasoning_output_tokens = int(usage.get("reasoning_output_tokens", 0))
                turns.append(
                    {
                        "event_line": line_number,
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "non_cached_input_tokens": input_tokens - cached_input_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_output_tokens": reasoning_output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    }
                )
    meta = read_meta(meta_path)
    totals = {
        key: sum(turn[key] for turn in turns)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "non_cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
    }
    return {
        "attempt": name,
        "status": "BLOCKED_AT_S0_BWRAP_SANDBOX",
        "start_utc": meta.get("START_UTC"),
        "end_utc": meta.get("END_UTC"),
        "process_exit_code": int(meta.get("EXIT_CODE", "-1")),
        "codex_version": meta.get("CODEX_VERSION"),
        "private_events_sha256": sha256_file(events_path),
        "usage_available": bool(turns),
        "turns": turns,
        "totals": totals,
    }


def main() -> int:
    attempts = [parse_attempt(*item) for item in ATTEMPTS]
    aggregate = {
        key: sum(attempt["totals"][key] for attempt in attempts)
        for key in (
            "input_tokens",
            "cached_input_tokens",
            "non_cached_input_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        )
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scope": "Two server2203 Codex CLI initial-deployment attempts only",
        "definition": {
            "total_tokens": "input_tokens + output_tokens",
            "cached_input_tokens": "subset of input_tokens; not added again",
            "reasoning_output_tokens": "reported separately; not added again",
        },
        "attempts": attempts,
        "aggregate": aggregate,
        "demo_inference_billing_tokens": 0,
        "demo_internal_counts_are_not_billing_tokens": [
            "ctc_frame_count",
            "greedy_token_id_count_before_ctc_collapse",
            "transcript_character_count",
        ],
        "current_desktop_codex_session_usage": "UNAVAILABLE_FROM_SERVER_JSONL",
        "privacy": "Raw JSONL remains in the private audit directory and is excluded from the project report package.",
    }
    output_path = PROJECT_ROOT / "reports" / "TB001-DEMO001_TOKEN_USAGE.json"
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_path)
    print(json.dumps(summary["aggregate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
