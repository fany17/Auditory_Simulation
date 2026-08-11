from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ALLOWED_SPLITS = {"train", "validation", "test"}
BASE_REQUIRED_COLUMNS = {
    "sample_id",
    "split",
    "subject_id",
    "session_id",
    "recording_id",
    "stimulus_id",
    "block_id",
    "language",
    "start_sec",
    "end_sec",
}


@dataclass(frozen=True)
class Assignment:
    sample_id: str
    split: str
    subject_id: str
    session_id: str
    recording_id: str
    stimulus_id: str
    block_id: str
    language: str
    speaker_id: str
    start_sec: float
    end_sec: float

    def value(self, key: str) -> str:
        return str(getattr(self, key))


def read_assignments(path: str | Path) -> list[Assignment]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(BASE_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError("missing split columns: " + ", ".join(missing))
        rows: list[Assignment] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid time at line {line_number}") from exc
            rows.append(
                Assignment(
                    sample_id=(row.get("sample_id") or "").strip(),
                    split=(row.get("split") or "").strip(),
                    subject_id=(row.get("subject_id") or "").strip(),
                    session_id=(row.get("session_id") or "").strip(),
                    recording_id=(row.get("recording_id") or "").strip(),
                    stimulus_id=(row.get("stimulus_id") or "").strip(),
                    block_id=(row.get("block_id") or "").strip(),
                    language=(row.get("language") or "").strip(),
                    speaker_id=(row.get("speaker_id") or "").strip(),
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
            )
    return rows


def validate_assignments(
    rows: Sequence[Assignment],
    required_group_keys: Iterable[str],
    optional_group_keys: Iterable[str] = (),
    stratification_keys: Iterable[str] = (),
    temporal_context_key: str = "recording_id",
    temporal_embargo_seconds: float = 2.0,
    require_all_splits: bool = True,
) -> list[str]:
    issues: list[str] = []
    if not rows:
        return ["split manifest has no rows"]
    if temporal_embargo_seconds < 0:
        return ["temporal_embargo_seconds must be non-negative"]

    seen_ids: set[str] = set()
    present_splits: set[str] = set()
    for row in rows:
        if not row.sample_id:
            issues.append("empty sample_id")
        elif row.sample_id in seen_ids:
            issues.append(f"duplicate sample_id: {row.sample_id}")
        seen_ids.add(row.sample_id)
        if row.split not in ALLOWED_SPLITS:
            issues.append(f"invalid split for {row.sample_id}: {row.split}")
        else:
            present_splits.add(row.split)
        if row.end_sec <= row.start_sec:
            issues.append(f"non-positive interval: {row.sample_id}")

    if require_all_splits:
        missing_splits = sorted(ALLOWED_SPLITS - present_splits)
        if missing_splits:
            issues.append("missing splits: " + ", ".join(missing_splits))

    for key in required_group_keys:
        required_groups_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.value(key).strip()
            if not value:
                issues.append(f"missing required group {key}: {row.sample_id}")
                continue
            required_groups_by_value[value].add(row.split)
        for value, splits in sorted(required_groups_by_value.items()):
            if len(splits) > 1:
                issues.append(f"group leakage {key}={value}: {sorted(splits)}")

    for key in optional_group_keys:
        optional_groups_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.value(key).strip()
            if value:
                optional_groups_by_value[value].add(row.split)
        for value, splits in sorted(optional_groups_by_value.items()):
            if len(splits) > 1:
                issues.append(f"optional group leakage {key}={value}: {sorted(splits)}")

    for key in stratification_keys:
        stratified_splits_by_value: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            value = row.value(key).strip()
            if not value or value == "UNKNOWN":
                issues.append(f"missing stratification value {key}: {row.sample_id}")
                continue
            stratified_splits_by_value[value].add(row.split)
        for value, splits in sorted(stratified_splits_by_value.items()):
            missing = sorted(ALLOWED_SPLITS - splits)
            if missing:
                issues.append(f"stratification coverage missing {key}={value}: {missing}")

    by_context: dict[str, list[Assignment]] = defaultdict(list)
    for row in rows:
        context = row.value(temporal_context_key).strip()
        if context:
            by_context[context].append(row)
    for context, context_rows in by_context.items():
        active: list[Assignment] = []
        for current in sorted(context_rows, key=lambda item: (item.start_sec, item.end_sec)):
            active = [
                previous
                for previous in active
                if previous.end_sec + temporal_embargo_seconds
                >= current.start_sec - temporal_embargo_seconds
            ]
            for previous in active:
                separated = (
                    previous.end_sec + temporal_embargo_seconds < current.start_sec
                    or current.end_sec + temporal_embargo_seconds < previous.start_sec
                )
                if not separated and previous.split != current.split:
                    issues.append(
                        "temporal leakage "
                        f"{temporal_context_key}={context}: "
                        f"{previous.sample_id}/{previous.split} vs {current.sample_id}/{current.split}"
                    )
            active.append(current)
    return sorted(set(issues))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit split leakage for M6A-PUBLIC-001.")
    parser.add_argument("split_csv", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/m6a_public_001.json"))
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    split_config = config["split"]
    rows = read_assignments(args.split_csv)
    issues = validate_assignments(
        rows,
        required_group_keys=split_config["required_group_keys"],
        optional_group_keys=split_config.get("optional_group_keys", []),
        stratification_keys=split_config.get("stratification_keys", []),
        temporal_context_key=split_config["temporal_context_key"],
        temporal_embargo_seconds=float(split_config["temporal_embargo_seconds"]),
    )
    report = {"status": "PASS" if not issues else "FAIL", "rows": len(rows), "issues": issues}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
