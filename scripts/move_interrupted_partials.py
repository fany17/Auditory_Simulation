from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.public_s3_range_download import (
    move_to_backup,
    require_strictly_within,
    safe_destination,
)


def move_interrupted_partials(
    dataset_root: Path,
    backup_root: Path,
    allowed_backup_root: Path,
    expected_relative_paths: list[str],
) -> dict[str, Any]:
    dataset = dataset_root.resolve()
    backup_boundary = allowed_backup_root.resolve()
    backup = require_strictly_within(backup_root, backup_boundary, "interrupted backup root")
    if len(expected_relative_paths) != len(set(expected_relative_paths)):
        raise ValueError("expected partial paths must be unique")
    expected = {
        require_strictly_within(
            safe_destination(dataset, relative),
            dataset,
            "expected interrupted partial",
        ): relative
        for relative in expected_relative_paths
    }
    discovered = {
        require_strictly_within(path, dataset, "discovered interrupted partial")
        for path in dataset.rglob("*.partial-*")
        if path.is_file()
    }
    if discovered != set(expected):
        missing = sorted(str(path) for path in set(expected) - discovered)
        unexpected = sorted(str(path) for path in discovered - set(expected))
        raise ValueError(
            f"partial set mismatch: missing={missing}, unexpected={unexpected}"
        )
    if backup.exists() and any(backup.rglob("*")):
        raise ValueError("interrupted backup root must be absent or empty")

    before = [
        (path, expected[path], path.stat().st_size)
        for path in sorted(expected)
    ]
    moved: list[dict[str, Any]] = []
    for source, relative_text, byte_count in before:
        relative = Path(relative_text)
        destination = move_to_backup(
            source,
            dataset,
            backup,
            backup_boundary,
            relative,
        )
        moved.append(
            {
                "source": str(source),
                "destination": str(destination),
                "relative_path": relative_text,
                "bytes": byte_count,
            }
        )

    remaining = [path for path in dataset.rglob("*.partial-*") if path.is_file()]
    backup_files = [path for path in backup.rglob("*") if path.is_file()]
    if remaining or len(backup_files) != len(expected):
        raise RuntimeError(
            f"post-move count mismatch: remaining={len(remaining)}, backup={len(backup_files)}"
        )
    return {
        "status": "PASS",
        "dataset_root": str(dataset),
        "backup_root": str(backup),
        "expected_count": len(expected),
        "moved_count": len(moved),
        "remaining_partial_count": 0,
        "moved": moved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Move an exact set of interrupted partials.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--allowed-backup-root", type=Path, required=True)
    parser.add_argument("--expected-relative", action="append", required=True, dest="paths")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = move_interrupted_partials(
        args.dataset_root,
        args.backup_root,
        args.allowed_backup_root,
        args.paths,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
