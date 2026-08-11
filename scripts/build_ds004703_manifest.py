from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from m6a_public.dataset_manifest import build_ds004703_manifests


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ds004703 language/stimulus and block split manifests.")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--segment-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--preliminary-minimum-embargo-seconds", type=float, default=2.0)
    args = parser.parse_args()
    segments, split_rows, summary = build_ds004703_manifests(
        args.dataset_root,
        preliminary_minimum_embargo_seconds=args.preliminary_minimum_embargo_seconds,
    )
    write_csv(args.segment_manifest, segments)
    write_csv(args.split_manifest, split_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["manifest_status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
