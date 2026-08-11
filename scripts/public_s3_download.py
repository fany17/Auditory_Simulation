from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def safe_destination(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {relative}") from exc
    return candidate


def move_to_backup(path: Path, backup_root: Path, dataset_root: Path) -> Path:
    relative = path.resolve().relative_to(dataset_root.resolve())
    destination = backup_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def download_one(
    row: dict[str, str],
    dataset_root: Path,
    mismatch_backup: Path,
    failed_root: Path,
) -> dict[str, Any]:
    relative = row["path"]
    expected_bytes = int(row["bytes"])
    destination = safe_destination(dataset_root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file():
        actual_bytes = destination.stat().st_size
        if actual_bytes == expected_bytes:
            return {"path": relative, "status": "SKIPPED_SIZE_MATCH", "bytes": actual_bytes}
        move_to_backup(destination, mismatch_backup, dataset_root)

    temporary = destination.with_name(destination.name + f".partial-{uuid.uuid4().hex}")
    transferred = 0
    try:
        request = urllib.request.Request(
            row["source_url"],
            headers={"User-Agent": "M6A-PUBLIC-non-hash-download/1"},
        )
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("xb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                transferred += len(block)
        if transferred != expected_bytes:
            raise IOError(f"byte count mismatch: expected {expected_bytes}, received {transferred}")
        os.utime(temporary, (parse_time(row["modified_at_utc"]),) * 2)
        temporary.replace(destination)
        return {"path": relative, "status": "DOWNLOADED", "bytes": transferred}
    except Exception as exc:
        failed_root.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            failed_name = failed_root / (relative.replace("/", "__") + f".{uuid.uuid4().hex}.partial")
            shutil.move(str(temporary), str(failed_name))
        return {
            "path": relative,
            "status": "FAILED",
            "bytes": transferred,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public S3 files using non-hash evidence.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-level-only", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    project_root = dataset_root.parents[2]
    mismatch_backup = project_root / "logs" / "preexisting_mismatch" / run_id
    failed_root = project_root / "logs" / "failed_downloads" / run_id

    with args.inventory.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.top_level_only:
        rows = [row for row in rows if "/" not in row["path"]]

    results: list[dict[str, Any]] = []
    progress_lock = Lock()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, row, dataset_root, mismatch_backup, failed_root)
            for row in rows
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with progress_lock:
                completed += 1
                if completed % 25 == 0 or completed == len(rows):
                    print(f"completed={completed}/{len(rows)}", flush=True)

    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    report = {
        "run_id": run_id,
        "dataset_root": str(dataset_root),
        "integrity_policy": "NON_HASH_AUDIT",
        "requested_files": len(rows),
        "requested_bytes": sum(int(row["bytes"]) for row in rows),
        "status_counts": dict(sorted(counts.items())),
        "results": sorted(results, key=lambda item: item["path"]),
        "status": "PASS" if not counts.get("FAILED") else "FAIL",
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
