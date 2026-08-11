from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote


CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
DEFAULT_CHUNK_BYTES = 16 * 1024 * 1024
MAX_WORKERS = 8


@dataclass(frozen=True)
class RangeChunk:
    start: int
    end: int
    object_total_bytes: int

    @property
    def expected_bytes(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ObjectPlan:
    relative_path: str
    source_url: str
    expected_bytes: int
    modified_at_utc: str
    chunks: tuple[RangeChunk, ...]


def parse_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def safe_destination(root: Path, relative: str) -> Path:
    candidate = (root / Path(relative)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes root: {relative}") from exc
    return candidate


def require_strictly_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} is outside its allowed root: {resolved}") from exc
    if relative == Path("."):
        raise ValueError(f"{label} must be strictly inside its allowed root")
    return resolved


def require_separate_staging(dataset_root: Path, staging_root: Path) -> None:
    dataset = dataset_root.resolve()
    staging = staging_root.resolve()
    if dataset == staging:
        raise ValueError("staging root must be outside the dataset root")
    try:
        staging.relative_to(dataset)
    except ValueError:
        pass
    else:
        raise ValueError("staging root must be outside the dataset root")


def chunk_ranges(total_bytes: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> tuple[RangeChunk, ...]:
    if total_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("total_bytes and chunk_bytes must be positive")
    return tuple(
        RangeChunk(
            start=start,
            end=min(start + chunk_bytes, total_bytes) - 1,
            object_total_bytes=total_bytes,
        )
        for start in range(0, total_bytes, chunk_bytes)
    )


def validate_range_response(
    http_status: int,
    content_range: str | None,
    chunk: RangeChunk,
) -> None:
    if http_status != 206:
        raise IOError(f"range request returned HTTP {http_status}, expected 206")
    if content_range is None:
        raise IOError("Content-Range is missing")
    match = CONTENT_RANGE_PATTERN.fullmatch(content_range)
    if match is None:
        raise IOError(f"invalid Content-Range: {content_range}")
    actual = tuple(int(match.group(index)) for index in (1, 2, 3))
    expected = (chunk.start, chunk.end, chunk.object_total_bytes)
    if actual != expected:
        raise IOError(f"Content-Range mismatch: expected {expected}, received {actual}")


def safe_mib_per_second(byte_count: int, elapsed_seconds: float) -> float | None:
    if elapsed_seconds <= 0:
        return None
    return byte_count / (1024 * 1024) / elapsed_seconds


def chunk_path(staging_root: Path, relative_path: str, chunk: RangeChunk) -> Path:
    range_directory = safe_destination(staging_root / "chunks", relative_path + ".ranges")
    return range_directory / f"{chunk.start:012d}-{chunk.end:012d}.chunk"


def chunk_is_reusable(path: Path, chunk: RangeChunk) -> bool:
    return path.is_file() and path.stat().st_size == chunk.expected_bytes


def object_chunks_ready(plan: ObjectPlan, staging_root: Path) -> bool:
    return all(chunk_is_reusable(chunk_path(staging_root, plan.relative_path, chunk), chunk) for chunk in plan.chunks)


def move_to_backup(
    path: Path,
    source_root: Path,
    backup_root: Path,
    backup_boundary_root: Path,
    relative: Path,
) -> Path:
    source = require_strictly_within(path, source_root, "move source")
    if not source.exists():
        raise FileNotFoundError(source)
    destination = require_strictly_within(
        backup_root / relative,
        backup_boundary_root,
        "move destination",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(destination.name + f".{uuid.uuid4().hex}")
        require_strictly_within(destination, backup_boundary_root, "deduplicated move destination")
    shutil.move(str(source), str(destination))
    return destination


def download_chunk(
    plan: ObjectPlan,
    chunk: RangeChunk,
    staging_root: Path,
    run_id: str,
) -> dict[str, Any]:
    destination = chunk_path(staging_root, plan.relative_path, chunk)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if chunk_is_reusable(destination, chunk):
        return {
            "path": plan.relative_path,
            "start": chunk.start,
            "end": chunk.end,
            "bytes": chunk.expected_bytes,
            "status": "REUSED_SIZE_MATCH",
        }
    if destination.exists():
        move_to_backup(
            destination,
            staging_root,
            staging_root / "backup" / "chunk_size_mismatch" / run_id,
            staging_root,
            destination.relative_to(staging_root),
        )

    temporary = destination.with_name(destination.name + f".partial-{uuid.uuid4().hex}")
    transferred = 0
    content_range: str | None = None
    http_status: int | None = None
    started = time.monotonic()
    try:
        request = urllib.request.Request(
            plan.source_url,
            headers={
                "Range": f"bytes={chunk.start}-{chunk.end}",
                "User-Agent": "M6A-PUBLIC-resumable-range/1",
            },
        )
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("xb") as handle:
            http_status = int(response.status)
            content_range = response.headers.get("Content-Range")
            validate_range_response(http_status, content_range, chunk)
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                transferred += len(block)
        if transferred != chunk.expected_bytes or temporary.stat().st_size != chunk.expected_bytes:
            raise IOError(
                f"chunk byte mismatch: expected {chunk.expected_bytes}, received {transferred}"
            )
        temporary.replace(destination)
        elapsed = time.monotonic() - started
        return {
            "path": plan.relative_path,
            "start": chunk.start,
            "end": chunk.end,
            "http_status": http_status,
            "content_range": content_range,
            "bytes": transferred,
            "elapsed_seconds": elapsed,
            "mib_per_second": safe_mib_per_second(transferred, elapsed),
            "status": "DOWNLOADED",
        }
    except Exception as exc:
        if temporary.exists():
            move_to_backup(
                temporary,
                staging_root,
                staging_root / "backup" / "failed_chunks" / run_id,
                staging_root,
                temporary.relative_to(staging_root),
            )
        return {
            "path": plan.relative_path,
            "start": chunk.start,
            "end": chunk.end,
            "http_status": http_status,
            "content_range": content_range,
            "bytes": transferred,
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }


def assemble_object(
    plan: ObjectPlan,
    dataset_root: Path,
    staging_root: Path,
    run_id: str,
) -> dict[str, Any]:
    if not object_chunks_ready(plan, staging_root):
        return {"path": plan.relative_path, "status": "ASSEMBLY_BLOCKED_INCOMPLETE_CHUNKS"}
    destination = safe_destination(dataset_root, plan.relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == plan.expected_bytes:
        return {
            "path": plan.relative_path,
            "status": "FINAL_SKIPPED_SIZE_MATCH",
            "bytes": plan.expected_bytes,
        }
    if destination.exists():
        move_to_backup(
            destination,
            dataset_root,
            staging_root / "backup" / "preexisting_final_size_mismatch" / run_id,
            staging_root,
            Path(plan.relative_path),
        )
    for stale in destination.parent.glob(destination.name + ".partial-range-*"):
        move_to_backup(
            stale,
            dataset_root,
            staging_root / "backup" / "interrupted_assembly" / run_id,
            staging_root,
            Path(plan.relative_path + ".stale") / stale.name,
        )

    temporary = destination.with_name(destination.name + f".partial-range-{run_id}")
    written = 0
    try:
        with temporary.open("xb") as output:
            for chunk in plan.chunks:
                source = chunk_path(staging_root, plan.relative_path, chunk)
                with source.open("rb") as handle:
                    while True:
                        block = handle.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)
        if written != plan.expected_bytes or temporary.stat().st_size != plan.expected_bytes:
            raise IOError(
                f"assembled byte mismatch: expected {plan.expected_bytes}, received {written}"
            )
        timestamp = parse_time(plan.modified_at_utc)
        os.utime(temporary, (timestamp, timestamp))
        temporary.replace(destination)
        return {"path": plan.relative_path, "status": "ASSEMBLED", "bytes": written}
    except Exception as exc:
        if temporary.exists():
            move_to_backup(
                temporary,
                dataset_root,
                staging_root / "backup" / "failed_assembly" / run_id,
                staging_root,
                Path(plan.relative_path + f".partial-range-{run_id}"),
            )
        return {
            "path": plan.relative_path,
            "status": "ASSEMBLY_FAILED",
            "bytes": written,
            "error": f"{type(exc).__name__}: {exc}",
        }


def interleaved_tasks(plans: list[ObjectPlan]) -> list[tuple[ObjectPlan, RangeChunk]]:
    groups = [[(plan, chunk) for chunk in plan.chunks] for plan in plans]
    return [item for group in zip_longest(*groups) for item in group if item is not None]


def validate_inventory_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    rows_by_path: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        relative_path = (row.get("path") or "").strip()
        source_url = (row.get("source_url") or "").strip()
        size_text = (row.get("bytes") or "").strip()
        modified_at = (row.get("modified_at_utc") or "").strip()
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or path.is_absolute()
            or path.as_posix() != relative_path
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"inventory row {index} has unsafe or empty path")
        if relative_path in rows_by_path:
            raise ValueError(f"inventory has duplicate path: {relative_path}")
        if not source_url:
            raise ValueError(f"inventory row {index} has empty source URL")
        expected_url = (
            "https://s3.amazonaws.com/openneuro.org/ds004703/"
            + quote(relative_path, safe="/")
        )
        if source_url != expected_url:
            raise ValueError(f"inventory row {index} source is outside official ds004703 S3")
        try:
            size = int(size_text)
        except ValueError as exc:
            raise ValueError(f"inventory row {index} has invalid size") from exc
        if size <= 0:
            raise ValueError(f"inventory row {index} size must be positive")
        if not modified_at:
            raise ValueError(f"inventory row {index} has empty modification time")
        parse_time(modified_at)
        rows_by_path[relative_path] = row
    return rows_by_path


def load_plans(
    inventory_path: Path,
    selected_paths: list[str],
    chunk_bytes: int,
) -> list[ObjectPlan]:
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        rows = validate_inventory_rows(list(csv.DictReader(handle)))
    if len(selected_paths) != len(set(selected_paths)):
        raise ValueError("selected paths must be unique")
    missing = sorted(set(selected_paths) - set(rows))
    if missing:
        raise ValueError("selected paths are absent from inventory: " + ", ".join(missing))
    plans: list[ObjectPlan] = []
    for relative_path in selected_paths:
        row = rows[relative_path]
        expected_bytes = int(row["bytes"])
        plans.append(
            ObjectPlan(
                relative_path=relative_path,
                source_url=row["source_url"],
                expected_bytes=expected_bytes,
                modified_at_utc=row["modified_at_utc"],
                chunks=chunk_ranges(expected_bytes, chunk_bytes),
            )
        )
    return plans


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable public S3 range downloader.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--path", action="append", required=True, dest="paths")
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--benchmark-chunks", type=int)
    parser.add_argument("--benchmark-offset-chunks", type=int, default=0)
    args = parser.parse_args()

    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    if args.benchmark_chunks is not None and args.benchmark_chunks <= 0:
        raise ValueError("benchmark-chunks must be positive")
    if args.benchmark_offset_chunks < 0:
        raise ValueError("benchmark-offset-chunks must be non-negative")
    if args.benchmark_chunks is None and args.benchmark_offset_chunks != 0:
        raise ValueError("benchmark-offset-chunks requires benchmark-chunks")
    require_separate_staging(args.dataset_root, args.staging_root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plans = load_plans(args.inventory, args.paths, args.chunk_bytes)
    tasks = interleaved_tasks(plans)
    if args.benchmark_chunks is not None:
        start = args.benchmark_offset_chunks
        tasks = tasks[start : start + args.benchmark_chunks]

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_chunk, plan, chunk, args.staging_root, run_id)
            for plan, chunk in tasks
        ]
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed_chunks={completed}/{len(futures)}", flush=True)
    download_elapsed = time.monotonic() - started
    failed = [item for item in results if item["status"] == "FAILED"]
    downloaded_bytes = sum(
        int(item["bytes"]) for item in results if item["status"] == "DOWNLOADED"
    )
    assembly_results: list[dict[str, Any]] = []
    if args.benchmark_chunks is None and not failed:
        assembly_results = [
            assemble_object(plan, args.dataset_root, args.staging_root, run_id) for plan in plans
        ]
    assembly_failed = [
        item
        for item in assembly_results
        if item["status"] not in {"ASSEMBLED", "FINAL_SKIPPED_SIZE_MATCH"}
    ]
    status = "PASS" if not failed and not assembly_failed else "FAIL"
    mode = "BENCHMARK_ONLY_NO_ASSEMBLY" if args.benchmark_chunks is not None else "FULL"
    report = {
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "integrity_policy": "NON_HASH_RANGE_AND_BYTE_AUDIT",
        "dataset_root": str(args.dataset_root.resolve()),
        "staging_root": str(args.staging_root.resolve()),
        "chunk_bytes": args.chunk_bytes,
        "workers": args.workers,
        "selected_paths": args.paths,
        "requested_chunk_count": len(tasks),
        "benchmark_offset_chunks": args.benchmark_offset_chunks,
        "downloaded_chunk_count": sum(item["status"] == "DOWNLOADED" for item in results),
        "reused_chunk_count": sum(item["status"] == "REUSED_SIZE_MATCH" for item in results),
        "failed_chunk_count": len(failed),
        "downloaded_bytes": downloaded_bytes,
        "download_elapsed_seconds": download_elapsed,
        "aggregate_mib_per_second": safe_mib_per_second(downloaded_bytes, download_elapsed),
        "chunk_reuse_policy": "EXPECTED_START_END_AND_BYTE_SIZE_ONLY",
        "content_completeness_limitation": "SIZE_MATCH_REUSE_IS_NON_CRYPTOGRAPHIC_AND_DOES_NOT_PROVE_CONTENT_IDENTITY",
        "response_headers_used": ["Content-Range"],
        "chunk_results": sorted(results, key=lambda item: (item["path"], item["start"])),
        "assembly_results": assembly_results,
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "chunk_results"}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
