from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


def parse_content_range(value: str | None) -> tuple[int, int, int | None]:
    if value is None:
        raise ValueError("Content-Range is missing")
    match = CONTENT_RANGE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid Content-Range: {value}")
    start = int(match.group(1))
    end = int(match.group(2))
    if match.group(3) == "*":
        raise ValueError(f"object total is required in Content-Range: {value}")
    total = int(match.group(3))
    if end < start or end >= total:
        raise ValueError(f"inconsistent Content-Range: {value}")
    return start, end, total


def run_range_smoke(
    url: str,
    output: Path,
    start: int,
    end: int,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    if start < 0 or end < start:
        raise ValueError("range must satisfy 0 <= start <= end")
    expected_bytes = end - start + 1
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "M6A-PUBLIC-range-smoke/1",
        },
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        http_status = int(response.status)
        content_range = response.headers.get("Content-Range")
        body = response.read()
    elapsed_seconds = time.monotonic() - started
    actual_start, actual_end, total_bytes = parse_content_range(content_range)
    passed = (
        http_status == 206
        and actual_start == start
        and actual_end == end
        and len(body) == expected_bytes
    )
    if passed:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(body)
    return {
        "status": "PASS" if passed else "FAIL",
        "http_status": http_status,
        "content_range": content_range,
        "requested_start": start,
        "requested_end": end,
        "content_range_start": actual_start,
        "content_range_end": actual_end,
        "object_total_bytes": total_bytes,
        "expected_bytes": expected_bytes,
        "actual_bytes": len(body),
        "elapsed_seconds": elapsed_seconds,
        "mib_per_second": (
            None if elapsed_seconds <= 0 else len(body) / (1024 * 1024) / elapsed_seconds
        ),
        "response_headers_used": ["Content-Range"],
        "output": str(output) if passed else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a non-hash HTTP Range smoke request.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1024 * 1024 - 1)
    args = parser.parse_args()
    report = run_range_smoke(args.url, args.output, args.start, args.end)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
