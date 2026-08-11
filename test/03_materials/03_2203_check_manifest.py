#!/usr/bin/env python3
"""Read-only final hash check for the 2203 S0-S2 delivery manifest."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/home/fanyu/auditory_simulation_tb001_demo001_20260806")
MANIFEST = ROOT / "reports/TB001-DEMO001_INITIAL_DELIVERY_MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
mismatches = [
    record["relative_path"]
    for record in manifest["files"]
    if sha256_file(ROOT / record["relative_path"]) != record["sha256"]
]
if mismatches:
    raise SystemExit("MISMATCH: " + ", ".join(mismatches))
print(f"ALL_SELECTED_HASHES_MATCH files={len(manifest['files'])}")
