from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_DISTRIBUTIONS = [
    "torch",
    "transformers",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "mne",
    "soundfile",
    "pybids",
    "pytest",
    "jsonschema",
]


def distribution_versions() -> tuple[dict[str, str], list[str]]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    return versions, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Non-hash 2203 environment self-check.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    resources = config["resources"]
    project_root = Path(resources["remote_project_root"])
    versions, missing = distribution_versions()

    torch_report: dict[str, object]
    try:
        import torch

        torch_report = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "cuda_runtime": torch.version.cuda,
        }
    except Exception as exc:  # pragma: no cover - exercised on remote failures
        torch_report = {"cuda_available": False, "error": f"{type(exc).__name__}: {exc}"}

    stat = os.statvfs(project_root)
    report = {
        "task_id": config["task_id"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "project_root": str(project_root),
        "project_root_exists": project_root.is_dir(),
        "free_bytes": int(stat.f_bavail * stat.f_frsize),
        "packages": versions,
        "missing_packages": missing,
        "torch": torch_report,
        "integrity_policy": "NON_HASH_AUDIT",
    }
    errors: list[str] = []
    if report["conda_environment"] != resources["conda_environment"]:
        errors.append("unexpected Conda environment")
    if not report["project_root_exists"]:
        errors.append("remote project root missing")
    if missing:
        errors.append("required packages missing")
    if not torch_report.get("cuda_available"):
        errors.append("CUDA unavailable")
    if report["free_bytes"] < resources["minimum_free_bytes"]:
        errors.append("free space below configured minimum")
    report["status"] = "PASS" if not errors else "FAIL"
    report["errors"] = errors

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
