from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from m6a_public.audio_context_gate import (
    MODEL_FILES,
    MODEL_ID,
    MODEL_REVISION_LABEL,
    audit_model_config,
    audit_pytorch_weight_file,
)
from m6a_public.audio_context_gate import load_strict_json_object


MIRROR_ENDPOINT = "https://hf-mirror.com"
MIRROR_BASE_URL = f"{MIRROR_ENDPOINT}/facebook/wav2vec2-base/resolve/main"


def validate_download_authorization(config: dict[str, Any]) -> None:
    if config.get("model", {}).get("download_allowed") is not True:
        raise ValueError(
            "controlled model acquisition is closed: model.download_allowed must be true"
        )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _inventory(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.name,
        "bytes": stat.st_size,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def _semantic_readable(path: Path, expected_name: str) -> bool:
    if expected_name in {"config.json", "preprocessor_config.json"}:
        payload = load_strict_json_object(path)
        if expected_name == "config.json":
            return audit_model_config(path)["status"] == "PASS"
        return payload.get("sampling_rate") == 16000
    if expected_name == "README.md":
        return bool(path.read_text(encoding="utf-8")[:256].strip())
    if expected_name == "pytorch_model.bin":
        return audit_pytorch_weight_file(path)["status"] == "PASS"
    return False


def download_file(url: str, destination: Path) -> int:
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Auditory-Simulation-M6A/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status for {destination.name}: {response.status}")
        with partial.open("wb") as handle:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    size = partial.stat().st_size
    if size <= 0:
        raise RuntimeError(f"empty model download: {destination.name}")
    if not _semantic_readable(partial, destination.name):
        raise RuntimeError(f"model file is not semantically readable: {destination.name}")
    partial.replace(destination)
    return size


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache the single frozen M6A audio model without digests.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    args = parser.parse_args()

    config = load_strict_json_object(args.config)
    validate_download_authorization(config)
    project_root = Path(config["resources"]["remote_project_root"]).resolve()
    configured_model_dir = Path(config["model"]["remote_cache"]).resolve()
    model_dir = args.model_dir.resolve()
    backup_root = args.backup_root.resolve()
    if config["model"]["model_id"] != MODEL_ID or config["model"]["revision_label"] != MODEL_REVISION_LABEL:
        raise ValueError("only facebook/wav2vec2-base at revision label main is authorized")
    if config["model"].get("source_endpoint") != MIRROR_ENDPOINT:
        raise ValueError("only the fixed hf-mirror.com endpoint is authorized for this node")
    if model_dir != configured_model_dir or not _inside(model_dir, project_root):
        raise ValueError("model directory must equal the configured remote-only cache path")
    if not _inside(backup_root, project_root):
        raise ValueError("backup root must remain inside the dedicated remote project")

    model_dir.mkdir(parents=True, exist_ok=True)
    backup_root.mkdir(parents=True, exist_ok=True)
    moved_partials: list[str] = []
    reused: list[str] = []
    downloaded: list[str] = []
    for name in MODEL_FILES:
        destination = model_dir / name
        partial = destination.with_name(destination.name + ".partial")
        if partial.exists():
            backup = backup_root / partial.name
            if backup.exists():
                backup = backup_root / f"{partial.name}.{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
            shutil.move(str(partial), str(backup))
            moved_partials.append(str(backup))
        if destination.is_file() and destination.stat().st_size > 0 and _semantic_readable(destination, name):
            reused.append(name)
            continue
        if destination.exists():
            backup = backup_root / destination.name
            if backup.exists():
                backup = backup_root / f"{destination.name}.{datetime.now(timezone.utc).strftime('%H%M%S%f')}"
            shutil.move(str(destination), str(backup))
        download_file(f"{MIRROR_BASE_URL}/{name}?download=true", destination)
        downloaded.append(name)

    report = {
        "status": "CACHE_READY_FOR_SEMANTIC_MODEL_LOAD",
        "model_id": MODEL_ID,
        "revision_label": MODEL_REVISION_LABEL,
        "revision_limitation": "MUTABLE_MAIN_LABEL_NON_CRYPTOGRAPHIC_REPRODUCIBILITY_ONLY",
        "source_endpoint": MIRROR_ENDPOINT,
        "source_endpoint_role": "PUBLIC_HUGGING_FACE_ENDPOINT_MIRROR",
        "source_endpoint_limitation": (
            "THIRD_PARTY_MIRROR_PLUS_MUTABLE_MAIN_AND_NO_HASH_POLICY_DO_NOT_PROVIDE_"
            "CRYPTOGRAPHIC_INTEGRITY_OR_IMMUTABLE_PROVENANCE"
        ),
        "model_dir": str(model_dir),
        "downloaded": downloaded,
        "reused_after_semantic_readability_validation": reused,
        "moved_interrupted_partials": moved_partials,
        "inventory": [_inventory(model_dir / name) for name in MODEL_FILES],
        "integrity_policy": "NON_HASH_AUDIT",
        "semantic_model_load_still_required": True,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
