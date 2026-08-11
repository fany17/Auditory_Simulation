#!/usr/bin/env python3
"""Finalize the S0-S2 deployment manifest and append a token-free handoff entry."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/fanyu/auditory_simulation_tb001_demo001_20260806")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    selected = [
        "README.md",
        "docs/STANDALONE_EXECUTION_SPEC.md",
        "environment/preflight.json",
        "environment/environment.json",
        "environment/pip-freeze.txt",
        "models/model_manifest.json",
        "stimuli/TB001-DEMO001/source.flac",
        "stimuli/TB001-DEMO001/input_16k_mono.wav",
        "stimuli/TB001-DEMO001/manifest.json",
        "outputs/TB001-DEMO001/baseline/baseline_summary.json",
        "reports/TB001-DEMO001_INITIAL_DEPLOYMENT.md",
        "reports/TB001-DEMO001_INITIAL_VERIFY.json",
        "reports/TB001-DEMO001_TOKEN_USAGE.json",
        "scripts/run_initial_deployment.py",
        "scripts/verify_initial_deployment.py",
        "scripts/summarize_token_usage.py",
    ]
    records = []
    for relative in selected:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        records.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    verification = json.loads((ROOT / "reports/TB001-DEMO001_INITIAL_VERIFY.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("Offline verification is not PASS")
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "scope": "TB001-DEMO001 S0-S2 initial deployment only",
        "status": "INITIAL_DEPLOYMENT_SUCCESS",
        "s3_to_s6": "PENDING",
        "files": records,
        "exclusions": [
            "model cache blobs are covered by models/model_manifest.json",
            "append-only docs/CODEX_PROJECT_LOG.md is intentionally not self-hashed",
            "private Codex JSONL remains outside the project root",
            "S3-S6 outputs do not exist yet",
        ],
    }
    output = ROOT / "reports/TB001-DEMO001_INITIAL_DELIVERY_MANIFEST.json"
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output)

    log_path = ROOT / "docs/CODEX_PROJECT_LOG.md"
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {utc_now()}：初步部署复核与交接\n\n")
        handle.write("- 状态：S0-S2 INITIAL_DEPLOYMENT_SUCCESS；S3-S6 保持 PENDING。\n")
        handle.write("- 离线复核：reports/TB001-DEMO001_INITIAL_VERIFY.json 为 PASS。\n")
        handle.write("- 兼容性边界：Transformers 5.14.1 加载时报告 wav2vec2.masked_spec_embed 缺失并初始化；eval 模式离线输出已复现，进入 S3-S6 前仍须保留 warning。\n")
        handle.write("- token：脱敏摘要位于 reports/TB001-DEMO001_TOKEN_USAGE.json；按全局规则不在本日志记录 token 数值。\n")
        handle.write("- 交付 manifest：reports/TB001-DEMO001_INITIAL_DELIVERY_MANIFEST.json。\n")
        handle.write("- Git：未初始化、未提交、未推送。\n")
    print(json.dumps({"status": manifest["status"], "file_count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
