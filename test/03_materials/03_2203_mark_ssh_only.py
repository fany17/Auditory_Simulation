#!/usr/bin/env python3
"""Mark the deployed 2203 project as SSH-only and retire the remote-agent launcher."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
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


launcher = ROOT / "initial_deploy.sh"
history_dir = ROOT / "historical_remote_agent_attempts"
if launcher.exists():
    history_dir.mkdir(parents=True, exist_ok=True)
    target = history_dir / "initial_deploy_codex_exec.sh"
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(launcher), str(target))

readme = ROOT / "README.md"
readme_text = readme.read_text(encoding="utf-8")
marker = "## Execution method\n"
if marker not in readme_text:
    readme_text += (
        "\n## Execution method\n\n"
        "The only active deployment method is controlled SSH/SCP from the controlling session. "
        "This server runs Python, model inference, tests, and static serving only. "
        "Do not start `codex exec` or a second agent on this server.\n"
    )
    tmp = readme.with_name(readme.name + ".tmp")
    tmp.write_text(readme_text, encoding="utf-8")
    os.replace(tmp, readme)

log = ROOT / "docs/CODEX_PROJECT_LOG.md"
with log.open("a", encoding="utf-8", newline="\n") as handle:
    handle.write(f"\n## {utc_now()}：执行方式纠正为 SSH-only\n\n")
    handle.write("- 后续由控制端通过 SSH/SCP 驱动 2203；服务器只运行 Python、模型、测试和静态服务。\n")
    handle.write("- 不再在服务器上启动 codex exec 或第二个 Agent；Codex sandbox/bwrap 不属于项目依赖。\n")
    handle.write("- 旧 remote-agent launcher 已移动到 historical_remote_agent_attempts，仅作历史审计。\n")
    handle.write("- S0-S2 成功状态不变；S3-S6 仍为 PENDING。\n")

manifest_path = ROOT / "reports/TB001-DEMO001_INITIAL_DELIVERY_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for record in manifest["files"]:
    path = ROOT / record["relative_path"]
    record["size_bytes"] = path.stat().st_size
    record["sha256"] = sha256_file(path)
manifest["execution_method"] = "SSH_ONLY_NO_REMOTE_AGENT"
manifest["updated_at_utc"] = utc_now()
tmp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(tmp_manifest, manifest_path)

print(json.dumps({"status": "SSH_ONLY", "launcher_retired": not launcher.exists()}, ensure_ascii=False))
