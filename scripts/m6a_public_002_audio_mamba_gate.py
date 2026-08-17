"""Record the low-cost Audio-Mamba/SSAM pretrained-path gate on server2203."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_audio_mamba_gate.json"


def main() -> None:
    code_candidates = [
        ROOT / "cache/audio_mamba_ssam/code/audio-mamba-official-master",
        ROOT / "code/audio-mamba-official",
        ROOT / "code/audio_mamba_official",
    ]
    cache_candidates = [ROOT / "cache/audio_mamba", ROOT / "cache/ssam"]
    weight_candidates = [
        ROOT / "cache/audio_mamba_ssam/weights/checkpoints/checkpoint-99.pth",
    ]
    code_present = [str(path) for path in code_candidates if path.exists()]
    cache_present = [str(path) for path in cache_candidates if path.exists()]
    weights_present = [str(path) for path in weight_candidates if path.exists()]
    weight_inventory = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "timestamp": datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "source": "official Audio-Mamba/SSAM tiny checkpoint supplied to 2203",
        }
        for path in weight_candidates
        if path.exists()
    ]
    dependencies = {
        name: bool(importlib.util.find_spec(name))
        for name in ("mamba_ssm", "causal_conv1d", "hear_api")
    }
    runtime_ready = all(dependencies.values()) and bool(weights_present)
    result = {
        "model": "Audio-Mamba/SSAM",
        "config": "ssam_tiny_200_16x4",
        "status": "PASS_INFERENCE" if runtime_ready else "BLOCKED_BY_UPSTREAM_DEPENDENCY",
        "inference": runtime_ready,
        "temporal_probe": False,
        "zero_training": True,
        "patient_data_read": False,
        "official_source": "https://github.com/SarthakYadav/audio-mamba-official",
        "source_audit": {
            "official_readme_weight_source": "Google Drive folder",
            "github_releases": "NO_DIRECT_WEIGHT_RELEASE_FOUND",
            "huggingface_same_official_weight": "NOT_FOUND",
            "domestic_mirror_same_official_weight": "NOT_FOUND",
            "same_official_weight_direct_download": False,
            "prior_2203_result": "OFFICIAL_GOOGLE_DRIVE_NETWORK_UNREACHABLE",
            "credentials_or_login_requested": False,
            "bypass_attempted": False,
        },
        "code_candidates_present": code_present,
        "cache_candidates_present": cache_present,
        "weights_candidates_present": weights_present,
        "weights_inventory": weight_inventory,
        "weights_status": "PRESENT" if weights_present else "MISSING",
        "dependency_presence": dependencies,
        "dependency_install_attempted": True,
        "inference_attempted": False,
        "reason": (
            "official tiny checkpoint is present, but official mamba runtime dependencies "
            "are unavailable in the dedicated environment; no bypass or alternate model used"
            if not runtime_ready
            else "official tiny checkpoint and runtime dependencies are present"
        ),
        "download_attempted": True,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
