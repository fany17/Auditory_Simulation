"""Record the low-cost Audio-Mamba/SSAM pretrained-path gate on server2203."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
OUT = ROOT / "reports/m6a_public_002_audio_mamba_gate.json"


def main() -> None:
    code_candidates = [ROOT / "code/audio-mamba-official", ROOT / "code/audio_mamba_official"]
    cache_candidates = [ROOT / "cache/audio_mamba", ROOT / "cache/ssam"]
    code_present = [str(path) for path in code_candidates if path.exists()]
    cache_present = [str(path) for path in cache_candidates if path.exists()]
    dependencies = {
        name: bool(importlib.util.find_spec(name))
        for name in ("mamba_ssm", "causal_conv1d", "hear_api")
    }
    result = {
        "model": "Audio-Mamba/SSAM",
        "status": "BLOCKED_BY_UPSTREAM_DEPENDENCY",
        "inference": False,
        "temporal_probe": False,
        "zero_training": True,
        "patient_data_read": False,
        "official_source": "https://github.com/SarthakYadav/audio-mamba-official",
        "code_candidates_present": code_present,
        "cache_candidates_present": cache_present,
        "dependency_presence": dependencies,
        "reason": "official code/weights and the Mamba runtime chain are not present; obtaining the official weight folder is outside this low-cost pass",
        "download_attempted": False,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
