#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from route_b_temporal_models.wav2vec2_layer_demo.core import build_lightweight_delivery, create_manifest, read_json

current = read_json(ROOT / "outputs" / "TB001-DEMO001" / "current_run_group.json")
run_group = ROOT / current["relative_path"]
delivery = build_lightweight_delivery(ROOT, run_group)
manifest = create_manifest(delivery)
print(json.dumps({"status": "PACKAGED", "delivery": str(delivery), "manifest_files": manifest["file_count"], "size_bytes": manifest["total_size_bytes"]}, ensure_ascii=False, indent=2))
