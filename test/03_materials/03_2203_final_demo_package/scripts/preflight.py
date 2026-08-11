#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from route_b_temporal_models.wav2vec2_layer_demo.core import preflight

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, default=ROOT)
args = parser.parse_args()
result = preflight(args.root)
print(json.dumps(result, ensure_ascii=False, indent=2))
raise SystemExit(0 if result["status"] == "PASS" else 1)
