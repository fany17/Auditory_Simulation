#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))
from route_b_temporal_models.wav2vec2_layer_demo.core import main_run

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "configs" / "TB001-DEMO001.yaml")
parser.add_argument("--root", type=Path, default=ROOT)
args = parser.parse_args()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
raise SystemExit(main_run(args.root, args.config if args.config.is_absolute() else args.root / args.config))
