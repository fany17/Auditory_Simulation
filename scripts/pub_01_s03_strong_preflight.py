#!/usr/bin/env python3
"""2203-only source/protocol and existing-public-asset audit. Never trains.

Fetches only pinned source/text manifests, not signals, weights, labels or pickle.
Author code is inspected as text/AST, never imported or executed. Outputs are
metadata only. A new run directory is mandatory; existing evidence is untouched.
"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
import platform
import re
import socket
import subprocess
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes public project root")
    return path


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "PUB-01-S03-SS02-audit"})
    with urllib.request.urlopen(request, timeout=30) as response:
        # Source/text only; protect against accidental large payload.
        data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("source exceeds text limit")
        return data.decode("utf-8")


def source_audit(cfg, out):
    manifest, issues = [], []
    staging = Path(cfg["source_staging"]) if cfg.get("source_staging") else None
    if staging and not staging.resolve().is_relative_to(Path(cfg["server_root"]).resolve()):
        raise ValueError("source staging outside public project")
    for repo in cfg["repos"]:
        dest = out / "source_inspection_only" / repo["name"]
        dest.mkdir(parents=True)
        tree_url = f'https://api.github.com/repos/{repo["repo"]}/git/trees/{repo["commit"]}?recursive=1'
        try:
            tree = (json.loads((staging / repo["name"] / "tree_metadata.json").read_text())
                    if staging else json.loads(fetch(tree_url)))
            paths = [entry["path"] for entry in tree["tree"] if entry["type"] == "blob"]
            if tree.get("truncated"):
                raise ValueError("truncated source tree")
            license_paths = [p for p in paths if re.search(r"(^|/)(license|copying|notice)(\.|$)", p, re.I)]
            write_json(dest / "tree_metadata.json", {"url": tree_url, "paths": paths,
                                                       "license_paths": license_paths})
            issues.append({"repo": repo["name"], "license_paths": license_paths,
                           "license_gate": "NEEDS_TEXT_REVIEW" if license_paths else "HOLD_NO_LICENSE_FILE"})
        except Exception as exc:
            issues.append({"repo": repo["name"], "tree_error": repr(exc), "license_gate": "HOLD_UNVERIFIED"})
        for filename in repo["files"]:
            if Path(filename).suffix.lower() not in {".py", ".json", ".md", ".yml", ".txt", ""}:
                raise ValueError("not an approved source/text extension")
            url = f'https://raw.githubusercontent.com/{repo["repo"]}/{repo["commit"]}/{filename}'
            row = {"repo": repo["name"], "commit": repo["commit"], "file": filename, "url": url}
            try:
                body = (contained(staging / repo["name"], filename).read_text(encoding="utf-8-sig")
                        if staging else fetch(url))
                target = contained(dest, filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(body, encoding="utf-8")
                row.update(status="READ_OK", bytes=len(body.encode("utf-8")), lines=len(body.splitlines()),
                           transport="Windows_official_raw_source_then_scp" if staging else "server_https")
                if filename.endswith(".py"):
                    try:
                        ast.parse(body)
                        row["syntax"] = "PASS_AST_ONLY"
                    except SyntaxError as exc:
                        row["syntax"] = str(exc)
            except Exception as exc:
                row.update(status="READ_FAILED", error=repr(exc))
            manifest.append(row)
    write_json(out / "source_manifest.json", manifest)
    write_json(out / "license_gate.json", issues)
    return manifest


def array_smoke(index, root):
    import numpy as np
    subjects = sorted({r["subject"] for r in index if r.get("eeg_path")})
    selected = subjects[0]
    rows = []
    for record in sorted([r for r in index if r["subject"] == selected], key=lambda r: r["source_recording"]):
        item = {"subject": selected, "record": record["source_recording"],
                "story": record["story"], "sample_rate_hz": record["sample_rate_hz"],
                "rate_basis": record["rate_basis"], "physical_sync": record["physical_sync"]}
        try:
            for key in ("eeg_path", "envelope_path"):
                path = Path(record[key]).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("signal path outside public project")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                item[key] = str(path)
                item[key + "_shape"] = list(array.shape)
                item[key + "_dtype"] = str(array.dtype)
                item[key + "_finite"] = bool(np.isfinite(array).all())
                if key == "eeg_path":
                    if array.ndim != 2 or 64 not in array.shape:
                        raise ValueError("EEG shape is not a 64-channel matrix")
                    eeg_n = array.shape[1] if array.shape[0] == 64 else array.shape[0]
                else:
                    if array.ndim not in (1, 2):
                        raise ValueError("unsupported envelope dimensions")
                    env_n = array.shape[0]
            item["common_samples"] = min(eeg_n, env_n)
            if item["common_samples"] != int(record["samples"]) or float(record["sample_rate_hz"]) != 64:
                raise ValueError("upstream common support or 64Hz contract mismatch")
            item["status"] = "PASS_READABILITY" if item["eeg_path_finite"] and item["envelope_path_finite"] else "HOLD_NONFINITE"
        except Exception as exc:
            item.update(status="HOLD_READ", error=repr(exc))
        rows.append(item)
    return rows


def split_coverage(out, index):
    hdps_path = out / "source_inspection_only/HDPS/derivatives/SparrKULee/split.json"
    clip_path = out / "source_inspection_only/Transformer-CLIP-EEG/fold_split.json"
    existing = {(r["subject"], r["story"]) for r in index if r.get("eeg_path") and Path(r["eeg_path"]).is_file()}
    report = {}
    if hdps_path.exists():
        split = json.loads(hdps_path.read_text())
        rows = []
        for subject, phases in split.items():
            for phase, stories in phases.items():
                for story in stories:
                    rows.append({"subject": subject, "phase": phase, "story": story,
                                 "existing_64hz_record": (subject, story) in existing,
                                 "official_100hz_feature": "NOT_VERIFIED"})
        report["HDPS"] = {"subjects": len(split), "requested_record_phase_pairs": len(rows),
                          "existing_64hz_matches": sum(r["existing_64hz_record"] for r in rows),
                          "phase_counts": dict(Counter(r["phase"] for r in rows)),
                          "warning": "64Hz coverage is not official 100Hz readiness"}
        write_json(out / "hdps_split_coverage.json", rows)
    if clip_path.exists():
        split = json.loads(clip_path.read_text())
        report["Transformer"] = {"keys": list(split), "list_lengths": {k: len(v) for k, v in split.items()}}
        available = {s for s, _ in existing}
        for key in ("train_subs", "test_subs"):
            if key in split:
                report["Transformer"][key + "_available"] = sorted(set(split[key]) & available)
                report["Transformer"][key + "_missing"] = sorted(set(split[key]) - available)
    return report


def run(cfg):
    if platform.system() != "Linux" or cfg["patient_data"] != "FORBIDDEN":
        raise RuntimeError("execute only on authorized Linux server2203")
    root = Path(cfg["server_root"])
    if not root.is_dir():
        raise RuntimeError("public project missing; no local fallback")
    if cfg["payload_download"] or cfg["weight_download"]:
        raise ValueError("this runner forbids payload and weight downloads")
    out = contained(root, cfg["output_relative"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config_snapshot.json", cfg)
    packages = {}
    for name in ("numpy", "scipy", "torch", "torchaudio", "transformers", "mne", "brain-pipe", "brainpipe", "tensorflow", "librosa", "pytest", "braindecode", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    import torch
    gpu_smoke = {"available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                 "device_count": torch.cuda.device_count()}
    if torch.cuda.is_available():
        # Basic runtime check only, not model/metric or research evidence.
        gpu_smoke["finite_tensor_operation"] = bool(torch.isfinite(torch.ones((8, 8), device="cuda") @ torch.ones((8, 8), device="cuda")).all())
    write_json(out / "environment.json", {"host": socket.gethostname(), "platform": platform.platform(),
               "python": platform.python_version(), "packages": packages, "gpu_smoke": gpu_smoke,
               "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], capture_output=True, text=True).stdout})
    sources = source_audit(cfg, out)
    index = read_csv(contained(root, cfg["index_relative"]))
    split = read_csv(contained(root, cfg["split_relative"]))
    write_json(out / "array_smoke.json", array_smoke(index, root))
    # Inventory only known public project roots. No signal values in output.
    search_roots = [root / "cache", root / "models", root / "m6a_public_002/cache", root / "pub_01"]
    patterns = ["*wav2vec*", "*gpt*", "*clip*", "*TEST_task*", "*challenge*", "*100Hz*", "valid_seg_dict_6.pkl", "feat_len_dict.pkl"]
    found = {pattern: sorted({str(p) for base in search_roots for p in base.rglob(pattern)}) for pattern in patterns}
    write_json(out / "asset_inventory.json", {"index_records": len(index), "index_subjects": len({r["subject"] for r in index}),
              "split_records": len(split), "split_subjects": len({r["subject"] for r in split}),
              "split_counts": dict(Counter(r["split"] for r in split)), "matching_public_paths": found,
              "split_coverage": split_coverage(out, index)})
    write_json(out / "run_summary.json", {"run_id": cfg["run_id"], "finished_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PREFLIGHT_ONLY", "model_smoke": "NOT_RUN", "training": "NOT_RUN", "evaluation": "NOT_RUN",
              "source_files_read": sum(r["status"] == "READ_OK" for r in sources),
              "source_files_failed": sum(r["status"] != "READ_OK" for r in sources),
              "no_payload_download": True, "no_weight_download": True, "no_author_code_execution": True})
    print(str(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text(encoding="utf-8-sig")))
