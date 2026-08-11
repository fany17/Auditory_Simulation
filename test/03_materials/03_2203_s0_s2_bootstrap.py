#!/usr/bin/env python3
"""Bootstrap and verify only S0-S2 of the standalone wav2vec2 demo on 2203."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"
sys.path.insert(0, str(VENDOR))

MODEL_ID = "facebook/wav2vec2-base-960h"
DATASET_ID = "hf-internal-testing/librispeech_asr_demo"
DATASET_CONFIG = "clean"
DATASET_SPLIT = "validation"
DATASET_ROW = 8
TASK_ID = "TB001-DEMO001"
METRICS_SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def append_log(lines: list[str]) -> None:
    log_path = ROOT / "docs" / "CODEX_PROJECT_LOG.md"
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n## " + utc_now() + "：受控 SSH 执行 S0-S2\n\n")
        for line in lines:
            handle.write(f"- {line}\n")


def collect_preflight() -> dict:
    stat = shutil.disk_usage(ROOT)
    mem_kib = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                mem_kib = int(line.split()[1])
                break
    return {
        "checked_at_utc": utc_now(),
        "demo_root": str(ROOT),
        "root_is_git_repository": (ROOT / ".git").exists(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "disk_free_bytes": stat.free,
        "memory_available_kib": mem_kib,
        "nvidia_smi": shutil.which("nvidia-smi"),
        "requirements": {
            "python_3_10_or_3_11": sys.version_info[:2] in {(3, 10), (3, 11)},
            "disk_at_least_10_gib": stat.free >= 10 * 1024**3,
            "memory_at_least_8_gib": mem_kib is None or mem_kib >= 8 * 1024**2,
        },
    }


def freeze_environment() -> dict:
    import numpy
    import scipy
    import soundfile
    import torch
    import torchaudio
    import transformers

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    write_text(ROOT / "environment" / "pip-freeze.txt", freeze)
    return {
        "captured_at_utc": utc_now(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "soundfile": soundfile.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def acquire_one_audio(dataset_revision: str) -> tuple[Path, dict]:
    import requests
    import soundfile as sf

    endpoint = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "offset": DATASET_ROW,
        "length": 1,
    }
    response = requests.get(endpoint, params=params, timeout=60)
    response.raise_for_status()
    rows = response.json().get("rows", [])
    if len(rows) != 1 or rows[0].get("row_idx") != DATASET_ROW:
        raise RuntimeError("Dataset server did not return the requested single row")
    row = rows[0]["row"]
    audio_cell = row["audio"]
    if isinstance(audio_cell, list) and len(audio_cell) == 1:
        audio_url = audio_cell[0]["src"]
    elif isinstance(audio_cell, dict):
        audio_url = audio_cell["src"]
    else:
        raise RuntimeError(f"Unexpected dataset-server audio cell: {type(audio_cell).__name__}")

    stimuli_dir = ROOT / "stimuli" / TASK_ID
    stimuli_dir.mkdir(parents=True, exist_ok=True)
    source_path = stimuli_dir / "source.flac"
    if not source_path.exists():
        audio_response = requests.get(audio_url, timeout=120)
        audio_response.raise_for_status()
        source_path.write_bytes(audio_response.content)

    samples, sample_rate = sf.read(source_path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    duration = len(mono) / float(sample_rate)
    if not 2.0 <= duration <= 6.0:
        raise RuntimeError(f"Chosen single audio row duration {duration:.3f}s is outside 2-6s")

    wav_path = stimuli_dir / "input_16k_mono.wav"
    if sample_rate != 16000:
        raise RuntimeError(f"Expected LibriSpeech 16 kHz input, got {sample_rate}")
    if not wav_path.exists():
        sf.write(wav_path, mono, 16000, subtype="PCM_16")

    with wave.open(str(wav_path), "rb") as handle:
        wav_channels = handle.getnchannels()
        wav_rate = handle.getframerate()
        wav_frames = handle.getnframes()
    if wav_channels != 1 or wav_rate != 16000:
        raise RuntimeError("Converted WAV is not mono 16 kHz")

    manifest = {
        "audio_id": f"{DATASET_ID}:{DATASET_CONFIG}:{DATASET_SPLIT}:{DATASET_ROW}",
        "dataset_id": DATASET_ID,
        "dataset_revision": dataset_revision,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "row_idx": DATASET_ROW,
        "file": row.get("file"),
        "utterance_id": row.get("id"),
        "speaker_id": row.get("speaker_id"),
        "chapter_id": row.get("chapter_id"),
        "reference_text": row.get("text"),
        "source_endpoint": response.url,
        "resolved_audio_url": audio_url,
        "source_sha256": sha256_file(source_path),
        "wav_sha256": sha256_file(wav_path),
        "sample_rate": wav_rate,
        "channels": wav_channels,
        "frames": wav_frames,
        "duration_seconds": wav_frames / float(wav_rate),
        "normalization": "none; stereo mean only if needed",
        "license": "CC BY 4.0",
        "license_source": "https://www.openslr.org/12",
        "created_at_utc": utc_now(),
    }
    write_json(stimuli_dir / "manifest.json", manifest)
    return wav_path, manifest


def hash_snapshot(snapshot_path: Path) -> list[dict]:
    records = []
    for path in sorted(snapshot_path.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "relative_path": str(path.relative_to(snapshot_path)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path.resolve()),
                }
            )
    return records


def run_baseline(model_revision: str, wav_path: Path, audio_manifest: dict) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=model_revision,
            cache_dir=ROOT / "models" / "cache",
            allow_patterns=[
                "README.md",
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "model.safetensors",
            ],
        )
    )
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(snapshot, local_files_only=True)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)

    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    if waveform.ndim != 1 or sample_rate != 16000:
        raise RuntimeError("Baseline input is not mono 16 kHz")
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    with torch.inference_mode():
        first = model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        second = model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    logits_a = first.logits.detach().cpu().float()
    logits_b = second.logits.detach().cpu().float()
    max_abs_repeat_error = float(torch.max(torch.abs(logits_a - logits_b)).item())
    if max_abs_repeat_error > 1e-5:
        raise RuntimeError(f"Baseline repeatability failed: {max_abs_repeat_error}")
    hidden_states = first.hidden_states
    if hidden_states is None or len(hidden_states) != 13:
        raise RuntimeError(f"Expected 13 hidden states, got {0 if hidden_states is None else len(hidden_states)}")
    if not torch.isfinite(logits_a).all():
        raise RuntimeError("Baseline logits contain NaN/Inf")
    for index, hidden in enumerate(hidden_states):
        if not torch.isfinite(hidden).all():
            raise RuntimeError(f"Hidden state {index} contains NaN/Inf")

    token_ids = torch.argmax(logits_a, dim=-1)[0]
    transcript = processor.batch_decode(token_ids.unsqueeze(0))[0]
    probabilities = torch.softmax(logits_a, dim=-1)
    blank_id = int(model.config.pad_token_id)
    blank_ratio = float((token_ids == blank_id).float().mean().item())
    mean_frame_max_probability = float(probabilities.max(dim=-1).values.mean().item())

    model_files = hash_snapshot(snapshot)
    model_manifest = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "processor_revision": model_revision,
        "snapshot_path": str(snapshot),
        "license": "Apache-2.0",
        "files": model_files,
        "created_at_utc": utc_now(),
    }
    write_json(ROOT / "models" / "model_manifest.json", model_manifest)

    result = {
        "task_id": TASK_ID,
        "run_id": "baseline-smoke-001",
        "status": "SUCCESS",
        "created_at_utc": utc_now(),
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "processor_revision": model_revision,
        "audio_id": audio_manifest["audio_id"],
        "audio_sha256": audio_manifest["wav_sha256"],
        "device": str(device),
        "dtype": "float32",
        "input_sample_rate": 16000,
        "input_duration": audio_manifest["duration_seconds"],
        "logits_shape": list(logits_a.shape),
        "hidden_state_count": len(hidden_states),
        "hidden_state_shapes": [list(value.shape) for value in hidden_states],
        "transcript": transcript,
        "reference_text": audio_manifest["reference_text"],
        "ctc_frame_count": int(logits_a.shape[1]),
        "ctc_blank_id": blank_id,
        "ctc_blank_ratio": blank_ratio,
        "mean_frame_max_probability": mean_frame_max_probability,
        "greedy_token_id_count_before_ctc_collapse": int(token_ids.numel()),
        "greedy_token_ids_sha256": hashlib.sha256(np.asarray(token_ids, dtype=np.int64).tobytes()).hexdigest(),
        "repeat_max_abs_logit_error": max_abs_repeat_error,
        "repeat_tolerance": 1e-5,
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "nan_or_inf_detected": False,
        "pending": ["S3 layer intervention", "S4 24 non-baseline runs", "S5 final HTML", "S6 final report"],
    }
    write_json(ROOT / "outputs" / TASK_ID / "baseline" / "baseline_summary.json", result)
    return result


def write_readme() -> None:
    text = f"""# {TASK_ID} initial deployment\n\nThis isolated directory contains only the S0-S2 initial deployment of the standalone specification. It is not the completed demo.\n\n- Specification: `docs/STANDALONE_EXECUTION_SPEC.md`\n- Initial report: `reports/{TASK_ID}_INITIAL_DEPLOYMENT.md`\n- Baseline result: `outputs/{TASK_ID}/baseline/baseline_summary.json`\n- Audio manifest: `stimuli/{TASK_ID}/manifest.json`\n- Model manifest: `models/model_manifest.json`\n\nStatus: S0-S2 attempted; S3-S6 remain PENDING. No Git repository was initialized.\n"""
    write_text(ROOT / "README.md", text)


def write_report(status: str, preflight: dict, environment: dict | None, result: dict | None, error: str | None) -> None:
    lines = [
        f"# {TASK_ID} 2203 initial deployment",
        "",
        f"- Status: `{status}`",
        f"- Generated at: `{utc_now()}`",
        f"- DEMO_ROOT: `{ROOT}`",
        "- Scope: S0-S2 only; S3-S6 are PENDING.",
        "- Git: not initialized; no commit or push.",
        "",
        "## Preflight",
        "",
        f"- Python: `{preflight.get('python_version')}` at `{preflight.get('python_executable')}`",
        f"- Disk free bytes: `{preflight.get('disk_free_bytes')}`",
        f"- Memory available KiB: `{preflight.get('memory_available_kib')}`",
    ]
    if environment:
        lines += [
            f"- PyTorch: `{environment['torch']}`; Transformers: `{environment['transformers']}`",
            f"- CUDA available: `{environment['cuda_available']}`; GPUs: `{environment['gpu_names']}`",
        ]
    lines += ["", "## Baseline", ""]
    if result:
        lines += [
            f"- Run: `{result['run_id']}` / `{result['status']}`",
            f"- Model revision: `{result['model_revision']}`",
            f"- Input duration: `{result['input_duration']:.3f}` s",
            f"- Transcript: `{result['transcript']}`",
            f"- Logits shape: `{result['logits_shape']}`",
            f"- Hidden states: `{result['hidden_state_count']}`; shapes recorded in baseline JSON.",
            f"- Repeat max absolute logit error: `{result['repeat_max_abs_logit_error']}` (tolerance `{result['repeat_tolerance']}`).",
        ]
    else:
        lines.append("- Baseline was not completed.")
    if error:
        lines += ["", "## Error", "", "```text", error, "```"]
    lines += [
        "",
        "## Boundary",
        "",
        "This run only establishes that one pinned model and one pinned English utterance can produce a repeatable baseline with 13 hidden-state tensors on this server. It does not validate layer intervention, cross-utterance generalization, final HTML, or any neuroscientific interpretation.",
        "",
    ]
    write_text(ROOT / "reports" / f"{TASK_ID}_INITIAL_DEPLOYMENT.md", "\n".join(lines))


def main() -> int:
    started = time.time()
    write_readme()
    preflight = collect_preflight()
    write_json(ROOT / "environment" / "preflight.json", preflight)
    if not all(preflight["requirements"].values()):
        error = "Preflight requirements failed: " + json.dumps(preflight["requirements"])
        write_report("BLOCKED", preflight, None, None, error)
        append_log(["结果：S0 BLOCKED。", error])
        return 2

    environment = None
    result = None
    try:
        environment = freeze_environment()
        write_json(ROOT / "environment" / "environment.json", environment)
        from huggingface_hub import HfApi

        api = HfApi()
        model_info = api.model_info(MODEL_ID)
        dataset_info = api.dataset_info(DATASET_ID)
        model_revision = model_info.sha
        dataset_revision = dataset_info.sha
        if not model_revision or len(model_revision) != 40:
            raise RuntimeError(f"Invalid resolved model SHA: {model_revision}")
        if not dataset_revision or len(dataset_revision) != 40:
            raise RuntimeError(f"Invalid resolved dataset SHA: {dataset_revision}")
        wav_path, audio_manifest = acquire_one_audio(dataset_revision)
        result = run_baseline(model_revision, wav_path, audio_manifest)
        result["elapsed_seconds_including_download"] = time.time() - started
        write_json(ROOT / "outputs" / TASK_ID / "baseline" / "baseline_summary.json", result)
        write_report("INITIAL_DEPLOYMENT_SUCCESS", preflight, environment, result, None)
        append_log(
            [
                "结果：S0-S2 INITIAL_DEPLOYMENT_SUCCESS；S3-S6 保持 PENDING。",
                f"模型 revision：{model_revision}；数据集 revision：{dataset_revision}。",
                f"baseline：{result['run_id']}，hidden states={result['hidden_state_count']}，repeat max abs error={result['repeat_max_abs_logit_error']}。",
                "验证文件：reports/TB001-DEMO001_INITIAL_DEPLOYMENT.md 与 outputs/TB001-DEMO001/baseline/baseline_summary.json。",
                "未初始化 Git；未执行层干预、24 个非基线 run 或最终 HTML。",
            ]
        )
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        write_json(
            ROOT / "outputs" / TASK_ID / "baseline" / "failure.json",
            {"status": "BLOCKED", "created_at_utc": utc_now(), "error": error},
        )
        write_report("BLOCKED", preflight, environment, result, error)
        append_log(["结果：S1/S2 BLOCKED。", f"错误：{type(exc).__name__}: {exc}", "S3-S6 未执行。"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
