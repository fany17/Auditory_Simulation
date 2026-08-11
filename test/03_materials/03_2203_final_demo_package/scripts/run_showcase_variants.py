#!/usr/bin/env python3
"""Generate real speed × single-layer intervention results for the clean showcase."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "src"))

from route_b_temporal_models.wav2vec2_layer_demo.core import (
    base_record,
    decode_output,
    downsample,
    finish_record,
    forward,
    load_config,
    load_runtime,
    read_json,
    sha256_file,
    write_json,
    write_text,
)


TASK_ID = "TB001-DEMO001"
SPEED_FACTORS = [1.0, 1.25, 1.5, 1.75, 2.0]
ALPHAS = [0.0, 0.5]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def speed_id(factor: float) -> str:
    return f"speed-{factor:.2f}".replace(".", "p")


def pitch_preserving_speedup(waveform, factor: float, sample_rate: int):
    import torch
    import torchaudio.functional as AF

    if factor == 1.0:
        return waveform.clone()
    n_fft = 1024
    hop_length = 256
    win_length = 1024
    window = torch.hann_window(win_length, dtype=waveform.dtype, device=waveform.device)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    phase_advance = torch.linspace(0, math.pi * hop_length, spectrum.shape[-2], device=spectrum.device)[..., None]
    stretched = AF.phase_vocoder(spectrum, rate=factor, phase_advance=phase_advance)
    target_length = int(round(waveform.numel() / factor))
    output = torch.istft(
        stretched,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=target_length,
    )
    if not torch.isfinite(output).all() or output.numel() != target_length:
        raise RuntimeError(f"Invalid phase-vocoder output for factor {factor}")
    peak = output.abs().max().clamp_min(1e-12)
    if peak > 0.999:
        output = output * (0.999 / peak)
    return output


def compact_run(run: dict) -> dict:
    keys = [
        "run_id", "status", "layer_id", "alpha", "baseline_transcript", "adjusted_transcript",
        "transcript_edit_distance_from_baseline", "ctc_blank_ratio", "mean_frame_max_probability",
        "final_hidden_cosine_distance_from_baseline", "ctc_logit_divergence_from_baseline",
        "inference_time_ms", "error",
    ]
    return {key: run.get(key) for key in keys}


def main() -> int:
    import soundfile as sf
    import torch

    config_path = ROOT / "configs" / f"{TASK_ID}.yaml"
    config = load_config(config_path)
    config["alphas"] = ALPHAS
    audio_source_manifest = read_json(ROOT / "stimuli" / TASK_ID / "manifest.json")
    model, processor, _, _, audit = load_runtime(ROOT, config)
    original_waveform, sample_rate = sf.read(ROOT / config["audio_path"], dtype="float32")
    original = torch.from_numpy(original_waveform).float()
    group_id = datetime.now(timezone.utc).strftime("showcase-%Y%m%dT%H%M%SZ")
    group = ROOT / "outputs" / TASK_ID / "showcase_variants" / group_id
    group.mkdir(parents=True, exist_ok=False)
    log_lines = [f"{utc_now()} START {group_id}"]
    conditions = []
    all_runs = []

    for factor in SPEED_FACTORS:
        condition_id = speed_id(factor)
        waveform = pitch_preserving_speedup(original, factor, sample_rate).cpu()
        audio_dir = ROOT / "stimuli" / TASK_ID / "showcase"
        audio_path = audio_dir / f"input_{condition_id}.wav"
        audio_dir.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, waveform.numpy(), sample_rate, subtype="PCM_16")
        saved, saved_rate = sf.read(audio_path, dtype="float32")
        if saved_rate != 16000 or saved.ndim != 1:
            raise RuntimeError(f"Invalid saved showcase WAV: {audio_path}")
        audio_manifest = {
            "audio_id": f"{audio_source_manifest['audio_id']}:{condition_id}",
            "condition_id": condition_id,
            "speed_factor": factor,
            "label": "原速" if factor == 1.0 else f"{factor:g}× 语速",
            "source_audio_sha256": audio_source_manifest["wav_sha256"],
            "wav_sha256": sha256_file(audio_path),
            "sample_rate": saved_rate,
            "channels": 1,
            "frames": len(saved),
            "duration_seconds": len(saved) / float(saved_rate),
            "reference_text": audio_source_manifest["reference_text"],
            "transformation": "none" if factor == 1.0 else "pitch-preserving phase vocoder",
            "phase_vocoder": None if factor == 1.0 else {"n_fft": 1024, "hop_length": 256, "win_length": 1024, "rate": factor},
            "created_at_utc": utc_now(),
        }
        write_json(audio_path.with_suffix(".json"), audio_manifest)
        encoded = processor(saved, sampling_rate=saved_rate, return_tensors="pt")
        input_values = encoded.input_values.to(device=audit["device"], dtype=torch.float32)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(audit["device"])

        started = time.perf_counter()
        baseline_output, baseline_trace = forward(model, input_values, attention_mask)
        baseline_elapsed = (time.perf_counter() - started) * 1000.0
        baseline_decoded = decode_output(model, processor, baseline_output)
        baseline_id = f"{condition_id}-baseline"
        baseline_record = base_record(config, audio_manifest, audit, baseline_id, None, 1.0)
        baseline_record = finish_record(
            baseline_record,
            baseline_output,
            baseline_decoded,
            baseline_output,
            baseline_decoded,
            audio_manifest["reference_text"],
            baseline_trace,
            baseline_elapsed,
        )
        baseline_record.update(
            {
                "condition_id": condition_id,
                "speed_factor": factor,
                "audio_file": f"stimuli/{TASK_ID}/showcase/{audio_path.name}",
                "final_hidden_cosine_distance_from_baseline": 0.0,
                "ctc_logit_divergence_from_baseline": 0.0,
            }
        )
        condition_runs = [baseline_record]
        write_json(group / condition_id / baseline_id / "summary.json", baseline_record)

        for layer_id in range(1, 13):
            for alpha in ALPHAS:
                run_id = f"{condition_id}-layer-{layer_id:02d}-alpha-{str(alpha).replace('.', 'p')}"
                record = base_record(config, audio_manifest, audit, run_id, layer_id, alpha)
                record.update({"condition_id": condition_id, "speed_factor": factor, "audio_file": f"stimuli/{TASK_ID}/showcase/{audio_path.name}"})
                try:
                    started = time.perf_counter()
                    output, trace = forward(model, input_values, attention_mask, layer_id=layer_id, alpha=alpha)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    decoded = decode_output(model, processor, output)
                    record = finish_record(
                        record,
                        output,
                        decoded,
                        baseline_output,
                        baseline_decoded,
                        audio_manifest["reference_text"],
                        trace,
                        elapsed_ms,
                    )
                    log_lines.append(
                        f"{utc_now()} SUCCESS {run_id} edit={record['transcript_edit_distance_from_baseline']} transcript={json.dumps(record['adjusted_transcript'])}"
                    )
                except Exception as exc:
                    record.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
                    log_lines.append(f"{utc_now()} FAILED {run_id} {type(exc).__name__}: {exc}")
                write_json(group / condition_id / run_id / "summary.json", record)
                condition_runs.append(record)
        write_json(group / condition_id / "runs.json", condition_runs)
        all_runs.extend(condition_runs)
        conditions.append(
            {
                "id": condition_id,
                "label": audio_manifest["label"],
                "speed_factor": factor,
                "duration_seconds": audio_manifest["duration_seconds"],
                "audio_sha256": audio_manifest["wav_sha256"],
                "audio_source": f"assets/{audio_path.name}",
                "waveform": downsample(saved.tolist(), 480),
                "baseline": compact_run(baseline_record),
                "runs": [compact_run(run) for run in condition_runs if run["layer_id"] is not None],
            }
        )

    successful_interventions = [r for r in all_runs if r["layer_id"] is not None and r["status"] == "SUCCESS"]
    strongest = max(
        successful_interventions,
        key=lambda r: (
            r["transcript_edit_distance_from_baseline"],
            r["ctc_logit_divergence_from_baseline"],
            r["final_hidden_cosine_distance_from_baseline"],
        ),
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "group_id": group_id,
        "method": "ordinary Python over SSH; no remote Codex Agent",
        "speed_method": "pitch-preserving phase vocoder",
        "conditions": conditions,
        "default_selection": {
            "condition_id": strongest["condition_id"],
            "layer_id": strongest["layer_id"],
            "alpha": strongest["alpha"],
            "run_id": strongest["run_id"],
            "transcript_edit_distance": strongest["transcript_edit_distance_from_baseline"],
        },
        "counts": {
            "conditions": len(conditions),
            "unique_inferences": len(all_runs),
            "interventions": len(successful_interventions),
            "failed": sum(r["status"] != "SUCCESS" for r in all_runs),
            "text_changed": sum(r["transcript_edit_distance_from_baseline"] > 0 for r in successful_interventions),
        },
    }
    write_json(group / "showcase_results.json", summary)
    write_json(
        ROOT / "outputs" / TASK_ID / "current_showcase.json",
        {"relative_path": group.relative_to(ROOT).as_posix(), "group_id": group_id, "updated_at_utc": utc_now()},
    )

    demo = ROOT / "demo" / TASK_ID
    assets = demo / "assets"
    data_dir = demo / "data"
    assets.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        source = ROOT / "stimuli" / TASK_ID / "showcase" / Path(condition["audio_source"]).name
        target = assets / source.name
        target.write_bytes(source.read_bytes())
    write_json(data_dir / "showcase_results.json", summary)
    write_text(
        data_dir / "showcase_data.js",
        "window.TB001_SHOWCASE = " + json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n",
    )
    write_text(group / "run.log", "\n".join(log_lines) + "\n")
    print(json.dumps({"status": "SUCCESS", **summary["counts"], "default_selection": summary["default_selection"], "group": str(group)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raise SystemExit(main())
