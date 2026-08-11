#!/usr/bin/env python3
"""Offline verification for the S0-S2 TB001-DEMO001 deployment."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    checks: dict[str, object] = {}
    baseline = load_json(ROOT / "outputs" / "TB001-DEMO001" / "baseline" / "baseline_summary.json")
    audio = load_json(ROOT / "stimuli" / "TB001-DEMO001" / "manifest.json")
    model_manifest = load_json(ROOT / "models" / "model_manifest.json")
    snapshot = Path(model_manifest["snapshot_path"])
    wav_path = ROOT / "stimuli" / "TB001-DEMO001" / "input_16k_mono.wav"

    checks["root_is_not_git"] = not (ROOT / ".git").exists()
    checks["model_processor_revision_match"] = (
        model_manifest["model_revision"] == model_manifest["processor_revision"] == baseline["model_revision"]
    )
    checks["model_revision_is_full_sha"] = len(model_manifest["model_revision"]) == 40
    checks["dataset_revision_is_full_sha"] = len(audio["dataset_revision"]) == 40
    checks["audio_sha256_match"] = sha256_file(wav_path) == audio["wav_sha256"] == baseline["audio_sha256"]
    with wave.open(str(wav_path), "rb") as handle:
        checks["audio_mono_16k"] = handle.getnchannels() == 1 and handle.getframerate() == 16000
        checks["audio_duration_2_to_6s"] = 2 <= handle.getnframes() / handle.getframerate() <= 6

    file_checks = []
    for record in model_manifest["files"]:
        path = snapshot / record["relative_path"]
        file_checks.append(
            path.is_file()
            and path.stat().st_size == record["size_bytes"]
            and sha256_file(path.resolve()) == record["sha256"]
        )
    checks["all_model_file_hashes_match"] = all(file_checks) and len(file_checks) == 7

    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model, loading_info = Wav2Vec2ForCTC.from_pretrained(
        snapshot,
        local_files_only=True,
        output_loading_info=True,
    )
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)
    waveform, sample_rate = sf.read(wav_path, dtype="float32")
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        output = model(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    logits = output.logits.detach().cpu().float()
    token_ids = torch.argmax(logits, dim=-1)[0]
    transcript = processor.batch_decode(token_ids.unsqueeze(0))[0]
    token_hash = hashlib.sha256(np.asarray(token_ids, dtype=np.int64).tobytes()).hexdigest()

    checks["offline_transcript_match"] = transcript == baseline["transcript"]
    checks["offline_token_ids_hash_match"] = token_hash == baseline["greedy_token_ids_sha256"]
    checks["offline_logits_shape_match"] = list(logits.shape) == baseline["logits_shape"]
    checks["offline_hidden_shapes_match"] = [list(x.shape) for x in output.hidden_states] == baseline["hidden_state_shapes"]
    checks["offline_outputs_finite"] = bool(torch.isfinite(logits).all()) and all(
        bool(torch.isfinite(value).all()) for value in output.hidden_states
    )
    checks["model_eval_mode"] = model.training is False
    checks["s3_to_s6_pending"] = len(baseline.get("pending", [])) == 4

    passed = all(value is True for value in checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "verified_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "network_mode": "HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1",
        "device": str(device),
        "checks": checks,
        "loading_info": {
            "missing_keys": sorted(loading_info.get("missing_keys", [])),
            "unexpected_keys": sorted(loading_info.get("unexpected_keys", [])),
            "mismatched_keys": sorted(loading_info.get("mismatched_keys", [])),
        },
        "compatibility_note": (
            "Transformers 5.14.1 reports wav2vec2.masked_spec_embed as missing and initializes it. "
            "The deployed model is in eval mode and this verification reproduced the pinned CTC output offline; "
            "the warning must remain visible before S3-S6."
        ),
    }
    output_path = ROOT / "reports" / "TB001-DEMO001_INITIAL_VERIFY.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
