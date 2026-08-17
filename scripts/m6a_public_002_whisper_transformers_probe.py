"""Official Hugging Face Whisper large-v3-turbo synthetic encoder probe on 2203."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/fanyu/auditory_simulation_m6a/m6a_public_002")
MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_DIR = ROOT / "cache/whisper_transformers_large_v3_turbo"
PROBES = ROOT / "probes"
AUDIT_OUT = ROOT / "reports/m6a_public_002_whisper_transformers_download.json"
OUT = ROOT / "reports/m6a_public_002_whisper_transformers_probe.json"
NAMES = ("tone", "regular_clicks", "jitter_clicks", "omission", "phase_shift", "speech")


def load_probes() -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name in NAMES:
        value = np.load(PROBES / f"{name}.npy", allow_pickle=False)
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"invalid synthetic probe: {name}")
        values[name] = value
    return values


def summarize(outputs: dict[str, np.ndarray]) -> dict[str, object]:
    reference = outputs["tone"].reshape(-1).astype(np.float64)
    relative_distance: dict[str, float] = {}
    for name, value in outputs.items():
        if name == "tone":
            continue
        flat = value.reshape(-1).astype(np.float64)
        n = min(reference.size, flat.size)
        relative_distance[name] = float(
            np.linalg.norm(reference[:n] - flat[:n])
            / max(np.linalg.norm(reference[:n]), 1e-12)
        )

    persistence: dict[str, float | None] = {}
    event_response: dict[str, float | None] = {}
    for name, value in outputs.items():
        trace = np.mean(np.abs(value), axis=(0, 2))
        if trace.size < 2 or np.std(trace[:-1]) == 0 or np.std(trace[1:]) == 0:
            persistence[name] = None
        else:
            persistence[name] = float(np.corrcoef(trace[:-1], trace[1:])[0, 1])
        indexes = [
            max(0, min(trace.size - 1, int(fraction * trace.size)))
            for fraction in (0.25, 0.5, 0.75)
        ]
        width = max(1, trace.size // 20)
        selected = np.concatenate(
            [trace[index : min(trace.size, index + width)] for index in indexes]
        )
        event_response[name] = float(np.mean(selected)) if selected.size else None

    return {
        "probe_names": list(outputs),
        "shapes": {name: list(value.shape) for name, value in outputs.items()},
        "finite": all(bool(np.isfinite(value).all()) for value in outputs.values()),
        "relative_distance_to_tone": relative_distance,
        "persistence_lag1": persistence,
        "event_triggered_mean": event_response,
    }


def download() -> None:
    from huggingface_hub import snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision="main",
        local_dir=str(MODEL_DIR),
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "generation_config.json",
            "model*.safetensors",
            "*.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "normalizer.json",
            "added_tokens.json",
        ],
    )
    files = []
    for path in sorted(MODEL_DIR.rglob("*")):
        relative = path.relative_to(MODEL_DIR)
        if path.is_file() and ".cache" not in relative.parts:
            stat = path.stat()
            files.append(
                {
                    "relative_path": str(relative),
                    "bytes": stat.st_size,
                    "modified_time": stat.st_mtime,
                }
            )
    AUDIT_OUT.write_text(
        json.dumps(
            {
                "status": "PASS",
                "model_id": MODEL_ID,
                "revision_label": "main",
                "revision_limitation": "mutable label; no cryptographic provenance asserted",
                "endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                "cache_path": str(MODEL_DIR),
                "files": files,
                "safetensors_only": any(
                    item["relative_path"].endswith(".safetensors") for item in files
                ),
                "local_only_after_download": True,
                "trust_remote_code": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def probe() -> None:
    import torch
    from transformers import WhisperFeatureExtractor, WhisperModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Whisper inference")
    extractor = WhisperFeatureExtractor.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = WhisperModel.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float16,
    ).cuda().eval()
    if model.config.model_type != "whisper":
        raise RuntimeError("unexpected model_type")
    if len(model.encoder.layers) != 32 or model.config.d_model != 1280:
        raise RuntimeError("unexpected Whisper encoder configuration")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("gradients must remain disabled")

    outputs: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for name, audio in load_probes().items():
            features = extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).input_features.to(device="cuda", dtype=torch.float16)
            result = model.encoder(input_features=features)
            hidden = result.last_hidden_state.detach().float().cpu().numpy()
            if not np.isfinite(hidden).all():
                raise RuntimeError(f"non-finite representation: {name}")
            outputs[name] = hidden
    torch.cuda.synchronize()
    summary = summarize(outputs)
    summary.update(
        {
            "status": "PASS",
            "model_id": MODEL_ID,
            "revision_label": "main",
            "revision_limitation": "mutable label; no cryptographic provenance asserted",
            "representation": "Whisper encoder last_hidden_state",
            "sample_rate": 16000,
            "feature_extractor": {
                "feature_size": extractor.feature_size,
                "sampling_rate": extractor.sampling_rate,
                "hop_length": extractor.hop_length,
                "n_samples": extractor.n_samples,
                "padding": "max_length to model context",
            },
            "wall_seconds": time.perf_counter() - started,
            "zero_training": True,
            "patient_data_read": False,
            "downstream_probe": False,
            "local_files_only": True,
            "trust_remote_code": False,
            "safetensors_only": True,
        }
    )
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    if "--download" in sys.argv:
        download()
    if "--probe" in sys.argv:
        probe()
    if not {"--download", "--probe"}.intersection(sys.argv):
        raise SystemExit("use --download and/or --probe")


if __name__ == "__main__":
    main()
