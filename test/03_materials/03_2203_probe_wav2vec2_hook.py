#!/usr/bin/env python3
"""Read-only runtime probe for the pinned wav2vec2 layer hook on server2203."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))


def describe(value):
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [describe(item) for item in value]}
    return {"type": type(value).__name__}


def main() -> int:
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    manifest = json.loads((ROOT / "models" / "model_manifest.json").read_text(encoding="utf-8"))
    snapshot = Path(manifest["snapshot_path"])
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model, loading_info = Wav2Vec2ForCTC.from_pretrained(
        snapshot,
        local_files_only=True,
        output_loading_info=True,
    )
    model.eval()
    model.config.apply_spec_augment = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32)
    waveform, rate = sf.read(ROOT / "stimuli" / "TB001-DEMO001" / "input_16k_mono.wav", dtype="float32")
    inputs = processor(waveform, sampling_rate=rate, return_tensors="pt")
    captured = {}

    def hook(module, module_inputs, module_output):
        captured["module"] = type(module).__name__
        captured["inputs"] = describe(module_inputs)
        captured["output"] = describe(module_output)
        return module_output

    layer = model.wav2vec2.encoder.layers[0]
    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                inputs.input_values.to(device),
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        handle.remove()
    captured["layer_count"] = len(model.wav2vec2.encoder.layers)
    captured["hidden_state_count"] = len(output.hidden_states or ())
    captured["remaining_hooks"] = len(layer._forward_hooks)
    captured["loading_info"] = loading_info
    print(json.dumps(captured, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
