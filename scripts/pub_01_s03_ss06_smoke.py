#!/usr/bin/env python3
"""Synthetic forward/backward smoke only. No optimizer, loader or experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from m6a_public.continuity.objectives import dense_objective
from m6a_public.continuity.registry import MODELS, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg["phase"] != "PROTOTYPE_ONLY" or cfg["real_data_training_authorized"] is not False or cfg["optimizer_steps"] != 0:
        raise ValueError("this entry point supports only zero-update synthetic smoke")
    if cfg["data_manifest"] is not None or cfg["checkpoint"] is not None:
        raise ValueError("synthetic smoke cannot consume data or checkpoints")
    if cfg["frameworks"] != list(MODELS):
        raise ValueError("smoke must include all five declared frameworks")
    torch.set_num_threads(1)
    torch.manual_seed(cfg["synthetic_seed"])
    report = {"task_id": cfg["task_id"], "status": "PASS", "evidence_type": "SYNTHETIC_ENGINEERING_ONLY",
              "utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
              "python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
              "config": cfg, "optimizer_steps": 0, "real_data_reads": 0,
              "pretraining": "NOT_RUN", "real_training": "NOT_RUN", "formal_test": "NOT_RUN", "models": {}}
    for name in cfg["frameworks"]:
        kwargs = cfg.get(name, {})
        model = build_model(name, in_channels=cfg["input_channels"], hidden=cfg["hidden"], **kwargs)
        x = torch.randn(2, cfg["smoke_samples"], cfg["input_channels"], requires_grad=True)
        y = torch.randn(2, cfg["smoke_samples"], 1)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        start = time.perf_counter()
        output = model(x)
        loss = dense_objective(output.prediction, y)["loss"]
        loss.backward()
        active = {n: p for n, p in model.named_parameters() if p.grad is not None}
        if not active or not all(torch.isfinite(p.grad).all() for p in active.values()):
            raise RuntimeError(f"invalid gradients: {name}")
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError(f"invalid input gradient: {name}")
        short_seconds = time.perf_counter() - start
        with torch.no_grad():
            long_x = torch.randn(1, cfg["long_forward_samples"], cfg["input_channels"])
            start = time.perf_counter()
            long_output = model(long_x).prediction
            long_seconds = time.perf_counter() - start
            state, chunks = None, []
            for chunk in long_x.split(257, dim=1):
                streamed = model(chunk, state)
                chunks.append(streamed.prediction)
                state = streamed.state
            long_streamed = torch.cat(chunks, dim=1)
            torch.testing.assert_close(long_streamed, long_output, rtol=1e-5, atol=1e-6)
            chunk_max_error = (long_streamed - long_output).abs().max().item()
        if list(long_output.shape) != [1, cfg["long_forward_samples"], 1] or not torch.isfinite(long_output).all():
            raise RuntimeError(f"invalid long output: {name}")
        if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
            raise RuntimeError(f"parameters or buffers unexpectedly changed: {name}")
        report["models"][name] = {
            "short_output_shape": list(output.prediction.shape), "long_output_shape": list(long_output.shape),
            "registered_parameters": sum(p.numel() for p in model.parameters()),
            "requires_grad_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "gradient_participating_parameters": sum(p.numel() for p in active.values()),
            "nonzero_gradient_scalars": sum(int(torch.count_nonzero(p.grad).item()) for p in active.values()),
            "short_forward_backward_seconds": short_seconds, "long_forward_seconds": long_seconds,
            "long_chunk_max_abs_error": chunk_max_error,
            "all_finite": True, "parameters_unchanged": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "models": list(report["models"]), "optimizer_steps": 0,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
