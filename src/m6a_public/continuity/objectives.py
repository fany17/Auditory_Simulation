"""Proposed losses and causal output-only control; not a fitting pipeline."""
import math

import torch


def dense_objective(prediction, target, valid_mask=None, mse_weight=0.1):
    """1 - Pearson r over each continuous segment + weighted raw MSE.

    Call once on concatenated continuous outputs, not averaged 3-second r.
    Targets must eventually use train-only normalization. Mask excludes warmup
    or padding; it never legitimizes joining separate records into one segment.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 1:
        raise ValueError("prediction and target must share [B,T,1]")
    if mse_weight < 0 or not math.isfinite(mse_weight):
        raise ValueError("mse_weight must be finite and nonnegative")
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    if valid_mask.shape == target.shape[:2]:
        valid_mask = valid_mask.unsqueeze(-1)
    if valid_mask.shape != target.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean [B,T] or [B,T,1]")
    count = valid_mask.sum(dim=1)
    if (count < 2).any():
        raise ValueError("at least two valid samples per segment are required")
    # torch.where also prevents masked NaNs from contaminating the reduction.
    p = torch.where(valid_mask, prediction, 0.0)
    y = torch.where(valid_mask, target, 0.0)
    if not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("valid predictions/targets must be finite")
    pc = torch.where(valid_mask, p - p.sum(dim=1, keepdim=True) / count.unsqueeze(1), 0.0)
    yc = torch.where(valid_mask, y - y.sum(dim=1, keepdim=True) / count.unsqueeze(1), 0.0)
    yy = yc.square().sum(dim=1)
    if (yy <= 1e-12).any():
        raise ValueError("Pearson objective is undefined for a constant target")
    denom = (pc.square().sum(dim=1) * yy).clamp_min(1e-12).sqrt()
    correlation = (pc * yc).sum(dim=1) / denom
    mse = ((p - y).square().sum(dim=1) / count).mean()
    return {"loss": (1 - correlation).mean() + mse_weight * mse,
            "correlation": correlation.mean(), "mse": mse}


def causal_smooth(prediction, alpha=0.9, state=None):
    """A fixed output-only EMA control. No target/audio signal is consumed."""
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0,1)")
    if prediction.ndim != 3 or prediction.shape[1] < 1:
        raise ValueError("prediction must be nonempty [B,T,C]")
    previous = torch.zeros_like(prediction[:, 0]) if state is None else state
    if previous.shape != prediction[:, 0].shape:
        raise ValueError("smoother state shape mismatch")
    sequence = []
    for value in prediction.unbind(dim=1):
        previous = alpha * previous + (1 - alpha) * value
        sequence.append(previous)
    return torch.stack(sequence, dim=1), previous
