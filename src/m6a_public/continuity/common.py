"""Explicit-state causal building blocks for the SS06 engineering prototypes.

All temporal tensors use [batch, time, channels]. No module owns stream state:
the caller decides when to retain, detach, replace, or reset the returned cache.
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DecoderOutput:
    prediction: Tensor
    state: dict[str, Any] = field(default_factory=dict)
    aux: dict[str, Any] = field(default_factory=dict)


def detach_state(tree: Any) -> Any:
    """Detach a nested state without changing its values or stream position."""
    if isinstance(tree, Tensor):
        return tree.detach()
    if isinstance(tree, dict):
        return {key: detach_state(value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [detach_state(value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(detach_state(value) for value in tree)
    return tree


def validate_input(x: Tensor, channels: int) -> None:
    """Reject malformed stream inputs before a lower-level operator fails."""
    if not isinstance(x, Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 3 or x.shape[-1] != channels:
        raise ValueError(f"x must have shape [B,T,{channels}], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("x must have nonempty batch and time dimensions")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")


def validate_state_tensor(value: Tensor, x: Tensor, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be a tensor with shape {shape}")
    if value.device != x.device or value.dtype != x.dtype:
        raise ValueError(f"{name} must have the input device and dtype")


class CausalConv(nn.Module):
    """Causal convolution with an exact raw-input cache, including dilation gaps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, groups: int = 1):
        super().__init__()
        if min(in_channels, out_channels, kernel_size, dilation, groups) < 1:
            raise ValueError("convolution dimensions, dilation and groups must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.history = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, groups=groups)

    def forward(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        validate_input(x, self.in_channels)
        if cache is None:
            cache = x.new_zeros(x.shape[0], self.history, self.in_channels)
        else:
            validate_state_tensor(cache, x, (x.shape[0], self.history, self.in_channels),
                                  "convolution cache")
        extended = torch.cat((cache, x), dim=1)
        features = self.conv(extended.transpose(1, 2)).transpose(1, 2)
        next_cache = extended[:, -self.history:, :] if self.history else extended[:, :0, :]
        return features, next_cache


class LocalEncoder(nn.Module):
    """Dense local EEG encoder, with a causal receptive field of 17 samples."""

    def __init__(self, in_channels: int = 64, hidden: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.spatial_temporal = CausalConv(in_channels, hidden, 9)
        self.depthwise = CausalConv(hidden, hidden, 9, groups=hidden)

    def forward(self, x: Tensor, cache: dict[str, Tensor] | None = None
                ) -> tuple[Tensor, dict[str, Tensor]]:
        validate_input(x, self.in_channels)
        if cache is not None and not isinstance(cache, dict):
            raise TypeError("encoder cache must be a dict or None")
        cache = {} if cache is None else cache
        features, first = self.spatial_temporal(x, cache.get("spatial_temporal"))
        features, second = self.depthwise(F.gelu(features), cache.get("depthwise"))
        return F.gelu(features), {"spatial_temporal": first, "depthwise": second}
