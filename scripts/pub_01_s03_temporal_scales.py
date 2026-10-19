"""Matched-parameter dilated temporal controls for public EEG→envelope decoding.

Prototype only. This module does not read data, train models, or claim a score.
All variants share layer widths, convolution taps, and parameter counts.
Input [batch, time, 64]; output [batch, time, 1]; offline/noncausal.
"""
from __future__ import annotations
from collections.abc import Sequence
import torch
from torch import nn
from torch.nn import functional as F


DILATION_PRESETS = {
    "short": (1, 1, 1),
    "middle": (4, 4, 4),
    "long": (16, 16, 16),
    "multiscale": (1, 4, 16),
}


def effective_receptive_field(kernel_size: int, dilation: int) -> int:
    """Number of samples spanned by a single dilated temporal convolution."""
    if kernel_size < 1 or dilation < 1:
        raise ValueError("kernel_size and dilation must be positive")
    return (kernel_size - 1) * dilation + 1


class TemporalScaleDecoder(nn.Module):
    """Direct EEG-to-continuous-feature decoder; no candidate audio input.

    Width is split into independently learned depthwise temporal branches, then
    combined through pointwise channel mixing. Same-width variants differ ONLY
    in temporal dilations. The same 3-s signal is available to every variant.
    """

    def __init__(
        self,
        *,
        variant: str = "multiscale",
        eeg_channels: int = 64,
        width: int = 48,
        kernel_size: int = 9,
        output_channels: int = 1,
        dilations: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if variant not in DILATION_PRESETS:
            raise ValueError(f"unknown variant: {variant}")
        chosen = tuple(DILATION_PRESETS[variant] if dilations is None else dilations)
        if len(chosen) != 3 or any(int(d) != d or d < 1 for d in chosen):
            raise ValueError("three positive integer dilations required")
        if width < 3 or width % 3 != 0:
            raise ValueError("width must be divisible by three")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if eeg_channels < 1 or output_channels < 1:
            raise ValueError("channel counts must be positive")
        self.dilations = chosen
        self.eeg_channels = eeg_channels
        self.output_channels = output_channels
        self.spatial = nn.Conv1d(eeg_channels, width, kernel_size=1)
        branch_width = width // len(chosen)
        self.branches = nn.ModuleList(
            nn.Conv1d(
                branch_width, branch_width, kernel_size,
                groups=branch_width, dilation=d,
                padding=(kernel_size - 1) * d // 2,
            )
            for d in chosen
        )
        self.fusion = nn.Conv1d(width, width, kernel_size=1)
        self.predictor = nn.Conv1d(width, output_channels, kernel_size=1)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[-1] != self.eeg_channels:
            raise ValueError(f"expected [batch, time, {self.eeg_channels}] EEG")
        x = self.spatial(eeg.transpose(1, 2))
        subbands = torch.chunk(x, len(self.branches), dim=1)
        y = torch.cat(
            [F.gelu(conv(chunk)) for conv, chunk in zip(self.branches, subbands)],
            dim=1,
        )
        y = F.gelu(self.fusion(y) + x)
        return self.predictor(y).transpose(1, 2)


def model_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
