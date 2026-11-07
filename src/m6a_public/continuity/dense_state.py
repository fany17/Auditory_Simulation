"""M1 dense multiscale and M2 persistent fast/slow continuity prototypes."""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import (CausalConv, DecoderOutput, LocalEncoder, validate_input,
                     validate_state_tensor)


class DenseMultiscaleDecoder(nn.Module):
    """Preserve each EEG time point across parallel causal receptive fields.

    Short/long controls keep the same module inventory but only execute the
    selected branch. They therefore have fewer active parameters than multi;
    equal instantiated parameter counts must not be called a fair budget match.
    """

    DILATIONS = (1, 8, 64, 256)

    def __init__(self, in_channels: int = 64, hidden: int = 32,
                 branch_mode: str = "multi"):
        super().__init__()
        if branch_mode not in {"multi", "short", "long"}:
            raise ValueError("branch_mode must be multi, short, or long")
        self.in_channels = in_channels
        self.hidden = hidden
        self.branch_mode = branch_mode
        self.encoder = LocalEncoder(in_channels, hidden)
        self.branches = nn.ModuleDict({str(d): CausalConv(hidden, hidden, 5, dilation=d)
                                       for d in self.DILATIONS})
        self.fusion = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    @property
    def active_dilations(self) -> tuple[int, ...]:
        if self.branch_mode == "short":
            return (self.DILATIONS[0],)
        if self.branch_mode == "long":
            return (self.DILATIONS[-1],)
        return self.DILATIONS

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> DecoderOutput:
        validate_input(x, self.in_channels)
        if state is not None and not isinstance(state, dict):
            raise TypeError("state must be a dict or None")
        state = {} if state is None else state
        local, encoder_cache = self.encoder(x, state.get("encoder"))
        old_branches = state.get("branches", {})
        if not isinstance(old_branches, dict):
            raise TypeError("branch state must be a dict")
        branch_values, next_branches = [], {}
        for dilation in self.active_dilations:
            key = str(dilation)
            value, cache = self.branches[key](local, old_branches.get(key))
            branch_values.append(F.gelu(value))
            next_branches[key] = cache
        combined = torch.stack(branch_values, dim=0).mean(dim=0)
        dense = F.gelu(local + self.fusion(combined))
        return DecoderOutput(self.head(dense),
                             {"encoder": encoder_cache, "branches": next_branches},
                             {"active_dilations": self.active_dilations,
                              "receptive_field_samples": 17 + 4 * max(self.active_dilations)})


class PersistentStateDecoder(nn.Module):
    """Per-sample fast recurrence and stable learned slow-state accumulation.

    Reset by passing state=None. For truncated backpropagation, callers detach
    the returned state and pass it onward instead of resetting its values.
    """

    def __init__(self, in_channels: int = 64, hidden: int = 32,
                 slow_rate: float = 0.01):
        super().__init__()
        if not 1e-4 < slow_rate < 1 - 1e-4:
            raise ValueError("slow_rate must lie strictly between 0.0001 and 0.9999")
        self.in_channels = in_channels
        self.hidden = hidden
        self.encoder = LocalEncoder(in_channels, hidden)
        self.fast_cell = nn.GRUCell(hidden, hidden)
        self.slow_input = nn.Linear(hidden, hidden)
        unscaled_rate = (slow_rate - 1e-4) / (1 - 2e-4)
        self.slow_logit = nn.Parameter(torch.full((hidden,), math.log(unscaled_rate / (1 - unscaled_rate))))
        self.fusion = nn.Linear(3 * hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    @property
    def slow_coefficient(self) -> Tensor:
        # The explicit margin avoids sigmoid rounding to exactly 0 or 1.
        return 1e-4 + (1 - 2e-4) * torch.sigmoid(self.slow_logit)

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> DecoderOutput:
        validate_input(x, self.in_channels)
        if state is not None and not isinstance(state, dict):
            raise TypeError("state must be a dict or None")
        state = {} if state is None else state
        local, encoder_cache = self.encoder(x, state.get("encoder"))
        fast = state.get("fast", x.new_zeros(x.shape[0], self.hidden))
        slow = state.get("slow", x.new_zeros(x.shape[0], self.hidden))
        validate_state_tensor(fast, x, (x.shape[0], self.hidden), "fast state")
        validate_state_tensor(slow, x, (x.shape[0], self.hidden), "slow state")
        alpha = self.slow_coefficient
        fast_sequence, slow_sequence = [], []
        slow_inputs = torch.tanh(self.slow_input(local))
        for index in range(x.shape[1]):
            fast = self.fast_cell(local[:, index, :], fast)
            slow = (1 - alpha) * slow + alpha * slow_inputs[:, index, :]
            fast_sequence.append(fast)
            slow_sequence.append(slow)
        fast_features = torch.stack(fast_sequence, dim=1)
        slow_features = torch.stack(slow_sequence, dim=1)
        fused = F.gelu(self.fusion(torch.cat((local, fast_features, slow_features), dim=-1)))
        return DecoderOutput(self.head(fused),
                             {"encoder": encoder_cache, "fast": fast, "slow": slow},
                             {"fast": fast_features, "slow": slow_features,
                              "slow_coefficient": alpha})
