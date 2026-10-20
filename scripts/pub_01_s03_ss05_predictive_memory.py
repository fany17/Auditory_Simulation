"""Exploratory predictive-innovation multi-rate associative memory (PI-MRAM).

Research prototype, not a new established algorithm or a clinical claim.
The recurrence combines known ideas: input-conditioned state gating, predictive
innovation, and associative delta-rule memory. No prospective novelty claim.

Inputs: [batch, time, input_channels] EEG (default 64) or toy sequences.
Outputs: [batch, time, output_channels]. EEG only, causal input processing.
Persistent state must be reset at a recording/subject boundary by the caller.
A state can be carried across contiguous chunks from *one same record* only.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PredictiveMemoryState:
    prev2: torch.Tensor   # [batch, input_channels]
    prev1: torch.Tensor   # [batch, input_channels]
    rates: torch.Tensor   # [batch, 3, hidden]
    assoc: torch.Tensor   # [batch, key_dim, hidden]

    def detach(self) -> "PredictiveMemoryState":
        """Detach explicitly at a predeclared truncated-BPTT boundary."""
        return PredictiveMemoryState(
            self.prev2.detach(), self.prev1.detach(),
            self.rates.detach(), self.assoc.detach(),
        )


class PredictiveEventMemory(nn.Module):
    """Local causal filter + predictive multirate recurrence + delta memory.

    Ablations preserve the instantiated parameter inventory, not effective
    parameter use or runtime; comparisons must report this limitation.

    variants:
      local:             same 3-tap local input, no persistent state
      state:             three-rate recurrent state, no prediction or assoc read
      predictive_state:  recurrent state plus explicit innovation
      full:              predictive state plus associative delta-rule memory
      assoc_no_pred:     recurrent state/assoc with prediction replaced by input
    """

    ALLOWED_VARIANTS = ("local", "state", "predictive_state", "full", "assoc_no_pred")

    def __init__(
        self,
        *,
        input_channels: int = 64,
        hidden: int = 16,
        key_dim: int = 8,
        output_channels: int = 1,
        variant: str = "full",
        memory_decay: float = 0.995,
    ) -> None:
        super().__init__()
        if variant not in self.ALLOWED_VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        if min(input_channels, hidden, key_dim, output_channels) < 1:
            raise ValueError("all channel/hidden dimensions must be positive")
        if not (0.0 < memory_decay <= 1.0):
            raise ValueError("memory_decay must be in (0,1]")
        self.input_channels = input_channels
        self.hidden = hidden
        self.key_dim = key_dim
        self.output_channels = output_channels
        self.variant = variant
        self.memory_decay = memory_decay

        self.local_filter = nn.Linear(input_channels * 3, hidden)
        self.predict_local = nn.Linear(hidden * 3, hidden)
        self.query = nn.Linear(hidden, key_dim)
        self.key = nn.Linear(hidden, key_dim)
        self.value = nn.Linear(hidden * 2, hidden)
        self.write_gate = nn.Linear(hidden * 3, 1)
        self.update_gate = nn.Linear(hidden * 6, hidden * 3)
        self.update_proposal = nn.Linear(hidden * 3, hidden * 3)
        self.output = nn.Sequential(
            nn.Linear(hidden * 6, hidden), nn.GELU(),
            nn.Linear(hidden, output_channels),
        )
        self.register_buffer(
            "retention", torch.tensor([0.75, 0.94, 0.993], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "update_scale", torch.tensor([1.0, 0.3, 0.07], dtype=torch.float32),
            persistent=False,
        )
        # A conservative starting point: write infrequently, not on every frame.
        nn.init.constant_(self.write_gate.bias, -2.0)
        nn.init.constant_(self.update_gate.bias, -1.8)

    def empty_state(self, x: torch.Tensor) -> PredictiveMemoryState:
        if x.ndim != 2 or x.shape[1] != self.input_channels:
            raise ValueError("expected one timestep: [batch, input_channels]")
        b = x.shape[0]
        return PredictiveMemoryState(
            prev2=x.new_zeros(b, self.input_channels),
            prev1=x.new_zeros(b, self.input_channels),
            rates=x.new_zeros(b, 3, self.hidden),
            assoc=x.new_zeros(b, self.key_dim, self.hidden),
        )

    def step(
        self, x: torch.Tensor,
        state: PredictiveMemoryState | None = None,
    ) -> tuple[torch.Tensor, PredictiveMemoryState]:
        if x.ndim != 2 or x.shape[1] != self.input_channels:
            raise ValueError(f"expected x=[batch,{self.input_channels}]")
        if state is None:
            state = self.empty_state(x)
        if state.prev1.shape != x.shape or state.rates.shape != (len(x), 3, self.hidden):
            raise ValueError("incompatible recurrent state; reset at recording boundary")
        if state.assoc.shape != (len(x), self.key_dim, self.hidden):
            raise ValueError("incompatible associative memory")

        f = F.gelu(self.local_filter(torch.cat((state.prev2, state.prev1, x), -1)))
        persistent = self.variant != "local"
        use_predict = self.variant in ("predictive_state", "full")
        use_assoc = self.variant in ("full", "assoc_no_pred")
        previous = state.rates if persistent else torch.zeros_like(state.rates)
        predicted = self.predict_local(previous.flatten(1))
        # 'f' ablation keeps the innovation input width, avoids the literal zero
        # shortcut but does not use predicted history.
        innovation = (f - predicted) if use_predict else f

        if use_assoc:
            q = F.normalize(self.query(f), dim=-1)
            k = F.normalize(self.key(f), dim=-1)
            assoc_before = state.assoc if persistent else torch.zeros_like(state.assoc)
            retrieved = torch.einsum("bk,bkh->bh", q, assoc_before)
            value = torch.tanh(self.value(torch.cat((f, innovation), dim=-1)))
            read_at_key = torch.einsum("bk,bkh->bh", k, assoc_before)
            beta = torch.sigmoid(self.write_gate(torch.cat((f, innovation, retrieved), -1)))
            # Known delta-rule form, with a learnt surprise-conditioned write gate.
            assoc = (
                self.memory_decay * assoc_before
                + beta.unsqueeze(-1) * (k.unsqueeze(-1) * (value - read_at_key).unsqueeze(-2))
            )
        else:
            retrieved = torch.zeros_like(f)
            assoc = state.assoc if persistent else torch.zeros_like(state.assoc)

        gate_input = torch.cat((f, innovation, retrieved, previous.flatten(1)), -1)
        gates = torch.sigmoid(self.update_gate(gate_input)).reshape(len(x), 3, self.hidden)
        gates = gates * self.update_scale.view(1, 3, 1).to(gates.dtype)
        proposals = torch.tanh(
            self.update_proposal(torch.cat((f, innovation, retrieved), -1))
        ).reshape(len(x), 3, self.hidden)
        decays = self.retention.view(1, 3, 1).to(x.dtype)
        rates = decays * (1 - gates) * previous + gates * proposals

        features = torch.cat((f, innovation, retrieved, rates.flatten(1)), dim=-1)
        output = self.output(features)
        return output, PredictiveMemoryState(
            prev2=state.prev1, prev1=x, rates=rates, assoc=assoc,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        state: PredictiveMemoryState | None = None,
        *,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, PredictiveMemoryState]:
        if eeg.ndim != 3 or eeg.shape[-1] != self.input_channels or eeg.shape[1] < 1:
            raise ValueError(f"expected eeg=[batch,time,{self.input_channels}]")
        outputs = []
        for t in range(eeg.shape[1]):
            y, state = self.step(eeg[:, t, :], state)
            outputs.append(y)
        predicted = torch.stack(outputs, dim=1)
        return (predicted, state) if return_state else predicted


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
