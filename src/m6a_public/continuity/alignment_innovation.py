"""SS06 M4/M5 causal prototypes; no data access or training side effects.

Alignment is in EEG coordinates. The lag bank never advances EEG or reads an
audio target. Innovation is an explicit EEG-feature residual, not an asserted
physiological prediction error. Neither prototype has measured EEG efficacy.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .common import DecoderOutput, LocalEncoder


def _validate_input(x: Tensor, channels: int) -> None:
    if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != channels:
        raise ValueError(f"x must have shape [B,T,{channels}] with B,T >= 1")
    if not x.is_floating_point():
        raise ValueError("x must be floating point")


def _state_tensor(value: Tensor, reference: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"state[{name!r}] must have shape {shape}")
    if value.device != reference.device or value.dtype != reference.dtype:
        raise ValueError(f"state[{name!r}] must match input device and dtype")
    return value


class AdaptiveAlignmentDecoder(nn.Module):
    """Causal lag mixture with a slowly changing, EEG-only gate.

    Each time step updates gate logits as ``g = decay*g + (1-decay)*W*h``.
    Softmax gives a bounded simplex mixture over the *same* nonnegative lag
    bank in adaptive and fixed modes. The decay is learned but remains strictly
    between 0 and 1 in floating-point arithmetic through an epsilon margin.
    ``fixed_lag_weights`` is a prespecified, non-learned control (uniform by
    default); a one-hot vector implements a single fixed-lag control.

    State keys are encoder, lag, gate. Call with state=None at a new record or
    other metadata boundary; only the caller owns that boundary information.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden: int = 32,
        lags: Sequence[int] = (0, 4, 8, 16),
        mode: str = "adaptive",
        fixed_lag_weights: Sequence[float] | None = None,
        slow_decay: float = 0.98,
    ) -> None:
        super().__init__()
        if in_channels < 1 or hidden < 1:
            raise ValueError("in_channels and hidden must be positive")
        if mode not in {"adaptive", "fixed"}:
            raise ValueError("mode must be adaptive or fixed")
        lags = tuple(lags)
        if not lags or any(isinstance(lag, bool) or not isinstance(lag, int) or lag < 0 for lag in lags):
            raise ValueError("lags must be a nonempty sequence of nonnegative integers")
        if len(set(lags)) != len(lags):
            raise ValueError("lags must be unique")
        if not 1e-6 < slow_decay < 1 - 1e-6:
            raise ValueError("slow_decay must lie strictly between 1e-6 and 1-1e-6")
        self.in_channels, self.hidden, self.mode = in_channels, hidden, mode
        self.lags, self.max_lag = lags, max(lags)
        self.encoder = LocalEncoder(in_channels=in_channels, hidden=hidden)
        self.lag_logits = nn.Linear(hidden, len(lags))
        self.slow_decay_logit = nn.Parameter(torch.tensor(math.log(slow_decay / (1 - slow_decay))))
        self.readout = nn.Linear(hidden, 1)
        weights = torch.ones(len(lags)) if fixed_lag_weights is None else torch.tensor(fixed_lag_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() != len(lags):
            raise ValueError("fixed_lag_weights must have one entry per lag")
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or float(weights.sum()) <= 0:
            raise ValueError("fixed_lag_weights must be finite, nonnegative and have positive sum")
        self.register_buffer("fixed_weights", weights / weights.sum())

    @property
    def slow_decay(self) -> Tensor:
        return 1e-6 + (1 - 2e-6) * torch.sigmoid(self.slow_decay_logit)

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> DecoderOutput:
        _validate_input(x, self.in_channels)
        batch, steps, _ = x.shape
        if state is not None and not isinstance(state, dict):
            raise TypeError("state must be a dict or None")
        state = {} if state is None else state
        features, encoder_cache = self.encoder(x, state.get("encoder"))
        lag_cache = state.get("lag")
        if lag_cache is None:
            lag_cache = x.new_zeros(batch, self.max_lag, self.hidden)
        else:
            lag_cache = _state_tensor(lag_cache, x, (batch, self.max_lag, self.hidden), "lag")
        extended = torch.cat((lag_cache, features), dim=1)
        bank = torch.stack(
            [extended[:, self.max_lag - lag:self.max_lag - lag + steps] for lag in self.lags],
            dim=2,
        )
        gate = state.get("gate")
        if gate is None:
            gate = x.new_zeros(batch, len(self.lags))
        else:
            gate = _state_tensor(gate, x, (batch, len(self.lags)), "gate")
        if self.mode == "adaptive":
            logits = self.lag_logits(features)
            decay = self.slow_decay
            weights_per_step = []
            for index in range(steps):
                gate = decay * gate + (1 - decay) * logits[:, index]
                weights_per_step.append(torch.softmax(gate, dim=-1))
            weights = torch.stack(weights_per_step, dim=1)
        else:
            weights = self.fixed_weights.view(1, 1, -1).expand(batch, steps, -1)
        aligned = (bank * weights.unsqueeze(-1)).sum(dim=2)
        next_lag = extended[:, -self.max_lag:] if self.max_lag else extended[:, :0]
        return DecoderOutput(
            prediction=self.readout(aligned),
            state={"encoder": encoder_cache, "lag": next_lag, "gate": gate},
            aux={"weights": weights, "aligned_features": aligned, "slow_decay": self.slow_decay},
        )


class InnovationStateDecoder(nn.Module):
    """Latent prior corrected by an explicit EEG-feature innovation.

    ``prior = tanh(A*previous_latent)`` predicts a latent state. A learned
    observation projection predicts EEG features, ``innovation = h - H*prior``.
    A diagonal sigmoid gain then corrects the state before envelope readout.
    This is a small learned state-space scaffold, not a Kalman-filter claim.

    full: uses prior and observed EEG; prior_only: autonomous prior and readout,
    completely skipping the EEG encoder; eeg_only: local features and readout,
    with no recurrent latent state. All controls share the readout definition.
    No true previous envelope, future EEG or target audio enters forward.
    """

    def __init__(self, in_channels: int = 64, hidden: int = 32, mode: str = "full") -> None:
        super().__init__()
        if in_channels < 1 or hidden < 1:
            raise ValueError("in_channels and hidden must be positive")
        if mode not in {"full", "prior_only", "eeg_only"}:
            raise ValueError("mode must be full, prior_only or eeg_only")
        self.in_channels, self.hidden, self.mode = in_channels, hidden, mode
        self.encoder = LocalEncoder(in_channels=in_channels, hidden=hidden)
        self.prior_dynamics = nn.Linear(hidden, hidden)
        self.observation_predictor = nn.Linear(hidden, hidden)
        self.gain_net = nn.Linear(2 * hidden, hidden)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x: Tensor, state: dict[str, Any] | None = None) -> DecoderOutput:
        _validate_input(x, self.in_channels)
        batch, steps, _ = x.shape
        if state is not None and not isinstance(state, dict):
            raise TypeError("state must be a dict or None")
        state = {} if state is None else state
        if state.get("mode", self.mode) != self.mode:
            raise ValueError("state from another innovation control mode must be reset")
        if self.mode == "prior_only":
            features = None
            encoder_cache = None
        else:
            features, encoder_cache = self.encoder(x, state.get("encoder"))
        if self.mode == "eeg_only":
            latent_series = torch.tanh(features)
            zeros = torch.zeros_like(features)
            return DecoderOutput(
                prediction=self.readout(latent_series),
                state={"encoder": encoder_cache, "latent": None, "mode": self.mode},
                aux={
                    "prior": zeros, "predicted_observation": zeros,
                    "innovation": features, "gain": torch.ones_like(features),
                    "latent": latent_series,
                },
            )
        latent = state.get("latent")
        if latent is None:
            latent = x.new_zeros(batch, self.hidden)
        else:
            latent = _state_tensor(latent, x, (batch, self.hidden), "latent")
        priors, observations, innovations, gains, latent_steps = [], [], [], [], []
        for index in range(steps):
            prior = torch.tanh(self.prior_dynamics(latent))
            if self.mode == "full":
                observation = self.observation_predictor(prior)
                innovation = features[:, index] - observation
                gain = torch.sigmoid(self.gain_net(torch.cat((prior, innovation), dim=-1)))
                latent = torch.tanh(prior + gain * innovation)
            else:
                observation = torch.zeros_like(prior)
                innovation = torch.zeros_like(prior)
                gain = torch.zeros_like(prior)
                latent = prior
            priors.append(prior)
            observations.append(observation)
            innovations.append(innovation)
            gains.append(gain)
            latent_steps.append(latent)
        latent_series = torch.stack(latent_steps, dim=1)
        return DecoderOutput(
            prediction=self.readout(latent_series),
            state={"encoder": encoder_cache, "latent": latent, "mode": self.mode},
            aux={
                "prior": torch.stack(priors, dim=1),
                "predicted_observation": torch.stack(observations, dim=1),
                "innovation": torch.stack(innovations, dim=1),
                "gain": torch.stack(gains, dim=1), "latent": latent_series,
            },
        )
