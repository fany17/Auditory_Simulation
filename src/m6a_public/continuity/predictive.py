"""M3: causal EEG context with an explicit, separate predictive objective.

This is an untrained scaffold. ``mode`` specifies a downstream comparison,
not evidence that a model has actually been pretrained. Inputs are continuous
EEG-coordinate sequences; future EEG is permitted only as a loss target.
"""

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import DecoderOutput, LocalEncoder


class PredictiveContinuityDecoder(nn.Module):
    """Dense local features -> causal GRU context -> envelope.

    ``no_pretrain`` trains the downstream encoder, context and readout directly.
    ``frozen`` freezes both encoder and context; only the envelope head is
    trainable downstream. ``finetune`` permits both an explicit ``pretrain``
    stage and downstream adaptation. Construction never loads a checkpoint or
    performs pretraining. A future experiment must initialize frozen/finetune
    comparisons from the same approved pretrained encoder/context state.

    Downstream forward omits auxiliary heads entirely. Teacher EMA updates are
    explicit and never occur in forward, loss, or train/eval mode changes.
    """

    MODES = ("no_pretrain", "frozen", "finetune")
    STAGES = ("downstream", "pretrain")

    def __init__(
        self,
        in_channels: int = 64,
        hidden: int = 32,
        horizons: tuple[int, ...] = (4, 16, 64),
        mode: str = "finetune",
        stage: str = "downstream",
    ) -> None:
        super().__init__()
        if in_channels < 1 or hidden < 1:
            raise ValueError("in_channels and hidden must be positive")
        if not horizons or any(type(h) is not int or h < 1 for h in horizons):
            raise ValueError("horizons must contain positive integer sample offsets")
        if len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be unique")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.in_channels = in_channels
        self.hidden = hidden
        self.horizons = tuple(sorted(horizons))
        self.mode = mode
        self.encoder = LocalEncoder(in_channels=in_channels, hidden=hidden)
        self.context = nn.GRUCell(hidden, hidden)
        self.envelope_head = nn.Linear(hidden, 1)
        self.predictors = nn.ModuleDict(
            {str(h): nn.Linear(hidden, hidden) for h in self.horizons}
        )
        self.teacher = deepcopy(self.encoder)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.set_stage(stage)

    def set_stage(self, stage: str) -> "PredictiveContinuityDecoder":
        """Select computation/gradient paths; no optimizer or EMA operation."""
        if stage not in self.STAGES:
            raise ValueError(f"stage must be one of {self.STAGES}")
        if stage == "pretrain" and self.mode != "finetune":
            raise ValueError("pretrain stage requires finetune mode")
        self.stage = stage
        self.encoder.requires_grad_(self.mode != "frozen")
        self.context.requires_grad_(self.mode != "frozen")
        self.envelope_head.requires_grad_(stage == "downstream")
        self.predictors.requires_grad_(stage == "pretrain")
        self.teacher.requires_grad_(False)
        return self

    def train(self, mode: bool = True) -> "PredictiveContinuityDecoder":
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self, x: Tensor, state: dict[str, Any] | None = None
    ) -> DecoderOutput:
        if x.ndim != 3 or x.shape[1] < 1 or x.shape[2] != self.in_channels:
            raise ValueError("x must have shape [B,T>=1,in_channels]")
        if not torch.is_floating_point(x):
            raise ValueError("x must be floating point")
        if state is not None and set(state) != {"encoder", "context"}:
            raise ValueError("state must contain encoder and context")
        features, encoder_state = self.encoder(
            x, None if state is None else state["encoder"]
        )
        context = (
            x.new_zeros(x.shape[0], self.hidden)
            if state is None
            else state["context"]
        )
        if (
            context.shape != (x.shape[0], self.hidden)
            or context.device != x.device
            or context.dtype != x.dtype
        ):
            raise ValueError("context state must match input batch, dtype and device")
        states = []
        for feature in features.unbind(dim=1):
            context = self.context(feature, context)
            states.append(context)
        trajectory = torch.stack(states, dim=1)
        latent_predictions = (
            {h: self.predictors[str(h)](trajectory) for h in self.horizons}
            if self.stage == "pretrain"
            else {}
        )
        return DecoderOutput(
            prediction=self.envelope_head(trajectory),
            state={"encoder": encoder_state, "context": context},
            aux={
                "features": features,
                "context": trajectory,
                "latent_predictions": latent_predictions,
                "input_shape": tuple(x.shape),
                "starts_at_boundary": state is None,
            },
        )

    @torch.no_grad()
    def update_teacher(self, decay: float = 0.99) -> None:
        """Explicit EMA, intended only after a separately authorized update."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must lie in [0,1]")
        for teacher_param, online_param in zip(
            self.teacher.parameters(), self.encoder.parameters(), strict=True
        ):
            teacher_param.mul_(decay).add_(online_param, alpha=1.0 - decay)
        # LocalEncoder currently has no running statistics. This preserves an
        # explicit policy if non-learned buffers are added in a future revision.
        for teacher_buffer, online_buffer in zip(
            self.teacher.buffers(), self.encoder.buffers(), strict=True
        ):
            teacher_buffer.copy_(online_buffer)

    def parameter_counts(self) -> dict[str, int]:
        """Separate downstream inference, objective and frozen EMA copies."""
        def count(*modules: nn.Module) -> int:
            return sum(p.numel() for module in modules for p in module.parameters())

        return {
            "registered_total": count(self),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "inference_active": count(self.encoder, self.context, self.envelope_head),
            "pretraining_online_active": count(self.encoder, self.context, self.predictors),
            "teacher_frozen": count(self.teacher),
            "auxiliary_heads": count(self.predictors),
        }

    def predictive_loss(
        self,
        output: DecoderOutput,
        target_source: Tensor,
        valid_mask: Tensor | None = None,
        variance_weight: float = 0.1,
        variance_target: float = 1.0,
        epsilon: float = 1e-4,
    ) -> dict[str, Any]:
        """Predict detached future teacher latents within one continuous segment.

        ``target_source`` must be the same complete input trajectory used for
        ``output``. It is a separate argument to make target/inference isolation
        testable. The caller owns this identity contract; shape checks cannot
        certify it. Each row starts at a real stream boundary and may have only
        trailing padding. Internal gaps/record changes require separate calls.

        Loss on carried chunks is intentionally rejected: a correct future
        implementation would additionally carry version-consistent EMA teacher
        convolution state. Streaming downstream forward is already supported.

        Mean squared distance of L2-normalized predictions and target latents is
        averaged equally across horizons. The non-collapse term penalizes low
        per-channel variance in both online local features and causal context.
        This objective alone is not evidence of useful neural representation.
        """
        if self.stage != "pretrain":
            raise ValueError("predictive_loss requires an explicit pretrain stage")
        if not output.aux.get("starts_at_boundary", False):
            raise ValueError("predictive_loss requires a complete boundary-started segment")
        if tuple(target_source.shape) != output.aux.get("input_shape"):
            raise ValueError("target_source must match the complete forward input shape")
        features = output.aux["features"]
        context = output.aux["context"]
        if target_source.device != features.device or target_source.dtype != features.dtype:
            raise ValueError("target_source must match forward dtype and device")
        if variance_weight < 0 or variance_target <= 0 or epsilon <= 0:
            raise ValueError("variance_weight>=0, variance_target>0 and epsilon>0 required")
        batch, length = target_source.shape[:2]
        if valid_mask is None:
            valid_mask = torch.ones(batch, length, dtype=torch.bool, device=features.device)
        if (
            valid_mask.shape != (batch, length)
            or valid_mask.dtype != torch.bool
            or valid_mask.device != features.device
        ):
            raise ValueError("valid_mask must be bool [B,T] on the input device")
        if torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1]).item():
            raise ValueError("valid_mask allows trailing padding only; reset at gaps/boundaries")
        if int(valid_mask.sum().item()) < 2:
            raise ValueError("at least two valid samples are required")
        masks: dict[int, Tensor] = {}
        valid_pairs: dict[int, int] = {}
        for horizon in self.horizons:
            if length <= horizon:
                raise ValueError(f"insufficient valid support for horizon {horizon}")
            # Include source, target, and every intermediate point: no objective
            # may bridge a gap just because both endpoints are marked valid.
            mask = valid_mask.unfold(1, horizon + 1, 1).all(dim=-1)
            count = int(mask.sum().item())
            if count < 2:
                raise ValueError(f"insufficient valid support for horizon {horizon}")
            masks[horizon] = mask
            valid_pairs[horizon] = count
        with torch.no_grad():
            teacher_features, _ = self.teacher(target_source.detach(), None)
        horizon_losses: dict[int, Tensor] = {}
        for horizon in self.horizons:
            predicted = output.aux["latent_predictions"][horizon][:, :-horizon]
            target = teacher_features[:, horizon:]
            predicted = F.normalize(predicted[masks[horizon]], dim=-1, eps=epsilon)
            target = F.normalize(target[masks[horizon]], dim=-1, eps=epsilon)
            horizon_losses[horizon] = (predicted - target).square().sum(dim=-1).mean()
        prediction_loss = torch.stack(list(horizon_losses.values())).mean()
        penalties = []
        for representation in (features, context):
            std = torch.sqrt(representation[valid_mask].var(dim=0, unbiased=False) + epsilon)
            penalties.append(F.relu(variance_target - std).mean())
        variance_loss = torch.stack(penalties).mean()
        return {
            "loss": prediction_loss + variance_weight * variance_loss,
            "prediction_loss": prediction_loss,
            "variance_loss": variance_loss,
            "horizon_losses": horizon_losses,
            "valid_pairs": valid_pairs,
        }
