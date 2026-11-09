"""Explicit continuity boundaries; no dataset loader or implicit global state."""
from dataclasses import dataclass

import torch

from .common import detach_state


@dataclass(frozen=True)
class StreamIdentity:
    subject: str
    session: str
    record: str
    split: str


class StreamRunner:
    """Homogeneous-batch stream; any lane identity change resets the whole batch.

    A production loader must group boundary-aligned lanes or provide per-lane
    state management. Gaps within an identity require an explicit reset.
    """

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.state = None
        self.identities = None
        self.next_sample = None

    def detach(self):
        self.state = detach_state(self.state)

    def step(self, x, identities, start_sample):
        identities = tuple(identities)
        if len(identities) != x.shape[0] or not all(
            isinstance(key, StreamIdentity) for key in identities
        ):
            raise ValueError("one StreamIdentity is required for each batch lane")
        if not isinstance(start_sample, int) or start_sample < 0:
            raise ValueError("start_sample must be a nonnegative integer")
        changed = identities != self.identities
        if not changed and start_sample != self.next_sample:
            raise ValueError("noncontiguous samples: explicitly reset before a gap")
        previous = None if changed else self.state
        output = self.model(x, previous)
        self.state = output.state
        self.identities = identities
        self.next_sample = start_sample + x.shape[1]
        return output


def compare_histories(model, current, real_history, wrong_history):
    """Use identical current EEG with real, reset and mismatched prefixes.

    Caller supplies admissible histories from one split; no data lookup occurs.
    Returns differentiable outputs and does not change module parameters/state.
    This intervenes on the whole prefix, including the local encoder cache.
    Use compare_state_histories to isolate post-encoder history mechanisms.
    """
    if real_history.shape != wrong_history.shape:
        raise ValueError("history duration and channel/batch shape must match")
    real = model(real_history).state
    wrong = model(wrong_history).state
    return {
        "real": model(current, real),
        "reset": model(current, None),
        "mismatched": model(current, wrong),
    }


def compare_state_histories(model, current, real_history, wrong_history):
    """Hold both current EEG and real local-encoder cache fixed.

    Replace/reset only post-encoder state (branches, fast/slow, context,
    lag/gate, latent). This isolates its contribution from the first 16 samples
    of local-encoder boundary effects. Only registered SS06 models are accepted.
    """
    if real_history.shape != wrong_history.shape:
        raise ValueError("history duration and channel/batch shape must match")
    real = model(real_history).state
    wrong = model(wrong_history).state
    keys = set(real) - {"encoder", "mode"}
    allowed = {"branches", "fast", "slow", "context", "lag", "gate", "latent"}
    if not keys or not keys <= allowed or set(wrong) != set(real):
        raise ValueError("unsupported state schema for post-encoder intervention")

    def zeros(tree):
        if isinstance(tree, torch.Tensor):
            return torch.zeros_like(tree)
        if isinstance(tree, dict):
            return {key: zeros(value) for key, value in tree.items()}
        if tree is None:
            return None
        raise ValueError("post-encoder states must contain only tensors/dicts/None")

    reset = {**real, **{key: zeros(real[key]) for key in keys}}
    mismatch = {**real, **{key: wrong[key] for key in keys}}
    return {"real": model(current, real), "reset": model(current, reset),
            "mismatched": model(current, mismatch)}
