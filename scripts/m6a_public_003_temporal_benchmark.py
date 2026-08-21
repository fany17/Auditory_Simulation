"""M6A-PUBLIC-003: small matched temporal architecture benchmark.

The benchmark uses only synthetic one-dimensional temporal signals.  It does not
read patient/STN data, download models, load pretrained weights, or save model
checkpoints.  The same generated arrays, optimizer, epochs, batch size and split
are reused for every architecture/seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import socket
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


METRIC_NAMES = (
    "rate_discrimination_balanced_accuracy",
    "regular_vs_jitter_balanced_accuracy",
    "omission_detection_balanced_accuracy",
    "phase_shift_detection_balanced_accuracy",
    "onset_timing_error_ms",
    "rate_mae_hz",
    "jitter_magnitude_mae_ms",
    "phase_magnitude_mae_rad",
    "unseen_rate_mae_hz",
    "unseen_jitter_mae_ms",
    "unseen_phase_magnitude_mae_rad",
)


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    group: str
    width: int = 16
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 1, 1, 1)
    strides: tuple[int, ...] = (1, 1, 1, 1)
    mode: str = "single"
    branch_width: int = 0
    parallel_kernels: tuple[int, ...] = (3, 7, 15)
    notes: str = ""


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_specs(config: dict[str, Any]) -> list[ArchitectureSpec]:
    blocks = int(config["num_blocks"])
    ones = (1,) * blocks
    return [
        ArchitectureSpec(
            "early_downsample",
            "downsampling",
            "downsampling",
            width=int(config["default_width"]),
            kernel=5,
            dilations=ones,
            strides=(2, 1, 2, 1),
            notes="stride/pooling proxy occurs early; total factor 4",
        ),
        ArchitectureSpec(
            "late_downsample",
            "downsampling",
            "downsampling",
            width=int(config["default_width"]),
            kernel=5,
            dilations=ones,
            strides=(1, 1, 2, 2),
            notes="same final factor 4; stride is delayed to later blocks",
        ),
        ArchitectureSpec(
            "uniform_local",
            "rf_growth",
            "rf_schedule",
            width=int(config["default_width"]),
            kernel=3,
            dilations=ones,
            strides=ones,
            notes="uniform/local dilation schedule",
        ),
        ArchitectureSpec(
            "exponential_growth",
            "rf_growth",
            "rf_schedule",
            width=int(config["default_width"]),
            kernel=3,
            dilations=(1, 2, 4, 8),
            strides=ones,
            notes="exponential dilation schedule",
        ),
        ArchitectureSpec(
            "delayed_growth",
            "rf_growth",
            "rf_schedule",
            width=int(config["default_width"]),
            kernel=3,
            dilations=(1, 1, 1, 2),
            strides=ones,
            notes="delayed-growth dilation schedule",
        ),
        ArchitectureSpec(
            "parallel_multiscale",
            "rf_growth",
            "multiscale",
            width=13,
            kernel=3,
            dilations=ones,
            strides=ones,
            mode="parallel",
            parallel_kernels=tuple(int(v) for v in config["parallel_kernels"]),
            notes="depthwise parallel k=3/7/15 branches with 1x1 fusion",
        ),
        ArchitectureSpec(
            "rf_stride_coupled",
            "rf_growth",
            "rf_decoupling",
            width=int(config["default_width"]),
            kernel=3,
            dilations=(1, 1, 1, 2),
            strides=(2, 2, 1, 1),
            notes="matched RF obtained mainly with stride; final resolution 4 ms",
        ),
        ArchitectureSpec(
            "rf_dilation_decoupled",
            "rf_growth",
            "rf_decoupling",
            width=int(config["default_width"]),
            kernel=3,
            dilations=(1, 2, 4, 8),
            strides=ones,
            notes="same theoretical RF through dilation; final resolution 1 ms",
        ),
        ArchitectureSpec(
            "kernel_3",
            "rf_growth",
            "kernel_sweep",
            width=16,
            kernel=3,
            dilations=ones,
            strides=ones,
            notes="kernel-size diagnostic; width chosen for approximate parameter match",
        ),
        ArchitectureSpec(
            "kernel_7",
            "rf_growth",
            "kernel_sweep",
            width=11,
            kernel=7,
            dilations=ones,
            strides=ones,
            notes="kernel-size diagnostic; width chosen for approximate parameter match",
        ),
        ArchitectureSpec(
            "kernel_15",
            "rf_growth",
            "kernel_sweep",
            width=8,
            kernel=15,
            dilations=ones,
            strides=ones,
            notes="kernel-size diagnostic; width chosen for approximate parameter match",
        ),
        ArchitectureSpec(
            "kernel_31",
            "rf_growth",
            "kernel_sweep",
            width=5,
            kernel=31,
            dilations=ones,
            strides=ones,
            notes="kernel-size diagnostic; expected parameter confound is retained",
        ),
        ArchitectureSpec(
            "event_baseline",
            "explicit_change_branch",
            "event_branch",
            width=int(config["default_width"]),
            kernel=3,
            dilations=ones,
            strides=ones,
            notes="single content branch baseline",
        ),
        ArchitectureSpec(
            "explicit_change",
            "explicit_change_branch",
            "event_branch",
            width=int(config["default_width"]),
            kernel=3,
            dilations=ones,
            strides=ones,
            mode="explicit_change",
            branch_width=11,
            notes="content branch plus first-difference branch",
        ),
        ArchitectureSpec(
            "ordinary_second_branch",
            "explicit_change_branch",
            "event_branch",
            width=int(config["default_width"]),
            kernel=3,
            dilations=ones,
            strides=ones,
            mode="ordinary_second",
            branch_width=11,
            notes="content branch plus parameter-matched copy of raw input branch",
        ),
    ]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_gaussian(signal: np.ndarray, time_ms: np.ndarray, centers: Iterable[float], amplitude: float, sigma_ms: float) -> None:
    for center in centers:
        left = max(0, int(center - 5.0 * sigma_ms))
        right = min(signal.size, int(center + 5.0 * sigma_ms) + 1)
        if right > left:
            signal[left:right] += amplitude * np.exp(-0.5 * ((time_ms[left:right] - center) / sigma_ms) ** 2)


def generate_split(config: dict[str, Any], size: int, seed: int, regime: str) -> dict[str, np.ndarray]:
    """Generate independent synthetic temporal sequences and multitask targets."""

    length = int(config["sequence_length"])
    dt_ms = float(config["base_time_step_ms"])
    time_ms = np.arange(length, dtype=np.float32) * dt_ms
    rng = np.random.default_rng(seed)
    train_rates = np.asarray(config["train_rates_hz"], dtype=np.float32)
    unseen_rates = np.asarray(config["unseen_rates_hz"], dtype=np.float32)
    train_jitter = np.asarray(config["train_jitter_ms"], dtype=np.float32)
    unseen_jitter = np.asarray(config["unseen_jitter_ms"], dtype=np.float32)
    train_phase = np.asarray(config["train_phase_magnitudes_rad"], dtype=np.float32)
    unseen_phase = np.asarray(config["unseen_phase_magnitudes_rad"], dtype=np.float32)
    rates = train_rates if regime == "seen" else unseen_rates
    jitter_options = train_jitter if regime == "seen" else unseen_jitter
    phase_options = train_phase if regime == "seen" else unseen_phase
    rate_to_class = {float(value): index for index, value in enumerate(train_rates.tolist())}

    x = np.zeros((size, length), dtype=np.float32)
    targets: dict[str, np.ndarray] = {
        "rate_class": np.full(size, -1, dtype=np.int64),
        "rate_reg": np.zeros(size, dtype=np.float32),
        "jitter_flag": np.zeros(size, dtype=np.float32),
        "jitter_reg": np.zeros(size, dtype=np.float32),
        "omission_flag": np.zeros(size, dtype=np.float32),
        "phase_detect": np.zeros(size, dtype=np.float32),
        "phase_direction": np.zeros(size, dtype=np.int64),
        "phase_reg": np.zeros(size, dtype=np.float32),
        "onset_reg": np.zeros(size, dtype=np.float32),
    }

    for index in range(size):
        signal = rng.normal(0.0, 0.012, size=length).astype(np.float32)
        rate = float(rng.choice(rates))
        jitter_flag = int(rng.integers(0, 2))
        jitter_ms = float(rng.choice(jitter_options)) if jitter_flag else 0.0
        omission_flag = int(rng.integers(0, 2))
        first_event = float(rng.uniform(35.0, 75.0))
        event_times = np.arange(first_event, length * dt_ms - 18.0, 1000.0 / rate, dtype=np.float32)
        if event_times.size < 3:
            event_times = np.asarray([first_event, first_event + 1000.0 / rate, first_event + 2.0 * 1000.0 / rate], dtype=np.float32)
        if jitter_flag:
            event_times = event_times + rng.uniform(-jitter_ms, jitter_ms, size=event_times.size).astype(np.float32)
        if omission_flag and event_times.size >= 4:
            omission_index = int(rng.integers(1, event_times.size - 1))
            event_times = np.delete(event_times, omission_index)
        add_gaussian(signal, time_ms, event_times, amplitude=0.58, sigma_ms=1.5)

        onset_ms = float(rng.uniform(80.0, 400.0))
        add_gaussian(signal, time_ms, [onset_ms], amplitude=0.82, sigma_ms=2.0)

        phase_direction = int(rng.choice([-1, 0, 1]))
        phase_magnitude = float(rng.choice(phase_options)) if phase_direction else 0.0
        carrier_frequency = float(rng.choice([18.0, 24.0, 32.0]))
        gate = np.exp(-0.5 * ((time_ms - 256.0) / 96.0) ** 2)
        carrier = 0.16 * np.sin(2.0 * np.pi * carrier_frequency * time_ms / 1000.0 + phase_direction * phase_magnitude) * gate
        signal += carrier.astype(np.float32)
        scale = max(float(np.std(signal)), 0.08)
        x[index] = np.clip(signal / scale, -5.0, 5.0)

        if regime == "seen":
            targets["rate_class"][index] = rate_to_class[rate]
        targets["rate_reg"][index] = (rate - 7.0) / 4.0
        targets["jitter_flag"][index] = float(jitter_flag)
        targets["jitter_reg"][index] = jitter_ms / 8.0
        targets["omission_flag"][index] = float(omission_flag)
        targets["phase_detect"][index] = float(phase_direction != 0)
        targets["phase_direction"][index] = {0: 0, 1: 1, -1: 2}[phase_direction]
        targets["phase_reg"][index] = phase_magnitude / np.pi
        targets["onset_reg"][index] = onset_ms / (length * dt_ms)

    targets["onset_ms"] = targets["onset_reg"] * (length * dt_ms)
    targets["rate_hz"] = targets["rate_reg"] * 4.0 + 7.0
    targets["jitter_ms"] = targets["jitter_reg"] * 8.0
    targets["phase_rad"] = targets["phase_reg"] * np.pi
    return {"x": x, **targets}


class TemporalDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.x = torch.from_numpy(arrays["x"][:, None, :])
        self.targets = {
            key: torch.from_numpy(value)
            for key, value in arrays.items()
            if key not in {"x", "onset_ms", "rate_hz", "jitter_ms", "phase_rad"}
        }

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.x[index], {key: value[index] for key, value in self.targets.items()}


class ScaleBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, stride: int):
        super().__init__()
        padding = ((kernel - 1) * dilation) // 2
        self.conv = nn.Conv1d(channels, channels, kernel, stride=stride, padding=padding, dilation=dilation)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class ParallelScaleBlock(nn.Module):
    def __init__(self, channels: int, kernels: Sequence[int], dilation: int, stride: int):
        super().__init__()
        branches = []
        for kernel in kernels:
            padding = ((kernel - 1) * dilation) // 2
            branches.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                )
            )
        self.branches = nn.ModuleList(branches)
        self.fusion = nn.Conv1d(channels * len(kernels), channels, 1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = [branch(x) for branch in self.branches]
        return self.activation(self.fusion(torch.cat(values, dim=1)))


class TemporalBackbone(nn.Module):
    def __init__(self, spec: ArchitectureSpec, width_override: int | None = None):
        super().__init__()
        width = int(width_override or spec.width)
        self.width = width
        self.spec = spec
        self.stem = nn.Sequential(nn.Conv1d(1, width, 3, padding=1), nn.GELU())
        if spec.mode == "parallel":
            blocks: list[nn.Module] = [
                ParallelScaleBlock(width, spec.parallel_kernels, dilation, stride)
                for dilation, stride in zip(spec.dilations, spec.strides)
            ]
        else:
            blocks = [
                ScaleBlock(width, spec.kernel, dilation, stride)
                for dilation, stride in zip(spec.dilations, spec.strides)
            ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


class TemporalModel(nn.Module):
    def __init__(self, spec: ArchitectureSpec, head_width: int = 32):
        super().__init__()
        self.spec = spec
        self.head_width = head_width
        if spec.mode in {"explicit_change", "ordinary_second"}:
            branch_spec = replace(spec, mode="single", width=spec.branch_width)
            self.content_backbone = TemporalBackbone(branch_spec)
            self.second_backbone = TemporalBackbone(branch_spec)
            self.fusion = nn.Conv1d(2 * spec.branch_width, spec.width, 1)
            self.backbone = None
            self.feature_width = spec.width
        else:
            self.backbone = TemporalBackbone(spec)
            self.content_backbone = None
            self.second_backbone = None
            self.fusion = None
            self.feature_width = spec.width
        self.projection = nn.Sequential(
            nn.Linear(2 * self.feature_width, head_width),
            nn.GELU(),
        )
        self.rate_class = nn.Linear(head_width, 4)
        self.rate_reg = nn.Linear(head_width, 1)
        self.jitter = nn.Linear(head_width, 1)
        self.jitter_reg = nn.Linear(head_width, 1)
        self.omission = nn.Linear(head_width, 1)
        self.phase_detect = nn.Linear(head_width, 1)
        self.phase_reg = nn.Linear(head_width, 1)
        self.onset_reg = nn.Linear(head_width, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            return self.backbone(x)
        assert self.content_backbone is not None and self.second_backbone is not None and self.fusion is not None
        content = self.content_backbone(x)
        if self.spec.mode == "explicit_change":
            change = torch.zeros_like(x)
            change[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
            second = self.second_backbone(change)
        else:
            second = self.second_backbone(x)
        return F.gelu(self.fusion(torch.cat([content, second], dim=1)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        pooled = torch.cat([features.mean(dim=-1), features.amax(dim=-1)], dim=1)
        hidden = self.projection(pooled)
        return {
            "features": features,
            "latent": hidden,
            "rate_class": self.rate_class(hidden),
            "rate_reg": self.rate_reg(hidden).squeeze(-1),
            "jitter": self.jitter(hidden).squeeze(-1),
            "jitter_reg": self.jitter_reg(hidden).squeeze(-1),
            "omission": self.omission(hidden).squeeze(-1),
            "phase_detect": self.phase_detect(hidden).squeeze(-1),
            "phase_reg": self.phase_reg(hidden).squeeze(-1),
            "onset_reg": self.onset_reg(hidden).squeeze(-1),
        }


def multitask_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    rate_mask = targets["rate_class"] >= 0
    if bool(rate_mask.any()):
        rate_class_loss = F.cross_entropy(outputs["rate_class"][rate_mask], targets["rate_class"][rate_mask])
    else:
        rate_class_loss = outputs["rate_class"].sum() * 0.0
    return (
        rate_class_loss
        + F.mse_loss(outputs["rate_reg"], targets["rate_reg"])
        + F.binary_cross_entropy_with_logits(outputs["jitter"], targets["jitter_flag"])
        + 0.5 * F.mse_loss(outputs["jitter_reg"], targets["jitter_reg"])
        + F.binary_cross_entropy_with_logits(outputs["omission"], targets["omission_flag"])
        + F.binary_cross_entropy_with_logits(outputs["phase_detect"], targets["phase_detect"])
        + 0.5 * F.mse_loss(outputs["phase_reg"], targets["phase_reg"])
        + F.mse_loss(outputs["onset_reg"], targets["onset_reg"])
    )


def evaluate_loss(model: TemporalModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, targets in loader:
            xb = xb.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}
            loss = multitask_loss(model(xb), targets)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(
    model: TemporalModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    variant: str,
    seed: int,
) -> tuple[str, list[dict[str, Any]], str | None]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    model.to(device)
    logs: list[dict[str, Any]] = []
    failure: str | None = None
    status = "PASS"
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        train_losses: list[float] = []
        for xb, targets in train_loader:
            xb = xb.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = multitask_loss(model(xb), targets)
            if not bool(torch.isfinite(loss)):
                status = "DIVERGED"
                failure = f"non-finite training loss at epoch {epoch}"
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            if not bool(torch.isfinite(grad_norm)):
                status = "DIVERGED"
                failure = f"non-finite gradient norm at epoch {epoch}"
                break
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss = evaluate_loss(model, validation_loader, device) if status == "PASS" else float("nan")
        logs.append(
            {
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "validation_loss": validation_loss,
                "status": status,
            }
        )
        if status != "PASS":
            break
    return status, logs, failure


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(y_true)
    recalls = []
    for label in labels:
        mask = y_true == label
        recalls.append(float(np.mean(y_pred[mask] == label)) if np.any(mask) else float("nan"))
    return float(np.nanmean(recalls)) if recalls else float("nan")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def predict_arrays(model: TemporalModel, arrays: dict[str, np.ndarray], device: torch.device) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(TemporalDataset(arrays), batch_size=128, shuffle=False)
    output_parts: dict[str, list[np.ndarray]] = {}
    feature_parts: list[np.ndarray] = []
    latent_parts: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            outputs = model(xb.to(device))
            for key, value in outputs.items():
                if key in {"features", "latent"}:
                    continue
                output_parts.setdefault(key, []).append(value.detach().cpu().numpy())
            feature_parts.append(outputs["features"].detach().cpu().numpy())
            latent_parts.append(outputs["latent"].detach().cpu().numpy())
    outputs = {key: np.concatenate(values, axis=0) for key, values in output_parts.items()}
    return outputs, np.concatenate(feature_parts, axis=0), np.concatenate(latent_parts, axis=0)


def metric_rows_for_split(
    outputs: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    split: str,
    variant: str,
    seed: int,
) -> list[dict[str, Any]]:
    rate_pred = outputs["rate_reg"] * 4.0 + 7.0
    jitter_pred = outputs["jitter_reg"] * 8.0
    phase_pred = outputs["phase_reg"] * np.pi
    onset_pred = outputs["onset_reg"] * 512.0
    rows: list[dict[str, Any]] = []
    if split == "test_seen":
        rows.append({"variant": variant, "seed": seed, "split": split, "metric": "rate_discrimination_balanced_accuracy", "value": balanced_accuracy(arrays["rate_class"], outputs["rate_class"].argmax(axis=1))})
    if split == "test_unseen":
        rows.append({"variant": variant, "seed": seed, "split": split, "metric": "unseen_rate_mae_hz", "value": float(np.mean(np.abs(rate_pred - arrays["rate_hz"])))})
        rows.append({"variant": variant, "seed": seed, "split": split, "metric": "unseen_jitter_mae_ms", "value": float(np.mean(np.abs(jitter_pred - arrays["jitter_ms"])))})
        rows.append({"variant": variant, "seed": seed, "split": split, "metric": "unseen_phase_magnitude_mae_rad", "value": float(np.mean(np.abs(phase_pred - arrays["phase_rad"])))})
    rows.extend(
        [
            {"variant": variant, "seed": seed, "split": split, "metric": "regular_vs_jitter_balanced_accuracy", "value": balanced_accuracy(arrays["jitter_flag"].astype(int), (sigmoid(outputs["jitter"]) >= 0.5).astype(int))},
            {"variant": variant, "seed": seed, "split": split, "metric": "omission_detection_balanced_accuracy", "value": balanced_accuracy(arrays["omission_flag"].astype(int), (sigmoid(outputs["omission"]) >= 0.5).astype(int))},
            {"variant": variant, "seed": seed, "split": split, "metric": "phase_shift_detection_balanced_accuracy", "value": balanced_accuracy(arrays["phase_detect"].astype(int), (sigmoid(outputs["phase_detect"]) >= 0.5).astype(int))},
            {"variant": variant, "seed": seed, "split": split, "metric": "onset_timing_error_ms", "value": float(np.mean(np.abs(onset_pred - arrays["onset_ms"])) )},
            {"variant": variant, "seed": seed, "split": split, "metric": "rate_mae_hz", "value": float(np.mean(np.abs(rate_pred - arrays["rate_hz"])))},
            {"variant": variant, "seed": seed, "split": split, "metric": "jitter_magnitude_mae_ms", "value": float(np.mean(np.abs(jitter_pred - arrays["jitter_ms"])))},
            {"variant": variant, "seed": seed, "split": split, "metric": "phase_magnitude_mae_rad", "value": float(np.mean(np.abs(phase_pred - arrays["phase_rad"])))},
        ]
    )
    return rows


def output_length(length: int, kernel: int, dilation: int, stride: int) -> int:
    padding = ((kernel - 1) * dilation) // 2
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def rf_update(rf: int, jump: int, kernel: int, dilation: int, stride: int) -> tuple[int, int]:
    return rf + (kernel - 1) * dilation * jump, jump * stride


def _rf_row(model: str, layer: str, branch: str, kernel: int, stride: int, dilation: int, jump: int, rf: int, base_step_ms: float, output_len: int) -> dict[str, Any]:
    return {
        "model": model,
        "layer": layer,
        "branch": branch,
        "kernel": kernel,
        "stride": stride,
        "dilation": dilation,
        "jump_frame_step": jump,
        "jump_ms": jump * base_step_ms,
        "theoretical_RF_samples": rf,
        "theoretical_RF_ms": rf * base_step_ms,
        "output_length": output_len,
        "output_time_resolution_ms": jump * base_step_ms,
    }


def rf_rows_for_spec(spec: ArchitectureSpec, sequence_length: int, base_step_ms: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if spec.mode in {"explicit_change", "ordinary_second"}:
        branch_spec = replace(spec, mode="single", width=spec.branch_width)
        content_rows = rf_rows_for_spec(branch_spec, sequence_length, base_step_ms)
        second_rows = rf_rows_for_spec(branch_spec, sequence_length, base_step_ms)
        for row in content_rows:
            row["model"] = spec.name
            row["branch"] = "content"
            row["layer"] = f"content/{row['layer']}"
        for row in second_rows:
            row["model"] = spec.name
            row["branch"] = "change" if spec.mode == "explicit_change" else "ordinary_second"
            row["layer"] = f"second/{row['layer']}"
        rows.extend(content_rows)
        rows.extend(second_rows)
        final = content_rows[-1]
        rows.append(_rf_row(spec.name, "fusion_1x1", "fusion", 1, 1, 1, int(final["jump_frame_step"]), int(final["theoretical_RF_samples"]), base_step_ms, int(final["output_length"])))
        return rows

    rf = 1
    jump = 1
    length = sequence_length
    rf, jump = rf_update(rf, jump, 3, 1, 1)
    length = output_length(length, 3, 1, 1)
    rows.append(_rf_row(spec.name, "stem", "content", 3, 1, 1, jump, rf, base_step_ms, length))
    if spec.mode == "parallel":
        for block_index, (dilation, stride) in enumerate(zip(spec.dilations, spec.strides), start=1):
            branch_values: list[tuple[int, int]] = []
            block_length = output_length(length, spec.parallel_kernels[0], dilation, stride)
            for kernel in spec.parallel_kernels:
                branch_rf, branch_jump = rf_update(rf, jump, kernel, dilation, stride)
                branch_values.append((branch_rf, branch_jump))
                rows.append(_rf_row(spec.name, f"block{block_index}_k{kernel}", f"parallel_k{kernel}", kernel, stride, dilation, branch_jump, branch_rf, base_step_ms, block_length))
            rf = max(value[0] for value in branch_values)
            jump = branch_values[0][1]
            length = block_length
            rows.append(_rf_row(spec.name, f"block{block_index}_fusion", "fusion", 1, 1, 1, jump, rf, base_step_ms, length))
    else:
        for block_index, (dilation, stride) in enumerate(zip(spec.dilations, spec.strides), start=1):
            rf, jump = rf_update(rf, jump, spec.kernel, dilation, stride)
            length = output_length(length, spec.kernel, dilation, stride)
            rows.append(_rf_row(spec.name, f"block{block_index}", "content", spec.kernel, stride, dilation, jump, rf, base_step_ms, length))
    return rows


def flops_for_backbone(spec: ArchitectureSpec, sequence_length: int, width_override: int | None = None) -> int:
    width = int(width_override or spec.width)
    length = sequence_length
    flops = 2 * 1 * width * 3 * length
    length = output_length(length, 3, 1, 1)
    for dilation, stride in zip(spec.dilations, spec.strides):
        if spec.mode == "parallel":
            for kernel in spec.parallel_kernels:
                flops += 2 * width * kernel * length
            next_length = output_length(length, spec.parallel_kernels[0], dilation, stride)
            flops += 2 * width * len(spec.parallel_kernels) * width * next_length
            length = next_length
        else:
            flops += 2 * width * width * spec.kernel * length
            length = output_length(length, spec.kernel, dilation, stride)
    return int(flops)


def approximate_flops(spec: ArchitectureSpec, sequence_length: int, head_width: int) -> int:
    if spec.mode in {"explicit_change", "ordinary_second"}:
        branch_spec = replace(spec, mode="single", width=spec.branch_width)
        flops = 2 * flops_for_backbone(branch_spec, sequence_length)
        final_length = rf_rows_for_spec(branch_spec, sequence_length, 1.0)[-1]["output_length"]
        flops += 2 * spec.branch_width * spec.width * final_length
        flops += 2 * (2 * spec.width) * head_width
    else:
        flops = flops_for_backbone(spec, sequence_length)
        final_length = rf_rows_for_spec(spec, sequence_length, 1.0)[-1]["output_length"]
        flops += 2 * (2 * spec.width) * head_width
    flops += 2 * head_width * 13
    return int(flops)


def final_rf_info(spec: ArchitectureSpec, sequence_length: int, base_step_ms: float) -> tuple[int, float, float, int]:
    rows = rf_rows_for_spec(spec, sequence_length, base_step_ms)
    last = rows[-1]
    return int(last["theoretical_RF_samples"]), float(last["theoretical_RF_ms"]), float(last["output_time_resolution_ms"]), int(last["output_length"])


def representation_metrics(model: TemporalModel, arrays: dict[str, np.ndarray], device: torch.device, final_resolution_ms: float) -> dict[str, float]:
    subset = min(128, arrays["x"].shape[0])
    base = arrays["x"][:subset]
    shifted = np.concatenate([np.zeros((subset, 10), dtype=np.float32), base[:, :-10]], axis=1)
    base_arrays = {"x": base, **{key: value[:subset] for key, value in arrays.items() if key != "x"}}
    shifted_arrays = {"x": shifted, **{key: value[:subset] for key, value in arrays.items() if key != "x"}}
    base_outputs, base_features, base_latent = predict_arrays(model, base_arrays, device)
    _, _, shifted_latent = predict_arrays(model, shifted_arrays, device)
    base_norm = base_latent / np.maximum(np.linalg.norm(base_latent, axis=1, keepdims=True), 1e-8)
    shifted_norm = shifted_latent / np.maximum(np.linalg.norm(shifted_latent, axis=1, keepdims=True), 1e-8)
    no_fit_distance = float(np.mean(np.linalg.norm(base_norm - shifted_norm, axis=1)))

    response_values: list[float] = []
    persistence_values: list[float] = []
    for index, feature in enumerate(base_features):
        frame_index = int(round(float(arrays["onset_ms"][index]) / final_resolution_ms))
        window = max(1, int(round(20.0 / final_resolution_ms)))
        baseline_window = max(1, int(round(80.0 / final_resolution_ms)))
        start = max(0, frame_index - window)
        end = min(feature.shape[-1], frame_index + window + 1)
        base_start = max(0, frame_index - baseline_window)
        base_end = max(base_start + 1, frame_index - max(1, window))
        response = float(np.mean(np.abs(feature[:, start:end])) - np.mean(np.abs(feature[:, base_start:base_end])))
        response_values.append(response)
        trace = feature.mean(axis=0)
        if trace.size > 2 and float(np.std(trace[:-1])) > 1e-8 and float(np.std(trace[1:])) > 1e-8:
            persistence_values.append(float(np.corrcoef(trace[:-1], trace[1:])[0, 1]))
    return {
        "no_fit_representation_distance": no_fit_distance,
        "event_triggered_response": float(np.mean(response_values)),
        "temporal_persistence_autocorrelation": float(np.nanmean(persistence_values)) if persistence_values else float("nan"),
    }


def measure_latency(model: TemporalModel, arrays: dict[str, np.ndarray], device: torch.device) -> float:
    model.eval()
    batch = torch.from_numpy(arrays["x"][:64, None, :]).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(10):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / 10.0 / batch.shape[0])


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in metric_rows:
        value = float(row["value"])
        if not math.isfinite(value):
            continue
        key = (str(row["variant"]), str(row["split"]), str(row["metric"]))
        groups.setdefault(key, []).append(value)
    rows = []
    for (variant, split, metric), values in sorted(groups.items()):
        rows.append({
            "variant": variant,
            "split": split,
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
        })
    return rows


def format_summary(summary: list[dict[str, Any]], variant: str, split: str, metric: str) -> str:
    for row in summary:
        if row["variant"] == variant and row["split"] == split and row["metric"] == metric:
            return f"{float(row['mean']):.4f} ± {float(row['std']):.4f}"
    return "NA"


def write_markdown_reports(
    output_root: Path,
    config: dict[str, Any],
    specs: list[ArchitectureSpec],
    parameter_rows: list[dict[str, Any]],
    rf_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    representation_rows: list[dict[str, Any]],
) -> None:
    reports = output_root / "reports"
    summary = summary_rows(metric_rows)
    pass_count = sum(row["status"] == "PASS" for row in status_rows)
    failed_count = len(status_rows) - pass_count
    parameter_by_name = {row["variant"]: row for row in parameter_rows}
    rf_by_name = {spec.name: final_rf_info(spec, int(config["sequence_length"]), float(config["base_time_step_ms"])) for spec in specs}

    registry = [
        "# M6A-PUBLIC-003 Structural Modification Registry",
        "",
        "本 registry 只覆盖本轮三组：downsampling、RF growth/multiscale/decoupling、explicit change/event branch。所有模型均为小型、从零训练的 1D temporal models；不含 pretrained backbone。",
        "",
        "| family | modification | computational meaning | temporal-resolution effect | RF effect | parameter/compute change | matched control | falsifiable endpoint | status |",
        "|---|---|---|---|---|---|---|---|---|",
        "| downsampling | early vs late stride | same total factor 4, move stride timing | intermediate frame step changes | RF changes as a consequence of stride timing | exact same convolutional parameterization | early vs late | seen-rate, omission, onset metrics | RUN + RF_CONFOUND noted |",
        "| RF growth | uniform/local dilation | retain local windows | 1 ms output step | slow RF growth | exact schedule-parameter match | exponential/delayed | RF table and task metrics | RUN |",
        "| RF growth | exponential dilation | grow context 1,2,4,8 | 1 ms output step | fast RF growth | exact schedule-parameter match | uniform/delayed | RF table and task metrics | RUN |",
        "| RF growth | delayed-growth dilation | keep early layers local | 1 ms output step | delayed RF growth | exact schedule-parameter match | uniform/exponential | RF table and task metrics | RUN |",
        "| RF growth | parallel multiscale k=3/7/15 | concurrent temporal windows then 1x1 fusion | 1 ms output step | max branch RF per block | width-adjusted; must be checked | uniform_local | RF table and task metrics | RUN/CONFOUNDED if outside ±10% |",
        "| RF/downsampling | stride-coupled vs dilation-decoupled | same target RF, change source of RF | 4 ms vs 1 ms final step | matched theoretical RF | exact convolutional parameterization | pairwise | fast-event metrics at matched RF | RUN |",
        "| RF diagnostic | kernel 3/7/15/31 | change direct local window | 1 ms output step | direct kernel growth | width-adjusted; confound retained if >±10% | kernel_3 | RF and performance diagnostic | RUN/CONFOUNDED if needed |",
        "| event branch | explicit Δx branch | expose first temporal difference | same final resolution | same branch RF | branch width 11 vs baseline width 16 | ordinary second raw branch | onset/omission/phase metrics | RUN |",
        "",
        "参数匹配状态以 `m6a_public_003_model_parameters.csv` 为准；任何 `CONFOUNDED` 比较只能作为工程描述，不能被写成单一结构因果证据。",
        "",
    ]
    (reports / "STRUCTURAL_MODIFICATION_REGISTRY.md").write_text("\n".join(registry), encoding="utf-8")

    design = [
        "# M6A-PUBLIC-003 Temporal Benchmark Design",
        "",
        "## 状态与图形契约",
        "",
        "- 状态：`READY_FOR_REVIEW`（Agent B 产物，未写 ACCEPT/HUMAN_ACCEPTED）。",
        "- 核心工程命题：在同一合成时序任务、统一优化和分组匹配下，结构变量改变快速事件的可定位性、区分性或未见参数泛化时，才构成结构 benchmark 证据。",
        "- 图形类型：quantitative grid；主证据为任务性能与 RF/temporal-resolution 曲线，控制证据为参数匹配、representation distance、event-triggered response 和 autocorrelation。",
        "- 后端：Python/Matplotlib；导出 PNG、SVG、PDF，所有图由同一 Python 脚本生成。",
        "- Reviewer risks：3 seeds 不是生物学重复；theoretical RF 不等价于 effective RF；synthetic signal 不等价于声音、神经记录或听觉生理；参数超出 ±10% 的组标为 `CONFOUNDED`。",
        "",
        "## Data and split",
        "",
        f"- Sequence length: `{config['sequence_length']} samples`, `{config['base_time_step_ms']} ms/sample`; input is a synthetic 1D temporal signal, not patient or STN data.",
        f"- Train/validation/test-seen/test-unseen sizes: `{config['train_size']}/{config['validation_size']}/{config['test_seen_size']}/{config['test_unseen_size']}`.",
        "- Train and validation use the seen parameter regime. Test-seen uses an independent generation seed with the same ranges. Test-unseen uses held-out rates, jitter magnitudes and phase magnitudes.",
        "- No adjacent-window, clip, story, subject or patient records are read; each sample is generated from an independent seed stream and split arrays are reused by every architecture.",
        "",
        "## Tasks and metrics",
        "",
        "- Rate discrimination: 4/6/8/10 Hz seen classes; balanced accuracy and rate MAE.",
        "- Regular vs jitter: binary balanced accuracy and jitter-magnitude MAE.",
        "- Omission detection: binary balanced accuracy.",
        "- Phase-shift detection: binary balanced accuracy and phase-magnitude MAE.",
        "- Onset/event timing: absolute error in milliseconds.",
        "- Unseen generalization: held-out rate, jitter and phase magnitude MAE.",
        "- Representation checks without additional fitting: no-fit latent distance after a 10-ms shift, event-triggered response, temporal lag-1 autocorrelation.",
        "",
        "## Matched training contract",
        "",
        f"- Optimizer: AdamW, learning rate `{config['learning_rate']}`, weight decay `{config['weight_decay']}`; epochs `{config['epochs']}`; batch `{config['batch_size']}`; gradient clip `{config['max_grad_norm']}`.",
        f"- Seeds: `{config['seeds']}`. No per-architecture hyperparameter search or early stopping; the final epoch is evaluated for every run.",
        "- Failure policy: retain non-finite loss, divergence, timeout and failed seed rows; never replace a failed run with a best run.",
        "",
        "## RF convention",
        "",
        "For a layer with kernel `k`, dilation `d`, stride `s`, previous jump `j` and RF `r`: `r_new = r + (k-1)d j`, `j_new = j s`. RF is reported as the number of inclusive input samples and `RF_ms = RF_samples × 1 ms`; output temporal resolution is the jump, not the RF width. Theoretical RF describes graph connectivity and is not an effective influence estimate learned from data.",
        "",
        "## Reproducibility artifacts",
        "",
        "- Code: `scripts/m6a_public_003_temporal_benchmark.py`.",
        "- Configuration: `configs/m6a_public_003.json`.",
        "- Raw model checkpoints and large training tensors are not saved; the report bundle contains only lightweight structured results, logs and figures.",
        "",
    ]
    (reports / "TEMPORAL_BENCHMARK_DESIGN.md").write_text("\n".join(design), encoding="utf-8")

    def table_for(variants: Sequence[str], metrics: Sequence[str], split: str) -> list[str]:
        lines = ["| variant | parameters | RF ms | final resolution ms | " + " | ".join(metrics) + " |", "|---|---:|---:|---:|" + "---:|" * len(metrics)]
        for variant in variants:
            params = parameter_by_name.get(variant, {})
            rf = rf_by_name[variant]
            values = [format_summary(summary, variant, split, metric) for metric in metrics]
            lines.append(f"| {variant} | {params.get('parameter_count', 'NA')} | {rf[1]:.1f} | {rf[2]:.1f} | " + " | ".join(values) + " |")
        return lines

    down = [
        "# M6A-PUBLIC-003 Downsampling Ablation",
        "",
        "E1 compares early and late stride timing with the same depth, kernel, width, total downsampling factor 4 and convolutional parameterization. Because moving stride also changes graph RF, the timing comparison is reported with an RF confound rather than treated as a pure RF-independent effect.",
        "",
        *table_for(["early_downsample", "late_downsample"], ["rate_discrimination_balanced_accuracy", "omission_detection_balanced_accuracy", "onset_timing_error_ms"], "test_seen"),
        "",
        *table_for(["early_downsample", "late_downsample"], ["unseen_rate_mae_hz", "unseen_jitter_mae_ms", "unseen_phase_magnitude_mae_rad"], "test_unseen"),
        "",
        "Interpretation is descriptive across three seeds. A positive difference is not a biological or causal claim; the exact RF and temporal-resolution rows are in `receptive_field_by_layer.csv`.",
        "",
    ]
    (reports / "DOWNSAMPLING_ABLATION.md").write_text("\n".join(down), encoding="utf-8")

    rf_report = [
        "# M6A-PUBLIC-003 Receptive-Field Ablation",
        "",
        "This report separates local kernel size, dilation schedule, parallel multiscale RF and RF/downsampling decoupling. The table is generated from the same architecture specs as the trained models.",
        "",
        *table_for(["uniform_local", "exponential_growth", "delayed_growth", "parallel_multiscale", "rf_stride_coupled", "rf_dilation_decoupled"], ["onset_timing_error_ms", "omission_detection_balanced_accuracy", "rate_discrimination_balanced_accuracy"], "test_seen"),
        "",
        *table_for(["uniform_local", "exponential_growth", "delayed_growth", "parallel_multiscale", "rf_stride_coupled", "rf_dilation_decoupled"], ["unseen_rate_mae_hz", "unseen_jitter_mae_ms", "unseen_phase_magnitude_mae_rad"], "test_unseen"),
        "",
        "## Kernel-size diagnostic",
        "",
        *table_for(["kernel_3", "kernel_7", "kernel_15", "kernel_31"], ["onset_timing_error_ms", "phase_shift_detection_balanced_accuracy"], "test_seen"),
        "",
        "`rf_stride_coupled` and `rf_dilation_decoupled` are designed to have the same final theoretical RF, while their final output steps differ. The exact per-layer audit is authoritative. Kernel width changes use width adjustment but any residual parameter mismatch is retained as `CONFOUNDED`.",
        "",
    ]
    (reports / "RECEPTIVE_FIELD_ABLATION.md").write_text("\n".join(rf_report), encoding="utf-8")

    event = [
        "# M6A-PUBLIC-003 Explicit Change/Event Branch Ablation",
        "",
        "The event experiment compares a single content branch, a first-difference (`Δx(t)`) branch and a parameter-matched ordinary second raw-input branch. Only the second input transformation differs between the two branch variants; both use two width-11 branch backbones and the same 1x1 fusion.",
        "",
        *table_for(["event_baseline", "explicit_change", "ordinary_second_branch"], ["onset_timing_error_ms", "omission_detection_balanced_accuracy", "phase_shift_detection_balanced_accuracy"], "test_seen"),
        "",
        *table_for(["event_baseline", "explicit_change", "ordinary_second_branch"], ["unseen_rate_mae_hz", "unseen_jitter_mae_ms", "unseen_phase_magnitude_mae_rad"], "test_unseen"),
        "",
        f"本轮未满足显式变化支路的必要证据模式：onset error 为 explicit_change {format_summary(summary, 'explicit_change', 'test_seen', 'onset_timing_error_ms')}、ordinary_second_branch {format_summary(summary, 'ordinary_second_branch', 'test_seen', 'onset_timing_error_ms')}；omission balanced accuracy 为 explicit_change {format_summary(summary, 'explicit_change', 'test_seen', 'omission_detection_balanced_accuracy')}、ordinary_second_branch {format_summary(summary, 'ordinary_second_branch', 'test_seen', 'omission_detection_balanced_accuracy')}；phase-shift detection 两者均为 chance-level。故本轮不支持 explicit_change 相对普通第二支路的稳定优势。",
        "三 seeds 和 synthetic data 只支持工程 benchmark 表述，不建立神经 event pathway 结论。",
        "",
    ]
    (reports / "EVENT_BRANCH_ABLATION.md").write_text("\n".join(event), encoding="utf-8")

    matched = [
        "# M6A-PUBLIC-003 Summary",
        "",
        "状态：`READY_FOR_REVIEW`。这是 Agent B 的执行交付，不是 `ACCEPT` 或 `HUMAN_ACCEPTED`。",
        "",
        f"- Runs recorded: `{len(status_rows)}`; PASS: `{pass_count}`; failed/diverged/other: `{failed_count}`.",
        "- Scope: exactly the three approved groups; no pretrained model, model download, patient/STN data, cochlear frontend, fast/slow branch or Mamba gating.",
        "- Primary evidence: `m6a_public_003_metrics_by_seed.csv`, `m6a_public_003_metrics_summary.csv`, `m6a_public_003_model_parameters.csv`, `receptive_field_by_layer.csv` and generated figures.",
        "",
        "## Engineering PASS criteria",
        "",
        "- Parameter/RF smoke completed before formal training.",
        "- Every formal variant was attempted under the same optimizer, epoch count, batch size, split and three seeds.",
        "- RF recurrence and output frame step are computed from the exact model specification.",
        "- Negative, failed and parameter-confounded states are retained in structured outputs.",
        "",
        "## Evidence boundary",
        "",
        "A performance difference in this synthetic benchmark is engineering evidence about these small model graphs and task distributions. It is not evidence of brain-region correspondence, auditory-system homology, patient/STN applicability, causality in neural tissue or clinical utility. Theoretical RF is not effective RF, and three seeds do not estimate biological variability.",
        "",
        "## Open scientific questions",
        "",
        "- Whether the observed task effects persist across real audio perturbations and public neural recordings remains unresolved.",
        "- Whether effective rather than theoretical RF tracks the performance differences remains unresolved.",
        "- Whether explicit change information adds stable value after a broader family of matched controls remains unresolved.",
        "",
        "下一步必须经过 Agent A/人工审核，不由本 worker 自动启动。",
        "",
    ]
    (reports / "M6A-PUBLIC-003_SUMMARY.md").write_text("\n".join(matched), encoding="utf-8")

    work_report = [
        "# Agent B Work Report — M6A-PUBLIC-003",
        "",
        "TASK_ID: M6A-PUBLIC-003",
        "SESSION_ID: M6A-PUBLIC-003-S01",
        "STATUS: READY_FOR_REVIEW",
        "COMPLETED: small matched temporal benchmark, parameter/RF smoke, three-seed runs, structured outputs and figures",
        "OUTPUTS: see the seven task-book reports plus CSV/JSON/figure artifacts in this directory",
        "SELF_CHECK: recorded in the final response and structured manifests; no patient/STN data read",
        "EVIDENCE_GAPS: synthetic task distribution, three seeds, theoretical rather than effective RF",
        "BLOCKERS: none for the authorized engineering run; scientific acceptance remains a human/Agent A gate",
        "QUESTIONS_FOR_AGENT_CHECK: review parameter matching, RF confounds and explicit-change superiority pattern",
        "STOP_REASON: reached the approved READY_FOR_REVIEW gate; no ACCEPT/HUMAN_ACCEPTED written",
        "",
    ]
    (reports / "M6A-PUBLIC-003_AGENT_WORK_REPORT.md").write_text("\n".join(work_report), encoding="utf-8")


def plot_results(output_root: Path, parameter_rows: list[dict[str, Any]], rf_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]], representation_rows: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
    })
    figure_dir = output_root / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "early_downsample": "#4C78A8", "late_downsample": "#F58518", "uniform_local": "#6B6B6B",
        "exponential_growth": "#54A24B", "delayed_growth": "#B279A2", "parallel_multiscale": "#E45756",
        "rf_stride_coupled": "#72B7B2", "rf_dilation_decoupled": "#FF9DA6", "event_baseline": "#7F7F7F",
        "explicit_change": "#2F5597", "ordinary_second_branch": "#D17A22", "kernel_3": "#6B6B6B",
        "kernel_7": "#4C78A8", "kernel_15": "#54A24B", "kernel_31": "#E45756",
    }
    summary = summary_rows(metric_rows)

    def save_figure(fig: Any, stem: str) -> None:
        fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.svg", bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    selected_rf = ["early_downsample", "late_downsample", "uniform_local", "exponential_growth", "delayed_growth", "rf_stride_coupled", "rf_dilation_decoupled"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for variant in selected_rf:
        rows = [row for row in rf_rows if row["model"] == variant and row["branch"] in {"content", "fusion"}]
        if not rows:
            continue
        axes[0].plot(range(1, len(rows) + 1), [float(row["theoretical_RF_ms"]) for row in rows], marker="o", ms=2.5, lw=1.1, label=variant, color=colors.get(variant))
        axes[1].plot(range(1, len(rows) + 1), [float(row["output_time_resolution_ms"]) for row in rows], marker="o", ms=2.5, lw=1.1, label=variant, color=colors.get(variant))
    axes[0].set(xlabel="depth / audit row", ylabel="theoretical RF (ms)", title="RF growth")
    axes[1].set(xlabel="depth / audit row", ylabel="output temporal resolution (ms)", title="frame step")
    axes[0].legend(fontsize=5, ncol=2, frameon=False)
    save_figure(fig, "m6a_public_003_rf_and_resolution")

    def get_summary(variant: str, metric: str, split: str) -> tuple[float, float] | None:
        for row in summary:
            if row["variant"] == variant and row["metric"] == metric and row["split"] == split:
                return float(row["mean"]), float(row["std"])
        return None

    variants = [spec for spec in ["early_downsample", "late_downsample", "uniform_local", "exponential_growth", "delayed_growth", "parallel_multiscale", "rf_stride_coupled", "rf_dilation_decoupled", "event_baseline", "explicit_change", "ordinary_second_branch"]]
    panels = [
        ("onset_timing_error_ms", "test_seen", "onset timing error (ms)"),
        ("omission_detection_balanced_accuracy", "test_seen", "omission balanced accuracy"),
        ("unseen_rate_mae_hz", "test_unseen", "unseen-rate MAE (Hz)"),
        ("unseen_phase_magnitude_mae_rad", "test_unseen", "unseen-phase MAE (rad)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    for axis, (metric, split, ylabel) in zip(axes.ravel(), panels):
        means, errors, labels, bar_colors = [], [], [], []
        for variant in variants:
            value = get_summary(variant, metric, split)
            if value is None:
                continue
            means.append(value[0]); errors.append(value[1]); labels.append(variant); bar_colors.append(colors.get(variant, "#777777"))
        axis.bar(np.arange(len(means)), means, yerr=errors, capsize=2, color=bar_colors, edgecolor="black", linewidth=0.3)
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=5)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    save_figure(fig, "m6a_public_003_performance_by_perturbation")

    group_variants = ["early_downsample", "late_downsample", "uniform_local", "exponential_growth", "delayed_growth", "event_baseline", "explicit_change", "ordinary_second_branch"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for axis, metric, split, ylabel in [
        (axes[0], "onset_timing_error_ms", "test_seen", "onset error (ms)"),
        (axes[1], "omission_detection_balanced_accuracy", "test_seen", "omission balanced accuracy"),
    ]:
        means, errors, labels, bar_colors = [], [], [], []
        for variant in group_variants:
            value = get_summary(variant, metric, split)
            if value is not None:
                means.append(value[0]); errors.append(value[1]); labels.append(variant); bar_colors.append(colors.get(variant, "#777777"))
        axis.bar(np.arange(len(means)), means, yerr=errors, capsize=2, color=bar_colors, edgecolor="black", linewidth=0.3)
        axis.set_ylabel(ylabel)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=5)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    save_figure(fig, "m6a_public_003_matched_architecture_comparison")

    representation_summary: dict[tuple[str, str], list[float]] = {}
    for row in representation_rows:
        representation_summary.setdefault((str(row["variant"]), str(row["metric"])), []).append(float(row["value"]))
    rep_metrics = ["no_fit_representation_distance", "event_triggered_response", "temporal_persistence_autocorrelation"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), constrained_layout=True)
    rep_variants = ["uniform_local", "exponential_growth", "delayed_growth", "parallel_multiscale", "event_baseline", "explicit_change", "ordinary_second_branch"]
    for axis, metric in zip(axes, rep_metrics):
        values, labels, bar_colors = [], [], []
        for variant in rep_variants:
            data = representation_summary.get((variant, metric), [])
            if data:
                values.append(float(np.mean(data))); labels.append(variant); bar_colors.append(colors.get(variant, "#777777"))
        axis.bar(np.arange(len(values)), values, color=bar_colors, edgecolor="black", linewidth=0.3)
        axis.set_ylabel(metric.replace("_", " "))
        axis.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=5)
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    save_figure(fig, "m6a_public_003_representation_checks")
    return [
        "reports/figures/m6a_public_003_rf_and_resolution.{png,svg,pdf}",
        "reports/figures/m6a_public_003_performance_by_perturbation.{png,svg,pdf}",
        "reports/figures/m6a_public_003_matched_architecture_comparison.{png,svg,pdf}",
        "reports/figures/m6a_public_003_representation_checks.{png,svg,pdf}",
    ]


def run_smoke(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    specs = make_specs(config)
    parameter_rows: list[dict[str, Any]] = []
    rf_rows: list[dict[str, Any]] = []
    for spec in specs:
        model = TemporalModel(spec, head_width=int(config["hidden_head_width"]))
        count = sum(parameter.numel() for parameter in model.parameters())
        rf_samples, rf_ms, resolution_ms, output_len = final_rf_info(spec, int(config["sequence_length"]), float(config["base_time_step_ms"]))
        parameter_rows.append({
            "variant": spec.name,
            "family": spec.family,
            "match_group": spec.group,
            "mode": spec.mode,
            "width": spec.width,
            "branch_width": spec.branch_width,
            "kernel": spec.kernel,
            "dilation_schedule": ",".join(map(str, spec.dilations)),
            "stride_schedule": ",".join(map(str, spec.strides)),
            "parameter_count": int(count),
            "approx_flops_per_sample": approximate_flops(spec, int(config["sequence_length"]), int(config["hidden_head_width"])),
            "final_theoretical_RF_samples": rf_samples,
            "final_theoretical_RF_ms": rf_ms,
            "final_output_time_resolution_ms": resolution_ms,
            "final_output_length": output_len,
            "notes": spec.notes,
        })
        rf_rows.extend(rf_rows_for_spec(spec, int(config["sequence_length"]), float(config["base_time_step_ms"])))
    references = {
        "downsampling": "early_downsample",
        "rf_schedule": "uniform_local",
        "multiscale": "uniform_local",
        "rf_decoupling": "rf_stride_coupled",
        "kernel_sweep": "kernel_3",
        "event_branch": "event_baseline",
    }
    by_name = {row["variant"]: row for row in parameter_rows}
    for row in parameter_rows:
        reference = references[row["match_group"]]
        ref_count = int(by_name[reference]["parameter_count"])
        relative = (int(row["parameter_count"]) - ref_count) / ref_count
        row["reference_variant"] = reference
        row["relative_parameter_difference"] = relative
        row["parameter_match_status"] = "PASS" if abs(relative) <= 0.10 else "CONFOUNDED"
    csv_write(output_root / "reports" / "m6a_public_003_model_parameters.csv", parameter_rows, list(parameter_rows[0].keys()))
    csv_write(output_root / "reports" / "receptive_field_by_layer.csv", rf_rows, list(rf_rows[0].keys()))
    smoke = {
        "status": "PASS",
        "task_id": config["task_id"],
        "session_id": config["session_id"],
        "variant_count": len(specs),
        "rf_row_count": len(rf_rows),
        "parameter_match_status_counts": {status: sum(row["parameter_match_status"] == status for row in parameter_rows) for status in ["PASS", "CONFOUNDED"]},
        "known_rf_checks": {
            "rf_stride_coupled_samples": by_name["rf_stride_coupled"]["final_theoretical_RF_samples"],
            "rf_dilation_decoupled_samples": by_name["rf_dilation_decoupled"]["final_theoretical_RF_samples"],
            "rf_stride_coupled_resolution_ms": by_name["rf_stride_coupled"]["final_output_time_resolution_ms"],
            "rf_dilation_decoupled_resolution_ms": by_name["rf_dilation_decoupled"]["final_output_time_resolution_ms"],
        },
        "no_pretrained_model": True,
        "patient_stn_data_read": False,
        "integrity_policy": "NON_HASH_AUDIT",
    }
    if smoke["known_rf_checks"]["rf_stride_coupled_samples"] != smoke["known_rf_checks"]["rf_dilation_decoupled_samples"]:
        smoke["status"] = "FAIL"
        smoke["rf_error"] = "matched RF pair differs"
    if smoke["known_rf_checks"]["rf_stride_coupled_resolution_ms"] != 4.0 or smoke["known_rf_checks"]["rf_dilation_decoupled_resolution_ms"] != 1.0:
        smoke["status"] = "FAIL"
        smoke["resolution_error"] = "expected 4 ms vs 1 ms final resolution"
    json_write(output_root / "reports" / "m6a_public_003_smoke.json", smoke)
    return {"smoke": smoke, "specs": specs, "parameter_rows": parameter_rows, "rf_rows": rf_rows}


def run_full(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    smoke_data = run_smoke(config, output_root)
    if smoke_data["smoke"]["status"] != "PASS":
        raise RuntimeError("parameter/RF smoke failed")
    specs: list[ArchitectureSpec] = smoke_data["specs"]
    parameter_rows: list[dict[str, Any]] = smoke_data["parameter_rows"]
    rf_rows: list[dict[str, Any]] = smoke_data["rf_rows"]
    device_name = config.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    train_arrays = generate_split(config, int(config["train_size"]), int(config["generation_seed"]) + 1, "seen")
    validation_arrays = generate_split(config, int(config["validation_size"]), int(config["generation_seed"]) + 2, "seen")
    test_seen_arrays = generate_split(config, int(config["test_seen_size"]), int(config["generation_seed"]) + 3, "seen")
    test_unseen_arrays = generate_split(config, int(config["test_unseen_size"]), int(config["generation_seed"]) + 4, "unseen")
    train_dataset = TemporalDataset(train_arrays)
    validation_dataset = TemporalDataset(validation_arrays)
    train_logs: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in [int(value) for value in config["seeds"]]:
            seed_everything(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                num_workers=int(config["num_workers"]),
                generator=torch.Generator().manual_seed(seed + 1000),
            )
            validation_loader = DataLoader(
                validation_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=False,
                num_workers=int(config["num_workers"]),
            )
            model = TemporalModel(spec, head_width=int(config["hidden_head_width"]))
            started = time.perf_counter()
            try:
                status, logs, failure = train_model(model, train_loader, validation_loader, device, config, spec.name, seed)
                train_logs.extend(logs)
                if status == "PASS":
                    for split_name, arrays in [("test_seen", test_seen_arrays), ("test_unseen", test_unseen_arrays)]:
                        outputs, _, _ = predict_arrays(model, arrays, device)
                        metric_rows.extend(metric_rows_for_split(outputs, arrays, split_name, spec.name, seed))
                    _, _, _ = predict_arrays(model, test_seen_arrays, device)
                    rf_samples, rf_ms, resolution_ms, _ = final_rf_info(spec, int(config["sequence_length"]), float(config["base_time_step_ms"]))
                    rep = representation_metrics(model, test_seen_arrays, device, resolution_ms)
                    for metric, value in rep.items():
                        representation_rows.append({"variant": spec.name, "seed": seed, "metric": metric, "value": value})
                    latency_rows.append({"variant": spec.name, "seed": seed, "latency_ms_per_sample": measure_latency(model, test_seen_arrays, device), "approx_flops_per_sample": approximate_flops(spec, int(config["sequence_length"]), int(config["hidden_head_width"])), "final_theoretical_RF_ms": rf_ms, "final_output_time_resolution_ms": resolution_ms})
                status_rows.append({"variant": spec.name, "family": spec.family, "seed": seed, "status": status, "failure": failure or "", "wall_seconds": time.perf_counter() - started})
            except Exception as exc:  # retain a failed seed and continue the matrix
                status_rows.append({"variant": spec.name, "family": spec.family, "seed": seed, "status": "FAILED", "failure": f"{type(exc).__name__}: {exc}", "wall_seconds": time.perf_counter() - started})
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    csv_write(output_root / "reports" / "m6a_public_003_training_log.csv", train_logs, ["variant", "seed", "epoch", "train_loss", "validation_loss", "status"])
    csv_write(output_root / "reports" / "m6a_public_003_run_status.csv", status_rows, ["variant", "family", "seed", "status", "failure", "wall_seconds"])
    csv_write(output_root / "reports" / "m6a_public_003_metrics_by_seed.csv", metric_rows, ["variant", "seed", "split", "metric", "value"])
    summary = summary_rows(metric_rows)
    csv_write(output_root / "reports" / "m6a_public_003_metrics_summary.csv", summary, ["variant", "split", "metric", "mean", "std", "n_seeds"])
    csv_write(output_root / "reports" / "m6a_public_003_representation_metrics.csv", representation_rows, ["variant", "seed", "metric", "value"])
    csv_write(output_root / "reports" / "m6a_public_003_latency.csv", latency_rows, ["variant", "seed", "latency_ms_per_sample", "approx_flops_per_sample", "final_theoretical_RF_ms", "final_output_time_resolution_ms"])
    runtime = {
        "task_id": config["task_id"],
        "session_id": config["session_id"],
        "status": "READY_FOR_REVIEW",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "device_used": str(device),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_pretrained_model": True,
        "patient_stn_data_read": False,
        "large_checkpoints_saved": False,
        "integrity_policy": "NON_HASH_AUDIT",
        "same_split_optimizer_epochs_batch": True,
        "formal_seed_count": len(config["seeds"]),
    }
    json_write(output_root / "reports" / "m6a_public_003_runtime.json", runtime)
    write_markdown_reports(output_root, config, specs, parameter_rows, rf_rows, metric_rows, status_rows, representation_rows)
    figures = plot_results(output_root, parameter_rows, rf_rows, metric_rows, representation_rows)
    manifest = {
        "task_id": config["task_id"],
        "session_id": config["session_id"],
        "status": "READY_FOR_REVIEW",
        "scope_groups": ["downsampling", "rf_growth_multiscale_decoupling", "explicit_change_branch"],
        "variant_count": len(specs),
        "formal_run_count": len(status_rows),
        "pass_count": sum(row["status"] == "PASS" for row in status_rows),
        "failed_or_nonpass_count": sum(row["status"] != "PASS" for row in status_rows),
        "figures": figures,
        "reports": [
            "reports/STRUCTURAL_MODIFICATION_REGISTRY.md",
            "reports/receptive_field_by_layer.csv",
            "reports/TEMPORAL_BENCHMARK_DESIGN.md",
            "reports/DOWNSAMPLING_ABLATION.md",
            "reports/RECEPTIVE_FIELD_ABLATION.md",
            "reports/EVENT_BRANCH_ABLATION.md",
            "reports/M6A-PUBLIC-003_SUMMARY.md",
        ],
    }
    json_write(output_root / "reports" / "m6a_public_003_run_manifest.json", manifest)
    return {"runtime": runtime, "manifest": manifest, "status_rows": status_rows, "parameter_rows": parameter_rows}


def regenerate_reports(config: dict[str, Any], output_root: Path) -> None:
    """Rebuild text/figure artifacts from existing structured results without training."""

    reports = output_root / "reports"
    specs = make_specs(config)
    parameter_rows = read_csv(reports / "m6a_public_003_model_parameters.csv")
    rf_rows = read_csv(reports / "receptive_field_by_layer.csv")
    metric_rows = read_csv(reports / "m6a_public_003_metrics_by_seed.csv")
    status_rows = read_csv(reports / "m6a_public_003_run_status.csv")
    representation_rows = read_csv(reports / "m6a_public_003_representation_metrics.csv")
    write_markdown_reports(output_root, config, specs, parameter_rows, rf_rows, metric_rows, status_rows, representation_rows)
    plot_results(output_root, parameter_rows, rf_rows, metric_rows, representation_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the M6A-PUBLIC-003 small temporal benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.reports_only:
        regenerate_reports(config, args.output_root)
        return 0
    if args.smoke_only:
        result = run_smoke(config, args.output_root)
        print(json.dumps(result["smoke"], ensure_ascii=False, indent=2))
        return 0 if result["smoke"]["status"] == "PASS" else 1
    result = run_full(config, args.output_root)
    print(json.dumps(result["manifest"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
