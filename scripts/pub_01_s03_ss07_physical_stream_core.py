"""D19 physical data/decoder implementation. Run only through the reviewed 2203 gate."""

from collections import defaultdict
from dataclasses import dataclass
import sys

import numpy as np
import torch
from torch import nn
from torch.nn.functional import gelu

from pub_01_s03_ss07_physical_stream_meta import (
    BATCH, BIN_POINTS, BINS, CHANNELS, CONTEXT_CHANNELS, CONTEXT_WIDTH,
    FIRST_ACTIVE, FRAME, HISTORY, HOP, LEGACY_METRIC, NEAR, OLDER, ROLES, WIDTH,
    assets, covered_stop, read_json,
)

sys.path.insert(0, str(LEGACY_METRIC.parent))
import pub_01_s03_ss07_generalization_core as legacy


@dataclass
class Record:
    metadata: dict
    latent: np.ndarray
    frame_means: np.ndarray
    anchor: np.ndarray
    target: np.ndarray
    blocks: list

    @property
    def stop(self):
        return self.metadata["covered_stop"]


def frozen_assets(seed):
    files = assets(seed)
    with np.load(files["stats"], allow_pickle=False) as f:
        stats = [{key: f[f"{key}_{i}"].copy() for key in ("mean", "std", "constant")}
                 for i in range(3)]
    with np.load(files["coefficient"], allow_pickle=False) as f:
        coef, bias = f["coef"].copy(), f["bias"].copy()
    calibration = read_json(files["calibration"])["calibration"]
    if coef.shape != (3264, FRAME) or bias.shape != (FRAME,) or calibration["a"] <= 0:
        raise ValueError("Unexpected frozen Ccap layout")
    return stats, coef, bias, calibration


def anchor_frames(z, p, readout):
    """D8 concatenation and D18's 1024-window chunks, including first-four fallback."""
    stats, coef, bias, calibration = readout
    base = calibration["a"] * p.astype(np.float64) + calibration["b"]
    prediction = base.copy()
    for first in range(0, len(z), 1024):
        stop = min(first + 1024, len(z))
        free = max(first, 4)
        if free >= stop:
            continue
        u = legacy.normalize(z[free:stop].astype(np.float64).reshape(-1, 1536), stats[0])
        v = legacy.normalize(base[free:stop], stats[1])
        w = legacy.normalize(u * u, stats[2])
        features = np.concatenate((u / np.sqrt(1536), v / np.sqrt(192),
                                   w / np.sqrt(1536)), axis=1)
        prediction[free:stop] += features @ coef + bias
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite frozen Ccap output")
    return prediction


def fuse_latent(z, samples):
    """Same frame starts, Hann weights, and denominator as legacy output fusion."""
    stop = covered_stop(len(z))
    if samples < stop + 1:
        raise ValueError("Insufficient physical record length")
    h = np.hanning(FRAME)
    values = np.zeros((samples, 8), dtype=np.float64)
    weights = np.zeros(samples, dtype=np.float64)
    for i, frame in enumerate(z):
        start = i * HOP
        region = slice(start, start + FRAME)
        values[region] += frame.T * h[:, None]
        weights[region] += h
    keep = weights > 1e-8
    if keep[:1].any() or not keep[1:stop].all() or keep[stop:].any():
        raise ValueError("Unexpected positive Hann support")
    values[keep] /= weights[keep, None]
    if not np.isfinite(values).all():
        raise ValueError("Non-finite fused latent")
    return values.astype(np.float32), keep


def load_phase(metadata, seed, phase, guard, first_record_only=False):
    if phase not in ("train", "val") or (first_record_only and phase != "train"):
        raise ValueError("Unsupported phase")
    specification = metadata["phases"][phase]
    files = assets(seed)
    readout = frozen_assets(seed)
    z = np.load(files["folder"] / f"{phase}_latent.npy", mmap_mode="r", allow_pickle=False)
    p = np.load(files["folder"] / f"{phase}_P.npy", mmap_mode="r", allow_pickle=False)
    blocks = defaultdict(list)
    for block in specification["blocks"]:
        blocks[block["record"]].append(block)
    records = []
    rows = specification["records"][:1] if first_record_only else specification["records"]
    for row in rows:
        start = row["offset"]
        end = start + row["windows"]
        samples = row["common_samples"]
        anchor, anchor_keep = legacy.fuse(anchor_frames(z[start:end], p[start:end], readout), samples)
        latent, latent_keep = fuse_latent(z[start:end], samples)
        frame_means = (z[start:end].reshape(-1, CHANNELS, FRAME // BIN_POINTS, BIN_POINTS)
                       .mean(axis=3, dtype=np.float64).transpose(0, 2, 1).astype(np.float32))
        if not np.array_equal(anchor_keep, latent_keep):
            raise ValueError("Anchor/latent coordinates disagree")
        target = np.load(row["envelope_path"], mmap_mode="r", allow_pickle=False).reshape(-1)[:samples]
        target = target.astype(np.float64, copy=True)
        if not np.isfinite(target).all():
            raise ValueError("Non-finite target in frozen record")
        for array in (latent, frame_means, anchor, target):
            array.flags.writeable = False
        records.append(Record(row, latent, frame_means, anchor, target, blocks[row["record"]]))
        guard()
    return records


def rolling_mean(values, width):
    prefix = np.cumsum(values, axis=0, dtype=np.float64)
    answer = np.zeros_like(values, dtype=np.float32)
    answer[width - 1] = prefix[width - 1] / width
    answer[width:] = (prefix[width:] - prefix[:-width]) / width
    return answer


class Tables:
    """One row per physical point. Indexing never reaches a different record."""

    def __init__(self, records, device, guard):
        self.device = device
        self.starts = np.cumsum([0] + [r.stop for r in records], dtype=np.int64)
        self.stops = [r.stop for r in records]
        self.windows = [r.metadata["windows"] for r in records]
        self.frame_starts = np.cumsum([0] + self.windows, dtype=np.int64)
        n = int(self.starts[-1])
        self.current = torch.empty((n, CHANNELS), dtype=torch.float32, device=device)
        self.eeg = torch.empty_like(self.current)
        self.prior = torch.empty(n, dtype=torch.float32, device=device)
        self.anchor = torch.empty(n, dtype=torch.float64, device=device)
        self.target = torch.empty_like(self.anchor)
        self.frame_means = torch.empty((int(self.frame_starts[-1]), FRAME // BIN_POINTS, CHANNELS),
                                       dtype=torch.float32, device=device)
        for i, record in enumerate(records):
            end = record.stop
            region = slice(self.starts[i], self.starts[i + 1])
            self.current[region].copy_(torch.from_numpy(record.latent[:end].copy()))
            self.eeg[region].copy_(torch.from_numpy(rolling_mean(record.latent[:end], BIN_POINTS)))
            self.prior[region].copy_(torch.from_numpy(rolling_mean(record.anchor[:end], BIN_POINTS)))
            self.anchor[region].copy_(torch.from_numpy(record.anchor[:end].copy()))
            target = record.target[:end]
            # D9 verified this float32 rounding against all original TRAIN targets.
            target = target.astype(np.float32) if record.metadata["phase"] == "train" else target.copy()
            self.target[region].copy_(torch.from_numpy(target))
            frames = slice(self.frame_starts[i], self.frame_starts[i + 1])
            self.frame_means[frames].copy_(torch.from_numpy(record.frame_means.copy()))
            guard()
        self.bin_ends = torch.arange(BINS, device=device) * BIN_POINTS - (HISTORY - BIN_POINTS)

    def indices(self, spans):
        for record, start, stop in spans:
            if not FIRST_ACTIVE <= start < stop <= self.stops[record]:
                raise ValueError("Requested points outside frozen support")
        indices = np.concatenate([np.arange(self.starts[i] + lo, self.starts[i] + hi)
                                  for i, lo, hi in spans])
        return torch.from_numpy(indices).to(self.device)

    def history(self, indices, role, lag):
        if lag not in (NEAR, OLDER):
            raise ValueError("Unrecognized lag")
        end = indices[:, None] - lag + self.bin_ends
        if role == "H_past_eeg":
            return self.eeg[end]
        if role == "P_past_prediction":
            return self.prior[end]
        raise ValueError(f"Unrecognized context role {role}")

    def context_raw(self, indices, spans, role, lag=NEAR):
        if role == "C_current" and lag == NEAR:
            frames = []
            for i, lo, hi in spans:
                points = np.arange(lo, hi)
                first = np.maximum(0, (points - (FRAME - 2) + HOP - 1) // HOP)
                last = np.minimum(self.windows[i] - 1, (points - 1) // HOP)
                frames.append(np.stack((first, last), axis=1) + self.frame_starts[i])
            positions = torch.from_numpy(np.concatenate(frames)).to(self.device)
            return self.frame_means[positions].flatten(1, 2)
        return self.history(indices, role, lag)

    def context(self, indices, spans, role, stats, lag=NEAR):
        return normalized(self.context_raw(indices, spans, role, lag), stats[role])


def chunks(records):
    return [(i, start, min(start + BATCH, record.stop))
            for i, record in enumerate(records)
            for start in range(FIRST_ACTIVE, record.stop, BATCH)]


def batches(spans, order):
    """Shuffle whole record-local spans, then pack to one fixed-denominator batch."""
    group = []
    remaining = BATCH
    for index in order:
        record, start, stop = spans[index]
        while start < stop:
            end = min(stop, start + remaining)
            group.append((record, start, end))
            remaining -= end - start
            start = end
            if remaining == 0:
                yield group
                group, remaining = [], BATCH
    if group:
        yield group


def fit_statistics(records, table, guard):
    """TRAIN-only equal-point moments. Older uses the fixed near moments."""
    count = 0
    spans = chunks(records)
    sums = torch.zeros(CHANNELS, dtype=torch.float64, device=table.device)
    squares = torch.zeros_like(sums)
    for i, start, stop in spans:
        values = table.current[table.starts[i] + start:table.starts[i] + stop].double()
        sums += values.sum(dim=0)
        squares += values.square().sum(dim=0)
        count += len(values)
    stats = {"current": moments(sums, squares, count)}
    for role in ROLES:
        shape = () if role == "P_past_prediction" else (CHANNELS,)
        sums = torch.zeros(shape, dtype=torch.float64, device=table.device)
        squares = torch.zeros_like(sums)
        for i, start, stop in spans:
            indices = table.indices([(i, start, stop)])
            values = table.context_raw(indices, [(i, start, stop)], role).double()
            sums += values.sum(dim=(0, 1))
            squares += values.square().sum(dim=(0, 1))
            if stop == records[i].stop:
                guard()
        stats[role] = moments(sums, squares, count * BINS)
    return stats, count


def moments(sums, squares, count):
    mean = sums / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt()
    constant = std < 1e-8
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError("Invalid TRAIN moments")
    return mean.float(), std.masked_fill(constant, 1).float(), constant


def normalized(value, statistics):
    mean, std, constant = statistics
    return ((value - mean) / std).masked_fill(constant, 0)


def batch_inputs(table, spans, role, stats):
    indices = table.indices(spans)
    x = normalized(table.current[indices], stats["current"])
    v = table.context(indices, spans, role, stats)
    return indices, x, v, table.anchor[indices], table.target[indices]


class PhysicalDecoder(nn.Module):
    """Shared scalar head with trainable mixing of ordered history bins."""

    def __init__(self, seed, role):
        super().__init__()
        torch.manual_seed(seed)
        self.current = nn.Linear(CHANNELS, WIDTH)
        self.context_in = nn.Linear(BINS * CONTEXT_CHANNELS[role], CONTEXT_WIDTH)
        self.context_out = nn.Linear(CONTEXT_WIDTH, WIDTH)
        self.mix = nn.Linear(WIDTH, WIDTH)
        self.output = nn.Linear(WIDTH, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def current_code(self, x):
        return gelu(self.current(x))

    def decode(self, current, context):
        code = self.context_out(gelu(self.context_in(context.flatten(1))))
        interaction = current * (1.0 + code.tanh())
        return self.output(gelu(self.mix(interaction))).squeeze(-1)

    def forward(self, x, v, anchor):
        return anchor + self.decode(self.current_code(x), v).to(torch.float64)


def aggregate_blocks(segments):
    by_person = defaultdict(list)
    for block in segments:
        by_person[block["subject"]].append(block)
    people = [dict(subject=subject, blocks=len(rows),
                   r=float(np.mean([r["r"] for r in rows])),
                   MSE=float(np.mean([r["MSE"] for r in rows])))
              for subject, rows in sorted(by_person.items())]
    return dict(r=float(np.mean([r["r"] for r in people])),
                MSE=float(np.mean([r["MSE"] for r in people])),
                people=people, blocks=segments)


def anchor_autocorrelation(records):
    rows = []
    for record in records:
        wave = record.anchor
        current = wave[FIRST_ACTIVE:record.stop]
        row = dict(record=record.metadata["record"], subject=record.metadata["subject"])
        for lag in (NEAR, OLDER):
            prior = wave[FIRST_ACTIVE - lag:record.stop - lag]
            row[str(lag)] = None if np.ptp(current) == 0 or np.ptp(prior) == 0 else legacy.corr(current, prior)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(records, table, stats, model, role, guard, count, paired=False):
    if paired and (model is None or role == "C_current"):
        raise ValueError("Invalid paired evaluation")
    outputs = ("near", "older") if paired else ("near",)
    totals = {key: dict(segments=[], active_SSE=0., residual_SSQ=0.,
                       residual_max=0., active_points=0) for key in outputs}
    for i, record in enumerate(records):
        waves = {key: record.anchor.copy() for key in outputs}
        if model is not None:
            for start in range(FIRST_ACTIVE, record.stop, BATCH):
                stop = min(start + BATCH, record.stop)
                indices, x, v, b, _ = batch_inputs(table, [(i, start, stop)], role, stats)
                code = model.current_code(x)
                waves["near"][start:stop] = (b + model.decode(code, v).double()).cpu().numpy()
                count("current_stem_points", len(b))
                count("context_output_points", len(b))
                if paired:
                    # Same current tensor, code, anchor, coordinates and weights.
                    v_old = table.context(indices, [(i, start, stop)], role, stats, OLDER)
                    waves["older"][start:stop] = (b + model.decode(code, v_old).double()).cpu().numpy()
                    count("context_output_points", len(b))
        row = record.metadata
        for key, wave in waves.items():
            acc = totals[key]
            residual = wave[FIRST_ACTIVE:record.stop] - record.anchor[FIRST_ACTIVE:record.stop]
            target = record.target[FIRST_ACTIVE:record.stop]
            if row["phase"] == "train":
                target = target.astype(np.float32)
            error = wave[FIRST_ACTIVE:record.stop] - target
            if not np.isfinite(error).all():
                raise ValueError("Non-finite continuous prediction")
            acc["active_SSE"] += float(error @ error)
            acc["residual_SSQ"] += float(residual @ residual)
            acc["residual_max"] = max(acc["residual_max"], float(np.abs(residual).max()))
            acc["active_points"] += len(error)
            for block in record.blocks:
                lo, hi = max(1, block["start"]), min(record.stop, block["start"] + 3840)
                p, y = wave[lo:hi], record.target[lo:hi]
                if len(p) != block["samples"]:
                    raise ValueError("Legacy block mask mismatch")
                acc["segments"].append(dict(**block, subject=row["subject"],
                                            source_family=row["source_family"],
                                            r=legacy.corr(p, y),
                                            MSE=float(np.mean((p - y)**2))))
        guard()
    return {key: dict(**aggregate_blocks(acc["segments"]),
                      active_points=acc["active_points"],
                      active_MSE=acc["active_SSE"] / acc["active_points"],
                      residual_RMS=np.sqrt(acc["residual_SSQ"] / acc["active_points"]).item(),
                      residual_max=acc["residual_max"])
            for key, acc in totals.items()}
