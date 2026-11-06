"""D24 upstream worker. All source paths come from a TRAIN-only frozen allowlist."""
import argparse
import bisect
import importlib
import json
import os
import resource
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch

from pub_01_s03_ss07_crossfit_meta import BASE, D19, HOME, OLD, ROOT, SEEDS, digest, read

FRAME, HOP = 192, 96
GIB = 1024**3
HEAD_BATCH = 64
NATIVE_CFG = OLD / "training_seed1024_attempt04/config_frozen.json"
FEATURES = OLD / "features/manifest.json"


def imported():
    d4 = BASE.parent / "s03_conditional_history_probe_20261101/input"
    for path in (OLD / "input", BASE / "input", D19 / "input", d4):
        sys.path.insert(0, str(path))
    return tuple(importlib.import_module(name) for name in (
        "pub_01_s03_transformer_train_v2", "pub_01_s03_ss07_baseline",
        "pub_01_s03_ss07_finetune", "pub_01_s03_ss07_physical_stream_core",
        "pub_01_s03_ss07_conditional_probe",
    ))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exclusive(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=os.getpid(), uid=os.getuid(), startticks=int(fields[19]),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))


def status(event, **values):
    print(json.dumps(dict(event=event, monotonic_s=time.monotonic(), **values)), flush=True)


def guard():
    require(shutil.disk_usage(ROOT).free >= 64*GIB, "SATA free space reserve")
    require(sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) <= 64*GIB,
            "SATA D24 file cap")
    require(torch.cuda.max_memory_reserved() < 20*GIB, "per-worker CUDA reserve cap")


def raw_eeg(row):
    raw = np.load(row["eeg_path"], mmap_mode="r", allow_pickle=False)
    eeg = raw.T if raw.shape[0] == 64 else raw
    require(eeg.ndim == 2 and eeg.shape[1] == 64 and len(eeg) >= row["common_samples"],
            "Frozen EEG shape")
    return eeg[:row["common_samples"]]


def fit_normalization(rows):
    count = 0
    sums, squares = np.zeros(64), np.zeros(64)
    for row in rows:
        eeg = np.asarray(raw_eeg(row), dtype=np.float64)
        count += len(eeg)
        sums += eeg.sum(0)
        squares += np.square(eeg).sum(0)
    mean = sums / count
    std = np.sqrt(np.maximum(squares/count - mean**2, 1e-12))
    require(np.isfinite(mean).all() and np.isfinite(std).all(), "EEG scaling finite")
    return mean, std, count


def load_rows(rows, mean, std, labels):
    eeg, target = {}, {}
    for row in rows:
        key = row["record"]
        eeg[key] = ((np.asarray(raw_eeg(row), dtype=np.float32)-mean)/std).astype(np.float32)
        if labels:
            envelope = np.load(row["envelope_path"], mmap_mode="r", allow_pickle=False).reshape(-1)
            require(len(envelope) >= row["common_samples"], "Frozen envelope support")
            target[key] = envelope[:row["common_samples"]]
    return eeg, target


def record_windows(signal, count, dtype):
    view = np.lib.stride_tricks.sliding_window_view(signal, FRAME, axis=0)[::HOP][:count]
    if signal.ndim == 2:
        view = view.swapaxes(1, 2)
    require(len(view) == count, "Frozen readout grid")
    return np.asarray(view, dtype=dtype)


def plans_for(rows, prepared, index):
    allowed = {row["record"] for row in rows}
    starts, subjects = defaultdict(lambda: defaultdict(list)), defaultdict(set)
    for item in index:
        record = prepared[item["record_id"]]["record"]
        if record in allowed:
            starts[item["story"]][item["start"]].append(record)
            subjects[item["story"]].add(item["subject"])
    plans = []
    for story, times in sorted(starts.items()):
        positions = sorted(times)
        blocks = len(positions)//128
        if blocks:
            width = len(positions)//blocks
            for block in range(blocks):
                plans.extend([(story, positions[block*width:(block+1)*width], times)]*len(subjects[story]))
    return plans


def contrastive(model, native, rows, eeg, prepared, index, seed, budget, step_limit=None):
    plans = plans_for(rows, prepared, index)
    if step_limit is None:
        require(len(plans) == budget["batches_per_epoch"], "Contrastive batch count")
    stories = {row["story"] for row in rows}
    paths = {row["story"]: row["path"] for row in read(FEATURES) if row["story"] in stories}
    require(paths.keys() == stories, "Missing frozen XLSR")
    audio = {story: np.load(path, mmap_mode="r", allow_pickle=False) for story, path in paths.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=.1)
    visits, unique, history = Counter(), defaultdict(set), []
    for epoch in range(budget["epochs"] if step_limit is None else 1):
        rng = np.random.default_rng(seed+epoch)
        order = np.arange(len(plans))
        rng.shuffle(order)
        total = 0.
        used = Counter()
        model.train()
        for i in order[:step_limit]:
            story, candidates, times = plans[i]
            positions = rng.choice(candidates, size=128, replace=False)
            # Native samples an integer row index from each time's ordered list.
            keys = [times[int(start)][int(rng.choice(len(times[int(start)])))] for start in positions]
            x = np.stack([eeg[key][s:s+FRAME] for key, s in zip(keys, positions)])
            a = np.stack([audio[story][s:s+FRAME] for s in positions])
            optimizer.zero_grad(set_to_none=True)
            loss = native.contrastive(model, torch.from_numpy(x).cuda(), torch.from_numpy(a).cuda())
            require(torch.isfinite(loss).item(), "Nonfinite contrastive loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            used.update(keys)
            for key, pos in zip(keys, positions):
                unique[key].add(int(pos))
        visits.update(used)
        scheduler.step()
        history.append(dict(epoch=epoch, batches=len(order[:step_limit]), train_loss=total/len(order[:step_limit]),
                            record_presentations=dict(used)))
        status("contrastive_epoch", epoch=epoch, batches=len(order[:step_limit]))
        guard()
    return dict(history=history, record_presentations=dict(visits),
                unique_record_starts={key: sorted(values) for key, values in unique.items()},
                updates=sum(row["batches"] for row in history))


def targets_gpu(rows, target):
    return torch.from_numpy(np.concatenate([
        record_windows(target[row["record"]], row["windows"], np.float32)
        for row in rows
    ])).cuda()[:, None]


@torch.no_grad()
def encode_head(model, rows, eeg):
    result = torch.empty((sum(row["windows"] for row in rows), 8, FRAME), device="cuda")
    model.eval()
    offset = 0
    for row in rows:
        count = row["windows"]
        xx = record_windows(eeg[row["record"]], count, np.float32)
        for first in range(0, count, 256):
            x = torch.from_numpy(xx[first:first+256].copy()).cuda()
            result[offset+first:offset+first+len(x)] = model.eeg(x).transpose(1, 2)
        offset += count
    return result


def train_head(encoder, rows, eeg, y, seed, objective, epochs=250, step_limit=None):
    x = encode_head(encoder, rows, eeg)
    torch.manual_seed(seed)
    head = torch.nn.Sequential(torch.nn.Conv1d(8, 1, 32, padding="same"),
                               torch.nn.LeakyReLU(.01)).cuda()
    optimizer = torch.optim.Adam(head.parameters(), lr=.001)
    history = []
    for epoch in range(epochs):
        head.train()
        total, seen, updates = 0., 0, 0
        for first in range(0, len(x), HEAD_BATCH):
            xx, yy = x[first:first+HEAD_BATCH], y[first:first+HEAD_BATCH]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(head(xx), yy)
            require(torch.isfinite(loss).item(), "Nonfinite head loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(xx)
            seen += len(xx)
            updates += 1
            if updates == step_limit:
                break
        history.append(dict(epoch=epoch, target_windows=seen, updates=updates, train_loss=total/seen))
        if epoch % 10 == 0 or epoch == epochs-1:
            status("head_epoch", **history[-1])
            guard()
    return head, history


class Grid:
    """Record-bound windows in original order, with global 64-batch boundaries."""
    def __init__(self, rows, eeg):
        self.parts = []
        self.ends = []
        offset = 0
        for row in rows:
            view = record_windows(eeg[row["record"]], row["windows"], np.float32)
            self.parts.append(view)
            offset += len(view)
            self.ends.append(offset)

    def __len__(self):
        return self.ends[-1]

    def batch(self, first, last):
        spans = []
        while first < last:
            index = bisect.bisect_right(self.ends, first)
            offset = 0 if index == 0 else self.ends[index-1]
            end = min(last, self.ends[index])
            spans.append(self.parts[index][first-offset:end-offset])
            first = end
        return torch.from_numpy(np.concatenate(spans, axis=0)).cuda()


def train_ft(model, grid, y, objective, epochs=5, step_limit=None):
    optimizer = torch.optim.Adam([{"params": model.encoder.parameters(), "lr": 1e-5},
                                  {"params": model.head.parameters(), "lr": .001}], weight_decay=0)
    history = []
    for epoch in range(epochs):
        model.train()
        total, seen, updates = 0., 0, 0
        for first in range(0, len(grid), HEAD_BATCH):
            last = min(first+HEAD_BATCH, len(grid))
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(grid.batch(first, last)), y[first:last])
            require(torch.isfinite(loss).item(), "Nonfinite FT loss")
            loss.backward()
            require(all(torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    "Nonfinite FT gradient")
            optimizer.step()
            total += float(loss.detach())*(last-first)
            seen += last-first
            updates += 1
            if updates == step_limit:
                break
        history.append(dict(epoch=epoch, target_windows=seen, updates=updates, train_loss=total/seen))
        status("FT_epoch", **history[-1])
        guard()
    return history


@torch.no_grad()
def decode(model, grid):
    z = np.empty((len(grid), 8, FRAME), np.float32)
    p = np.empty((len(grid), FRAME), np.float32)
    model.eval()
    for first in range(0, len(grid), HEAD_BATCH):
        last = min(first+HEAD_BATCH, len(grid))
        latent = model.encoder(grid.batch(first, last)).transpose(1, 2)
        z[first:last] = latent.cpu().numpy()
        p[first:last] = model.head(latent)[:, 0].cpu().numpy()
    return z, p


def blocks(rows, z, p):
    offset = 0
    for row in rows:
        end = offset + row["windows"]
        yield row, z[offset:end], p[offset:end]
        offset = end
    require(offset == len(z) == len(p), "No extra decoded windows")


def readout(rows, z, p, target, d4):
    n = sx = sy = sxx = sxy = 0.
    for row, _, pred in blocks(rows, z, p):
        values, keep = d4.fused(pred.astype(np.float64), row)
        x, y = values[keep], np.asarray(target[row["record"]], dtype=np.float64)[keep]
        n += len(x)
        sx += x.sum()
        sy += y.sum()
        sxx += x@x
        sxy += x@y
    a = float((sxy-sx*sy/n)/(sxx-sx*sx/n))
    b = float(sy/n-a*sx/n)
    require(np.isfinite(a) and np.isfinite(b) and a > 0, "Fold TRAIN affine slope must be positive")
    def features():
        for row, latent, pred in blocks(rows, z, p):
            yield row, latent[4:].astype(np.float64).reshape(-1, 1536), a*pred[4:].astype(np.float64)+b
    sums = [np.zeros(1536), np.zeros(192)]
    squares = [v.copy() for v in sums]
    count = 0
    for _, latent, base in features():
        count += len(latent)
        for i, value in enumerate((latent, base)):
            sums[i] += value.sum(0)
            squares[i] += np.square(value).sum(0)
    stats = [d4.stats_finish(s, t, count) for s, t in zip(sums, squares)]
    qs = np.zeros(1536)
    qt = np.zeros(1536)
    for _, latent, _ in features():
        q = np.square(d4.normalize(latent, stats[0]))
        qs += q.sum(0)
        qt += np.square(q).sum(0)
    stats.append(d4.stats_finish(qs, qt, count))
    G, H = np.zeros((3264, 3264)), np.zeros((3264, FRAME))
    xs, rs = np.zeros(3264), np.zeros(FRAME)
    for row, latent, base in features():
        u = d4.normalize(latent, stats[0])
        v = d4.normalize(base, stats[1])
        w = d4.normalize(np.square(u), stats[2])
        x = np.concatenate((u/np.sqrt(1536), v/np.sqrt(192), w/np.sqrt(1536)), axis=1)
        residual = record_windows(target[row["record"]], row["windows"], np.float32)[4:].astype(np.float64)-base
        G += x.T@x
        H += x.T@residual
        xs += x.sum(0)
        rs += residual.sum(0)
    xm, rm = xs/count, rs/count
    G = G/count-np.outer(xm, xm)
    H = H/count-np.outer(xm, rm)
    eig, basis = la.eigh((G+G.T)*.5, driver="evr")
    require(eig.min() > -1e-8, "Invalid ridge eigenvalues")
    coef = basis@((basis.T@H)/(eig[:, None]+.01))
    bias = rm-xm@coef
    err = float(np.max(np.abs((G+.01*np.eye(len(G)))@coef-H)))
    require(np.isfinite(coef).all() and np.isfinite(bias).all()
            and err <= 1e-7*max(1., float(np.abs(H).max())), "Fold Ccap normal equation")
    return (stats, coef, bias, dict(a=a, b=b)), dict(
        TRAIN_physical_points=int(n), TRAIN_fit_windows=count, normal_equation_max_abs=err)


def save_predictions(output, kind, rows, model, eeg, fit, core):
    z, p = decode(model, Grid(rows, eeg))
    anchor = []
    info = []
    offset = 0
    for row, latent, pred in blocks(rows, z, p):
        frames = core.anchor_frames(latent, pred, fit)
        require(np.array_equal(frames[:4], fit[-1]["a"]*pred[:4].astype(np.float64)+fit[-1]["b"]),
                "First four windows must remain affine P")
        values, keep = core.legacy.fuse(frames, row["common_samples"])
        stop = row["covered_stop"]
        require(not keep[0] and keep[1:stop].all() and not keep[stop:].any(),
                "Fold anchor positive-Hann support")
        anchor.append(values[:stop])
        info.append(dict(record=row["record"], offset=offset, stop=stop, windows=row["windows"]))
        offset += stop
    with (output / f"{kind}_ANCHOR.npy").open("xb") as stream:
        np.save(stream, np.concatenate(anchor), allow_pickle=False)
    exclusive(output / f"{kind}_ANCHOR.json", info)
    stored = np.load(output / f"{kind}_ANCHOR.npy", mmap_mode="r", allow_pickle=False)
    require(len(stored) == offset and read(output / f"{kind}_ANCHOR.json") == info
            and all(np.array_equal(stored[row["offset"]:row["offset"]+row["stop"]], actual)
                    for row, actual in zip(info, anchor)), "Anchor save/reload bridge")
    return sum(row["windows"] for row in rows)


def validate_history(history, expected):
    require(sum(row["target_windows"] for row in history) == expected["target_windows"],
            "Training window-presentation mismatch")
    require(sum(row["updates"] for row in history) == expected["updates"], "Update mismatch")


def main(run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--fold", required=True, type=int, choices=range(3))
    parser.add_argument("--stage", required=True, choices=("engineer", "formal"))
    args = parser.parse_args()
    started = time.monotonic()
    candidate_path = ROOT / "FOLD_AND_BUDGET_CANDIDATE_V2.json"
    candidate = read(candidate_path)
    if args.stage == "formal":
        gate = read(ROOT / "UPSTREAM_FORMAL_GATE.json")
        require(gate["passed"] and all(digest(Path(p)) == h for p, h in gate["sources"].items()),
                "Formal engineering/source pins")
        require(gate["candidate_sha256"] == digest(candidate_path), "Formal candidate pin")
        require(gate["worker_sha256"] == digest(Path(__file__)), "Formal worker pin")
    for item in candidate["input_files"].values():
        require(digest(Path(item["path"])) == item["sha256"], "Frozen metadata changed")
    path = Path(candidate["train_manifest"])
    require(digest(path) == candidate["train_manifest_sha256"], "Frozen TRAIN records changed")
    manifest = read(path)["records"]
    fit = [row for row in manifest if row["OOF_model_fold"] != args.fold]
    real_fixture_rows = fit[:12] if args.stage == "engineer" else []
    oof = [row for row in manifest if row["OOF_model_fold"] == args.fold]
    isin = [row for row in manifest if row["IS_model_fold"] == args.fold]
    require(all(row["phase"] == "train" for row in manifest), "TRAIN-only upstream")
    for field in ("record", "source_family", "eeg_path", "envelope_path", "story"):
        require({row[field] for row in fit}.isdisjoint(row[field] for row in oof),
                f"OOF separation for {field}")
    budget = next(job for job in candidate["jobs"] if job["seed"] == args.seed and job["model_fold"] == args.fold)
    modules = native, baseline, finetune, core, d4 = imported()
    if args.stage == "formal":
        require(all(str(Path(module.__file__).resolve()) in gate["sources"] for module in modules),
                "Actual upstream imports must be pinned")
    base = ROOT / ("engineer_v03" if args.stage == "engineer" else "formal")
    base.mkdir(exist_ok=True)
    output = base / f"seed{args.seed}_fold{args.fold}"
    output.mkdir()
    run["output"] = output
    process = identity()
    exclusive(output / "PROCESS.json", process)
    require(torch.cuda.device_count() == 1 and process["cuda_visible_devices"].startswith("GPU-"),
            "Worker must be pinned to a single GPU UUID")
    baseline.seed_all(args.seed, "cuda:0")
    precision = read(BASE / "BASELINE_PROTOCOL.json")["precision"]
    require(precision == read(BASE / "INCREMENTAL_PROTOCOL.json")["precision"], "Native precision")
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = precision["TF32_matmul"]
    torch.backends.cudnn.allow_tf32 = precision["TF32_cudnn"]
    guard()
    prepared = read(Path(candidate["input_files"]["prepared_manifest"]["path"]))["records"]
    index = read(Path(candidate["input_files"]["train_window_index"]["path"]))
    cl_plans = plans_for(fit, prepared, index)
    require(len(cl_plans) == budget["contrastive"]["batches_per_epoch"], "Plan count")
    if args.stage == "engineer":
        # Full metadata exclusion check above, bounded actual path on one large TRAIN story.
        plan_story = cl_plans[0][0]
        fit = [row for row in fit if row["story"] == plan_story][:2]
        require(fit and plans_for(fit, prepared, index), "Bounded engineering sampler")
    mean, std, n = fit_normalization(fit)
    eeg, target = load_rows(fit, mean, std, labels=True)
    cfg = read(NATIVE_CFG)
    require(cfg["patient_data"] == "FORBIDDEN", "Public only")
    model = native.TransformerCLIP(cfg).cuda()
    selected_init = sum(p.numel() for p in model.parameters())
    stage_cl = (time.monotonic(), time.process_time())
    cl = contrastive(model, native, fit, eeg, prepared, index, args.seed, budget["contrastive"],
                     step_limit=1 if args.stage == "engineer" else None)
    cl["seconds"] = time.monotonic()-stage_cl[0]
    cl["CPU_seconds"] = time.process_time()-stage_cl[1]
    cl["CUDA_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    allowed = {row["record"] for row in fit}
    require(cl["record_presentations"].keys() <= allowed, "Actual contrastive records")
    if args.stage == "formal":
        require(n == budget["fit_raw_channel_normalization_samples"]
                and cl["updates"] == budget["contrastive"]["updates"], "Pretrain budget")
        torch.save(model.state_dict(), output / "contrastive.pt")
    stage_head = (time.monotonic(), time.process_time())
    torch.cuda.reset_peak_memory_stats()
    y = targets_gpu(fit, target)
    head, head_log = train_head(model, fit, eeg, y, args.seed, finetune.objective,
                                epochs=250 if args.stage == "formal" else 1,
                                step_limit=None if args.stage == "formal" else 1)
    head_seconds = time.monotonic()-stage_head[0]
    head_cpu = time.process_time()-stage_head[1]
    head_peak = torch.cuda.max_memory_reserved()
    if args.stage == "formal":
        validate_history(head_log, budget["head"])
        torch.save(head.state_dict(), output / "head.pt")
    stage_ft = (time.monotonic(), time.process_time())
    baseline.seed_all(args.seed, "cuda:0")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = precision["TF32_matmul"]
    torch.backends.cudnn.allow_tf32 = precision["TF32_cudnn"]
    original = native.TransformerCLIP(cfg).cuda()
    original.load_state_dict(model.state_dict())
    decoder = finetune.Decoder(original.eeg).cuda()
    decoder.head.load_state_dict(head.state_dict())
    del model, original, head
    grid = Grid(fit, eeg)
    ft = train_ft(decoder, grid, y, finetune.objective,
                  epochs=5 if args.stage == "formal" else 1,
                  step_limit=None if args.stage == "formal" else 1)
    ft_seconds = time.monotonic()-stage_ft[0]
    ft_cpu = time.process_time()-stage_ft[1]
    ft_peak = torch.cuda.max_memory_reserved()
    del y
    evidence = dict(seed=args.seed, fold=args.fold, stage=args.stage,
                    fit_records=sorted(allowed), normalization_samples=n, contrastive=cl,
                    fresh_contrastive_parameters=selected_init, precision=precision, process=process,
                    head=head_log, head_seconds=head_seconds, head_CPU_seconds=head_cpu,
                    head_CUDA_peak_reserved_bytes=head_peak, head_encoder_windows=len(grid),
                    FT=ft, FT_seconds=ft_seconds, FT_CPU_seconds=ft_cpu, FT_CUDA_peak_reserved_bytes=ft_peak,
                    original_VAL_reads=0, old_test_reads=0)
    if args.stage == "formal":
        validate_history(ft, budget["FT"])
        torch.save(decoder.state_dict(), output / "FT.pt")
        stage_ccap = (time.monotonic(), time.process_time())
        z, p = decode(decoder, grid)
        fit_readout, ridge = readout(fit, z, p, target, d4)
        require(ridge["TRAIN_fit_windows"] == budget["Ccap"]["fit_windows"]
                and ridge["TRAIN_physical_points"] == budget["affine_physical_points"], "Ccap fit support")
        stats, coef, bias, affine = fit_readout
        np.savez(output / "READOUT.npz", mean=mean, std=std, coef=coef, bias=bias,
                 **{f"{key}_{i}": value for i, item in enumerate(stats) for key, value in item.items()})
        exclusive(output / "UPSTREAM_FROZEN.json", dict(affine=affine, Ccap=ridge,
                                                        fit_records=sorted(allowed)))
        evidence["Ccap_seconds"] = time.monotonic()-stage_ccap[0]
        evidence["Ccap_CPU_seconds"] = time.process_time()-stage_ccap[1]
        evidence["Ccap_FT_windows"] = len(grid)
        del z, p, grid, target
        # Held-out EEG enters only after all parameters and statistics are frozen.
        require(save_predictions(output, "IS", isin, decoder, eeg, fit_readout, core)
                == budget["IS_output_windows"], "IS output coverage")
        del eeg
        heldout_eeg, heldout_labels = load_rows(oof, mean, std, labels=False)
        require(not heldout_labels, "OOF labels must be unopened")
        evidence["OOF_label_reads"] = 0
        require(save_predictions(output, "OOF", oof, decoder, heldout_eeg, fit_readout, core)
                == budget["OOF_output_windows"], "OOF output coverage")
        evidence["prediction_FT_windows"] = {kind: budget[f"{kind}_output_windows"]
                                              for kind in ("IS", "OOF")}
    else:
        z, p = decode(decoder, grid)
        require(len(z) == len(p) == len(grid) and np.isfinite(p).all(), "Bounded FT encode bridge")
        # Synthetic positive-affine targets exercise the full Ccap algebra without
        # an exploratory fit to held-out or original validation data.
        fixture = {}
        for row, _, pred in blocks(fit, z, p):
            raw, _ = d4.fused(pred.astype(np.float64), row)
            fixture[row["record"]] = 0.5*raw+0.25
        estimate, diagnostics = readout(fit, z, p, fixture, d4)
        require(abs(estimate[-1]["a"]-.5) <= 1e-8 and abs(estimate[-1]["b"]-.25) <= 1e-8,
                "Synthetic positive-affine bridge")
        independent_rows = oof[:1]
        test_eeg, test_labels = load_rows(independent_rows, mean, std, labels=False)
        oz, op = decode(decoder, Grid(independent_rows, test_eeg))
        anchor, keep = core.legacy.fuse(core.anchor_frames(oz, op, estimate),
                                        independent_rows[0]["common_samples"])
        stop = independent_rows[0]["covered_stop"]
        require(not test_labels and not keep[0] and keep[1:stop].all() and not keep[stop:].any()
                and np.isfinite(anchor).all(), "Synthetic Ccap OOF output bridge")
        evidence["synthetic_readout"] = dict(affine=estimate[-1], heldout_labels_opened=0,
                                              OOF_decoded_windows=len(oz), **diagnostics)
        # Existing D8 frozen TRAIN outputs are a bounded canary only. Formal
        # initialization, normalization and prediction never open these caches.
        d8 = HOME / "s03_finetuned_readout_transfer_20261101" / f"seed{args.seed}"
        zz, pp = [np.load(d8 / name, mmap_mode="r", allow_pickle=False)
                  for name in ("train_latent.npy", "train_P.npy")]
        spans = [slice(row["offset"], row["offset"]+row["windows"]) for row in real_fixture_rows]
        labels = {row["record"]: np.load(row["envelope_path"], mmap_mode="r", allow_pickle=False)
                  .reshape(-1)[:row["common_samples"]] for row in real_fixture_rows}
        real_fit, real_diag = readout(real_fixture_rows, np.concatenate([zz[s] for s in spans]),
                                      np.concatenate([pp[s] for s in spans]), labels, d4)
        engineer_is = isin[:1]
        is_eeg, _ = load_rows(engineer_is, mean, std, labels=False)
        is_windows = save_predictions(output, "IS", engineer_is, decoder, is_eeg, real_fit, core)
        oof_windows = save_predictions(output, "OOF", independent_rows, decoder, test_eeg, real_fit, core)
        evidence["real_TRAIN_readout_canary"] = dict(records=[row["record"] for row in real_fixture_rows],
            affine=real_fit[-1], D8_fixture_only=True, first4_equal=True, saved_roundtrip=True,
            IS_windows=is_windows, OOF_windows=oof_windows, **real_diag)
    guard()
    evidence.update(cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    wall_seconds=time.monotonic()-started)
    exclusive(output / "COMPLETED.json", evidence)
    status("completed", stage=args.stage, seed=args.seed, fold=args.fold)


if __name__ == "__main__":
    worker_run = {}
    try:
        main(worker_run)
    except BaseException as error:
        if "output" in worker_run:
            exclusive(worker_run["output"] / "FAILURE.json",
                      dict(process=identity(), kind=type(error).__name__, message=str(error),
                           traceback=traceback.format_exc()))
        raise
