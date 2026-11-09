"""D21 output-only affine and FIR ridge controls on the frozen physical axis."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time


PARENT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
D19 = PARENT / "s03_physical_stream_20261106"
D20 = PARENT / "s03_recent_prediction_context_20261106"
ROOT = PARENT / "s03_output_ridge_20261106"
SEEDS = (1024, 2026, 31415)
ROLES = {"A": 1, "F": 97}
LAMBDAS = (1., .01, .0001)
BATCH, FIRST, HISTORY, BIN = 16384, 1344, 384, 4
GIB = 1024**3
LIMIT = {"engineer": 600, "fit": 7200}
PINNED = {
    D19 / "PROTOCOL.json": "133b60eeb6e937def9f25bdb8303ab80d9a2cf00d0f86ea3f6768c0f56a7b078",
    D19 / "PREPARATION_METADATA.json": "1068d1eede47893e074c7a67b7a57c1a967364cb974ffdf088a1d623121e24bd",
    D20 / "PROTOCOL.json": "535d06ed1e12ea1048293ca1be63bb2ad8dfd7c0b985dd9ca9503d6f64735540",
    D20 / "fit/COMPLETED.json": "40637c0a9f38bf47942b7a4e5268435a06f9f86f22421b41ba09faad1ebf7320",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(pins):
    for path, digest in pins.items():
        require(sha(Path(path)) == digest, f"Source or frozen receipt drift: {path}")


def prepare():
    verify(PINNED)
    meta = read(D19 / "PREPARATION_METADATA.json")
    d20 = read(D20 / "PROTOCOL.json")
    require(d20["near_lag"] == 1 and d20["first_active"] == FIRST
            and read(D20 / "fit/COMPLETED.json")["passed"], "Unexpected D20 parent")
    pins = {str(k): v for k, v in PINNED.items()}
    for name, digest in meta["prepared_source_sha256"].items():
        pins[str(D19 / "input" / name)] = digest
    for seed in SEEDS:
        for name in ("TRAIN_STATS", "ANCHOR"):
            path = D20 / "fit" / f"seed{seed}_{name}.json"
            pins[str(path)] = sha(path)
    verify(pins)
    train, val = [meta["phases"][p] for p in ("train", "val")]
    train_windows = sum(row["windows"] for row in train["records"])
    val_windows = sum(row["windows"] for row in val["records"])
    engineer_train = sum(row["windows"] for row in train["records"][:2])
    engineer_val = val["records"][0]["windows"] + val["records"][-1]["windows"]
    budget = dict(TRAIN_records=3 * len(train["records"]), VAL_records=3 * len(val["records"]),
                  TRAIN_cached_windows=3 * train_windows, VAL_cached_windows=3 * val_windows,
                  TRAIN_Gram_points=3 * train["active_points"], VAL_feature_points=3 * val["active_points"],
                  VAL_candidate_points=18 * val["active_points"], exact_ridge_solves=18,
                  VAL_anchor_blocks=3 * len(val["blocks"]), VAL_candidate_blocks=18 * len(val["blocks"]),
                  new_NN_forwards=0, optimizer_updates=0)
    engineering = dict(TRAIN_records=6, VAL_records=6, TRAIN_cached_windows=3 * engineer_train,
                       VAL_cached_windows=3 * engineer_val, TRAIN_Gram_points=3 * 512,
                       small_ridge_solves=18, independent_lstsq_solves=18, new_NN_forwards=0)
    require((train["active_points"], val["active_points"], len(val["blocks"])) ==
            (23703510, 2866799, 752), "D20 endpoint geometry drift")
    protocol = dict(candidate="D21_OUTPUT_RIDGE_V1", source_sha256=sha(Path(__file__)),
                    frozen_pins=pins, seed=SEEDS, features=ROLES, lambdas=LAMBDAS,
                    first_active=FIRST, lag=1, history=HISTORY, bin_points=BIN, batch=BATCH,
                    objective="mean((b+intercept+X beta-y)^2) + lambda * sum(beta^2)",
                    feature_precision="raw float32, D20 P TRAIN reference FP32 affine, then FP64 / sqrt(D)",
                    target_precision="TRAIN float32-rounded envelope, FP64 residual and sufficient statistics",
                    resources={"seconds": LIMIT, "RSS_GiB": 6, "SATA_min_free_GiB": 20,
                               "writer_MiB": 64, "BLAS_threads": 8, "device": "CPU"},
                    formal_budget=budget, engineering_budget=engineering,
                    fit_requires="Independent EXECUTION_REVIEW bound to this protocol and actual engineering")
    write(ROOT / "PROTOCOL.json", protocol)
    print(sha(ROOT / "PROTOCOL.json"), flush=True)


def admit(stage):
    path = ROOT / "PROTOCOL.json"
    protocol = read(path)
    protocol_sha = sha(path)
    require(protocol["candidate"] == "D21_OUTPUT_RIDGE_V1"
            and protocol["source_sha256"] == sha(Path(__file__)), "D21 source drift")
    verify(protocol["frozen_pins"])
    meta = read(D19 / "PREPARATION_METADATA.json")
    verify(meta["provenance"])
    for path, stamp in meta["inputs"].items():
        stat = Path(path).stat()
        require((stat.st_size, stat.st_mtime_ns) == (stamp["bytes"], stamp["mtime_ns"]),
                f"Frozen input changed: {path}")
    if stage == "fit":
        review = read(ROOT / "EXECUTION_REVIEW.json")
        eng = ROOT / "engineer/COMPLETED.json"
        require(review["stage"] == "fit" and review["decision"] == "PASS"
                and review["protocol_sha256"] == protocol_sha
                and review["engineering_sha256"] == sha(eng) and read(eng)["passed"],
                "Missing bound independent fit review")
    return protocol, protocol_sha, meta


class Receipt:
    def __init__(self, stage, folder, protocol):
        self.stage = stage
        self.folder = folder
        self.protocol = protocol
        self.wall = time.monotonic()
        self.cpu = time.process_time()
        start = time.time()
        ticks = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19]
        self.identity = dict(pid=os.getpid(), uid=os.getuid(), argv=sys.argv, linux_startticks=int(ticks),
                             start_unix=start, deadline_unix=start + LIMIT[stage])
        self.count = Counter()
        self.state = {}

    def snapshot(self):
        return dict(stage=self.stage, protocol_sha256=self.protocol, process=self.identity,
                    location=self.state, counters=dict(self.count), CPU_seconds=time.process_time() - self.cpu,
                    wall_seconds=time.monotonic() - self.wall,
                    new_NN_forwards=0, new_optimizer_updates=0, new_encoder_head_windows=0,
                    resources=self.resources)

    def guard(self, checkpoint=False):
        self.resources = dict(
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            writer_free_bytes=shutil.disk_usage(ROOT).free,
            writer_bytes=sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file()),
        )
        require(self.resources["peak_RSS_bytes"] <= 6 * GIB, "RSS cap")
        require(self.resources["writer_free_bytes"] >= 20 * GIB, "SATA free space cap")
        require(self.resources["writer_bytes"] <= 64 * 1024**2, "Output cap")
        if checkpoint:
            tmp = self.folder / "CURRENT_STATUS.json.tmp"
            write(tmp, self.snapshot())
            tmp.replace(self.folder / "CURRENT_STATUS.json")


class Gram:
    def __init__(self, np):
        self.np = np
        self.n = 0
        self.x = np.zeros(97)
        self.xx = np.zeros((97, 97))
        self.xr = np.zeros(97)
        self.r = self.rr = 0.

    def add(self, x, r):
        self.n += len(r)
        self.x += x.sum(axis=0)
        self.xx += x.T @ x
        self.xr += x.T @ r
        self.r += float(r.sum())
        self.rr += float(r @ r)

    def solve(self, d, penalty):
        np = self.np
        scale = np.sqrt(d)
        mx = self.x[:d] / (self.n * scale)
        mr = self.r / self.n
        xx = self.xx[:d, :d] / (self.n * d)
        xr = self.xr[:d] / (self.n * scale)
        cov = xx - np.outer(mx, mx)
        rhs = xr - mx * mr
        regularized = cov + penalty * np.eye(d)
        factor = np.linalg.cholesky(regularized)
        beta = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
        intercept = float(mr - mx @ beta)
        grad = np.r_[intercept + mx @ beta - mr, xx @ beta + mx * intercept - xr + penalty * beta]
        normal_error = float(np.max(np.abs(grad)))
        require(normal_error <= 1e-10 and np.isfinite(beta).all(), "Ridge normal equation failed")
        mse = (self.rr / self.n + intercept**2 + beta @ xx @ beta + 2 * intercept * (mx @ beta - mr)
               - 2 * beta @ xr)
        return beta, dict(intercept=intercept, train_active_MSE=float(mse),
                          objective=float(mse + penalty * (beta @ beta)), normal_abs=normal_error,
                          beta_l2=float(np.linalg.norm(beta)))


def normalizer(seed, np):
    stats = read(D20 / "fit" / f"seed{seed}_TRAIN_STATS.json")["P_past_prediction"]
    require(not stats["constant"] and stats["std"] > 0, "Unexpected D20 history scale")
    return np.float32(stats["mean"]), np.float32(stats["std"])


def records(seed, phase, meta, core, receipt, indices=None):
    np = core.np
    folder = core.assets(seed)["folder"]
    z = np.load(folder / f"{phase}_latent.npy", mmap_mode="r", allow_pickle=False)
    p = np.load(folder / f"{phase}_P.npy", mmap_mode="r", allow_pickle=False)
    readout = core.frozen_assets(seed)
    spec = meta["phases"][phase]
    blocks = defaultdict(list)
    for row in spec["blocks"]:
        blocks[row["record"]].append(row)
    rows = spec["records"] if indices is None else [spec["records"][i] for i in indices]
    for row in rows:
        start, windows = row["offset"], row["windows"]
        stop, samples = row["covered_stop"], row["common_samples"]
        pred, keep = core.legacy.fuse(
            core.anchor_frames(z[start:start + windows], p[start:start + windows], readout), samples)
        require(not keep[0] and keep[1:stop].all() and not keep[stop:].any(), "Positive Hann geometry")
        target = np.load(row["envelope_path"], mmap_mode="r", allow_pickle=False).reshape(-1)[:stop]
        if phase == "train":
            target = target.astype(np.float32).astype(np.float64)
        else:
            target = target.astype(np.float64, copy=True)
        require(np.isfinite(target).all(), "Nonfinite frozen target")
        receipt.count[phase.upper() + "_records"] += 1
        receipt.count[phase.upper() + "_cached_windows"] += windows
        yield row, pred[:stop].copy(), target, blocks[row["record"]]


def features(b, rolling, lo, hi, norm, np):
    """Unscaled normalized F features. A is the first column of the same matrix."""
    points = np.arange(lo, hi, dtype=np.int64)
    ends = points[:, None] - 1 - (HISTORY - BIN) + np.arange(HISTORY // BIN) * BIN
    x = np.empty((hi - lo, 97), dtype=np.float32)
    x[:, 0] = b[lo:hi]
    x[:, 1:] = rolling[ends]
    x -= norm[0]
    x /= norm[1]
    return x.astype(np.float64)


def summary_metrics(rows, active_sse, active_count, full, core):
    out = core.aggregate_blocks(rows)
    people = defaultdict(list)
    for subject, r in full:
        people[subject].append(r)
    out.update(active_MSE=active_sse / active_count, active_points=active_count,
               full_record_r=float(core.np.mean([core.np.mean(rs) for rs in people.values()])))
    return out


def score_record(row, blocks, wave, target, core):
    out = []
    for block in blocks:
        lo = max(1, block["start"])
        hi = min(len(wave), block["start"] + 3840)
        p, y = wave[lo:hi], target[lo:hi]
        require(len(p) == block["samples"], "Frozen score mask")
        out.append(dict(**block, subject=row["subject"], source_family=row["source_family"],
                        r=core.legacy.corr(p, y), MSE=float(core.np.mean((p - y)**2))))
    err = wave[FIRST:] - target[FIRST:]
    return out, float(err @ err), (row["subject"], core.legacy.corr(wave[1:], target[1:]))


def anchors(seed, val, core):
    reference = read(D20 / "fit" / f"seed{seed}_ANCHOR.json")["anchor"]["val"]
    expected = {(b["record"], b["start"]): b["r"] for b in reference["blocks"]}
    rows, full, sse, n = [], [], 0., 0
    for row, b, y, blocks in val:
        rr, ee, record = score_record(row, blocks, b, y, core)
        rows.extend(rr)
        full.append(record)
        sse += ee
        n += len(b) - FIRST
    error = max(abs(r["r"] - expected[r["record"], r["start"]]) for r in rows)
    require(error <= 1e-10, "Anchor block bridge")
    metric = summary_metrics(rows, sse, n, full, core)
    if len(val) == 49:
        require(len(rows) == 752 and len(metric["people"]) == 49
                and abs(metric["r"] - reference["r"]) <= 1e-10
                and abs(metric["active_MSE"] - reference["active_MSE"]) <= 1e-10,
                "Full anchor bridge")
    return metric, error


def engineer(protocol, meta, core, metadata, receipt):
    np = core.np
    checks = []
    for seed in SEEDS:
        receipt.state = dict(seed=seed, phase="bounded_engineering")
        norm = normalizer(seed, np)
        gram = Gram(np)
        independent_x, independent_r = [], []
        max_error = 0.
        train = list(records(seed, "train", meta, core, receipt, (0, 1)))
        for i, (row, b, y, _) in enumerate(train):
            lo = FIRST if i == 0 else len(b) - 256
            hi = lo + 256
            rr = y[lo:hi] - b[lo:hi]
            xx = features(b, core.rolling_mean(b, BIN), lo, hi, norm, np)
            manual = []
            for t in range(lo, hi):
                require(t - HISTORY >= 1, "Left history boundary")
                raw_stop = lambda u: metadata.raw_support(row["windows"], u)[1]
                legal = lambda u: max(j for j in range(row["windows"]) if 1 <= u - j * 96 <= 190) * 96 + 192
                require(raw_stop(t - 1) == legal(t - 1) <= raw_stop(t) == legal(t), "Raw future expanded")
                bins = b[t - HISTORY:t].reshape(-1, BIN).mean(axis=1, dtype=np.float64)
                manual.append(np.r_[b[t], bins])
            manual = np.asarray(manual, dtype=np.float32)
            manual = ((manual - norm[0]) / norm[1]).astype(np.float64)
            max_error = max(max_error, float(np.max(np.abs(manual - xx))))
            independent_x.append(manual)
            independent_r.append(rr)
            gram.add(xx, rr)
            receipt.count["TRAIN_Gram_points"] += len(rr)
        require(max_error <= 2e-6, "Explicit history bin bridge")
        val = list(records(seed, "val", meta, core, receipt, (0, -1)))
        metric, anchor_error = anchors(seed, val, core)
        x = np.concatenate(independent_x)
        y = np.concatenate(independent_r)
        gram_error = max(float(np.max(np.abs(gram.xx - x.T @ x))),
                         float(np.max(np.abs(gram.x - x.sum(axis=0)))),
                         float(np.max(np.abs(gram.xr - x.T @ y))),
                         abs(gram.r - float(y.sum())), abs(gram.rr - float(y @ y)))
        require(gram_error / len(x) <= 2e-6, "Sufficient-statistics bridge")
        max_pred = max_normal = 0.
        for d in ROLES.values():
            design = np.column_stack([np.ones(len(x)), x[:, :d] / np.sqrt(d)])
            for lam in LAMBDAS:
                beta, fit = gram.solve(d, lam)
                receipt.count["small_ridge_solves"] += 1
                ridge = np.column_stack([np.zeros(d), np.eye(d) * np.sqrt(lam * len(x))])
                other, *_ = np.linalg.lstsq(np.vstack([design, ridge]), np.r_[y, np.zeros(d)], rcond=None)
                receipt.count["independent_lstsq_solves"] += 1
                err = float(np.max(np.abs(design @ (other - np.r_[fit["intercept"], beta]))))
                max_pred = max(max_pred, err)
                max_normal = max(max_normal, fit["normal_abs"])
        require(max_pred <= 2e-6, "Independent ridge solution bridge")
        checks.append(dict(seed=seed, explicit_feature_max_abs=max_error, raw_Gram_max_abs=gram_error,
                           ridge_prediction_max_abs=max_pred, normal_abs=max_normal,
                           anchor_block_max_abs=anchor_error, anchor_blocks=len(metric["blocks"])))
        receipt.guard(True)
    for name, n in protocol["engineering_budget"].items():
        require(receipt.count[name] == n, f"Engineering budget drift: {name}")
    return dict(passed=True, checks=checks)


def write_coefficients(folder, seed, gram, models, norm, np):
    arrays = {k: getattr(gram, k) for k in ("x", "xx", "xr", "r", "rr", "n")}
    for name, (d, beta, fitted) in models.items():
        slopes = beta / (np.sqrt(d) * float(norm[1]))
        kernel = np.zeros(HISTORY + 1)
        kernel[-1] = 1 + slopes[0]
        if d > 1:
            kernel[:-1] = np.repeat(slopes[1:] / BIN, BIN)
        arrays[name + "_beta"] = beta
        arrays[name + "_intercept"] = fitted["intercept"]
        arrays[name + "_physical_intercept"] = fitted["intercept"] - float(norm[0]) * slopes.sum()
        arrays[name + "_physical_kernel"] = kernel
        fitted.update(current_gain=float(kernel[-1]), past_gain_sum=float(kernel[:-1].sum()),
                      past_gain_L1=float(np.abs(kernel[:-1]).sum()))
    with (folder / f"seed{seed}_COEFFICIENTS_AND_GRAM.npz").open("xb") as stream:
        np.savez(stream, **arrays)


def fit(protocol, meta, core, receipt):
    np = core.np
    selections = {}
    for seed in SEEDS:
        receipt.state = dict(seed=seed, phase="VAL_anchor_bridge")
        val = list(records(seed, "val", meta, core, receipt))
        anchor, anchor_error = anchors(seed, val, core)
        receipt.count["VAL_anchor_blocks"] += len(anchor["blocks"])
        receipt.state["phase"] = "TRAIN_Gram"
        norm = normalizer(seed, np)
        gram = Gram(np)
        for row, b, y, _ in records(seed, "train", meta, core, receipt):
            rolling = core.rolling_mean(b, BIN)
            for lo in range(FIRST, len(b), BATCH):
                hi = min(lo + BATCH, len(b))
                gram.add(features(b, rolling, lo, hi, norm, np), y[lo:hi] - b[lo:hi])
                receipt.count["TRAIN_Gram_points"] += hi - lo
            receipt.guard(True)
        require(gram.n == meta["phases"]["train"]["active_points"], "TRAIN coverage")
        models = {}
        for role, d in ROLES.items():
            for lam in LAMBDAS:
                beta, fitted = gram.solve(d, lam)
                models[f"{role}_{lam:g}"] = (d, beta, dict(role=role, penalty=lam, **fitted))
                receipt.count["exact_ridge_solves"] += 1
        write_coefficients(receipt.folder, seed, gram, models, norm, np)
        receipt.state["phase"] = "VAL_candidates"
        totals = {key: dict(rows=[], sse=0., full=[]) for key in models}
        n = 0
        names = list(models)
        matrix = np.zeros((97, len(names)))
        offsets = np.empty(len(names))
        for i, key in enumerate(names):
            d, beta, fitted = models[key]
            matrix[:d, i] = beta / np.sqrt(d)
            offsets[i] = fitted["intercept"]
        for row, b, y, blocks in val:
            wave = np.repeat(b[:, None], len(models), axis=1)
            rolling = core.rolling_mean(b, BIN)
            for lo in range(FIRST, len(b), BATCH):
                hi = min(lo + BATCH, len(b))
                wave[lo:hi] += offsets + features(b, rolling, lo, hi, norm, np) @ matrix
                receipt.count["VAL_feature_points"] += hi - lo
                receipt.count["VAL_candidate_points"] += (hi - lo) * len(models)
            n += len(b) - FIRST
            for i, key in enumerate(names):
                rr, ee, full = score_record(row, blocks, wave[:, i], y, core)
                acc = totals[key]
                acc["rows"].extend(rr)
                acc["sse"] += ee
                acc["full"].append(full)
                receipt.count["VAL_candidate_blocks"] += len(rr)
            receipt.guard(True)
        require(n == meta["phases"]["val"]["active_points"], "VAL coverage")
        metrics = {key: dict(**models[key][2], val=summary_metrics(v["rows"], v["sse"], n, v["full"], core))
                   for key, v in totals.items()}
        best = {}
        for role in ROLES:
            selected, chosen_r = None, anchor["r"]
            for key in (k for k in names if k.startswith(role + "_")):
                if metrics[key]["val"]["r"] > chosen_r:
                    selected, chosen_r = key, metrics[key]["val"]["r"]
            best[role] = dict(selected=selected, r=chosen_r, delta_r=chosen_r - anchor["r"])
        selections[seed] = best
        write(receipt.folder / f"seed{seed}_METRICS.json",
              dict(seed=seed, anchor=anchor, train_anchor_active_MSE=gram.rr / gram.n,
                   anchor_block_max_abs=anchor_error, selection=best, models=metrics))
    for name, n in protocol["formal_budget"].items():
        require(receipt.count[name] == n, f"Formal budget drift: {name}")
    return dict(passed=True, selections=selections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "engineer", "fit"))
    stage = parser.parse_args().stage
    require(ROOT.resolve() == ROOT and Path(__file__).resolve().parent == ROOT / "input",
            "Use the independent 2203 SATA root")
    if stage == "prepare":
        prepare()
        return
    protocol, protocol_sha, meta = admit(stage)
    folder = ROOT / stage
    folder.mkdir()
    receipt = Receipt(stage, folder, protocol_sha)
    try:
        def timeout(*_):
            raise TimeoutError("D21 CPU wall-clock cap")

        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(LIMIT[stage])
        os.environ.update(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8",
                          MKL_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="")
        sys.path.insert(0, str(D19 / "input"))
        import pub_01_s03_ss07_physical_stream_core as core
        import pub_01_s03_ss07_physical_stream_meta as metadata
        receipt.guard(True)
        result = (engineer(protocol, meta, core, metadata, receipt) if stage == "engineer"
                  else fit(protocol, meta, core, receipt))
        receipt.guard()
        write(folder / "COMPLETED.json", dict(receipt.snapshot(), **result))
    except BaseException as error:
        receipt.guard()
        write(folder / "FAILURE.json", dict(receipt.snapshot(), error=repr(error)))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
