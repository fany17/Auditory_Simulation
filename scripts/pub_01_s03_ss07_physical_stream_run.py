"""Run one reviewed D19 stage on 2203. Admission precedes numerical imports."""

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

import pub_01_s03_ss07_physical_stream_meta as m


GIB = 1024**3
MAX_SECONDS = 6 * 3600
MAX_ENGINEERING_SECONDS = 600
INPUT = m.ROOT / "input"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def admit(stage):
    protocol_path = m.ROOT / "PROTOCOL.json"
    protocol = m.read_json(protocol_path)
    protocol_sha = m.digest(protocol_path)
    preparation = m.ROOT / "PREPARATION_METADATA.json"
    require(m.digest(preparation) == protocol["metadata_sha256"], "Unbound metadata")
    metadata = m.read_json(preparation)
    sources = [Path(__file__).name, m.__name__ + ".py",
               "pub_01_s03_ss07_physical_stream_core.py"]
    require(set(protocol["source_sha256"]) == set(sources)
            and protocol["source_sha256"] == metadata["prepared_source_sha256"],
            "Incomplete source binding")
    for name in sources:
        require(m.digest(INPUT / name) == protocol["source_sha256"][name], "Source drift")
    for path, sha in metadata["provenance"].items():
        require(m.digest(path) == sha, f"Provenance drift {path}")
    for path, stamp in metadata["inputs"].items():
        actual = m.identity(Path(path))
        require(all(stamp[key] == value for key, value in actual.items()), f"Input drift {path}")
    require(m.ROOT.resolve() == m.ROOT and Path(__file__).resolve().parent == INPUT
            and Path(m.__file__).resolve().parent == INPUT, "Wrong SATA execution path")
    require(metadata["stage"] == "SOURCE_METADATA_ONLY", "Unexpected prep provenance")
    require(protocol["candidate"] == metadata["candidate"] == m.CANDIDATE, "Wrong candidate")
    require(protocol["authorized_stages"] == ["engineer", "fit"], "Unexpected protocol scope")
    gate = "ENGINEERING_REVIEW.json" if stage == "engineer" else "EXECUTION_REVIEW.json"
    review = m.read_json(m.ROOT / gate)
    require(review["protocol_sha256"] == protocol_sha and review["decision"] == "PASS"
            and review["stage"] == stage, "Stage review missing or unbound")
    if stage == "fit":
        receipt_path = m.ROOT / "engineer/COMPLETED.json"
        receipt = m.read_json(receipt_path)
        require(receipt["protocol_sha256"] == protocol_sha and receipt["passed"]
                and m.digest(receipt_path) == review["engineering_sha256"],
                "Formal execution requires the reviewed engineering receipt")
    gpu = protocol["cuda_uuid"]
    require(gpu.startswith("GPU-") and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu,
            "CUDA UUID not pinned")
    used = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    require(used.isdecimal() and int(used) <= 256, "Selected GPU is busy")
    require(shutil.disk_usage(m.ROOT).free >= 20 * GIB, "Insufficient space on SATA")
    return metadata, protocol_sha


class Receipt:
    def __init__(self, folder, stage, protocol_sha):
        self.folder = folder
        self.start = time.monotonic()
        self.start_wall = time.time()
        self.start_cpu = time.process_time()
        self.counters = Counter()
        self.scope = {}
        self.stage = stage
        self.protocol_sha = protocol_sha
        self.runtime = {}
        self.identity = dict(pid=os.getpid(), uid=os.getuid(), argv=sys.argv,
                             linux_startticks=int(Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19]),
                             stage_started_unix=self.start_wall,
                             deadline_unix=self.start_wall + (MAX_ENGINEERING_SECONDS if stage == "engineer"
                                                             else MAX_SECONDS),
                             cuda_uuid=os.environ["CUDA_VISIBLE_DEVICES"])
        self.versions = {"python": sys.version}

    def count(self, field, amount=1):
        self.counters[field] += int(amount)

    def snapshot(self):
        return dict(stage=self.stage, protocol_sha256=self.protocol_sha, process=self.identity,
                    runtime_versions=self.versions, CPU_seconds=time.process_time() - self.start_cpu,
                    elapsed_seconds=time.monotonic() - self.start, counters=dict(self.counters),
                    location=self.scope, resources=self.runtime,
                    new_encoder_head_windows=0, new_signal_feature_prediction_array_files=0,
                    parameter_checkpoint_files=self.counters["parameter_checkpoint_files"],
                    training_statistics_files=self.counters["training_statistics_files"])

    def checkpoint(self):
        atomic_json(self.folder / "CURRENT_STATUS.json", self.snapshot())

    def guard(self, torch):
        self.runtime = dict(peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            writer_free_bytes=shutil.disk_usage(m.ROOT).free,
                            writer_bytes=sum(f.stat().st_size for f in m.ROOT.rglob("*") if f.is_file()))
        require(self.runtime["peak_RSS_bytes"] <= 12 * GIB, "RSS cap")
        require(self.runtime["cuda_peak_allocated_bytes"] <= 8 * GIB, "CUDA memory cap")
        require(self.runtime["writer_free_bytes"] >= 20 * GIB, "SATA free space floor")
        require(self.runtime["writer_bytes"] <= 256 * 1024**2, "SATA output cap")


def atomic_json(path, value):
    tmp = path.with_suffix(".json.tmp")
    m.write_json(tmp, value)
    tmp.replace(path)


def engineer(meta, receipt, core, torch, guard):
    np = core.np
    checks = []
    for seed in m.SEEDS:
        receipt.scope = dict(seed=seed, phase="first_two_TRAIN_records")
        records = core.load_phase(meta, seed, "train", guard, train_records_limit=2)
        record = records[0]
        receipt.count("TRAIN_records_loaded", len(records))
        receipt.count("TRAIN_cached_windows_loaded", sum(r.metadata["windows"] for r in records))
        receipt.scope["phase"] = "VAL_frozen_anchor_bridge"
        val = core.load_phase(meta, seed, "val", guard)
        receipt.count("VAL_records_loaded", len(val))
        receipt.count("VAL_cached_windows_loaded", sum(r.metadata["windows"] for r in val))
        anchor = core.evaluate(val, None, None, None, None, guard, receipt.count)["near"]
        anchor_error = anchor_bridge(anchor, seed)
        del val
        receipt.scope["phase"] = "first_two_TRAIN_records"
        table = core.Tables(records, "cuda:0", guard)
        stats, count = core.fit_statistics(records, table, guard)
        receipt.count("TRAIN_statistics_points", count)
        spans = [(0, m.FIRST_ACTIVE, m.FIRST_ACTIVE + 2048),
                 (1, records[1].stop - 2048, records[1].stop)]
        # Interleave two records in one batch, including the final dense point.
        dense = np.column_stack((np.arange(2048), np.arange(count - 2048, count))).astype(np.int32)
        batch_indices = table.active_indices(dense.ravel())
        expected_indices = table.indices(spans).reshape(2, 2048).T.contiguous().flatten()
        require(torch.equal(batch_indices, expected_indices), "Cross-record dense gather differs")
        boundary = record.stop - m.FIRST_ACTIVE
        check_indices = table.active_indices(np.array([boundary - 1, boundary], dtype=np.int32))
        require(torch.equal(check_indices,
                            table.indices([(0, record.stop - 1, record.stop),
                                           (1, m.FIRST_ACTIVE, m.FIRST_ACTIVE + 1)])),
                "Dense boundary switches records incorrectly")
        row = record.metadata
        cache = np.load(m.assets(seed)["folder"] / "train_latent.npy", mmap_mode="r",
                        allow_pickle=False)
        coords = sorted({1, 95, 96, 97, m.FIRST_ACTIVE, m.FIRST_ACTIVE + 1,
                         m.FIRST_ACTIVE + 95, m.FIRST_ACTIVE + 96,
                         record.stop - 97, record.stop - 2, record.stop - 1})
        h = np.hanning(m.FRAME)
        fuse_error = 0.
        for point in coords:
            first, last = m.raw_support(row["windows"], point)
            frame = np.arange(first, last - m.FRAME + 1, m.HOP)
            weights = h[point - frame]
            reference = np.average(cache[row["offset"] + frame // m.HOP, :, point - frame],
                                   axis=0, weights=weights).astype(np.float32)
            fuse_error = max(fuse_error, float(np.max(np.abs(reference - record.latent[point]))))
        require(fuse_error <= 2e-6, "Cached latent fusion bridge differs")
        probes = [p for p in coords if p >= m.FIRST_ACTIVE]
        probe_spans = [(0, point, point + 1) for point in probes]
        indices = table.indices(probe_spans)
        current_ref = []
        frame_ids = np.arange(row["windows"])
        for point in probes:
            local = point - m.HOP * frame_ids
            eligible = (local > 0) & (local < m.FRAME - 1)
            contributors = frame_ids[eligible]
            if len(contributors) == 1:
                contributors = np.repeat(contributors, 2)
            raw = cache[row["offset"] + contributors]
            means = raw.reshape(2, m.CHANNELS, m.FRAME // m.BIN_POINTS, m.BIN_POINTS).mean(
                axis=-1, dtype=np.float64).transpose(0, 2, 1).reshape(m.BINS, m.CHANNELS)
            current_ref.append(means)
        got = table.context_raw(indices, "C_current").cpu().numpy()
        current_error = float(np.max(np.abs(np.asarray(current_ref, dtype=np.float32) - got)))
        require(current_error <= 2e-6, "Positive-contributor current context bridge differs")
        history_error = 0.
        for lag in (m.NEAR, m.OLDER):
            for role, wave in (("H_past_eeg", record.latent), ("P_past_prediction", record.anchor)):
                expected = np.stack([wave[p - lag - m.HISTORY + 1:p - lag + 1]
                                     .reshape((m.BINS, m.BIN_POINTS) + wave.shape[1:])
                                     .mean(axis=1, dtype=np.float64) for p in probes]).astype(np.float32)
                got = table.history(indices, role, lag).cpu().numpy()
                history_error = max(history_error, float(np.max(np.abs(expected - got))))
        require(history_error <= 2e-6, "Ordered history bridge differs")
        edge_records = ((record, record.stop - 1), (records[1], m.FIRST_ACTIVE))
        for role in m.ROLES:
            for lag in ((m.NEAR,) if role == "C_current" else (m.NEAR, m.OLDER)):
                expected = []
                for row_record, point in edge_records:
                    if role == "C_current":
                        first, last = m.raw_support(row_record.metadata["windows"], point)
                        frame_ids = [first // m.HOP, (last - m.FRAME) // m.HOP]
                        value = row_record.frame_means[frame_ids].reshape(m.BINS, m.CHANNELS)
                    else:
                        wave = row_record.latent if role == "H_past_eeg" else row_record.anchor
                        value = (wave[point - lag - m.HISTORY + 1:point - lag + 1]
                                 .reshape((m.BINS, m.BIN_POINTS) + wave.shape[1:])
                                 .mean(axis=1, dtype=np.float64))
                    expected.append(value)
                got = table.context_raw(check_indices, role, lag).cpu().numpy()
                require(np.max(np.abs(got - np.asarray(expected, dtype=np.float32))) <= 2e-6,
                        "Cross-record context leaked or changed order")
        initialization, current_initialization = None, None
        for role in m.ROLES:
            receipt.scope["role"] = role
            model = core.PhysicalDecoder(seed, role).cuda()
            require(sum(p.numel() for p in model.parameters()) == m.PARAMETERS[role], "Parameter count")
            flat = torch.cat([p.detach().flatten() for p in model.parameters()]).cpu()
            current = torch.cat([p.detach().flatten() for p in model.current.parameters()]).cpu()
            if initialization is None:
                initialization, current_initialization = flat, current
            if role != "P_past_prediction":
                require(torch.equal(flat, initialization), "C/H initialization differs")
            require(torch.equal(current, current_initialization), "Current initialization differs")
            x, v, b, y = core.batch_inputs(table, batch_indices, role, stats)
            gradients = []
            for nonzero in (False, True):
                model.zero_grad(set_to_none=True)
                if nonzero:
                    with torch.no_grad():
                        model.output.weight.copy_(torch.linspace(-.025, .025, m.WIDTH, device="cuda"))
                prediction = (torch.cat([model(x[i:i + 2048], v[i:i + 2048], b[i:i + 2048])
                                         for i in (0, 2048)]) if nonzero else model(x, v, b))
                receipt.count("current_stem_points", len(b))
                receipt.count("context_output_points", len(b))
                if not nonzero:
                    require(torch.equal(prediction, b), "Fresh residual changed the anchor")
                prediction.retain_grad()
                ((prediction - y).square().sum() / m.BATCH).backward()
                receipt.count("backward_points", len(b))
                require(torch.equal(prediction.grad, 2 * (prediction.detach() - y) / m.BATCH),
                        "Short batch target weights differ")
                norms = {name: float(layer.weight.grad.norm())
                         for name, layer in model.named_children()}
                require(all(np.isfinite(v) for v in norms.values()) and norms["output"] > 0,
                        "Invalid engineering gradients")
                require(all((value > 0) == nonzero for name, value in norms.items()
                            if name != "output"), "Unexpected hidden gradients")
                gradients.append(norms)
            pair_delta = None
            with torch.no_grad():
                code = model.current_code(x)
                full = b + model.decode(code, v).double()
                receipt.count("current_stem_points", len(b))
                receipt.count("context_output_points", len(b))
                split_error = float((prediction - full).abs().max())
                require(split_error <= 2e-6 and
                        abs(float((prediction - y).square().mean() - (full - y).square().mean())) <= 2e-6,
                        "Batch split changed physical output or loss")
                if role != "C_current":
                    old_v = table.context(batch_indices, role, stats, m.OLDER)
                    old = b + model.decode(code, old_v).double()
                    receipt.count("context_output_points", len(b))
                    pair_delta = float((full - old).abs().max())
                    require(np.isfinite(pair_delta), "Non-finite paired substitution")
            checks.append(dict(seed=seed, role=role, ranges=spans, gradients=gradients,
                               latent_max_error=fuse_error, history_max_error=history_error,
                               current_context_max_error=current_error, anchor_block_r_error=anchor_error,
                               split_prediction_max_abs=split_error, temporary_pair_max_abs=pair_delta))
            guard()
    for key in ("current_stem_points", "context_output_points", "backward_points"):
        require(receipt.counters[key] == meta["proposal"]["engineering_" + key], "Wrong engineering size")
    return {"passed": True, "checks": checks, "optimizer_updates": 0, "formal_fits": 0}


def anchor_bridge(metric, seed):
    reference = m.read_json(m.assets(seed)["anchor_metrics"])["metrics"]
    block_r = {(row["record"], row["start"]): row["r"] for row in reference["segments"]}
    computed = {(row["record"], row["start"]): row["r"] for row in metric["blocks"]}
    require(computed.keys() == block_r.keys() and len(computed) == 752
            and len(metric["people"]) == 49, "Frozen VAL coverage changed")
    error = max(abs(r - block_r[key]) for key, r in computed.items())
    require(error <= 1e-10 and abs(metric["r"] - reference["phases"]["val"]["r"]) <= 1e-10,
            "Frozen VAL anchor score differs")
    return error


def small_metric(metric):
    return {key: val for key, val in metric.items() if key not in ("people", "blocks")}


def initialization_sha(model):
    sha = hashlib.sha256()
    for name, value in model.state_dict().items():
        sha.update(name.encode())
        sha.update(value.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def fit_role(meta, receipt, core, torch, guard, seed, role, train, val, tables, stats, anchor):
    train_table, val_table = tables
    model = core.PhysicalDecoder(seed, role).cuda()
    require(sum(p.numel() for p in model.parameters()) == m.PARAMETERS[role], "Parameter count")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8)
    output = receipt.folder / f"seed{seed}_{role}"
    output.mkdir()
    recovery = output / "recovery"
    recovery.mkdir()
    m.write_json(output / "CONFIG.json", dict(seed=seed, role=role, batch=m.BATCH, epochs=m.EPOCHS,
                 trainable_parameters=m.PARAMETERS[role], initialization_sha256=initialization_sha(model),
                 TRAIN_stats=f"../seed{seed}_TRAIN_STATS.json", optimizer="Adam",
                 lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0,
                 shuffle="global_without_replacement_int32", shuffle_seed_recipe=[seed, 20261106, "epoch"],
                 protocol_sha256=receipt.protocol_sha))
    score = lambda records, table, net, paired=False: core.evaluate(
        records, table, stats, net, role, guard, receipt.count, paired=paired)
    best_epoch, best, best_state = -1, anchor["val"], None
    epochs = [dict(epoch=-1, train=small_metric(anchor["train"]), val=small_metric(best))]
    planned = meta["phases"]["train"]
    for epoch in range(m.EPOCHS):
        receipt.scope = dict(seed=seed, role=role, epoch=epoch, phase="TRAIN")
        model.train()
        rng = core.np.random.default_rng(core.np.random.SeedSequence([seed, 20261106, epoch]))
        permutation = core.np.arange(planned["active_points"], dtype=core.np.int32)
        rng.shuffle(permutation)
        total, updates, sse = 0, 0, torch.zeros((), dtype=torch.float64, device="cuda")
        for start in range(0, len(permutation), m.BATCH):
            indices = train_table.active_indices(permutation[start:start + m.BATCH])
            x, v, b, y = core.batch_inputs(train_table, indices, role, stats)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, v, b)
            error = prediction - y
            receipt.count("current_stem_points", len(b))
            receipt.count("context_output_points", len(b))
            ssq = error.square().sum()
            (ssq / m.BATCH).backward()
            receipt.count("backward_points", len(b))
            optimizer.step()
            receipt.count("optimizer_updates")
            if updates == 0:
                with torch.no_grad():
                    first = dict(epoch=epoch, active_points=len(b), physical_MSE=float(error.square().mean()),
                                 residual_RMS=float((prediction - b).square().mean().sqrt()),
                                 gradients={name: float(p.grad.detach().norm())
                                            for name, p in model.named_parameters()})
                m.write_json(output / f"epoch{epoch:02}_FIRST_BATCH.json", first)
                receipt.checkpoint()
            total += len(b)
            updates += 1
            sse += ssq.detach()
            if updates % 64 == 0:
                require(bool(torch.isfinite(sse)), "Non-finite training loss")
                guard()
        require(total == planned["active_points"] and updates == planned["batches_per_pass"]
                and len(b) == planned["last_batch"],
                "TRAIN coverage or update count differs")
        require(bool(torch.isfinite(sse)), "Non-finite training loss")
        model.eval()
        receipt.scope["phase"] = "VAL"
        val_metric = score(val, val_table, model)["near"]
        epoch_row = dict(epoch=epoch, online_TRAIN_MSE=float(sse) / total, val=small_metric(val_metric))
        if epoch in m.FIXED_TRAIN:
            receipt.scope["phase"] = "fixed_TRAIN"
            epoch_row["train"] = small_metric(score(train, train_table, model)["near"])
        improved = val_metric["r"] > best["r"]
        if improved:
            best_epoch, best = epoch, val_metric
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
        epochs.append(epoch_row)
        checkpoint = recovery / f"epoch{epoch:02}.pt"
        partial = checkpoint.with_suffix(".pt.tmp")
        torch.save(dict(epoch=epoch, next_epoch=epoch + 1, seed=seed, role=role, best_epoch=best_epoch,
                        model=model.state_dict(), optimizer=optimizer.state_dict(),
                        numpy_bit_generator=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        torch_cuda_rng=torch.cuda.get_rng_state(), scores=epochs,
                        stage=receipt.snapshot()), partial)
        partial.replace(checkpoint)
        receipt.count("parameter_checkpoint_files")
        pointer = dict(epoch=epoch, path=str(checkpoint.relative_to(output)), best_epoch=best_epoch,
                       r=val_metric["r"], protocol_sha256=receipt.protocol_sha)
        atomic_json(output / "LAST.json", pointer)
        if improved:
            atomic_json(output / "BEST.json", pointer)
        guard()
        receipt.checkpoint()
    torch.save(model.state_dict(), output / "last.pt")
    receipt.count("parameter_checkpoint_files")
    near_older = None
    if best_state is not None:
        torch.save(best_state, output / "selected.pt")
        receipt.count("parameter_checkpoint_files")
        if role != "C_current":
            receipt.scope["phase"] = "selected_VAL_pair"
            model.load_state_dict(best_state)
            near_older = score(val, val_table, model, paired=True)
            require(abs(near_older["near"]["r"] - best["r"]) <= 1e-12,
                    "Selected near score did not reproduce")
    else:
        if role != "C_current":
            near_older = {"near": best, "older": best}
    result = dict(seed=seed, role=role, selected_epoch=best_epoch, selected=best,
                  val_pair=near_older, epochs=epochs)
    m.write_json(output / "RESULT.json", result)
    receipt.count("formal_fits_completed")
    guard()
    return {"epoch": best_epoch, "r": best["r"],
            "older_r": None if near_older is None else near_older["older"]["r"]}


def fit(meta, receipt, core, torch, guard):
    summary = {}
    for seed in m.SEEDS:
        receipt.scope = dict(seed=seed, phase="frozen_inputs")
        train = core.load_phase(meta, seed, "train", guard)
        val = core.load_phase(meta, seed, "val", guard)
        for phase, records in (("TRAIN", train), ("VAL", val)):
            receipt.count(phase + "_records_loaded", len(records))
            receipt.count(phase + "_cached_windows_loaded", sum(r.metadata["windows"] for r in records))
        anchor = {phase: core.evaluate(records, None, None, None, None, guard, receipt.count)["near"]
                  for phase, records in (("train", train), ("val", val))}
        bridge = anchor_bridge(anchor["val"], seed)
        autocorrelation = core.anchor_autocorrelation(val)
        m.write_json(receipt.folder / f"seed{seed}_ANCHOR.json",
                     dict(anchor=anchor, max_block_r_error=bridge, autocorrelation=autocorrelation))
        tables = (core.Tables(train, "cuda:0", guard), core.Tables(val, "cuda:0", guard))
        stats, count = core.fit_statistics(train, tables[0], guard)
        require(count == meta["phases"]["train"]["active_points"], "TRAIN statistics count")
        receipt.count("TRAIN_statistics_points", count)
        m.write_json(receipt.folder / f"seed{seed}_TRAIN_STATS.json",
                     {name: dict(zip(("mean", "std", "constant"), (v.cpu().tolist() for v in values)))
                      for name, values in stats.items()})
        receipt.count("training_statistics_files")
        results = {}
        for role in m.ROLES:
            results[role] = fit_role(meta, receipt, core, torch, guard, seed, role,
                                     train, val, tables, stats, anchor)
        old, current, prior = (results[k] for k in ("H_past_eeg", "C_current", "P_past_prediction"))
        baseline = anchor["val"]["r"]
        margins = dict(H_minus_C=old["r"] - current["r"], H_minus_P=old["r"] - prior["r"],
                       H_minus_anchor=old["r"] - baseline, H_near_minus_older=old["r"] - old["older_r"])
        summary[seed] = dict(anchor_r=baseline, roles=results, margins=margins,
                             all_H_margins_positive=all(value > 0 for value in margins.values()))
        del tables, stats, train, val, anchor
        torch.cuda.empty_cache()
    plan = meta["proposal"]
    for name, expected in (("optimizer_updates", plan["optimizer_updates"]),
                           ("backward_points", plan["train_target_presentations"]),
                           ("formal_fits_completed", 9)):
        require(receipt.counters[name] == expected, f"Budget differs for {name}")
    for key in ("current_stem_points", "context_output_points"):
        require(receipt.counters[key] <= plan[key + "_max"], f"Exceeded point budget for {key}")
    return dict(passed=True, seeds=summary,
                H_advancement_gate=all(row["all_H_margins_positive"] for row in summary.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("engineer", "fit"))
    stage = parser.parse_args().stage
    meta, protocol_sha = admit(stage)
    folder = m.ROOT / stage
    folder.mkdir()
    receipt = Receipt(folder, stage, protocol_sha)
    try:
        def timeout(*_):
            raise TimeoutError("D19 stage wall-clock cap")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(MAX_ENGINEERING_SECONDS if stage == "engineer" else MAX_SECONDS)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["OPENBLAS_NUM_THREADS"] = "8"
        os.environ["OMP_NUM_THREADS"] = "8"
        import pub_01_s03_ss07_physical_stream_core as core
        torch = core.torch
        require(torch.cuda.device_count() == 1, "Expected exactly one reviewed visible GPU")
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        receipt.versions.update(torch=str(torch.__version__), numpy=core.np.__version__,
                                torch_cuda=torch.version.cuda,
                                driver=subprocess.check_output(
                                    ["nvidia-smi", "-i", receipt.identity["cuda_uuid"],
                                     "--query-gpu=driver_version", "--format=csv,noheader"],
                                    text=True).strip())
        guard = lambda: receipt.guard(torch)
        receipt.checkpoint()
        worker = engineer if stage == "engineer" else fit
        result = worker(meta, receipt, core, torch, guard)
        guard()
        m.write_json(folder / "COMPLETED.json", dict(receipt.snapshot(), **result))
    except BaseException as error:
        m.write_json(folder / "FAILURE.json", dict(receipt.snapshot(), error=repr(error)))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
