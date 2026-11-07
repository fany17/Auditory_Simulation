"""D20: reuse the pinned D19 P runner, changing only the context lag."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys


PARENT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
D19 = PARENT / "s03_physical_stream_20261106"
ROOT = PARENT / "s03_recent_prediction_context_20261106"
ROLE = "P_past_prediction"
SEEDS = (1024, 2026, 31415)
RECENT, FAR = 1, 288
SOURCE = Path(__file__).name
GIB = 1024**3
PINS = {
    "PROTOCOL.json": "133b60eeb6e937def9f25bdb8303ab80d9a2cf00d0f86ea3f6768c0f56a7b078",
    "PREPARATION_METADATA.json": "1068d1eede47893e074c7a67b7a57c1a967364cb974ffdf088a1d623121e24bd",
    "fit/COMPLETED.json": "5183983f18c96a771b66bece03b04d3226ceaa55f0d583df198564fd2c7964b9",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def far_folder(seed):
    return D19 / "fit" / f"seed{seed}_{ROLE}"


def verify_pins(pins):
    for path, expected in pins.items():
        require(sha(Path(path)) == expected, f"Frozen source or receipt drift: {path}")


def prepare(uuid):
    verify_pins({str(D19 / key): value for key, value in PINS.items()})
    base = read(D19 / "PROTOCOL.json")
    meta = read(D19 / "PREPARATION_METADATA.json")
    fit = read(D19 / "fit/COMPLETED.json")
    require(base["source_sha256"] == meta["prepared_source_sha256"]
            and fit["passed"] and fit["counters"]["formal_fits_completed"] == 9,
            "D19 base is incomplete")
    inputs = {str(D19 / key): value for key, value in PINS.items()}
    for name, value in base["source_sha256"].items():
        inputs[str(D19 / "input" / name)] = value
    for seed in SEEDS:
        for path in (far_folder(seed) / "CONFIG.json", far_folder(seed) / "RESULT.json",
                     far_folder(seed) / "recovery/epoch00.pt", D19 / "fit" / f"seed{seed}_TRAIN_STATS.json"):
            inputs[str(path)] = sha(path)
    verify_pins(inputs)
    train, val = (meta["phases"][key] for key in ("train", "val"))
    backward = 3 * 20 * train["active_points"]
    fixed = 3 * 4 * train["active_points"]
    validation = 3 * 20 * val["active_points"]
    paired = 3 * val["active_points"]
    result = dict(candidate="D20_RECENT_P_ONLY_V1", source_sha256=sha(Path(__file__)),
                  d19_pins=inputs, seed=SEEDS, role=ROLE, near_lag=RECENT, far_lag=FAR,
                  history_points=384, first_active=meta["geometry"]["first_active"], cuda_uuid=uuid,
                  geometry=dict(TRAIN_records=len(train["records"]), VAL_records=len(val["records"]),
                                TRAIN_active=train["active_points"], VAL_active=val["active_points"],
                                batches_per_pass=train["batches_per_pass"], last_batch=train["last_batch"]),
                  engineering_budget=dict(current_stem_points=3 * 2 * 4096,
                                          context_output_points=3 * 3 * 4096,
                                          backward_points=3 * 2 * 4096, optimizer_updates=0,
                                          TRAIN_records_loaded=6, TRAIN_cached_windows_loaded=3447,
                                          VAL_records_loaded=0),
                  formal_budget=dict(fits=3, optimizer_updates=3 * 20 * train["batches_per_pass"],
                                     backward_points=backward, fixed_TRAIN_forwards=fixed,
                                     epoch_VAL_forwards=validation, paired_current_points_max=paired,
                                     current_stem_points_max=backward + fixed + validation + paired,
                                     context_output_points_max=backward + fixed + validation + 2 * paired),
                  status="ENGINEERING_ONLY_UNTIL_SEPARATE_EXECUTION_REVIEW",
                  new_model_weight_or_data_downloads=0, old_external_data_read=0)
    require(result["formal_budget"]["optimizer_updates"] == 86820 and backward == 1422210600,
            "Unexpected D20 budget")
    write(ROOT / "PROTOCOL.json", result)
    print(sha(ROOT / "PROTOCOL.json"), flush=True)


def admit(stage):
    require(Path(__file__).resolve().parent == ROOT / "input" and ROOT.resolve() == ROOT,
            "D20 must run at its own SATA root")
    path = ROOT / "PROTOCOL.json"
    protocol = read(path)
    protocol_sha = sha(path)
    require(protocol["source_sha256"] == sha(Path(__file__)), "Source changed after freeze")
    require(protocol["candidate"] == "D20_RECENT_P_ONLY_V1"
            and protocol["near_lag"] == RECENT and protocol["far_lag"] == FAR,
            "Unexpected D20 protocol")
    verify_pins(protocol["d19_pins"])
    meta = read(D19 / "PREPARATION_METADATA.json")
    verify_pins(meta["provenance"])
    for path, stamp in meta["inputs"].items():
        s = Path(path).stat()
        require((s.st_size, s.st_mtime_ns) == (stamp["bytes"], stamp["mtime_ns"]),
                f"D19 input stamp changed: {path}")
    if stage == "fit":
        gate = read(ROOT / "EXECUTION_REVIEW.json")
        receipt = ROOT / "engineer/COMPLETED.json"
        require(gate["stage"] == "fit" and gate["decision"] == "PASS"
                and gate["protocol_sha256"] == protocol_sha and gate["engineering_sha256"] == sha(receipt)
                and read(receipt)["passed"], "Missing actual bound formal admission")
    uuid = protocol["cuda_uuid"]
    require(uuid.startswith("GPU-") and os.environ.get("CUDA_VISIBLE_DEVICES") == uuid, "CUDA UUID")
    used = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    require(used.isdecimal() and int(used) <= 256, "Selected GPU is busy")
    require(shutil.disk_usage(ROOT).free >= 20 * GIB, "SATA free space floor")
    return meta, protocol, protocol_sha


def types(core, legacy):
    class RecentTables(core.Tables):
        def history(self, indices, role, lag):
            require(role == ROLE and lag in (core.NEAR, core.OLDER), "Unexpected D20 history route")
            # D19 names these two calls near and older. This is the only input change.
            delay = RECENT if lag == core.NEAR else FAR
            return self.prior[indices[:, None] - delay + self.bin_ends]

    class D20Receipt(legacy.Receipt):
        def guard(self, torch):
            self.runtime = dict(
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                writer_free_bytes=shutil.disk_usage(ROOT).free,
                writer_bytes=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()),
            )
            require(self.runtime["peak_RSS_bytes"] <= 12 * GIB, "RSS cap")
            require(self.runtime["cuda_peak_allocated_bytes"] <= 8 * GIB, "CUDA cap")
            require(self.runtime["writer_free_bytes"] >= 20 * GIB, "SATA space cap")
            require(self.runtime["writer_bytes"] <= 256 * 1024**2, "SATA output cap")

    return RecentTables, D20Receipt


def train_stats(seed, train, table, core, guard):
    torch = core.torch
    # Current stream is unchanged. Use the actual frozen D19 TRAIN statistics.
    current = read(D19 / "fit" / f"seed{seed}_TRAIN_STATS.json")["current"]
    current = tuple(torch.tensor(current[k], dtype=dtype, device=table.device)
                    for k, dtype in (("mean", torch.float32), ("std", torch.float32), ("constant", torch.bool)))
    sums = torch.zeros((), dtype=torch.float64, device=table.device)
    squares = torch.zeros_like(sums)
    count = 0
    for i, start, stop in core.chunks(train):
        idx = table.indices([(i, start, stop)])
        value = table.history(idx, ROLE, core.NEAR).double()
        sums += value.sum()
        squares += value.square().sum()
        count += stop - start
        if stop == train[i].stop:
            guard()
    return {"current": current, ROLE: core.moments(sums, squares, count * core.BINS)}, count


def engineer(meta, protocol, receipt, core, legacy, Table, guard):
    np, torch, m = core.np, core.torch, legacy.m
    checks = []
    for seed in SEEDS:
        receipt.scope = dict(seed=seed, phase="two_TRAIN_records")
        train = core.load_phase(meta, seed, "train", guard, train_records_limit=2)
        receipt.count("TRAIN_records_loaded", len(train))
        receipt.count("TRAIN_cached_windows_loaded", sum(r.metadata["windows"] for r in train))
        table = Table(train, "cuda:0", guard)
        stats, count = train_stats(seed, train, table, core, guard)
        receipt.count("TRAIN_statistics_points", count)
        first = train[0].stop - m.FIRST_ACTIVE
        dense = np.column_stack((np.arange(2048), np.arange(count - 2048, count))).astype(np.int32)
        index = table.active_indices(dense.ravel())
        want = table.indices([(0, m.FIRST_ACTIVE, m.FIRST_ACTIVE + 2048),
                              (1, train[1].stop - 2048, train[1].stop)]).reshape(2, 2048).T.flatten()
        require(torch.equal(index, want), "Cross-record dense gather")
        probe = [(i, point, point + 1) for i, record in enumerate(train)
                 for point in (m.FIRST_ACTIVE, m.FIRST_ACTIVE + 1, m.FIRST_ACTIVE + 95,
                               record.stop - 2, record.stop - 1)]
        probe_index = table.indices(probe)
        maximum_error = 0.
        for delay, key in ((RECENT, core.NEAR), (FAR, core.OLDER)):
            reference = []
            for i, point, _ in probe:
                record = train[i]
                width = record.metadata["windows"]
                context_stop = point - delay + 1
                context_start = context_stop - m.HISTORY
                require(1 <= context_start < context_stop <= point, "Recent history included current point")
                require(m.raw_support(width, context_stop - 1)[1] <= m.raw_support(width, point)[1],
                        "Past support expands current future permission")
                reference.append(record.anchor[context_start:context_stop].reshape(m.BINS, m.BIN_POINTS)
                                 .mean(axis=1, dtype=np.float64))
            observed = table.history(probe_index, ROLE, key).cpu().numpy()
            maximum_error = max(maximum_error, float(np.max(np.abs(observed - np.asarray(reference, np.float32)))))
        require(maximum_error <= 2e-6, "Explicit history bins differ")
        boundary = table.active_indices(np.array([first - 1, first], dtype=np.int32))
        require(torch.equal(boundary, table.indices([(0, train[0].stop - 1, train[0].stop),
                                                     (1, m.FIRST_ACTIVE, m.FIRST_ACTIVE + 1)])),
                "Global ID crosses the wrong record")
        model = core.PhysicalDecoder(seed, ROLE).cuda()
        model_sha = legacy.initialization_sha(model)
        config = read(far_folder(seed) / "CONFIG.json")
        require(model_sha == config["initialization_sha256"], "D19 P initial parameters differ")
        require(config["shuffle"] == "global_without_replacement_int32"
                and config["shuffle_seed_recipe"] == [seed, 20261106, "epoch"], "Shuffle recipe drift")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261106, 0]))
        permutation = np.arange(meta["phases"]["train"]["active_points"], dtype=np.int32)
        rng.shuffle(permutation)
        old_checkpoint = torch.load(far_folder(seed) / "recovery/epoch00.pt",
                                    map_location="cpu", weights_only=True)
        require(rng.bit_generator.state == old_checkpoint["numpy_bit_generator"], "D19 P epoch0 RNG differs")
        del permutation, old_checkpoint
        x, v, b, y = core.batch_inputs(table, index, ROLE, stats)
        grads = []
        for nonzero in (False, True):
            model.zero_grad(set_to_none=True)
            if nonzero:
                with torch.no_grad():
                    model.output.weight.copy_(torch.linspace(-.025, .025, m.WIDTH, device="cuda"))
            code = model.current_code(x)
            prediction = b + model.decode(code, v).double()
            receipt.count("current_stem_points", len(b))
            receipt.count("context_output_points", len(b))
            if not nonzero:
                require(torch.equal(prediction, b), "Fresh residual changed the anchor")
            ((prediction - y).square().sum() / m.BATCH).backward()
            receipt.count("backward_points", len(b))
            norms = {name: float(layer.weight.grad.norm()) for name, layer in model.named_children()}
            require(all(np.isfinite(norm) for norm in norms.values()) and norms["output"] > 0
                    and all((v > 0) == nonzero for k, v in norms.items() if k != "output"),
                    "Unexpected decoder gradient")
            grads.append(norms)
        with torch.no_grad():
            far = table.context(index, ROLE, stats, core.OLDER)
            prior = b + model.decode(code.detach(), far).double()
            receipt.count("context_output_points", len(b))
            pair = float((prediction - prior).abs().max())
            require(np.isfinite(pair) and pair > 0, "Recent/far replacement did not reach output")
        checks.append(dict(seed=seed, initialization_sha256=model_sha, D19_P_epoch0_rng_equal=True,
                           explicit_history_max_abs=maximum_error, same_query_pair_max_abs=pair,
                           gradients=grads))
        guard()
    for key, value in protocol["engineering_budget"].items():
        require(receipt.counters[key] == value, f"Engineering count differs: {key}")
    return dict(passed=True, checks=checks, optimizer_updates=0, formal_fits=0)


def fit(meta, protocol, receipt, core, legacy, Table, guard):
    torch = core.torch
    results = {}
    for seed in SEEDS:
        receipt.scope = dict(seed=seed, phase="frozen_D19_inputs")
        train = core.load_phase(meta, seed, "train", guard)
        val = core.load_phase(meta, seed, "val", guard)
        for name, records in (("TRAIN", train), ("VAL", val)):
            receipt.count(name + "_records_loaded", len(records))
            receipt.count(name + "_cached_windows_loaded", sum(r.metadata["windows"] for r in records))
        anchor = {phase: core.evaluate(records, None, None, None, None, guard, receipt.count)["near"]
                  for phase, records in (("train", train), ("val", val))}
        bridge = legacy.anchor_bridge(anchor["val"], seed)
        write(receipt.folder / f"seed{seed}_ANCHOR.json", dict(anchor=anchor, max_block_r_error=bridge))
        tables = Table(train, "cuda:0", guard), Table(val, "cuda:0", guard)
        stats, count = train_stats(seed, train, tables[0], core, guard)
        require(count == meta["phases"]["train"]["active_points"], "TRAIN stats coverage")
        receipt.count("TRAIN_statistics_points", count)
        write(receipt.folder / f"seed{seed}_TRAIN_STATS.json",
              {name: dict(zip(("mean", "std", "constant"), (v.cpu().tolist() for v in values)))
               for name, values in stats.items()})
        receipt.count("training_statistics_files")
        legacy.fit_role(meta, receipt, core, torch, guard, seed, ROLE, train, val, tables, stats, anchor)
        near = read(receipt.folder / f"seed{seed}_{ROLE}/RESULT.json")
        far = read(far_folder(seed) / "RESULT.json")
        best = lambda run: max(run["epochs"][1:], key=lambda r: r["val"]["r"])
        selected = near["selected"]["r"]
        results[seed] = dict(selected_epoch=near["selected_epoch"], selected_r=selected,
                             anchor_r=anchor["val"]["r"], far_selected_r=far["selected"]["r"],
                             recent_best_trained_r=best(near)["val"]["r"],
                             far_best_trained_r=best(far)["val"]["r"],
                             recent_last_r=near["epochs"][-1]["val"]["r"],
                             far_last_r=far["epochs"][-1]["val"]["r"],
                             paired_far_r=None if near["selected_epoch"] == -1 else near["val_pair"]["older"]["r"])
        del tables, stats, train, val, anchor
        torch.cuda.empty_cache()
    limit = protocol["formal_budget"]
    require(receipt.counters["formal_fits_completed"] == 3, "Incomplete D20 fit count")
    for field in ("optimizer_updates", "backward_points"):
        require(receipt.counters[field] == limit[field], f"Unexpected {field}")
    paired = sum(r["selected_epoch"] != -1 for r in results.values()) * meta["phases"]["val"]["active_points"]
    standard = limit["backward_points"] + limit["fixed_TRAIN_forwards"] + limit["epoch_VAL_forwards"]
    require(receipt.counters["current_stem_points"] == standard + paired
            and receipt.counters["context_output_points"] == standard + 2 * paired,
            "Actual counts do not match selected pairing")
    return dict(passed=True, seeds=results, paired_recent_lag=RECENT, paired_far_lag=FAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "engineer", "fit"))
    parser.add_argument("--cuda-uuid")
    args = parser.parse_args()
    if args.stage == "prepare":
        require(args.cuda_uuid and args.cuda_uuid.startswith("GPU-"), "Specify reviewed GPU UUID")
        prepare(args.cuda_uuid)
        return
    meta, protocol, protocol_sha = admit(args.stage)
    folder = ROOT / args.stage
    folder.mkdir()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8")
    sys.path.insert(0, str(D19 / "input"))
    import pub_01_s03_ss07_physical_stream_core as core
    import pub_01_s03_ss07_physical_stream_run as legacy
    Table, Receipt = types(core, legacy)
    receipt = Receipt(folder, args.stage, protocol_sha)
    torch = core.torch
    try:
        def timeout(*_):
            raise TimeoutError("D20 stage wall-clock cap")
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(600 if args.stage == "engineer" else 21600)
        require(torch.cuda.device_count() == 1, "Expected exactly one CUDA device")
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        receipt.versions.update(torch=str(torch.__version__), numpy=core.np.__version__, torch_cuda=torch.version.cuda)
        guard = lambda: receipt.guard(torch)
        receipt.checkpoint()
        worker = engineer if args.stage == "engineer" else fit
        result = worker(meta, protocol, receipt, core, legacy, Table, guard)
        guard()
        write(folder / "COMPLETED.json", dict(receipt.snapshot(), **result))
    except BaseException as error:
        write(folder / "FAILURE.json", dict(receipt.snapshot(), error=repr(error)))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
