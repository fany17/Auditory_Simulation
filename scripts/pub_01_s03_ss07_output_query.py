"""D22 nonlinear output-only control on the pinned D20 endpoint."""

import argparse
from dataclasses import replace
import inspect
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
from types import FunctionType, SimpleNamespace


PARENT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
D19 = PARENT / "s03_physical_stream_20261106"
D20 = PARENT / "s03_recent_prediction_context_20261106"
ROOT = PARENT / "s03_output_query_20261106"
ROLE = "P_past_prediction"
SEEDS = (1024, 2026, 31415)
PARAMETERS = 9441
GIB = 1024**3
D20_PINS = {
    D20 / "PROTOCOL.json": "535d06ed1e12ea1048293ca1be63bb2ad8dfd7c0b985dd9ca9503d6f64735540",
    D20 / "fit/COMPLETED.json": "40637c0a9f38bf47942b7a4e5268435a06f9f86f22421b41ba09faad1ebf7320",
    D20 / "input/pub_01_s03_ss07_recent_prediction_context.py":
        "31b9dc7756b95e555d6775b26fe8b8d59fc5d0e96ac84c2a947f099ffc2a04c3",
}


def prepare(d20, uuid):
    d20.verify_pins(D20_PINS)
    parent = d20.read(D20 / "PROTOCOL.json")
    require = d20.require
    require(parent["near_lag"] == 1 and parent["far_lag"] == 288
            and parent["first_active"] == 1344, "Unexpected D20 geometry")
    pins = parent["d19_pins"] | {str(k): v for k, v in D20_PINS.items()}
    for seed in SEEDS:
        paths = [D20 / "fit" / f"seed{seed}_TRAIN_STATS.json"]
        folder = D20 / "fit" / f"seed{seed}_{ROLE}"
        paths.extend(folder / name for name in ("CONFIG.json", "RESULT.json", "recovery/epoch00.pt"))
        for path in paths:
            pins[str(path)] = d20.sha(path)
    d20.verify_pins(pins)
    result = dict(candidate="D22_OUTPUT_QUERY_V1", source_sha256=d20.sha(Path(__file__)),
                  frozen_pins=pins, seeds=SEEDS, stored_role_slot=ROLE, role="B_output_only",
                  parameters=PARAMETERS, current_dimension=1, history_points=384,
                  near_lag=1, paired_far_lag=288, first_active=1344, cuda_uuid=uuid,
                  normalization="Both inputs use actual frozen same-seed D20 P TRAIN scalar statistics",
                  initialization="D20 common tensors unchanged; current.weight is W[:,0:1] times sqrt(8)",
                  engineering_budget=dict(TRAIN_records_loaded=6, TRAIN_cached_windows_loaded=3447,
                                          VAL_records_loaded=0, current_stem_points=36864 + 96,
                                          context_output_points=49152, backward_points=24576,
                                          optimizer_updates=0),
                  formal_budget=parent["formal_budget"], resources=dict(GPU_count=1, fit_seconds=21600,
                      engineering_seconds=600, RSS_GiB=12, torch_allocated_GiB=8,
                      writer_MiB=256, SATA_min_free_GiB=20),
                  fit_requires="Independent EXECUTION_REVIEW bound to actual engineering and protocol")
    d20.write(ROOT / "PROTOCOL.json", result)
    print(d20.sha(ROOT / "PROTOCOL.json"), flush=True)


def admit(stage, d20):
    require = d20.require
    path = ROOT / "PROTOCOL.json"
    p, identity = d20.read(path), d20.sha(path)
    require(p["candidate"] == "D22_OUTPUT_QUERY_V1" and p["source_sha256"] == d20.sha(Path(__file__)),
            "D22 source drift")
    d20.verify_pins(p["frozen_pins"])
    meta = d20.read(D19 / "PREPARATION_METADATA.json")
    d20.verify_pins(meta["provenance"])
    for path, stamp in meta["inputs"].items():
        state = Path(path).stat()
        require((state.st_size, state.st_mtime_ns) == (stamp["bytes"], stamp["mtime_ns"]), "D20 input drift")
    if stage == "fit":
        gate = d20.read(ROOT / "EXECUTION_REVIEW.json")
        receipt = ROOT / "engineer/COMPLETED.json"
        require(gate["stage"] == "fit" and gate["decision"] == "PASS"
                and gate["protocol_sha256"] == identity and gate["engineering_sha256"] == d20.sha(receipt)
                and d20.read(receipt)["passed"], "Missing bound independent fit review")
    uuid = p["cuda_uuid"]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-"), "CUDA UUID")
    used = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    require(used.isdecimal() and int(used) <= 256, "Selected GPU is busy")
    require(shutil.disk_usage(ROOT).free >= 20 * GIB, "SATA space cap")
    return p, identity, meta


def adapters(d20, core, legacy):
    torch, require = core.torch, d20.require
    table, _ = d20.types(core, legacy)

    class OutputTable(table):
        def __init__(self, records, device, guard):
            super().__init__(records, device, guard)
            self.current = self.anchor.float()[:, None]

    class OutputDecoder(core.PhysicalDecoder):
        def __init__(self, seed, role):
            super().__init__(seed, role)
            current = self.current
            # The original constructor seeds CPU/CUDA. Preserve its resulting RNG state.
            with torch.random.fork_rng(devices=[]):
                self.current = torch.nn.Linear(1, legacy.m.WIDTH)
            with torch.no_grad():
                self.current.weight.copy_(current.weight[:, :1] * (8 ** .5))
                self.current.bias.copy_(current.bias)

    class Receipt(legacy.Receipt):
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
            require(self.runtime["writer_bytes"] <= 256 * 1024**2, "Writer cap")

    # Keep the byte-pinned optimizer, batching, evaluation and checkpoint routine.
    # Only its local parameter-count assertion and supplied constructor change.
    meta = SimpleNamespace(**(vars(legacy.m) | {"PARAMETERS": legacy.m.PARAMETERS | {ROLE: PARAMETERS}}))
    original = legacy.fit_role
    fit_role = FunctionType(original.__code__, original.__globals__ | {"m": meta}, original.__name__)
    backend = SimpleNamespace(**(vars(core) | {"PhysicalDecoder": OutputDecoder}))
    require(fit_role.__code__ is original.__code__
            and inspect.signature(fit_role) == inspect.signature(original)
            and legacy.m.PARAMETERS[ROLE] == 9665 and meta.PARAMETERS[ROLE] == PARAMETERS
            and backend.evaluate is core.evaluate and backend.batch_inputs is core.batch_inputs,
            "Pinned common training and evaluation path drift")
    return OutputTable, OutputDecoder, Receipt, fit_role, backend


def input_stats(seed, table, core, d20):
    stats = d20.read(D20 / "fit" / f"seed{seed}_TRAIN_STATS.json")["P_past_prediction"]
    require = d20.require
    require(not stats["constant"] and stats["std"] > 0, "D20 P stats")
    torch = core.torch
    history = tuple(torch.tensor(stats[k], dtype=dtype, device=table.device)
                    for k, dtype in (("mean", torch.float32), ("std", torch.float32), ("constant", torch.bool)))
    return {"current": tuple(v.reshape(1) for v in history), ROLE: history}


def engineer(meta, p, receipt, core, legacy, d20, Table, Decoder, guard):
    np, torch, m = core.np, core.torch, legacy.m
    require = d20.require
    checks = []
    for seed in SEEDS:
        receipt.scope = dict(seed=seed, phase="first_two_TRAIN_records")
        train = core.load_phase(meta, seed, "train", guard, train_records_limit=2)
        receipt.count("TRAIN_records_loaded", len(train))
        receipt.count("TRAIN_cached_windows_loaded", sum(r.metadata["windows"] for r in train))
        table = Table(train, "cuda:0", guard)
        stats = input_stats(seed, table, core, d20)
        count = sum(record.stop - m.FIRST_ACTIVE for record in train)
        dense = np.column_stack((np.arange(2048), np.arange(count - 2048, count))).astype(np.int32)
        idx = table.active_indices(dense.ravel())
        spans = [(0, m.FIRST_ACTIVE, m.FIRST_ACTIVE + 2048), (1, train[1].stop - 2048, train[1].stop)]
        require(torch.equal(idx, table.indices(spans).reshape(2, 2048).T.flatten()), "Cross-record IDs")
        x, v, b, y = core.batch_inputs(table, idx, ROLE, stats)
        mu, sigma = [np.float32(t.item()) for t in stats[ROLE][:2]]
        x_ref = ((b.cpu().numpy().astype(np.float32) - mu) / sigma)[:, None]
        current_error = float(np.max(np.abs(x.cpu().numpy() - x_ref)))
        points = [(i, t, t + 1) for i, record in enumerate(train)
                  for t in (m.FIRST_ACTIVE, m.FIRST_ACTIVE + 1, record.stop - 2, record.stop - 1)]
        probe = table.indices(points)
        history_error = 0.
        for lag, key in ((1, core.NEAR), (288, core.OLDER)):
            bins = np.stack([train[i].anchor[t - lag - 383:t - lag + 1].reshape(96, 4)
                             .mean(axis=1, dtype=np.float64) for i, t, _ in points]).astype(np.float32)
            actual = table.context(probe, ROLE, stats, key).cpu().numpy()
            history_error = max(history_error, float(np.max(np.abs(actual - (bins - mu) / sigma))))
        require(current_error == 0 and history_error <= 2e-6, "Explicit scalar or history normalization")
        parent = core.PhysicalDecoder(seed, ROLE)
        cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        model = Decoder(seed, ROLE)
        require(torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state()),
                "Constructor changed the D20 RNG stream")
        require(sum(t.numel() for t in model.parameters()) == PARAMETERS, "D22 parameter count")
        for name, tensor in model.state_dict().items():
            other = parent.state_dict()[name]
            expected = other[:, :1] * (8 ** .5) if name == "current.weight" else other
            require(torch.equal(tensor, expected), f"Initialization mismatch {name}")
        parent_folder = D20 / "fit" / f"seed{seed}_{ROLE}"
        config = d20.read(parent_folder / "CONFIG.json")
        require(config["initialization_sha256"] == legacy.initialization_sha(parent), "D20 initial checkpoint ID")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261106, 0]))
        order = np.arange(meta["phases"]["train"]["active_points"], dtype=np.int32)
        rng.shuffle(order)
        prior = torch.load(parent_folder / "recovery/epoch00.pt", map_location="cpu", weights_only=True)
        require(rng.bit_generator.state == prior["numpy_bit_generator"]
                and config["shuffle_seed_recipe"] == [seed, 20261106, "epoch"], "D20 shuffle state")
        del order, prior
        model = model.cuda()
        gradients = []
        for nonzero in (False, True):
            model.zero_grad(set_to_none=True)
            if nonzero:
                with torch.no_grad():
                    model.output.weight.copy_(torch.linspace(-.025, .025, m.WIDTH, device="cuda"))
            code = model.current_code(x)
            out = b + model.decode(code, v).double()
            receipt.count("current_stem_points", len(b))
            receipt.count("context_output_points", len(b))
            if not nonzero:
                require(torch.equal(out, b), "Zero residual changed frozen anchor")
            ((out - y).square().sum() / m.BATCH).backward()
            receipt.count("backward_points", len(b))
            norms = {name: float(layer.weight.grad.norm()) for name, layer in model.named_children()}
            require(all(np.isfinite(z) for z in norms.values()) and norms["output"] > 0
                    and all((v > 0) == nonzero for k, v in norms.items() if k != "output"),
                    "Unexpected D22 gradient")
            gradients.append(norms)
        changed = [replace(record, latent=record.latent + np.float32(1.5)) for record in train]
        changed_table = Table(changed, "cuda:0", guard)
        x2, v2, b2, _ = core.batch_inputs(changed_table, idx, ROLE, stats)
        with torch.no_grad():
            again = model(x2, v2, b2)
            receipt.count("current_stem_points", len(b))
            receipt.count("context_output_points", len(b))
            require(torch.equal(x2, x) and torch.equal(v2, v) and torch.equal(again, out),
                    "Perturbed g reached B")
            shift = core.normalized((b[:16].float() + .25)[:, None], stats["current"])
            current_change = float((model.current_code(shift) - code[:16]).abs().max())
            receipt.count("current_stem_points", 16)
            # Evaluate the current-only canary from an independently assembled FP32 reference.
            check_x = torch.from_numpy(x_ref[:16]).to("cuda")
            reference_error = float((model.current_code(check_x) - code[:16]).abs().max())
            receipt.count("current_stem_points", 16)
            require(current_change > 0 and reference_error <= 2e-6, "B current code bridge")
            far = table.context(idx, ROLE, stats, core.OLDER)
            pair = float((out - (b + model.decode(code, far).double())).abs().max())
            receipt.count("context_output_points", len(b))
            require(np.isfinite(pair) and pair > 0, "Same-code history intervention")
        checks.append(dict(seed=seed, shared_initial_tensors_exact=True, CPU_CUDA_RNG_exact=True,
                           D20_epoch0_shuffle_state_exact=True, perturbed_latent_output_max_abs=0.,
                           current_explicit_max_abs=current_error, history_explicit_max_abs=history_error,
                           current_b_perturbation_code_max_abs=current_change, current_code_ref_max_abs=reference_error,
                           same_code_near_far_max_abs=pair, gradients=gradients))
        guard()
    for name, value in p["engineering_budget"].items():
        require(receipt.counters[name] == value, f"Engineering count {name}")
    return dict(passed=True, checks=checks, optimizer_updates=0, formal_fits=0,
                common_fit_code_and_signature_exact=True, frozen_module_unchanged=True)


def fit(meta, p, receipt, core, legacy, d20, Table, fit_role, backend, guard):
    results = {}
    torch = core.torch
    for seed in SEEDS:
        receipt.scope = dict(seed=seed, phase="frozen_D20_inputs")
        train = core.load_phase(meta, seed, "train", guard)
        val = core.load_phase(meta, seed, "val", guard)
        for phase, records in (("TRAIN", train), ("VAL", val)):
            receipt.count(phase + "_records_loaded", len(records))
            receipt.count(phase + "_cached_windows_loaded", sum(r.metadata["windows"] for r in records))
        anchor = {name: core.evaluate(records, None, None, None, None, guard, receipt.count)["near"]
                  for name, records in (("train", train), ("val", val))}
        error = legacy.anchor_bridge(anchor["val"], seed)
        d20.write(receipt.folder / f"seed{seed}_ANCHOR.json", dict(anchor=anchor, max_block_r_error=error))
        tables = Table(train, "cuda:0", guard), Table(val, "cuda:0", guard)
        stats = input_stats(seed, tables[0], core, d20)
        d20.write(receipt.folder / f"seed{seed}_TRAIN_STATS.json",
                  {key: dict(zip(("mean", "std", "constant"), (v.cpu().tolist() for v in s)))
                   for key, s in stats.items()})
        receipt.count("training_statistics_files")
        fit_role(meta, receipt, backend, torch, guard, seed, ROLE, train, val, tables, stats, anchor)
        f = d20.read(receipt.folder / f"seed{seed}_{ROLE}/RESULT.json")
        baseline = d20.read(D20 / "fit" / f"seed{seed}_{ROLE}/RESULT.json")
        best = max(f["epochs"][1:], key=lambda r: r["val"]["r"])
        results[seed] = dict(selected_epoch=f["selected_epoch"], selected_r=f["selected"]["r"],
                             D20_r=baseline["selected"]["r"], anchor_r=anchor["val"]["r"],
                             best_trained_epoch=best["epoch"], best_trained_r=best["val"]["r"],
                             last_r=f["epochs"][-1]["val"]["r"],
                             paired_far_r=None if f["selected_epoch"] == -1 else f["val_pair"]["older"]["r"])
        del tables, stats, train, val, anchor
        torch.cuda.empty_cache()
    b = p["formal_budget"]
    for key, expected in (("formal_fits_completed", 3), ("optimizer_updates", b["optimizer_updates"]),
                          ("backward_points", b["backward_points"])):
        d20.require(receipt.counters[key] == expected, f"Formal count {key}")
    paired = sum(r["selected_epoch"] != -1 for r in results.values()) * meta["phases"]["val"]["active_points"]
    standard = b["backward_points"] + b["fixed_TRAIN_forwards"] + b["epoch_VAL_forwards"]
    d20.require(receipt.counters["current_stem_points"] == standard + paired
                and receipt.counters["context_output_points"] == standard + 2 * paired, "Paired point count")
    return dict(passed=True, seeds=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "engineer", "fit"))
    parser.add_argument("--cuda-uuid")
    args = parser.parse_args()
    if ROOT.resolve() != ROOT or Path(__file__).resolve().parent != ROOT / "input":
        raise RuntimeError("Use the independent 2203 SATA root")
    sys.path.insert(0, str(D20 / "input"))
    import pub_01_s03_ss07_recent_prediction_context as d20
    if args.stage == "prepare":
        d20.require(args.cuda_uuid and args.cuda_uuid.startswith("GPU-"), "Specify CUDA UUID")
        prepare(d20, args.cuda_uuid)
        return
    p, identity, meta = admit(args.stage, d20)
    folder = ROOT / args.stage
    folder.mkdir()
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8")
    sys.path.insert(0, str(D19 / "input"))
    import pub_01_s03_ss07_physical_stream_core as core
    import pub_01_s03_ss07_physical_stream_run as legacy
    Table, Decoder, Receipt, fit_role, backend = adapters(d20, core, legacy)
    receipt = Receipt(folder, args.stage, identity)
    torch = core.torch
    try:
        def timeout(*_):
            raise TimeoutError("D22 wall-clock cap")

        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(600 if args.stage == "engineer" else 21600)
        d20.require(torch.cuda.device_count() == 1, "Expected one CUDA device")
        torch.cuda.set_device(0)
        torch.set_num_threads(8)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        receipt.versions.update(torch=str(torch.__version__), numpy=core.np.__version__, torch_cuda=torch.version.cuda)
        guard = lambda: receipt.guard(torch)
        receipt.checkpoint()
        result = (engineer(meta, p, receipt, core, legacy, d20, Table, Decoder, guard)
                  if args.stage == "engineer"
                  else fit(meta, p, receipt, core, legacy, d20, Table, fit_role, backend, guard))
        guard()
        d20.write(folder / "COMPLETED.json", dict(receipt.snapshot(), **result))
    except BaseException as error:
        d20.write(folder / "FAILURE.json", dict(receipt.snapshot(), error=repr(error)))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
