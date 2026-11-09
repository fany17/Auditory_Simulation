"""D23 memoryless nonlinear scalar control, using the pinned D22 runner."""

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
D22 = PARENT / "s03_output_query_20261106"
ROOT = PARENT / "s03_memoryless_output_20261106"
SEEDS = (1024, 2026, 31415)
ROLE = "P_past_prediction"
PARAMETERS = 3361
GIB = 1024**3
D22_PINS = {
    D22 / "PROTOCOL.json": "40de86545cdcd69264a308a68cc02c653b211f9688511b2cea9df2976dbcc74e",
    D22 / "fit/COMPLETED.json": "95e36f45fe3c4fc1334c6524e69c64511a6d13508b564331c4b654005f2f6480",
    D22 / "input/pub_01_s03_ss07_output_query.py":
        "9425b886fed962ebfc349cf60c5dee3b56aa374ab13f7e7c231829f993f7afcb",
}


def prepare(d20, uuid):
    d20.verify_pins(D22_PINS)
    parent = d20.read(D22 / "PROTOCOL.json")
    d20.require(parent["role"] == "B_output_only" and parent["parameters"] == 9441
                and parent["first_active"] == 1344, "Unexpected D22 parent")
    pins = parent["frozen_pins"] | {str(k): v for k, v in D22_PINS.items()}
    for seed in SEEDS:
        folder = D22 / "fit" / f"seed{seed}_{ROLE}"
        for path in (folder / "CONFIG.json", folder / "RESULT.json", folder / "recovery/epoch00.pt",
                     D22 / "fit" / f"seed{seed}_ANCHOR.json"):
            pins[str(path)] = d20.sha(path)
    d20.verify_pins(pins)
    result = dict(candidate="D23_MEMORYLESS_OUTPUT_V1", source_sha256=d20.sha(Path(__file__)),
                  frozen_pins=pins, seeds=SEEDS, role="S_memoryless", stored_role_slot=ROLE,
                  parameters=PARAMETERS, current_dimension=1, context_dimension=1, first_active=1344,
                  cuda_uuid=uuid, normalization="Both branches use actual same-seed D20 P TRAIN scalar statistics",
                  initialization="D22 shared tensors unchanged; context_in.weight is W[:,0:1] times sqrt(96)",
                  engineering_budget=dict(TRAIN_records_loaded=6, TRAIN_cached_windows_loaded=3447,
                                          VAL_records_loaded=0, current_stem_points=25056,
                                          context_output_points=24960, context_code_canary_points=96,
                                          backward_points=24576, optimizer_updates=0),
                  formal_budget=parent["formal_budget"], resources=parent["resources"],
                  paired_evaluation="Original runner duplicate slot, identical inputs/outputs; not a history intervention",
                  fit_requires="Independent EXECUTION_REVIEW bound to actual engineering and protocol")
    d20.write(ROOT / "PROTOCOL.json", result)
    print(d20.sha(ROOT / "PROTOCOL.json"), flush=True)


def admit(stage, d20):
    path = ROOT / "PROTOCOL.json"
    p, identity = d20.read(path), d20.sha(path)
    require = d20.require
    require(p["candidate"] == "D23_MEMORYLESS_OUTPUT_V1" and p["source_sha256"] == d20.sha(Path(__file__)),
            "D23 source drift")
    d20.verify_pins(p["frozen_pins"])
    meta = d20.read(D19 / "PREPARATION_METADATA.json")
    d20.verify_pins(meta["provenance"])
    for path, stamp in meta["inputs"].items():
        s = Path(path).stat()
        require((s.st_size, s.st_mtime_ns) == (stamp["bytes"], stamp["mtime_ns"]), "Frozen input drift")
    if stage == "fit":
        gate = d20.read(ROOT / "EXECUTION_REVIEW.json")
        eng = ROOT / "engineer/COMPLETED.json"
        require(gate["stage"] == "fit" and gate["decision"] == "PASS"
                and gate["protocol_sha256"] == identity and gate["engineering_sha256"] == d20.sha(eng)
                and d20.read(eng)["passed"], "Missing bound independent fit review")
    uuid = p["cuda_uuid"]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-"), "CUDA UUID")
    used = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    require(used.isdecimal() and int(used) <= 256, "Selected GPU is busy")
    require(shutil.disk_usage(ROOT).free >= 20 * GIB, "SATA space cap")
    return p, identity, meta


def adapters(d20, d22, core, legacy):
    torch, require = core.torch, d20.require
    table, parent, _, _, _ = d22.adapters(d20, core, legacy)

    class ScalarTable(table):
        def context_raw(self, indices, role, lag=core.NEAR):
            require(role == ROLE and lag in (core.NEAR, core.OLDER), "Unexpected S input slot")
            return self.current[indices]

    class ScalarDecoder(parent):
        def __init__(self, seed, role):
            super().__init__(seed, role)
            context = self.context_in
            with torch.random.fork_rng(devices=[]):
                self.context_in = torch.nn.Linear(1, context.out_features)
            with torch.no_grad():
                self.context_in.weight.copy_(context.weight[:, :1] * (96 ** .5))
                self.context_in.bias.copy_(context.bias)

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

    base = legacy.fit_role
    meta = SimpleNamespace(**(vars(legacy.m) | {"PARAMETERS": legacy.m.PARAMETERS | {ROLE: PARAMETERS}}))
    fit_role = FunctionType(base.__code__, base.__globals__ | {"m": meta}, base.__name__)
    backend = SimpleNamespace(**(vars(core) | {"PhysicalDecoder": ScalarDecoder}))
    require(fit_role.__code__ is base.__code__ and inspect.signature(fit_role) == inspect.signature(base)
            and legacy.m.PARAMETERS[ROLE] == 9665 and meta.PARAMETERS[ROLE] == PARAMETERS
            and backend.evaluate is core.evaluate and backend.batch_inputs is core.batch_inputs,
            "Common frozen runner drift")
    return ScalarTable, parent, ScalarDecoder, Receipt, fit_role, backend


def anchor_bridge(seed, train, core, d20):
    reference = d20.read(D22 / "fit" / f"seed{seed}_ANCHOR.json")["anchor"]["train"]
    by_key = {(r["record"], r["start"]): r["r"] for r in reference["blocks"]}
    actual = core.evaluate(train, None, None, None, None, lambda: None, lambda *_: None)["near"]
    error = max(abs(r["r"] - by_key[r["record"], r["start"]]) for r in actual["blocks"])
    d20.require(error <= 1e-10, "Bounded TRAIN anchor scoring bridge")
    return len(actual["blocks"]), error


def engineer(meta, p, receipt, core, legacy, d20, d22, Table, Parent, Model, guard):
    np, torch, m = core.np, core.torch, legacy.m
    require = d20.require
    checks = []
    for seed in SEEDS:
        receipt.scope = dict(seed=seed, phase="first_two_TRAIN_records")
        train = core.load_phase(meta, seed, "train", guard, train_records_limit=2)
        receipt.count("TRAIN_records_loaded", len(train))
        receipt.count("TRAIN_cached_windows_loaded", sum(r.metadata["windows"] for r in train))
        block_count, anchor_error = anchor_bridge(seed, train, core, d20)
        table = Table(train, "cuda:0", guard)
        stats = d22.input_stats(seed, table, core, d20)
        count = sum(r.stop - m.FIRST_ACTIVE for r in train)
        dense = np.column_stack((np.arange(2048), np.arange(count - 2048, count))).astype(np.int32)
        idx = table.active_indices(dense.ravel())
        spans = [(0, m.FIRST_ACTIVE, m.FIRST_ACTIVE + 2048), (1, train[1].stop - 2048, train[1].stop)]
        require(torch.equal(idx, table.indices(spans).reshape(2, 2048).T.flatten()), "Cross-record IDs")
        x, v, b, y = core.batch_inputs(table, idx, ROLE, stats)
        mu, sigma = [np.float32(t.item()) for t in stats[ROLE][:2]]
        expected = ((b.cpu().numpy().astype(np.float32) - mu) / sigma)[:, None]
        require(torch.equal(x, v) and x.shape == (4096, 1)
                and np.array_equal(x.cpu().numpy(), expected)
                and torch.equal(table.context(idx, ROLE, stats, core.OLDER), v), "Scalar input contract")
        parent = Parent(seed, ROLE)
        cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        model = Model(seed, ROLE)
        require(torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state()),
                "D22 RNG stream changed")
        require(sum(t.numel() for t in model.parameters()) == PARAMETERS, "S parameter count")
        for name, tensor in model.state_dict().items():
            original = parent.state_dict()[name]
            expected = original[:, :1] * (96 ** .5) if name == "context_in.weight" else original
            require(torch.equal(tensor, expected), f"S initialization mismatch {name}")
        frozen = D22 / "fit" / f"seed{seed}_{ROLE}"
        config = d20.read(frozen / "CONFIG.json")
        require(legacy.initialization_sha(parent) == config["initialization_sha256"], "D22 initial identity")
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261106, 0]))
        order = np.arange(meta["phases"]["train"]["active_points"], dtype=np.int32)
        rng.shuffle(order)
        old = torch.load(frozen / "recovery/epoch00.pt", map_location="cpu", weights_only=True)
        require(rng.bit_generator.state == old["numpy_bit_generator"]
                and config["shuffle_seed_recipe"] == [seed, 20261106, "epoch"], "D22 shuffle identity")
        del order, old
        model = model.cuda()
        gradients = []
        for nonzero in (False, True):
            model.zero_grad(set_to_none=True)
            if nonzero:
                with torch.no_grad():
                    model.output.weight.copy_(torch.linspace(-.025, .025, m.WIDTH, device="cuda"))
            pred = model(x, v, b)
            receipt.count("current_stem_points", len(b))
            receipt.count("context_output_points", len(b))
            if not nonzero:
                require(torch.equal(pred, b), "Fresh S residual not zero")
            ((pred - y).square().sum() / m.BATCH).backward()
            receipt.count("backward_points", len(b))
            norms = {name: float(layer.weight.grad.norm()) for name, layer in model.named_children()}
            require(all(np.isfinite(n) for n in norms.values()) and norms["output"] > 0
                    and all((n > 0) == nonzero for name, n in norms.items() if name != "output"),
                    "Unexpected S gradient")
            gradients.append(norms)
        # Each probe is separated by >384 points. Change its full past and keep its own b fixed.
        probes = [m.FIRST_ACTIVE + 512 * i for i in range(32)]
        probe_spans = [(i, t, t + 1) for i in range(2) for t in probes]
        probe_idx = table.indices(probe_spans)
        changed = []
        for record in train:
            wave = record.anchor.copy()
            for t in probes:
                wave[t - 384:t] += 1.5
            changed.append(replace(record, anchor=wave))
        past = Table(changed, "cuda:0", guard)
        a, h, base, _ = core.batch_inputs(table, probe_idx, ROLE, stats)
        a2, h2, base2, _ = core.batch_inputs(past, probe_idx, ROLE, stats)
        require(bool((past.prior[probe_idx - 1] != table.prior[probe_idx - 1]).all())
                and torch.equal(base, base2) and torch.equal(a, a2) and torch.equal(h, h2),
                "Historical perturbation did not isolate current b")
        wave = changed[0].anchor.copy()
        wave[probes[:16]] += .25
        altered = Table([replace(changed[0], anchor=wave), changed[1]], "cuda:0", guard)
        x3, v3, _, _ = core.batch_inputs(altered, probe_idx[:16], ROLE, stats)
        with torch.no_grad():
            same = torch.equal(model(a, h, base), model(a2, h2, base2))
            receipt.count("current_stem_points", 128)
            receipt.count("context_output_points", 128)
            require(same, "True prior samples altered S predictions")
            q_delta = float((model.current_code(x3) - model.current_code(a[:16])).abs().max())
            context = lambda z: model.context_out(torch.nn.functional.gelu(model.context_in(z)))
            c_delta = float((context(v3) - context(h[:16])).abs().max())
            receipt.count("current_stem_points", 32)
            receipt.count("context_code_canary_points", 32)
            require(torch.equal(x3, v3) and q_delta > 0 and c_delta > 0, "Current b did not reach both branches")
        checks.append(dict(seed=seed, anchor_train_blocks=block_count, anchor_block_max_abs=anchor_error,
                           shared_initial_tensors_exact=True, CPU_CUDA_RNG_exact=True, D22_shuffle_exact=True,
                           changed_prior_probes=64, same_output_after_past_perturbation=True,
                           scalar_normalization_exact=True, query_code_delta=q_delta, context_code_delta=c_delta,
                           gradients=gradients))
        guard()
    for key, value in p["engineering_budget"].items():
        require(receipt.counters[key] == value, f"Engineering count {key}")
    return dict(passed=True, checks=checks, formal_fits=0, optimizer_updates=0,
                common_fit_code_and_signature_exact=True, frozen_module_unchanged=True)


def fit(meta, p, receipt, core, legacy, d20, d22, Table, fit_role, backend, guard):
    result = d22.fit(meta, p, receipt, core, legacy, d20, Table, fit_role, backend, guard)
    for seed, row in result["seeds"].items():
        previous = d20.read(D22 / "fit" / f"seed{seed}_{ROLE}/RESULT.json")["selected"]["r"]
        row["D22_r"] = previous
        row["delta_D22_r"] = row["selected_r"] - previous
        if row["selected_epoch"] != -1:
            d20.require(row["selected_r"] == row["paired_far_r"], "Memoryless duplicate scores differ")
            record = d20.read(receipt.folder / f"seed{seed}_{ROLE}/RESULT.json")
            pair = record["val_pair"]
            d20.require(pair["near"] == pair["older"], "Memoryless duplicate saved metrics differ")
        row["duplicate_slot_r"] = row.pop("paired_far_r")
    result["history_intervention"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "engineer", "fit"))
    parser.add_argument("--cuda-uuid")
    args = parser.parse_args()
    if ROOT.resolve() != ROOT or Path(__file__).resolve().parent != ROOT / "input":
        raise RuntimeError("Use the independent 2203 SATA root")
    sys.path[:0] = [str(D20 / "input"), str(D22 / "input")]
    import pub_01_s03_ss07_recent_prediction_context as d20
    import pub_01_s03_ss07_output_query as d22
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
    Table, Parent, Model, Receipt, fit_role, backend = adapters(d20, d22, core, legacy)
    receipt = Receipt(folder, args.stage, identity)
    torch = core.torch
    try:
        def timeout(*_):
            raise TimeoutError("D23 wall-clock cap")

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
        result = (engineer(meta, p, receipt, core, legacy, d20, d22, Table, Parent, Model, guard)
                  if args.stage == "engineer"
                  else fit(meta, p, receipt, core, legacy, d20, d22, Table, fit_role, backend, guard))
        guard()
        d20.write(folder / "COMPLETED.json", dict(receipt.snapshot(), **result))
    except BaseException as error:
        d20.write(folder / "FAILURE.json", dict(receipt.snapshot(), error=repr(error)))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
