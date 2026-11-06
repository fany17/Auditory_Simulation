"""D24 downstream: same frozen B/S trainer, paired IS and OOF scalar inputs."""
import argparse
import os
import resource
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pub_01_s03_ss07_crossfit_meta import D19, ROOT, SATA, SEEDS, digest, read

D20 = SATA / "s03_recent_prediction_context_20261106"
D22 = SATA / "s03_output_query_20261106"
D23 = SATA / "s03_memoryless_output_20261106"
REPLAY = ROOT / "coordinator_upstream_audit_v01"
REPLAY_SOURCE = ROOT / "COORDINATOR_UPSTREAM_REPLAY.py"
REPLAY_SOURCE_SHA = "d8016bbd5b171e2893fd80d6366552a5cd32115549137c25970b3efd7c877a14"
FORMAL_GATE = ROOT / "DOWNSTREAM_FORMAL_GATE_V04.json"
ROLE = "P_past_prediction"
GIB = 1024**3


def get_modules():
    sys.path[:0] = [str(path / "input") for path in (D19, D20, D22, D23)]
    import pub_01_s03_ss07_physical_stream_core as core
    import pub_01_s03_ss07_physical_stream_run as legacy
    import pub_01_s03_ss07_recent_prediction_context as d20
    import pub_01_s03_ss07_output_query as d22
    import pub_01_s03_ss07_memoryless_output as d23
    return core, legacy, d20, d22, d23


@dataclass
class ScalarRecord:
    metadata: dict
    anchor: object
    target: object
    blocks: list

    @property
    def stop(self):
        return self.metadata["covered_stop"]


def tables(core):
    np, torch = core.np, core.torch

    class ScalarTable(core.Tables):
        """Keep the original physical coordinate methods and discard unused EEG fields."""
        def __init__(self, records, variant, device, guard):
            self.variant = variant
            self.device = device
            self.starts = np.cumsum([0]+[r.stop for r in records], dtype=np.int64)
            self.stops = [r.stop for r in records]
            self.active_ends = torch.from_numpy(self.starts[1:]-core.FIRST_ACTIVE*
                                                np.arange(1, len(records)+1)).to(device)
            n = int(self.starts[-1])
            self.anchor = torch.empty(n, dtype=torch.float64, device=device)
            self.target = torch.empty_like(self.anchor)
            self.prior = torch.empty(n, dtype=torch.float32, device=device)
            for i, record in enumerate(records):
                region = slice(self.starts[i], self.starts[i+1])
                anchor = record.anchor[:record.stop]
                self.anchor[region].copy_(torch.from_numpy(anchor.copy()))
                self.prior[region].copy_(torch.from_numpy(core.rolling_mean(anchor, core.BIN_POINTS)))
                target = record.target[:record.stop]
                dtype = np.float32 if record.metadata["phase"] == "train" else np.float64
                self.target[region].copy_(torch.from_numpy(target.astype(dtype)))
                guard()
            self.current = self.anchor.float()[:, None]
            self.bin_ends = torch.arange(core.BINS, device=device)*core.BIN_POINTS-(
                core.HISTORY-core.BIN_POINTS)

        def history(self, indices, role, lag):
            if role != ROLE or lag not in (core.NEAR, core.OLDER):
                raise ValueError("Unexpected B history route")
            delay = 1 if lag == core.NEAR else 288
            return self.prior[indices[:, None]-delay+self.bin_ends]

        def context_raw(self, indices, role, lag=core.NEAR):
            if role != ROLE or lag not in (core.NEAR, core.OLDER):
                raise ValueError("Unexpected B/S context route")
            if self.variant == "S":
                return self.current[indices]
            return self.history(indices, role, lag)

    return ScalarTable


def payload_pin(path):
    before = path.stat()
    digestor = sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4*1024**2), b""):
            digestor.update(data)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Payload changed while hashing: {path}")
    return dict(bytes=after.st_size, mtime_ns=after.st_mtime_ns, sha256=digestor.hexdigest())


def historical_inputs(meta, d20):
    wall, cpu = time.monotonic(), time.process_time()
    d19 = read(D19/"PROTOCOL.json")
    d20.require(d19["metadata_sha256"] == digest(D19/"PREPARATION_METADATA.json"), "D19 metadata")
    pins = dict(meta["provenance"])
    pins.update({str(D19/"input"/name): value for name, value in d19["source_sha256"].items()})
    for run, name, key in ((D20, "recent_prediction_context", "d19_pins"),
                           (D22, "output_query", "frozen_pins"),
                           (D23, "memoryless_output", "frozen_pins")):
        protocol = read(run/"PROTOCOL.json")
        for path, sha in protocol[key].items():
            d20.require(pins.setdefault(path, sha) == sha, "Contradictory historical pin")
        pins[str(run/"input"/f"pub_01_s03_ss07_{name}.py")] = protocol["source_sha256"]
    bytes_hashed = 0
    for path, sha in pins.items():
        result = payload_pin(Path(path))
        d20.require(result["sha256"] == sha, f"Historical pin drift: {path}")
        bytes_hashed += result["bytes"]
    for path, expected in meta["inputs"].items():
        stat = Path(path).stat()
        d20.require((stat.st_size, stat.st_mtime_ns) == (expected["bytes"], expected["mtime_ns"]),
                    f"D19 frozen input stat drift: {path}")
    return dict(historical_sha256_pins=len(pins), D19_input_stats=len(meta["inputs"]),
                historical_sha256_bytes=bytes_hashed, history_payload_scoring=0,
                wall_seconds=time.monotonic()-wall, CPU_seconds=time.process_time()-cpu)


def lane_data(meta, seed, lane, manifest, core, require):
    np = core.np
    index, blocks = {}, defaultdict(list)
    route_key = f"{lane}_model_fold"
    for fold in range(3):
        folder = ROOT / f"formal/seed{seed}_fold{fold}"
        completed = read(folder / "COMPLETED.json")
        require(completed["stage"] == "formal" and completed["seed"] == seed
                and completed["fold"] == fold and completed["original_VAL_reads"] == 0
                and completed["OOF_label_reads"] == 0, "Upstream identity")
        fit = set(completed["fit_records"])
        expected_fit = {row["record"] for row in manifest if row["OOF_model_fold"] != fold}
        expected_output = [row for row in manifest if row[route_key] == fold]
        require(fit == expected_fit, "Actual upstream fit-record coverage")
        entries = read(folder / f"{lane}_ANCHOR.json")
        array = np.load(folder / f"{lane}_ANCHOR.npy", mmap_mode="r", allow_pickle=False)
        require(array.dtype == np.float64 and len(entries) == len(expected_output), "Lane schema")
        end = 0
        for out, row in zip(entries, expected_output):
            key = row["record"]
            require(out == dict(record=key, offset=end, stop=row["covered_stop"], windows=row["windows"])
                    and key not in index and (key in fit) == (lane == "IS"), "Lane routing/geometry")
            index[key] = array[end:end+out["stop"]]
            end += out["stop"]
        require(end == len(array), "No trailing anchor values")
    for block in meta["phases"]["train"]["blocks"]:
        blocks[block["record"]].append(block)
    output = []
    for row in meta["phases"]["train"]["records"]:
        key = row["record"]
        target = np.load(row["envelope_path"], mmap_mode="r", allow_pickle=False).reshape(-1)
        wave = index[key]
        require(np.isfinite(wave).all() and len(wave) == row["covered_stop"], "Finite lane anchor")
        output.append(ScalarRecord(row, wave, target[:row["common_samples"]].astype(np.float64),
                                   blocks[key]))
    require(len(index) == len(output) == 426, "Complete paired TRAIN lane")
    return output


def moments(train, table, core, guard):
    torch = core.torch
    sums = torch.zeros((), dtype=torch.float64, device=table.device)
    squares = torch.zeros_like(sums)
    count = 0
    for i, start, stop in core.chunks(train):
        idx = table.indices([(i, start, stop)])
        value = table.history(idx, ROLE, core.NEAR).double()
        sums += value.sum()
        squares += value.square().sum()
        count += len(idx)
        if stop == train[i].stop:
            guard()
    stats = core.moments(sums, squares, count*core.BINS)
    return {"current": tuple(value.reshape(1) for value in stats), ROLE: stats}, count


def wire_guard(receipt, torch, require):
    def guard():
        receipt.runtime = dict(
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            writer_free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize,
            writer_bytes=sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()),
        )
        require(receipt.runtime["peak_RSS_bytes"] <= 12*GIB
                and receipt.runtime["cuda_peak_reserved_bytes"] <= 8*GIB, "D24 meta memory cap")
        require(receipt.runtime["writer_free_bytes"] >= 64*GIB
                and receipt.runtime["writer_bytes"] <= 64*GIB, "D24 SATA cap")
    return guard


def engineer(meta, seed, core, legacy, d20, d22, d23, output):
    torch = core.torch
    receipt = legacy.Receipt(output, "engineer", "bounded_coordinate_bridge")
    guard = wire_guard(receipt, torch, d20.require)
    train = core.load_phase(meta, seed, "train", guard, train_records_limit=2)
    Table = tables(core)
    BTable, BModel, *_ = d22.adapters(d20, core, legacy)
    STable, _, SModel, *_ = d23.adapters(d20, d22, core, legacy)
    checkpoints = []
    for variant, original, model in (("B", BTable, BModel), ("S", STable, SModel)):
        old = original(train, "cuda:0", guard)
        new = Table(train, variant, "cuda:0", guard)
        stats, n = moments(train, new, core, guard)
        idx = new.indices([(i, core.FIRST_ACTIVE, core.FIRST_ACTIVE+256)
                           for i in range(len(train))])
        net = model(seed, ROLE).cuda()
        for lag in (core.NEAR, core.OLDER):
            d20.require(torch.equal(old.context_raw(idx, ROLE, lag), new.context_raw(idx, ROLE, lag)),
                        "Original history/table bridge")
        a = core.batch_inputs(old, idx, ROLE, stats)
        b = core.batch_inputs(new, idx, ROLE, stats)
        d20.require(all(torch.equal(x, y) for x, y in zip(a, b)), "Original physical input bridge")
        pred = net(*b[:3])
        d20.require(torch.equal(pred, b[2]), "Zero initial physical residual")
        ((pred-b[3]).square().sum()/legacy.m.BATCH).backward()
        d20.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()),
                    "Original loss/backward bridge")
        checkpoints.append(dict(variant=variant, identical=True, compared_points=len(idx),
                                moments_points=n, parameters=sum(p.numel() for p in net.parameters()),
                                initialization_sha256=legacy.initialization_sha(net)))
    d20.write(output/"COMPLETED.json", dict(passed=True, checks=checkpoints, updates=0,
                                          process=receipt.identity, runtime=receipt.runtime))


def formal(meta, seed, core, legacy, d20, d22, d23, output, gate):
    torch = core.torch
    Table = tables(core)
    _, _, _, fit_b, backend_b = d22.adapters(d20, core, legacy)
    _, _, _, _, fit_s, backend_s = d23.adapters(d20, d22, core, legacy)
    manifest = read(ROOT / "TRAIN_RECORD_ASSIGNMENT.json")["records"]
    receipt = legacy.Receipt(output, "fit", digest(gate))
    frozen_gate = read(gate)
    wall, cpu = time.monotonic(), time.process_time()
    historical = historical_inputs(meta, d20)
    required = {path: pin for path, pin in frozen_gate["upstream_payloads"].items()
                if Path(path).parent.name.startswith(f"seed{seed}_") and path.endswith("_ANCHOR.npy")}
    d20.require(len(required) == 6, "Exact lane prediction inputs")
    for path, pin in required.items():
        d20.require(payload_pin(Path(path)) == pin, f"Prediction identity drift: {path}")
    d20.write(output/"VERIFIED_INPUTS.json", dict(prediction_files=len(required),
              prediction_bytes=sum(pin["bytes"] for pin in required.values()),
              historical=historical, wall_seconds=time.monotonic()-wall,
              CPU_seconds=time.process_time()-cpu, new_scoring=0))
    guard = wire_guard(receipt, torch, d20.require)
    val = core.load_phase(meta, seed, "val", guard)
    anchor_val = core.evaluate(val, None, None, None, None, guard, receipt.count)["near"]
    bridge = legacy.anchor_bridge(anchor_val, seed)
    results = {}
    for lane in ("IS", "OOF"):
        train = lane_data(meta, seed, lane, manifest, core, d20.require)
        train_table = Table(train, "B", "cuda:0", guard)
        val_table = Table(val, "B", "cuda:0", guard)
        stats, count = moments(train, train_table, core, guard)
        d20.require(count == meta["phases"]["train"]["active_points"], "Complete TRAIN moments")
        saved = dict(mean=float(stats[ROLE][0]), std=float(stats[ROLE][1]),
                     constant=bool(stats[ROLE][2]), points=count, context_values=count*core.BINS)
        d20.require(saved["std"] > 0 and not saved["constant"], "Nonconstant scalar TRAIN moments")
        d20.write(output / f"{lane}_TRAIN_STATS.json", saved)
        anchor_train = core.evaluate(train, None, None, None, None, guard, receipt.count)["near"]
        anchor = {"train": anchor_train, "val": anchor_val}
        for variant, fit, backend in (("B", fit_b, backend_b), ("S", fit_s, backend_s)):
            arm = f"{variant}_{lane}"
            folder = output / arm
            folder.mkdir()
            receipt.folder = folder
            d20.write(folder / f"seed{seed}_TRAIN_STATS.json", saved)
            train_table.variant = val_table.variant = variant
            results[arm] = fit(meta, receipt, backend, torch, guard, seed, ROLE,
                               train, val, (train_table, val_table), stats, anchor)
            if variant == "S":
                pair = read(folder / f"seed{seed}_{ROLE}/RESULT.json")["val_pair"]
                d20.require(pair["near"] == pair["older"], "S has no history dependency")
        del train, train_table, val_table
    d20.require(receipt.counters["optimizer_updates"] == 115760
                and receipt.counters["backward_points"] == 1896280800, "Exact downstream work")
    receipt.folder = output
    d20.write(output/"COMPLETED.json", dict(passed=True, seed=seed, results=results,
                                           validation_anchor_bridge=bridge, **receipt.snapshot()))


def freeze(meta, d20):
    """Admit only a completed and exact upstream matrix, without reading its tensors."""
    upstream_gate = ROOT / "UPSTREAM_FORMAL_GATE_V04.json"
    upstream = read(upstream_gate)
    terminal = ROOT / "queue/TERMINAL.json"
    d20.require(read(terminal)["passed"], "Upstream queue has not completed")
    replay_done_path = REPLAY / "COMPLETED.json"
    replay_done = read(replay_done_path)
    replay_count = replay_done["counters"]
    d20.require(replay_done["passed"] and replay_done["normalization_folds"] == 3
                and len(replay_done["jobs"]) == 9 and replay_count["NN_windows"] <= 16384
                and replay_count["fitting_updates"] == replay_count["labels_read"] == 0,
                "Independent upstream replay has not passed")
    d20.require(replay_done["process"]["source_sha256"] == digest(REPLAY_SOURCE) == REPLAY_SOURCE_SHA,
                "Independent upstream replay source drift")
    candidate_path = ROOT / "FOLD_AND_BUDGET_CANDIDATE_V2.json"
    d20.require(upstream["candidate_sha256"] == digest(candidate_path), "Upstream candidate drift")
    candidate = read(candidate_path)
    assignment = ROOT / "TRAIN_RECORD_ASSIGNMENT.json"
    d20.require(candidate["train_manifest_sha256"] == digest(assignment), "TRAIN assignment drift")
    manifest = read(assignment)["records"]
    sources = {upstream_gate, terminal, candidate_path, assignment, Path(__file__).resolve(),
               D19 / "PREPARATION_METADATA.json",
               D19 / "PROTOCOL.json", D20 / "PROTOCOL.json", D22 / "PROTOCOL.json", D23 / "PROTOCOL.json",
               ROOT / "downstream_engineer/seed1024/COMPLETED.json",
               ROOT / "DOWNSTREAM_INPUT_PREFLIGHT_V03.json", replay_done_path, REPLAY_SOURCE}
    module_paths = {Path(module.__file__).resolve() for module in list(sys.modules.values())
                    if getattr(module, "__file__", None)
                    and Path(module.__file__).name.startswith("pub_01_")}
    sources.update(module_paths)
    d20.require(read(ROOT / "downstream_engineer/seed1024/COMPLETED.json")["passed"], "Downstream canary")
    historical = historical_inputs(meta, d20)
    coverage, payloads = {}, {}
    payload_wall, payload_cpu = time.monotonic(), time.process_time()
    for job in candidate["jobs"]:
        seed, fold = job["seed"], job["model_fold"]
        job_key = f"seed{seed}_fold{fold}"
        folder = ROOT / "formal" / job_key
        independent_path = REPLAY / f"{job_key}.json"
        independent = read(independent_path)
        d20.require(job_key in replay_done["jobs"] and independent["passed"]
                    and independent["seed"] == seed and independent["fold"] == fold,
                    "Independent actual job identity")
        sources.add(independent_path)
        completed_path = folder / "COMPLETED.json"
        frozen_path = folder / "UPSTREAM_FROZEN.json"
        completed, frozen = read(completed_path), read(frozen_path)
        ids = {row["record"] for row in manifest if row["OOF_model_fold"] != fold}
        d20.require(set(completed["fit_records"]) == set(frozen["fit_records"]) == ids
                    and set(completed["contrastive"]["record_presentations"]) <= ids, "Actual TRAIN fence")
        d20.require(completed["seed"] == seed and completed["fold"] == fold and
                    completed["original_VAL_reads"] == completed["OOF_label_reads"] ==
                    completed["old_test_reads"] == 0, "Upstream phase boundary")
        d20.require(completed["normalization_samples"] == job["fit_raw_channel_normalization_samples"]
                    and frozen["Ccap"]["TRAIN_physical_points"] == job["affine_physical_points"]
                    and frozen["Ccap"]["TRAIN_fit_windows"] == job["Ccap"]["fit_windows"]
                    and frozen["affine"]["a"] > 0, "Actual calibration support")
        d20.require(completed["contrastive"]["updates"] == job["contrastive"]["updates"],
                    "Actual contrastive budget")
        for stage in ("head", "FT"):
            for key in ("updates", "target_windows"):
                d20.require(sum(epoch[key] for epoch in completed[stage]) == job[stage][key],
                            "Actual supervised budget")
        sources.update((completed_path, frozen_path))
        for lane in ("IS", "OOF"):
            path = folder / f"{lane}_ANCHOR.json"
            row = read(path)
            expected = [item for item in manifest if item[f"{lane}_model_fold"] == fold]
            d20.require([item["record"] for item in row] == [item["record"] for item in expected],
                        "Fold output mapping")
            d20.require(sum(item["windows"] for item in row) == job[f"{lane}_output_windows"],
                        "Fold output windows")
            sources.add(path)
        for name in ("IS_ANCHOR.npy", "OOF_ANCHOR.npy", "READOUT.npz", "FT.pt", "head.pt", "contrastive.pt"):
            path = folder / name
            pin = payload_pin(path)
            other = independent["payloads"][name]
            d20.require(all(pin[key] == other[key] for key in ("bytes", "sha256")),
                        "Upstream payload differs from the independent replay")
            payloads[str(path)] = pin
        coverage[f"{seed}:{fold}"] = dict(fit_records=len(ids),
            contrastive_actual_records=len(completed["contrastive"]["record_presentations"]),
            Ccap_fit_windows=frozen["Ccap"]["TRAIN_fit_windows"],
            total_CPU_seconds=completed["total_CPU_seconds"],
            total_peak_reserved=completed["total_peak_reserved"])
    d20.require(len(coverage) == 9 and len(meta["phases"]["val"]["records"]) == 49,
                "Exact completed matrix")
    d20.require(len(payloads) == 54, "Exact upstream payload inventory")
    d20.write(FORMAL_GATE,
              dict(passed=True, upstream=coverage, upstream_payloads=payloads,
                   sources={str(path): digest(path) for path in sorted(sources)},
                   historical=historical, independent_replay=dict(receipt=str(replay_done_path),
                       jobs=9, counters=replay_count, source_sha256=REPLAY_SOURCE_SHA),
                   payload_verification=dict(bytes=sum(p["bytes"] for p in payloads.values()),
                       wall_seconds=time.monotonic()-payload_wall,
                       CPU_seconds=time.process_time()-payload_cpu, new_scoring=0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--stage", choices=("engineer", "preflight", "freeze", "formal"), required=True)
    args = parser.parse_args()
    core, legacy, d20, d22, d23 = get_modules()
    meta = read(D19 / "PREPARATION_METADATA.json")
    if args.stage == "preflight":
        result = historical_inputs(meta, d20)
        d20.write(ROOT/"DOWNSTREAM_INPUT_PREFLIGHT_V03.json", result)
        print(result)
        return
    if args.stage == "freeze":
        freeze(meta, d20)
        return
    d20.require(args.seed in SEEDS, "Specify a seed for GPU work")
    torch = core.torch
    d20.require(torch.cuda.device_count() == 1 and os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-"),
                "Pin single CUDA UUID")
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    gate = FORMAL_GATE
    if args.stage == "formal":
        pins = read(gate)
        d20.require(pins["passed"] and all(digest(Path(p)) == h for p, h in pins["sources"].items()),
                    "Frozen downstream gate drift")
    d20.require(legacy.m.BATCH == core.BATCH == 16384 and legacy.m.EPOCHS == 20
                and meta["phases"]["train"]["last_batch"] == 12246, "Frozen training schedule")
    root = ROOT / f"downstream_{args.stage}"
    root.mkdir(exist_ok=True)
    folder = root / f"seed{args.seed}"
    folder.mkdir()
    try:
        if args.stage == "engineer":
            engineer(meta, args.seed, core, legacy, d20, d22, d23, folder)
        else:
            formal(meta, args.seed, core, legacy, d20, d22, d23, folder, gate)
    except BaseException as error:
        d20.write(folder/"FAILURE.json", dict(pid=os.getpid(), uid=os.getuid(),
                                             message=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
