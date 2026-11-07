"""D27 independent full-cohort CPU replay, after all formal fits and saved audit.

Implements inference directly from stored tensors and convolution primitives.
Does not import the trained model, loader, report, or scoring implementation.
Self-test imports only the hash-pinned author class and uses synthetic inputs.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_seawave_seen_reference_20261106")
GATE = "8442767e6fb1a220bab129d2c2163321e760610102fdefb986f43a06079f7476"
AUTHOR_SHA = "bac621abfd5c4553d9b4be790da7e5af81d26932ee3e0dede86564503950f0f0"
TOLERANCE = 1e-4
SEEDS = (1024, 2026, 31415)
CPUS = {40, 41, 42, 43}
COUNTS = Counter()
HASHES = {}


def need(ok, label):
    if not ok:
        raise AssertionError(label)


def configure():
    os.sched_setaffinity(0, CPUS)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    need(not torch.cuda.is_initialized() and torch.get_num_threads() == 4, "CPU-only bounded replay")


def digest(path):
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    HASHES[str(path)] = h
    return h


def read(path):
    data = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=os.getpid(), uid=os.getuid(), startticks=int(fields[19]))


def shapes():
    expected = {}
    def normalized(prefix, outputs, inputs, kernel):
        expected[prefix + ".weight_g"] = (outputs, 1, 1)
        expected[prefix + ".weight_v"] = (outputs, inputs, kernel)
        expected[prefix + ".bias"] = (outputs,)
    normalized("init_conv.0.conv", 32, 64, 1)
    for i in range(20):
        prefix = f"residual_layer.residual_blocks.{i}"
        normalized(prefix + ".dilated_conv_layer.conv", 64, 32, 3)
        normalized(prefix + ".res_conv", 32, 32, 1)
    normalized("final_conv.0.conv", 32, 32, 1)
    expected["final_conv.2.conv.weight"] = (1, 32, 1)
    expected["final_conv.2.conv.bias"] = (1,)
    return expected


class DirectWave:
    def __init__(self, state):
        expected = shapes()
        need(state.keys() == expected.keys(), "Exact 128 selected state tensors")
        need(sum(t.numel() for t in state.values()) == 150433, "Exact author Small parameter count")
        for name, size in expected.items():
            tensor = state[name]
            need(tuple(tensor.shape) == size and tensor.dtype == torch.float32 and tensor.device.type == "cpu"
                 and torch.isfinite(tensor).all().item(), "Finite CPU FP32 selected tensor " + name)
        self.params = {}
        for name in expected:
            if name.endswith(".weight_v"):
                prefix = name.removesuffix(".weight_v")
                v, g = state[name], state[prefix + ".weight_g"]
                length = torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True)
                need(torch.all(length > 0).item(), "Nondegenerate weight normalization")
                self.params[prefix] = (v * (g / length), state[prefix + ".bias"])
        self.final_weight = state["final_conv.2.conv.weight"]
        self.final_bias = state["final_conv.2.conv.bias"]

    def conv(self, x, name, dilation=1):
        weight, bias = self.params[name]
        return F.conv1d(x, weight, bias, dilation=dilation,
                        padding=dilation * (weight.shape[-1] - 1) // 2)

    def __call__(self, x):
        COUNTS["independent_model_calls"] += 1
        COUNTS["EEG_sample_time_presentations"] += x.shape[-1] * x.shape[0]
        skip_input = F.gelu(self.conv(x, "init_conv.0.conv"), approximate="none")
        h, accumulated = skip_input, None
        for layer in range(20):
            if layer > 0 and layer % 5 == 0:
                h = (h + skip_input) * math.sqrt(.5)
            prefix = f"residual_layer.residual_blocks.{layer}"
            gates = self.conv(h, prefix + ".dilated_conv_layer.conv", 2 ** (layer % 5))
            activation = torch.tanh(gates[:, :32]) * torch.sigmoid(gates[:, 32:])
            residual = self.conv(activation, prefix + ".res_conv")
            h = (h + residual) * math.sqrt(.5)
            accumulated = residual if accumulated is None else accumulated + residual
        hidden = F.gelu(self.conv(accumulated * math.sqrt(1 / 20), "final_conv.0.conv"), approximate="none")
        return F.conv1d(hidden, self.final_weight, self.final_bias)


def continuous(model, x):
    n = x.shape[1]
    prediction = np.empty(n, dtype=np.float64)
    with torch.inference_mode():
        for core in range(0, n, 4096):
            last_core = min(n, core + 4096)
            first, last = max(0, core - 124), min(n, last_core + 124)
            y = model(x[None, :, first:last])[0, 0].numpy()
            prediction[core:last_core] = y[core-first:last_core-first]
    need(np.isfinite(prediction).all(), "Finite independent full record")
    return prediction


def metrics(prediction, target):
    prediction, target = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    need(prediction.shape == target.shape and len(target) > 1
         and np.isfinite(target).all() and np.isfinite(prediction).all(), "Finite exact metric support")
    # Independent covariance implementation; MSE uses compensated scalar summation.
    r = float(np.corrcoef(prediction, target)[0, 1])
    mse = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(prediction, target)) / len(target)
    need(math.isfinite(r) and math.isfinite(mse), "Defined independent metrics")
    return dict(r=r, MSE=mse)


def load_model(folder, expected):
    path = folder / "SELECTED.pt"
    need(digest(path) == expected, "Actual final selected checkpoint digest")
    COUNTS["checkpoint_load_attempts"] += 1
    state = torch.load(path, map_location="cpu", weights_only=True)
    COUNTS["formal_checkpoints_decoded"] += 1
    COUNTS["decoded_checkpoint_file_bytes"] += path.stat().st_size
    return DirectWave(state)


def check_stat(path, pins):
    st = Path(path).stat()
    need(pins[str(path)] == dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns), "Frozen physical input " + str(path))


def replay_case(seed, stage, inv, norm, masks):
    start, cpu = time.monotonic(), time.process_time()
    before = COUNTS.copy()
    base = ROOT / "formal" / f"seed{seed}"
    initial = read(base / "initial/COMPLETED.json")
    model = None
    if stage == "initial":
        model = load_model(base / "initial", initial["selected_sha256"])
    blocks, models = [], {}
    mean, std = norm
    for row in inv["phase_records"]["val"]:
        folder = base / "initial" if stage == "initial" else base / "personal" / row["subject"]
        completed, score = read(folder / "COMPLETED.json"), read(folder / "SELECTED_VAL.json")
        need(completed["passed"] and completed["stage"] == stage and completed["seed"] == seed
             and completed["old_test_reads"] == 0 and completed["parameters"] == 150433, "Completed registered case")
        if stage == "personal":
            need(completed["subject"] == row["subject"] and completed["initial_sha256"] == initial["selected_sha256"],
                 "Original own initial checkpoint")
            model = load_model(folder, completed["selected_sha256"])
        models[str(folder / "SELECTED.pt")] = completed["selected_sha256"]
        check_stat(row["eeg_path"], inv["inputs"])
        check_stat(row["envelope_path"], inv["inputs"])
        raw = np.load(row["eeg_path"], mmap_mode="r", allow_pickle=False)
        eeg = raw.T if raw.shape[0] == 64 else raw
        envelope = np.load(row["envelope_path"], mmap_mode="r", allow_pickle=False).reshape(-1)
        n = row["common_samples"]
        need(eeg.ndim == 2 and eeg.shape[1] == 64 and len(eeg) >= n and len(envelope) >= n, "Exact common timeline")
        normalized = ((np.asarray(eeg[:n], dtype=np.float32) - mean) / std).astype(np.float32)
        need(np.isfinite(normalized).all(), "Finite unchanged TRAIN-only normalization")
        x = torch.from_numpy(np.ascontiguousarray(normalized.T))
        COUNTS["physical_record_passes"] += 1
        COUNTS["EEG_record_sample_times_read"] += n
        pred = continuous(model, x)
        expected = {b["start"]:b for b in score["blocks"] if b["record"] == row["record"]}
        eligible = {b["start"]:b for b in masks[row["record"]] if b["eligible"]}
        need(expected.keys() == eligible.keys(), "Full saved record blocks on original mask")
        last_positive_exclusive = range(0, n - 192 + 1, 96)[-1] + 191
        for start_block, mask in sorted(eligible.items()):
            first, last = max(1, start_block), min(last_positive_exclusive, start_block + 3840)
            saved = expected[start_block]
            need(last-first == mask["samples"] == saved["samples"]
                 and saved["subject"] == row["subject"] and saved["source_family"] == row["source_family"],
                 "Exact saved sample and participant support")
            COUNTS["target_scored_sample_time_presentations"] += last-first
            observed = metrics(pred[first:last], envelope[first:last])
            errors = {m:abs(observed[m]-saved[m]) for m in ("r","MSE")}
            passed = {m:errors[m] <= TOLERANCE for m in errors}
            blocks.append(dict(record=row["record"],subject=row["subject"],source_family=row["source_family"],
                               start=start_block,samples=last-first,expected={m:saved[m] for m in errors},
                               observed=observed,errors=errors,original_checks=passed,strict_passed=all(passed.values())))
    need(len(blocks) == 752 and len({b["subject"] for b in blocks}) == 49, "All 752 blocks and 49 participants")
    return dict(seed=seed,stage=stage,execution_completed=True,strict_passed=all(b["strict_passed"] for b in blocks),
                original_absolute_tolerance=TOLERANCE,blocks=blocks,selected_checkpoint_sha256=models,
                actual_counts=dict(COUNTS-before),CPU_seconds=time.process_time()-cpu,
                wall_seconds=time.monotonic()-start)


def admitted():
    # Gate before all EEG/normalization/checkpoint reads.
    formal = ROOT / "formal/COMPLETED.json"
    audited = ROOT / "COORDINATOR_SAVED_EVIDENCE_AUDIT.json"
    need(formal.is_file() and audited.is_file()
         and (ROOT / "resource_concurrency_v01/REPORT_WAITER_COMPLETED.json").is_file(),
         "Wait for full formal matrix, report, and independent saved-evidence audit")
    end, audit = read(formal), read(audited)
    need(end["passed"] and end["initial_fits"] == 3 and end["personal_fits"] == 147
         and end["old_test_reads"] == 0 and end["protocol_sha256"] == GATE, "Complete original matrix")
    need(audit["saved_evidence_audit_passed"] and audit["formal_completed_sha256"] == digest(formal)
         and audit["protocol_sha256"] == GATE, "Actual independent saved-evidence acceptance")
    protocol = read(ROOT / "PROTOCOL_V04.json")
    need(digest(ROOT / "PROTOCOL_V04.json") == GATE, "Unchanged original science")
    for path, fingerprint in protocol["sources"].items():
        need(digest(Path(path)) == fingerprint, "Unchanged science source " + path)
    inv = read(ROOT / "INVENTORY.json")
    need(digest(ROOT / "INVENTORY.json") == protocol["inventory_sha256"]
         and len(inv["phase_records"]["val"]) == len(inv["subjects"]) == 49, "Original fixed VAL records")
    for path in inv["inputs"]:
        check_stat(path, inv["inputs"])
    mask = read(Path(protocol["VAL_mask"]))
    need(mask["prediction_independent"] and mask["records"] == [r["record"] for r in inv["phase_records"]["val"]]
         and sum(b["eligible"] for b in mask["blocks"]) == 752, "Original fixed eligibility")
    return protocol, inv, mask, digest(formal)


def run(output):
    protocol, inv, mask, terminal_sha = admitted()
    need(not output.exists(), "A replay directory is never overwritten or silently resumed")
    output.mkdir()
    wall, cpu = time.monotonic(), time.process_time()
    initial = dict(process=identity(),source_sha256=digest(Path(__file__)),protocol_sha256=GATE,
                   formal_completed_sha256=terminal_sha,original_absolute_tolerance=TOLERANCE,
                   planned_cases=[dict(seed=s,stage=t) for s in SEEDS for t in ("initial","personal")],
                   device="cpu",CPUs=sorted(CPUS),threads=torch.get_num_threads(),
                   torch_version=torch.__version__,numpy_version=np.__version__,
                   new_fits=0,new_optimizer_updates=0,old_test_reads=0,auto_retry=False)
    write(output / "STARTED.json", initial)
    cases = []
    try:
        norm_path = Path(protocol["train_only_norm"])
        check_stat(norm_path, inv["inputs"])
        with np.load(norm_path, allow_pickle=False) as norm:
            mean, std = norm["mean"].copy(), norm["std"].copy()
        need(mean.shape == std.shape == (64,) and np.all(std > 0), "Original normalization arrays")
        masks = {r["record"]:[b for b in mask["blocks"] if b["record"]==r["record"]] for r in inv["phase_records"]["val"]}
        for seed in SEEDS:
            for stage in ("initial","personal"):
                result = replay_case(seed,stage,inv,(mean,std),masks)
                name = f"REPLAY_{seed}_{stage}.json"
                write(output / name, result)
                cases.append(dict(seed=seed,stage=stage,receipt=name,sha256=digest(output/name),
                                  strict_passed=result["strict_passed"]))
        errors = [dict(seed=c["seed"],stage=c["stage"],**b) for c in cases
                  for b in read(output/c["receipt"])["blocks"] if not b["strict_passed"]]
        need(COUNTS["formal_checkpoints_decoded"] == 150 and COUNTS["physical_record_passes"] == 294,
             "Full original selected-model and record-pass matrix")
        completed = dict(**initial,execution_completed=True,passed=not errors,strict_all_metrics_passed=not errors,
                         cases=cases,failed_blocks=errors,total_blocks=4512,actual_counts=dict(COUNTS),
                         CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
                         evidence_sha256=HASHES,
                         scope="All 150 selected models and all 4512 original block scores; independent CPU convolution path.",
                         counting_note="Sample-time totals are repeated record passes and include halos, not a global union.",
                         scientific_generalization_or_SOTA_established=False)
        write(output/"COMPLETED.json",completed)
        print(json.dumps({k:v for k,v in completed.items() if k not in ("failed_blocks","evidence_sha256")}))
    except BaseException as exc:
        write(output/"FAILURE.json",dict(**initial,error=str(exc),completed_cases=cases,actual_counts=dict(COUNTS),
              CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,automatic_retry=False))
        raise


def self_test():
    configure()
    author = ROOT / "author/EEGWaveRegressor.py"
    need(digest(author) == AUTHOR_SHA, "Pinned original author implementation")
    spec = importlib.util.spec_from_file_location("author_fixture",author)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(1107)
    with contextlib.redirect_stdout(io.StringIO()):
        native = module.WaveNet_regressor(64,32,32,1,20,5,.65).eval()
    with torch.no_grad():
        native.final_conv[2].conv.weight.normal_(0,.05)
        native.final_conv[2].conv.bias.fill_(.03)
    direct = DirectWave(native.state_dict())
    generator = torch.Generator().manual_seed(571)
    errors = []
    with torch.inference_mode():
        for length in (1,127,249,4312):
            x=torch.randn(1,64,length,generator=generator)
            expected=native(x);observed=direct(x)
            error=(expected-observed).abs().max().item();errors.append(error)
            need(error <= 1e-5 and expected.abs().max().item() > .001, "Nonzero synthetic author/independent parity")
        x=torch.randn(64,8471,generator=generator)
        whole=direct(x[None])[0,0].numpy().astype(np.float64)
        tiled=continuous(direct,x)
        tile_error=float(np.max(np.abs(whole-tiled)))
        need(tile_error <= 1e-5, "Full/chunk equality including actual halo boundaries")
    removed=dict(native.state_dict());removed.pop("final_conv.2.conv.bias")
    try: DirectWave(removed)
    except AssertionError: rejected=True
    else: raise AssertionError("Incomplete state accepted")
    need(metrics(np.array([1.,2.,3.]),np.array([1.,3.,5.]))==dict(r=1.,MSE=5/3), "Independent metric arithmetic")
    print(json.dumps(dict(self_test_passed=True,author_forward_calls=4,
                         independent_forward_calls=COUNTS["independent_model_calls"],author_error_max=max(errors),
                         full_chunk_error_max=tile_error,incomplete_state_rejected=rejected,
                         formal_checkpoint_reads=0,formal_signal_payload_reads=0,new_fits=0,new_optimizer_updates=0,
                         source_sha256=digest(Path(__file__)))))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--self-test",action="store_true")
    cli.add_argument("--output",type=Path,default=ROOT/"coordinator_full_replay_v01")
    args=cli.parse_args()
    if args.self_test:
        self_test()
    else:
        configure()
        run(args.output)
