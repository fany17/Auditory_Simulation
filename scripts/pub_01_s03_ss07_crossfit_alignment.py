"""Run the fixed-weight D25 source-alignment contrast without fitting or selecting."""
import argparse
import csv
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pub_01_s03_ss07_crossfit_alignment_meta import (
    D19, D24, EPOCHS, FIRST_ACTIVE, ROOT, SEEDS, VARIANTS, load, sha,
)

CANDIDATE = ROOT/"ROUTING_AND_BUDGET_CANDIDATE.json"
GATE = ROOT/"FORMAL_GATE.json"
ROUTING_REVIEW = ROOT/"COORDINATOR_ROUTING_REVIEW.json"
ENGINEERING = ROOT/"ENGINEERING_COMPLETED.json"
D24_UP = D24/"input_v04"
D24_META = D24/"input_downstream_v06"
ROLE = "P_past_prediction"
SCALE_KEYS = tuple(f"standardized_{x}_{y}" for x in ("current","context") for y in ("mean","RMS"))
GIB = 1024**3


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, item):
    with path.open("x") as stream:
        json.dump(item, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stat(path):
    result = Path(path).stat()
    return {"bytes": result.st_size, "mtime_ns": result.st_mtime_ns}


def verified_source():
    candidate = load(CANDIDATE)
    require(all(sha(Path(path)) == h for path, h in candidate["sources"].items()), "Frozen metadata")
    route = Path(candidate["routing_manifest"])
    require(sha(route) == candidate["routing_manifest_sha256"], "Frozen VAL routing")
    return candidate, route


def named_budget(candidate):
    budget = dict(candidate["budget"])
    budget["upstream_encoder_fold_jobs"] = budget.pop("upstream_encoder_fold_batches")
    budget["upstream_encoder_minibatches"] = len(SEEDS)*sum(math.ceil(row["windows"]/64)
                                                          for row in candidate["folds"])
    return budget


def cuda_admit(cuda_uuid):
    require(cuda_uuid and cuda_uuid == os.environ.get("CUDA_VISIBLE_DEVICES"), "GPU UUID")
    used = subprocess.check_output(
        ["nvidia-smi","-i",cuda_uuid,"--query-gpu=memory.used","--format=csv,noheader,nounits"],
        text=True).strip()
    require(used.isdecimal() and int(used) <= 256, "Selected GPU is busy")


def review_weights(candidate, route):
    review = load(ROUTING_REVIEW)
    require(review["passed"] and review["candidate_sha256"] == sha(CANDIDATE)
            and review["routing_sha256"] == sha(route), "Independent routing review")
    return {row["path"]:row["sha256"] for row in review["weights"]}


def dependency_pins():
    result = {}
    for gate in (D24/"UPSTREAM_FORMAL_GATE_V04.json", D24/"DOWNSTREAM_FORMAL_GATE_V06.json"):
        data = load(gate)
        require(data["passed"], "D24 dependency gate")
        for path,value in data["sources"].items():
            require(result.setdefault(path,value) == value, "Contradictory transitive pin")
    require(all(sha(Path(path)) == value for path,value in result.items()), "Transitive source drift")
    return result


def validate_import_paths(pins):
    allowed = {Path(path).resolve() for path in pins}
    allowed.add(Path(__file__).with_name("pub_01_s03_ss07_crossfit_alignment_meta.py").resolve())
    for name,module in list(sys.modules.items()):
        if name.startswith("pub_01_") and getattr(module,"__file__",None):
            path = Path(module.__file__).resolve()
            require(path in allowed, f"Unpinned actual import: {name} {path}")


def freeze(cuda_uuid):
    candidate, route = verified_source()
    pinned_weights = review_weights(candidate,route)
    engineering = load(ENGINEERING)
    require(engineering["passed"] and engineering["source_sha256"] == sha(Path(__file__)), "Exact engineering source")
    writer = shutil.disk_usage(ROOT)
    require(writer.free >= 32*GIB, "SATA reserve")
    require(cuda_uuid.startswith("GPU-"), "Explicit single-GPU UUID")
    d24_gate = load(D24/"DOWNSTREAM_FORMAL_GATE_V06.json")
    require(d24_gate["passed"] and all(sha(Path(path)) == h for path, h in d24_gate["sources"].items()),
            "D24 frozen evidence drift")
    require(load(D24/"downstream_queue_v03/TERMINAL.json")["fits"] == 12, "Complete D24 downstream")
    scores, weights, val_inputs = {}, {}, {}
    for row in load(route)["records"]:
        require(row["phase"] == "val", "Original VAL only")
        for field in ("eeg_path", "envelope_path"):
            path = row[field]
            val_inputs[path] = stat(path)
    for row in candidate["weights"]:
        path = Path(row["path"])
        info = stat(path)
        require(info["bytes"] == row["bytes"], "Frozen fixed weight file size")
        digest = sha(path)
        require(pinned_weights[str(path)] == digest, "Independently pinned fixed checkpoint")
        weights[str(path)] = dict(**info, sha256=digest)
        scores[str(path.parents[1]/"RESULT.json")] = sha(path.parents[1]/"RESULT.json")
        stats = D24/"downstream_formal"/f"seed{row['seed']}"/"OOF_TRAIN_STATS.json"
        scores[str(stats)] = sha(stats)
    upstream = {}
    for seed in SEEDS:
        for fold in range(3):
            folder = D24/"formal"/f"seed{seed}_fold{fold}"
            for name in ("FT.pt", "READOUT.npz"):
                path = folder/name
                pin = d24_gate["upstream_payloads"][str(path)]
                require(stat(path) == {k: pin[k] for k in ("bytes","mtime_ns")} and sha(path) == pin["sha256"],
                        "D24 upstream payload identity")
                upstream[str(path)] = pin
            path = folder/"UPSTREAM_FROZEN.json"
            scores[str(path)] = sha(path)
    actual_sources = {
        Path(__file__).resolve(), Path(__file__).with_name("pub_01_s03_ss07_crossfit_alignment_meta.py"),
        D24_UP/"pub_01_s03_ss07_crossfit_upstream.py", D24_META/"pub_01_s03_ss07_crossfit_downstream.py",
        D24/"DOWNSTREAM_FORMAL_GATE_V06.json", CANDIDATE, route, ROUTING_REVIEW, ENGINEERING,
    }
    pins = {str(path):sha(path) for path in actual_sources}
    pins.update(dependency_pins())
    pins.update(scores)
    save(GATE, dict(candidate_sha256=sha(CANDIDATE), passed=True, cuda_uuid=cuda_uuid,
                    code_and_JSON=pins, OOF_fixed_weights=weights, upstream_payloads=upstream,
                    VAL_raw_stats=val_inputs, budget=named_budget(candidate),
                    authorized_scope="One existing VAL fixed-weight diagnostic only"))


def admit():
    candidate, route = verified_source()
    gate = load(GATE)
    require(gate["passed"] and gate["candidate_sha256"] == sha(CANDIDATE)
            and gate["budget"] == named_budget(candidate), "D25 formal gate")
    cuda_admit(gate["cuda_uuid"])
    require(all(sha(Path(path)) == h for path,h in gate["code_and_JSON"].items()), "Source/JSON drift")
    for path, pin in gate["VAL_raw_stats"].items():
        require(stat(path) == pin, "Frozen VAL metadata drift")
    for group in ("OOF_fixed_weights","upstream_payloads"):
        for path,pin in gate[group].items():
            require(stat(path) == {k:pin[k] for k in ("bytes","mtime_ns")} and sha(Path(path)) == pin["sha256"],
                    "Frozen weight/readout drift")
    return candidate, load(route)["records"], gate["code_and_JSON"]


def inference_modules():
    sys.path[:0] = [str(D24_UP),str(D24_META)]
    import pub_01_s03_ss07_crossfit_upstream as upstream
    import pub_01_s03_ss07_crossfit_downstream as downstream
    native, _, ft, _, _ = upstream.imported()
    core, legacy, d20, d22, d23 = downstream.get_modules()
    _, B, *_ = d22.adapters(d20,core,legacy)
    _, _, S, *_ = d23.adapters(d20,d22,core,legacy)
    return upstream, downstream, native, ft, core, {"B":B,"S":S}


def guard(torch):
    require(shutil.disk_usage(ROOT).free >= 32*GIB, "SATA free space")
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 12*GIB, "RSS cap")
    require(torch.cuda.max_memory_reserved() <= 8*GIB, "GPU cap")


def fold_records(seed, fold, rows, mask, targets, up, down, native, ft, core, counters, canary=False):
    np, torch = core.np, core.torch
    folder = D24/"formal"/f"seed{seed}_fold{fold}"
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    native_model = native.TransformerCLIP(load(up.NATIVE_CFG))
    model = ft.Decoder(native_model.eeg).cuda()
    del native_model
    model.load_state_dict(torch.load(folder/"FT.pt", map_location="cpu", weights_only=True), strict=True)
    with np.load(folder/"READOUT.npz", allow_pickle=False) as f:
        mean_eeg, std_eeg, coef, bias = [f[k].copy() for k in ("mean","std","coef","bias")]
        transforms = [{k:f[f"{k}_{i}"].copy() for k in ("mean","std","constant")} for i in range(3)]
    require((std_eeg > 0).all(), "Positive frozen EEG scale")
    affine = load(folder/"UPSTREAM_FROZEN.json")["affine"]
    require(affine["a"] > 0, "Positive frozen affine scale")
    eeg, labels = up.load_rows(rows, mean_eeg, std_eeg, labels=False)
    require(not labels, "No envelope access in encoder path")
    grid = up.Grid(rows,eeg)
    z,p = up.decode(model,grid)
    counters["encoder_windows"] += len(grid)
    counters["encoder_batches"] += math.ceil(len(grid)/64)
    records = []
    for row, latent, pred in up.blocks(rows,z,p):
        frames = core.anchor_frames(latent,pred,(transforms,coef,bias,affine))
        require(np.array_equal(frames[:4],affine["a"]*pred[:4].astype(np.float64)+affine["b"]), "Frozen fallback")
        wave, keep = core.legacy.fuse(frames,row["common_samples"])
        stop = row["covered_stop"]
        require(not keep[0] and keep[1:stop].all() and not keep[stop:].any(), "Frozen Hann coordinates")
        if canary:
            hann = np.hanning(192)
            for point in (1,FIRST_ACTIVE,stop-1):
                window = [i for i in range(row["windows"]) if 0 <= point-i*96 < 192]
                weights = [hann[point-i*96] for i in window]
                manual = sum(frames[i,point-i*96]*w for i,w in zip(window,weights))/sum(weights)
                require(abs(wave[point]-manual) < 1e-12,"Independent three-point Hann bridge")
                counters["Hann_bridge_points"] += 1
        records.append(down.ScalarRecord(row,wave,targets[row["record"]],mask[row["record"]]))
    guard(torch)
    return records


def models(seed, variant_types, candidate, torch):
    result = {}
    for row in candidate["weights"]:
        if row["seed"] != seed:
            continue
        variant, epoch = row["variant"],row["epoch"]
        checkpoint = torch.load(row["path"], map_location="cpu", weights_only=True)
        require(checkpoint["seed"] == seed and checkpoint["epoch"] == epoch
                and checkpoint["role"] == ROLE, "Actual fixed epoch and role")
        model = variant_types[variant](seed,ROLE).cuda()
        model.load_state_dict(checkpoint["model"],strict=True)
        model.eval()
        result[variant,epoch] = model
    require(len(result) == 4,"Complete fixed model matrix per seed")
    return result


def reduce_metrics(by_record, route, core):
    merged, sum_sse, residual_sse, points = [], 0., 0., 0
    for row in route:
        records = by_record[row["record"]]
        require(sorted(records) == row["folds"], "Exactly the allowed fold scores")
        rows = [records[fold] for fold in row["folds"]]
        base = rows[0]["blocks"]
        require(len(base) == row["VAL_blocks"], "Original block count")
        for i, block in enumerate(base):
            peers = [item["blocks"][i] for item in rows]
            require(all((item["record"],item["start"],item["samples"]) ==
                        (block["record"],block["start"],block["samples"]) for item in peers),
                    "Identical fold scoring coordinates")
            merged.append(dict(block,r=mean(item["r"] for item in peers),
                               MSE=mean(item["MSE"] for item in peers)))
        require(all(item["active_points"] == row["active_points"] for item in rows), "Active coordinate count")
        points += row["active_points"]
        sum_sse += mean(item["active_MSE"] for item in rows)*row["active_points"]
        residual_sse += mean(item["residual_RMS"]**2 for item in rows)*row["active_points"]
    res = core.aggregate_blocks(merged)
    return dict(r=res["r"], MSE=res["MSE"], people=len(res["people"]),
                blocks=len(res["blocks"]), active_points=points, active_MSE=sum_sse/points,
                residual_RMS=math.sqrt(residual_sse/points))


def shared_inputs(core, route):
    np = core.np
    mask = defaultdict(list)
    meta = load(D19/"PREPARATION_METADATA.json")["phases"]["val"]
    for block in meta["blocks"]:
        mask[block["record"]].append(block)
    targets = {r["record"]: np.load(r["envelope_path"],mmap_mode="r",allow_pickle=False).reshape(-1)
                             [:r["common_samples"]].astype(np.float64) for r in route}
    return mask, targets


def trained_stats(seed, torch):
    info = load(D24/"downstream_formal"/f"seed{seed}"/"OOF_TRAIN_STATS.json")
    require(not info["constant"] and info["std"] > 0, "Frozen OOF statistics")
    normal = tuple(torch.tensor(info[k],dtype=dtype,device="cuda")
                   for k,dtype in (("mean",torch.float32),("std",torch.float32),("constant",torch.bool)))
    return {"current":tuple(x.reshape(1) for x in normal),ROLE:normal}


class ObserveInput:
    """Measure standardized input already computed by the fixed evaluation path."""
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.values = {name: [0, torch.zeros((),device="cuda",dtype=torch.float64),
                              torch.zeros((),device="cuda",dtype=torch.float64)]
                       for name in ("current","context")}

    def add(self, name, value):
        result = self.values[name]
        result[0] += value.numel()
        result[1] += value.sum(dtype=self.torch.float64)
        result[2] += value.square().sum(dtype=self.torch.float64)

    def current_code(self, value):
        self.add("current", value)
        return self.model.current_code(value)

    def decode(self, code, value):
        self.add("context", value)
        return self.model.decode(code,value)

    def receipt(self):
        return {f"standardized_{name}_{key}": value
                for name,(n,total,squares) in self.values.items()
                for key,value in (("mean",float(total)/n),("RMS",math.sqrt(float(squares)/n)))}


def model_digest(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def engineer():
    wall,cpu = time.monotonic(),time.process_time()
    candidate,path = verified_source()
    pins = review_weights(candidate,path)
    cuda_admit(os.environ.get("CUDA_VISIBLE_DEVICES"))
    route = load(path)["records"]
    first = next(row for row in route if row["shared_train_family"])
    require(first["folds"] == [1], "Shared canary route")
    chosen = [first]
    seed, fold = SEEDS[0], 1
    d24_gate = load(D24/"DOWNSTREAM_FORMAL_GATE_V06.json")
    for name in ("FT.pt","READOUT.npz"):
        file = D24/"formal"/f"seed{seed}_fold{fold}"/name
        require(sha(file) == d24_gate["upstream_payloads"][str(file)]["sha256"], "Actual canary upstream")
    up,down,native,ft,core,variants = inference_modules()
    validate_import_paths(dependency_pins())
    torch = core.torch
    for row in candidate["weights"]:
        require(sha(Path(row["path"])) == pins[row["path"]], "Actual fixed downstream canary")
        checkpoint = torch.load(row["path"],map_location="cpu",weights_only=True)
        require((checkpoint["seed"],checkpoint["epoch"],checkpoint["role"]) ==
                (row["seed"],row["epoch"],ROLE), "Actual seed/epoch/role for all 12 fixed checkpoints")
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    mask, targets = shared_inputs(core,chosen)
    counters = defaultdict(int)
    records = fold_records(seed,fold,chosen,mask,targets,up,down,native,ft,core,counters,canary=True)
    record = records[0]
    weights = models(seed,variants,candidate,torch)
    initial = {key:model_digest(value) for key,value in weights.items()}
    stats = trained_stats(seed,torch)
    table_type = down.tables(core)
    tables = table_type(records,"B","cuda:0",lambda:guard(torch))
    torch.use_deterministic_algorithms(True)
    anchor = core.evaluate(records,tables,stats,None,ROLE,lambda:None,lambda *_:None)["near"]
    base = reduce_metrics({first["record"]:{1:anchor}},chosen,core)
    require(base["r"] == anchor["r"] and base["blocks"] == first["VAL_blocks"], "Single route preserved")
    adjusted = {}
    for fold_index,delta in enumerate((-.03,.02,.04)):
        adjusted[fold_index] = dict(anchor,blocks=[dict(block,r=block["r"]+delta)
                                                    for block in anchor["blocks"]])
    sim = reduce_metrics({first["record"]:adjusted},[dict(first,folds=[0,1,2])],core)
    require(abs(sim["r"]-anchor["r"]-.01) < 1e-12 and sim["people"] == 1,
            "Independent score average before subject aggregation")
    def count(key, amount):
        counters[key] += amount
    for (variant,epoch),model in weights.items():
        tables.variant = variant
        observed = ObserveInput(model,torch) if epoch == 0 else model
        metric = core.evaluate(records,tables,stats,observed,ROLE,lambda:None,count)["near"]
        require(metric["active_points"] == first["active_points"] and len(metric["blocks"]) == base["blocks"],
                "Shared fixed score mask")
        if epoch == 0:
            require(all(math.isfinite(x) for x in observed.receipt().values()),"Finite scale diagnostic")
    require(initial == {key:model_digest(value) for key,value in weights.items()},"Fixed states unchanged")
    require(counters["current_stem_points"] == counters["context_output_points"] == 4*first["active_points"],
            "Real engineering activity counters")
    save(ENGINEERING,dict(passed=True, source_sha256=sha(Path(__file__)), pid=os.getpid(),
                          one_shared_VAL_record=first["record"], fold=fold, actual_fixed_models=4,
                          all_12_checkpoint_identities_checked=True,
                          records_and_mask_verified=True, weights_unchanged=True,
                          synthetic_three_route_score_average=True, counters=dict(counters),
                          downstream_active_points=4*first["active_points"],
                          wall_seconds=time.monotonic()-wall, CPU_seconds=time.process_time()-cpu,
                          new_optimizer_updates=0, new_statistics_fits=0, VAL_labels=1, old_test_reads=0))


def run():
    started, cpu = time.monotonic(), time.process_time()
    candidate, route, pins = admit()
    up, down, native, ft, core, variants = inference_modules()
    validate_import_paths(pins)
    np, torch = core.np, core.torch
    require(torch.cuda.device_count() == 1 and core.FIRST_ACTIVE == FIRST_ACTIVE,"Frozen physical axis")
    torch.cuda.set_device(0)
    torch.set_num_threads(8)
    torch.cuda.reset_peak_memory_stats()
    table_type = down.tables(core)
    mask, targets = shared_inputs(core,route)
    counters = defaultdict(int)
    aggregates, record_scores, block_scores = [], [], []
    def count(key, amount):
        counters[key] += amount
    identity = up.identity()
    save(ROOT/"PROCESS.json",identity)
    for seed in SEEDS:
        scores = defaultdict(lambda:defaultdict(dict))
        weights = models(seed,variants,candidate,torch)
        stats = trained_stats(seed,torch)
        for fold in range(3):
            eligible = [r for r in route if fold in r["folds"]]
            records = fold_records(seed,fold,eligible,mask,targets,up,down,native,ft,core,counters)
            torch.use_deterministic_algorithms(True)
            for record in records:
                key = record.metadata["record"]
                tables = table_type([record],"B","cuda:0",lambda:guard(torch))
                evaluations = [(("anchor",None),None),*list(weights.items())]
                for (variant,epoch),model in evaluations:
                    tables.variant = variant
                    observe = ObserveInput(model,torch) if model is not None and epoch == 0 else None
                    result = core.evaluate([record],tables,stats,observe or model,ROLE,lambda:None,count)["near"]
                    scores[variant,epoch][key][fold] = result
                    record_scores.append(dict(seed=seed, fold=fold, variant=variant, epoch=epoch,
                        subject=record.metadata["subject"], record=key,
                        family=record.metadata["source_family"], r=result["r"], MSE=result["MSE"],
                        active_MSE=result["active_MSE"], residual_RMS=result["residual_RMS"],
                        active_points=result["active_points"],
                        **(observe.receipt() if observe is not None else dict.fromkeys(SCALE_KEYS))))
                    for block in result["blocks"]:
                        block_scores.append(dict(seed=seed,fold=fold,variant=variant,epoch=epoch,
                            record=key,subject=block["subject"],start=block["start"],samples=block["samples"],
                            r=block["r"],MSE=block["MSE"]))
                    if model is not None:
                        counters["downstream_active_points"] += result["active_points"]
                    else:
                        counters["anchor_score_points"] += result["active_points"]
            counters["logical_fold_readout_routes"] += len(weights)
            guard(torch)
        for group, rows in (("all",route),
                ("shared",[r for r in route if r["shared_train_family"]]),
                ("unseen",[r for r in route if not r["shared_train_family"]])):
            anchor = reduce_metrics(scores["anchor",None],rows,core)
            for variant,epoch in weights:
                folded = reduce_metrics(scores[variant,epoch],rows,core)
                fixed = next(row for row in candidate["weights"] if
                             (row["seed"],row["variant"],row["epoch"]) == (seed,variant,epoch))
                full_delta = fixed["full_model_VAL_score"]-fixed["full_anchor_VAL_score"]
                difference = folded["r"]-anchor["r"]
                aggregates.append(dict(seed=seed, group=group,variant=variant,epoch=epoch,
                    fold_anchor_r=anchor["r"],fold_anchor_active_MSE=anchor["active_MSE"],
                    fold_model_r=folded["r"],fold_model_active_MSE=folded["active_MSE"],
                    residual_RMS=folded["residual_RMS"], full_anchor_r=fixed["full_anchor_VAL_score"] if group=="all" else None,
                    fold_delta_r=difference,full_r=fixed["full_model_VAL_score"] if group=="all" else None,
                    full_delta_r=full_delta if group=="all" else None,
                    delta_of_deltas=difference-full_delta if group=="all" else None,
                    people=folded["people"],blocks=folded["blocks"],active_points=folded["active_points"]))
    for key in ("encoder_windows","downstream_active_points","logical_fold_readout_routes"):
        want = {"encoder_windows":"upstream_encoder_windows"}.get(key,key)
        require(counters[key] == candidate["budget"][want],f"Exact executed budget {key}")
    require(counters["encoder_batches"] == named_budget(candidate)["upstream_encoder_minibatches"] and len(aggregates) == 36
            and len(record_scores) == 1335 and len(block_scores) == 20970,"Executed matrix and routing count")
    require(counters["current_stem_points"] == counters["context_output_points"] == 64189524,
            "Actual production current/context call counts")
    for name, rows in (("RESULTS.csv",aggregates),("RECORD_SCORES.csv",record_scores),("BLOCK_SCORES.csv",block_scores)):
        with (ROOT/name).open("x",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save(ROOT/"COMPLETED.json",dict(passed=True, process=identity, gate_sha256=sha(GATE),
        counters=dict(counters), fixed_epochs=EPOCHS, scores=len(aggregates), record_scores=len(record_scores),
        block_scores=len(block_scores), no_fit=True,
        new_optimizer_updates=0, new_statistics_fits=0, VAL_labels=len(targets), old_test_reads=0,
        wall_seconds=time.monotonic()-started, CPU_seconds=time.process_time()-cpu,
        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("engineer","freeze","run"))
    parser.add_argument("--cuda-uuid")
    args = parser.parse_args()
    if args.stage == "engineer":
        try:
            engineer()
        except BaseException as error:
            save(ROOT/"ENGINEERING_FAILURE.json",dict(pid=os.getpid(),message=str(error),traceback=traceback.format_exc()))
            raise
    elif args.stage == "freeze":
        require(args.cuda_uuid,"Supply the intended CUDA UUID")
        freeze(args.cuda_uuid)
    else:
        try:
            run()
        except BaseException as error:
            save(ROOT/"FAILURE.json",dict(pid=os.getpid(), message=str(error), traceback=traceback.format_exc()))
            raise
