"""Independent D27 final saved-evidence audit; run on 2203, stdlib only.

Does not import the training/report implementation, decode weights, run models,
read EEG/audio payloads, fit anything, or infer SOTA from development scores.
Final execution requires the complete 150-fit matrix and successor report.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_seawave_seen_reference_20261106")
GATE = "8442767e6fb1a220bab129d2c2163321e760610102fdefb986f43a06079f7476"
RESOURCE_GATE = "9bce22984751e21fd8980f3a1032a2e1d64d39b01a5a41d11820906883300c75"
RESOURCE_REVIEW = "17a03102af3b7ef9a729c2ea9fdf23d5b10caaa56d139489adc3cdf8f6a369b7"
REPORT_REVIEW = "9c0831201ccf120d69ffccc503e91cc2712bf81536704856217d68303e0560ec"
SEEDS = (1024, 2026, 31415)
METRICS = ("r", "MSE")
PARAMETERS = 150433
TRAIN_PY = "/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python"
HASHES, CHECKS = {}, Counter()


def need(ok, label):
    if not ok:
        raise AssertionError(label)
    CHECKS["assertions"] += 1


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    HASHES[str(path)] = h.hexdigest()
    return h.hexdigest()


def read(path):
    data = path.read_bytes()
    HASHES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def mean(values):
    values = list(values)
    need(bool(values), "Nonempty mean")
    return math.fsum(values) / len(values)


def compare(expected, observed, label):
    if isinstance(expected, dict):
        need(isinstance(observed, dict), label + " mapping")
        for key, value in expected.items():
            need(key in observed, label + "/" + key + " present")
            compare(value, observed[key], label + "/" + key)
    elif isinstance(expected, list):
        need(isinstance(observed, list) and len(expected) == len(observed), label + " list")
        for a, b in zip(expected, observed):
            compare(a, b, label)
    elif expected is None:
        need(observed is None or observed == "", label)
    elif isinstance(expected, bool):
        need(observed == expected or observed == str(expected), label)
    elif isinstance(expected, (int, float)):
        need(not isinstance(observed, bool) and math.isfinite(float(observed))
             and abs(expected - float(observed)) <= 1e-11 * max(1., abs(expected)), label)
    else:
        need(expected == observed, label)


def index(rows, keys):
    def cell(value):
        return "" if value is None else str(value)
    result = {tuple(cell(row[k]) for k in keys): row for row in rows}
    need(len(result) == len(rows), "Unique row keys " + str(keys))
    return result


def table(report, name, rows, keys):
    path = report / (name + ".csv")
    digest(path)
    with path.open(newline="") as stream:
        observed = index(list(csv.DictReader(stream)), keys)
    expected = index(rows, keys)
    need(expected.keys() == observed.keys(), name + " exact complete keys")
    for k, row in expected.items():
        compare(row, observed[k], name + str(k))
    CHECKS[name + "_rows"] = len(rows)


def people(blocks):
    groups = defaultdict(list)
    for block in blocks:
        groups[block["subject"]].append(block)
    return [dict(subject=p, blocks=len(bs), **{m: mean(b[m] for b in bs) for m in METRICS})
            for p, bs in sorted(groups.items())]


def macro(blocks):
    rows = people(blocks)
    return dict(people=len(rows), blocks=len(blocks), records=len({b["record"] for b in blocks}),
                **{m: mean(p[m] for p in rows) for m in METRICS})


def blockset(saved, roster, eligible):
    points = []
    keys = set()
    for row in saved:
        k = row["record"], row["start"]
        need(k in eligible and k not in keys, "Original block key, no duplicates")
        keys.add(k)
        meta = roster[k[0]]
        need(row["subject"] == meta["subject"] and row["source_family"] == meta["source_family"],
             "Actual participant and source family")
        need(row["samples"] == eligible[k], "Exact prediction-independent sample support")
        need(all(math.isfinite(row[m]) for m in METRICS)
             and -1.000000000001 <= row["r"] <= 1.000000000001 and row["MSE"] >= 0, "Valid metrics")
        points.append(dict(row, story=meta["story"], phase="val"))
    need(keys == set(eligible), "Complete expected block set")
    return points


def score(saved, roster, eligible):
    blocks = blockset(saved["blocks"], roster, eligible)
    compare({m: macro(blocks)[m] for m in METRICS}, saved, "Saved score macro")
    actual = index(saved["people"], ("subject",))
    expected = index(people(blocks), ("subject",))
    need(actual.keys() == expected.keys(), "All saved participants")
    for key, row in expected.items():
        compare(row, actual[key], "Saved participant")
    return blocks


def strata(blocks, families):
    groups = defaultdict(list)
    for row in blocks:
        for key in (("all", "all"), ("story", row["story"]),
                    ("family", "shared" if row["source_family"] in families else "unseen"),
                    ("exact_source_family", row["source_family"])):
            groups[key].append(row)
    return [dict(stratum=k[0], label=k[1], **macro(rows)) for k, rows in sorted(groups.items())]


def geometry(train, val, spec):
    frame, hop, batch = (spec[k] for k in ("frame", "hop", "batch"))
    grids = [range(0, r["common_samples"] - frame + 1, hop) for r in train]
    need(all(len(g) for g in grids) and 0 < hop <= frame, "Overlapping legal record grids")
    windows = sum(map(len, grids))
    unique = sum(g[-1] + frame for g in grids)
    training = dict(records=len(train), windows=windows, updates=(windows + batch - 1) // batch,
                    tail=(windows - 1) % batch + 1, unique_record_sample_times=unique,
                    target_sample_presentations=windows * frame,
                    repeated_target_sample_presentations=windows * frame - unique)
    sizes = []
    for row in val:
        n = row["common_samples"]
        for core in range(0, n, 4096):
            sizes.append(min(n, core + 4096 + 124) - max(0, core - 124))
    validation = dict(unique=sum(r["common_samples"] for r in val), calls=len(sizes), presented=sum(sizes))
    return training, validation


def exposure(g, v, epochs, personal):
    passes = epochs + int(personal)
    displayed = g["target_sample_presentations"] * epochs
    distinct = g["unique_record_sample_times"] if epochs else 0
    return dict(training_epochs=epochs, optimizer_updates=g["updates"] * epochs,
                training_window_presentations=g["windows"] * epochs,
                training_EEG_and_target_sample_time_presentations_each=displayed,
                training_distinct_record_sample_times=distinct,
                training_repeated_sample_time_presentations=displayed - distinct,
                validation_passes=passes, validation_calls=v["calls"] * passes,
                validation_EEG_sample_time_presentations=v["presented"] * passes,
                validation_distinct_record_sample_times=v["unique"],
                validation_halo_repeat_sample_times=(v["presented"] - v["unique"]) * passes,
                validation_total_repeat_sample_times=v["presented"] * passes - v["unique"])


def select(history, incumbent, patience):
    best = incumbent
    stale = 0
    for i, row in enumerate(history):
        need(row["epoch"] == i and math.isfinite(row["r"]), "Sequential finite epoch")
        if best is None or row["r"] > best["r"]:
            best, stale = row, 0
        else:
            stale += 1
        need(row["stale"] == stale, "Actual early-tie selection/stale count")
        if patience is not None and i < len(history) - 1:
            need(stale < patience, "No epochs after first patience boundary")
    need(best is not None, "A selected model exists")
    return best, stale


def process_key(row):
    return tuple(row[k] for k in ("pid", "uid", "startticks"))


def still_live(key):
    try:
        p = Path("/proc") / str(key[0])
        fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
        return (key[0], p.stat().st_uid, int(fields[19])) == tuple(key) and fields[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


def audit_fit(folder, seed, subject, train, val, specs, frozen_geometry, initial_sha):
    stage = "initial" if subject is None else "personal"
    personal = subject is not None
    end, selected = read(folder / "COMPLETED.json"), read(folder / "SELECTED_VAL.json")
    compare(dict(passed=True, seed=seed, subject=subject, stage=stage, initial_sha256=initial_sha,
                 parameters=PARAMETERS, old_test_reads=0, population_global_FT=False), end, "Fit identity")
    g, v = geometry(train, val, specs)
    compare(g, end["geometry"], "Actual geometry")
    need(g == frozen_geometry, "Frozen geometry")
    n = end["complete_epochs"]
    need(isinstance(n, int) and 1 <= n <= specs["epochs"] and (personal or n == 201), "Full epoch budget")
    need({f.name for f in folder.glob("EPOCH_[0-9][0-9][0-9].json")}
         == {f"EPOCH_{i:03}.json" for i in range(n)}, "Every actual epoch receipt")
    history = [read(folder / f"EPOCH_{i:03}.json") for i in range(n)]
    incumbent = read(folder / "EPOCH_MINUS1.json") if personal else None
    if personal:
        compare(dict(epoch=-1, initial_sha256=initial_sha, updates=0), incumbent, "Eligible original incumbent")
    best, stale = select(history, incumbent, specs.get("patience"))
    need(not personal or n == specs["epochs"] or stale >= specs["patience"], "Bounded stop reached")
    for i, row in enumerate(history):
        compare(dict(updates=(i + 1) * g["updates"], windows=g["windows"],
                     VAL_model_calls=v["calls"], VAL_model_input_sample_times=v["presented"]),
                row, "Actual epoch work")
    actual, prefix = exposure(g, v, n, personal), exposure(g, v, best["epoch"] + 1, personal)
    compare(dict(selected_epoch=best["epoch"], selected_r=best["r"], actual_updates=actual["optimizer_updates"],
                 selected_prefix_updates=prefix["optimizer_updates"], windows=actual["training_window_presentations"],
                 VAL_model_calls=actual["validation_calls"],
                 VAL_model_input_sample_times=actual["validation_EEG_sample_time_presentations"]),
            end, "Final fit work and model selection")
    compare({m: best[m] for m in METRICS}, selected, "Selected actual epoch metrics")
    compare(dict(VAL_model_calls=v["calls"], VAL_model_input_sample_times=v["presented"],
                 VAL_unique_sample_times=v["unique"]), selected, "Whole-record selected inference")
    need(digest(folder / "SELECTED.pt") == end["selected_sha256"], "Actual selected weight bytes")
    CHECKS["checkpoint_bytes_hashed"] += (folder / "SELECTED.pt").stat().st_size
    need(all(math.isfinite(end[k]) and end[k] >= 0 for k in ("CPU_seconds", "elapsed_seconds", "peak_cuda_reserved")),
         "Nonnegative measured fit costs")
    ids = dict(seed=seed, stage=stage, subject=subject)
    cost = dict(**ids, selected_epoch=best["epoch"], selected_r=selected["r"], selected_MSE=selected["MSE"],
                initial_sha256=initial_sha, selected_sha256=end["selected_sha256"], parameters=PARAMETERS,
                checkpoint_selected_bytes=(folder / "SELECTED.pt").stat().st_size,
                checkpoint_final_bytes=(folder / "FINAL.pt").stat().st_size,
                peak_cuda_reserved=end["peak_cuda_reserved"], process=end["process"],
                committed_CPU_seconds=end["CPU_seconds"], committed_elapsed_seconds=end["elapsed_seconds"],
                stopped="epoch_cap" if n == specs["epochs"] else "personal_patience",
                training_geometry=g, actual=actual, used_prefix=prefix)
    CHECKS["fits"] += 1
    return end, selected, cost, [dict(**ids, **r) for r in ([incumbent] if personal else []) + history]


def audit_workers(formal, resource, fits, summary):
    ledger = summary["process_ledger"]
    expected = set(fits)
    registrations = [p for a in sorted(formal.glob("ATTEMPT_*")) for p in sorted(a.glob("*_PROCESS.json"))]
    need(len(registrations) == len(expected) == 150, "Exactly original 150 workers, no missing attempt")
    represented = index(ledger["workers"], ("seed", "stage", "subject"))
    need(len(represented) == 150, "Full worker ledger")
    attempts = [dict(attempt=a.name, process=read(a / "PROCESS.json"),
                     spawned=len(list(a.glob("*_PROCESS.json")))) for a in sorted(formal.glob("ATTEMPT_*"))]
    compare(attempts, ledger["attempts"], "Every actual attempt registration")
    wall, cpu, starts, ends, seen = [], [], [], [], set()
    for path in registrations:
        reg = read(path)
        argv = reg["argv"]
        seed = int(argv[argv.index("--seed") + 1])
        stage = argv[3]
        subject = argv[argv.index("--subject") + 1] if "--subject" in argv else None
        key = seed, stage, subject
        need(key in expected and key not in seen, "One real worker per actual fit")
        seen.add(key)
        required = [TRAIN_PY, "-B", str(ROOT / "input_v04/pub_01_s03_ss07_seawave_seen_train.py"),
                    stage, "--seed", str(seed)] + (["--subject", subject] if subject else [])
        need(argv == required and process_key(reg) == process_key(fits[key]["process"]), "Exact native worker registration")
        need(not still_live(process_key(reg)), "Worker actually ended")
        ex = read(path.with_name(path.name.replace("_PROCESS.json", "_EXIT.json")))
        need(ex["pid"] == reg["pid"] and ex["returncode"] == 0 and ex["complete"], "True clean exit")
        events = [json.loads(line) for line in path.with_name(path.name.replace("_PROCESS.json", ".log")).read_text().splitlines()
                  if line.startswith("{")]
        timings = [r for r in events if r.get("event") == "worker_timing"]
        need(len(timings) == 1 and process_key(timings[0]["process"]) == process_key(reg), "Unique actual final worker timing")
        timing = timings[0]
        last = [e["updates"] for e in events if "epoch" in e and "updates" in e][-1]
        compare(dict(process=reg, exit=ex, timing=timing, success=True,
                     last_logged_cumulative_updates=last, uncommitted_extra_updates=0),
                represented[str(seed), stage, subject or ""], "Reported true worker")
        need(last == fits[key]["actual_updates"] and ex["worker_wall_seconds_observed"] > 0
             and timing["CPU_seconds"] >= 0, "Worker duration and final updates")
        if path.parent.name == "ATTEMPT_000":
            proof = read(Path(ex["independent_exit_receipt"]))
            st = proof["proc_state"]
            need(proof["proc_stat_field"] == 52 and st["state"] == "Z"
                 and st["ppid"] == 1748466 and process_key(st) == process_key(reg)
                 and os.waitstatus_to_exitcode(st["wait_status_field52"]) == ex["returncode"] == proof["actual_returncode"],
                 "Adopted original kernel exit, no invented Popen wait")
            need(proof["completed_receipt_sha256"] == digest(formal / f"seed{seed}" / "initial/COMPLETED.json"),
                 "Adopted original completion binding")
        wall.append(ex["worker_wall_seconds_observed"])
        cpu.append(timing["CPU_seconds"])
        starts.append(reg["started_monotonic"])
        ends.append(reg["started_monotonic"] + ex["worker_wall_seconds_observed"])
    need(seen == expected, "All original jobs covered")
    compare(dict(completed_fit_identity_count=150, successful_case_count=150, worker_attempt_count=150,
                 completed_cases_without_clean_exit=[], failures=[], actual_uncommitted_failure_work=0,
                 workers_without_CPU_timing=0, elapsed_sum_not_parallel_wall=True,
                 observed_worker_wall_seconds_sum=math.fsum(wall), reported_worker_CPU_seconds_sum=math.fsum(cpu)),
            ledger, "Complete worker ledger sums")
    failures = list(ROOT.glob("FAILURE_*.json")) + list(formal.glob("ATTEMPT_*/FAILURE.json")) + list(resource.glob("*FAILURE*.json"))
    need(not failures, "No undisclosed actual failure receipts")
    return dict(worker_wall_sum=math.fsum(wall), worker_CPU_sum=math.fsum(cpu),
                observed_first_launch_to_last_exit=max(ends)-min(starts), worker_attempts=150)


def main():
    cpu, wall = time.process_time(), time.monotonic()
    formal, report, resource = ROOT / "formal", ROOT / "report01", ROOT / "resource_concurrency_v01"
    # Refuse unfinished runs before any scores/checkpoints can be inspected.
    need((formal / "COMPLETED.json").is_file() and (report / "REPORT_ARTIFACTS.json").is_file()
         and (resource / "REPORT_WAITER_COMPLETED.json").is_file(), "Full matrix and final report not complete")
    protocol, terminal = read(ROOT / "PROTOCOL_V04.json"), read(formal / "COMPLETED.json")
    need(digest(ROOT / "PROTOCOL_V04.json") == GATE and protocol["frozen"], "Original science protocol")
    for name, expected in protocol["sources"].items():
        need(digest(Path(name)) == expected, "Frozen science source " + name)
    inv = read(ROOT / "INVENTORY.json")
    need(digest(ROOT / "INVENTORY.json") == protocol["inventory_sha256"], "Original input inventory")
    need(len(inv["inputs"]) == 522, "Complete original physical input pins")
    for filename, pin in inv["inputs"].items():
        st = Path(filename).stat()
        need(pin == dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns), "Input stat identity " + filename)
    need(digest(Path(inv["source_inventory"])) == inv["source_inventory_sha256"], "Original parent inventory")
    maskpath = Path(protocol["VAL_mask"])
    mask = read(maskpath)
    need(mask["prediction_independent"] and digest(maskpath) == protocol["sources"][str(maskpath)], "Frozen mask")
    need(terminal["passed"] and terminal["protocol_sha256"] == GATE and terminal["old_test_reads"] == 0
         and terminal["initial_fits"] == 3 and terminal["personal_fits"] == 147, "Complete original scientific matrix")
    need(digest(resource / "RESOURCE_PROTOCOL_V01.json") == RESOURCE_GATE, "Frozen resource amendment")
    operational = read(resource / "RESOURCE_PROTOCOL_V01.json")
    for name, expected in operational["sources"].items():
        need(digest(resource / name) == expected, "Frozen resource source")
    approval = read(resource / "RESOURCE_EXECUTION_REVIEW.json")
    need(digest(resource / "RESOURCE_EXECUTION_REVIEW.json") == RESOURCE_REVIEW
         and approval["passed"] and approval["resource_protocol_sha256"] == RESOURCE_GATE, "Approved resource amendment")
    lease, resource_end = read(resource / "LEASE.json"), read(resource / "TERMINAL.json")
    need(tuple(lease["process"]) == process_key(terminal["process"]) == tuple(resource_end["process"])
         and not still_live(tuple(lease["process"])), "Actual finished successor identity")
    need(resource_end["formal_completed_sha256"] == digest(formal / "COMPLETED.json")
         and resource_end["resource_protocol_sha256"] == RESOURCE_GATE, "Successor scientific terminal binding")
    old_parent, old_report = read(resource / "OLD_QUEUE_RETIRED.json"), read(resource / "OLD_REPORT_SUPERSEDED.json")
    for row, expected in ((old_parent, [1748466,1012,791000426]), (old_report, [1812007,1012,791424876])):
        need(row["old_process"] == expected and row["observed_exited"] and not still_live(expected), "Exact old process gone")
    need(old_parent["exit_code"] is None and old_report["old_exit_code"] is None, "Unknown coordinator exit codes retained")
    reporter = read(resource / "REPORT_WAITER_COMPLETED.json")
    need(reporter["upstream"] == lease["process"] and reporter["resource_protocol_sha256"] == RESOURCE_GATE
         and old_report["replacement_process"] == reporter["process"] and not still_live(reporter["process"]),
         "Actual completed successor report")
    manifest = read(report / "REPORT_ARTIFACTS.json")
    need(manifest["passed"] and manifest["formal_completed_sha256"] == digest(formal / "COMPLETED.json")
         and reporter["report_artifacts_sha256"] == digest(report / "REPORT_ARTIFACTS.json"), "Report terminal chain")
    review = read(ROOT / "COORDINATOR_REPORT_SOURCE_REVIEW_V02.json")
    need(digest(ROOT / "COORDINATOR_REPORT_SOURCE_REVIEW_V02.json") == REPORT_REVIEW, "Pinned independent report review")
    need(manifest["sources"] == review["sources"], "Reviewed final reporting sources")
    need(manifest["D26_helper_hashes"] == review["reviewed_common_support_helpers"], "Reviewed common-support helpers")
    for name, expected in review["reviewed_common_support_helpers"].items():
        need(digest(ROOT.parent / "s03_contrastive_history_20261106/report_source_v03" / name) == expected,
             "Actual pinned common-support helper")
    for name, expected in manifest["artifacts"].items():
        need(digest(report / name) == expected, "Actual reported artifact " + name)
    for name, expected in review["sources"].items():
        need(digest(ROOT / "report_source_v02" / name) == expected, "Actual report source " + name)
    summary, saved_costs = read(report / "RESULT_SUMMARY.json"), read(report / "FIT_COSTS.json")
    train, val = inv["phase_records"]["train"], inv["phase_records"]["val"]
    roster = {r["record"]:r for r in val}
    eligible = {(r["record"],r["start"]):r["samples"] for r in mask["blocks"] if r["eligible"]}
    families = {r["source_family"] for r in train}
    need(len(train)==426 and len(val)==len(roster)==49 and len(eligible)==752 and mask["records"]==list(roster),
         "Original cohort and exact block roster")
    need(len({r["subject"] for r in val})==49 and {r["subject"] for r in val}<={r["subject"] for r in train},
         "All VAL people TRAIN seen, one record each")
    need(Counter(r["story"] for r in val)=={"audiobook_13_2":6,"audiobook_5_2":23,"podcast_15":4,"podcast_21":7,"podcast_30":9}
         and sum(r["source_family"] in families for r in val)==29, "Five stories and original 29/20 families")
    by_subject = {r["subject"]:r for r in val}
    need(set(inv["subjects"])==set(by_subject), "Exact personal branches")
    fits, costs, histories, blocks_out, persons_out, strata_out, seed_scores, pairs = {}, [], [], [], [], [], [], []
    native = index(terminal["results"], ("seed",))
    need(set(native)=={(str(s),) for s in SEEDS}, "All three seed terminal rows")
    for seed in SEEDS:
        base = formal / f"seed{seed}"
        end, selected, cost, history = audit_fit(base/"initial", seed, None, train, val,
                protocol["settings"]["initial"], inv["initial"], None)
        fits[seed,"initial",None] = end; costs.append(cost); histories.extend(history)
        initial = score(selected, roster, eligible)
        initial_people = {r["subject"]:r for r in people(initial)}
        personal = []
        for subject in inv["subjects"]:
            own_val = [by_subject[subject]]
            own_train = [r for r in train if r["subject"]==subject]
            end, selected, cost, history = audit_fit(base/"personal"/subject, seed, subject, own_train, own_val,
                    protocol["settings"]["personal"], inv["personal"][subject], fits[seed,"initial",None]["selected_sha256"])
            fits[seed,"personal",subject] = end; costs.append(cost); histories.extend(history)
            ownmask = {k:n for k,n in eligible.items() if roster[k[0]]["subject"]==subject}
            personal.extend(score(selected, roster, ownmask))
            incumbent = history[0]
            p = dict(seed=seed,subject=subject,story=own_val[0]["story"],source_family=own_val[0]["source_family"],
                     blocks=initial_people[subject]["blocks"], personal_epoch=cost["selected_epoch"],
                     personal_epochs_run=cost["actual"]["training_epochs"],personal_updates=cost["actual"]["optimizer_updates"],
                     selected_personal_prefix_updates=cost["used_prefix"]["optimizer_updates"])
            for m in METRICS:
                p.update({f"initial_{m}":initial_people[subject][m],f"epoch_minus1_{m}":incumbent[m],
                          f"personal_{m}":selected[m],f"delta_{m}":selected[m]-initial_people[subject][m],
                          f"delta_to_personal_incumbent_{m}":selected[m]-incumbent[m]})
            pairs.append(p)
        combined = score(read(base/"PERSONAL_SELECTED_VAL.json"), roster, eligible)
        need(index(combined,("record","start"))==index(personal,("record","start")), "Exactly 49 actual personal selections")
        for stage, blocks in (("initial",initial),("personal",personal)):
            labels = dict(seed=seed,stage=stage)
            seed_scores.append(dict(**labels,**macro(blocks)))
            blocks_out.extend(dict(**labels,**b) for b in blocks)
            persons_out.extend(dict(**labels,**p,story=by_subject[p["subject"]]["story"],
                                    source_family=by_subject[p["subject"]]["source_family"]) for p in people(blocks))
            strata_out.extend(dict(**labels,**row) for row in strata(blocks,families))
        compare(dict(initial_no_globalFT_r=macro(initial)["r"],personal_r=macro(personal)["r"],
                     initial_updates=fits[seed,"initial",None]["actual_updates"],
                     personal_updates=sum(fits[seed,"personal",s]["actual_updates"] for s in inv["subjects"])),
                native[str(seed),], "Actual seed scientific terminal")
    need(len(fits)==150 and len(blocks_out)==4512 and len(pairs)==147, "Full intended matrix")
    for name,rows,keys in (("SEED_SCORES",seed_scores,("seed","stage")),("BLOCKS",blocks_out,("seed","stage","record","start")),
                          ("PEOPLE",persons_out,("seed","stage","subject")),("STRATA",strata_out,("seed","stage","stratum","label")),
                          ("PAIRS",pairs,("seed","subject")),("TRAINING_EPOCHS",histories,("seed","stage","subject","epoch"))):
        table(report,name,rows,keys)
    old_costs = index(saved_costs,("seed","stage","subject"))
    need(len(old_costs)==150, "All 150 reported fit costs")
    for cost in costs:
        compare(cost,old_costs[str(cost["seed"]),cost["stage"],cost["subject"] or ""],"Independently recomputed cost")
    overall = [dict(stage=stage,**{m:mean(r[m] for r in seed_scores if r["stage"]==stage) for m in METRICS})
               for stage in ("initial","personal")]
    compare(overall,summary["overall"],"Two complete methods, no seed mosaics")
    differences = [dict(seed=s,improved_person_r=sum(p["delta_r"]>1e-11 for p in pairs if p["seed"]==s),
                        reduced_person_MSE=sum(p["delta_MSE"]< -1e-11 for p in pairs if p["seed"]==s),
                        **{f"mean_delta_{m}":mean(p[f"delta_{m}"] for p in pairs if p["seed"]==s) for m in METRICS})
                   for s in SEEDS]
    compare(differences,summary["seed_differences"],"Seed differences")
    histogram={str(k):v for k,v in Counter(p["personal_epoch"] for p in pairs).items()}
    need(histogram.keys()==summary["epoch_histogram"].keys(),"Exact selected epoch histogram")
    compare(histogram,summary["epoch_histogram"],"Personal chosen epochs")
    compare({m:max(abs(p[f"epoch_minus1_{m}"]-p[f"initial_{m}"]) for p in pairs) for m in METRICS},
            summary["shared_to_personal_incumbent_max_abs"],"Shared to personal evaluation discrepancy")
    totals = {}
    for stage in ("initial","personal"):
        group = [r for r in costs if r["stage"]==stage]
        total = dict(fits=len(group),committed_CPU_seconds=math.fsum(r["committed_CPU_seconds"] for r in group),
                     committed_elapsed_seconds_sum=math.fsum(r["committed_elapsed_seconds"] for r in group),
                     max_cuda_reserved=max(r["peak_cuda_reserved"] for r in group),
                     selected_checkpoint_bytes=sum(r["checkpoint_selected_bytes"] for r in group))
        for bucket in ("actual","used_prefix"):
            total[bucket]={k:sum(r[bucket][k] for r in group) for k in group[0][bucket]}
        compare(total,summary["totals"][stage],"Stage costs including actual search")
        totals[stage]=total
    for row in summary["seed_budget"]:
        s=row["seed"]; initial=next(c for c in costs if c["seed"]==s and c["stage"]=="initial")
        personal=[c for c in costs if c["seed"]==s and c["stage"]=="personal"]
        compare(dict(actual_initial_updates_once=initial["actual"]["optimizer_updates"],
                selected_initial_prefix_updates_once=initial["used_prefix"]["optimizer_updates"],
                actual_personal_updates=sum(c["actual"]["optimizer_updates"] for c in personal),
                selected_personal_prefix_updates=sum(c["used_prefix"]["optimizer_updates"] for c in personal),
                initial_equal_person_allocation_updates=initial["actual"]["optimizer_updates"]/49,
                used_initial_prefix_equal_person_allocation_updates=initial["used_prefix"]["optimizer_updates"]/49,
                epoch_minus1_selected=sum(c["selected_epoch"]==-1 for c in personal)),row,"Shared initial charged once")
    need({r["seed"] for r in summary["seed_budget"]}==set(SEEDS) and len(summary["seed_budget"])==3,"Complete seed budget")
    workers=audit_workers(formal,resource,fits,summary)
    compare(read(ROOT/"ENGINEERING.json"),summary["engineering"],"Original engineering cost disclosure")
    references = []
    for s in SEEDS:
        path=ROOT.parent/f"s03_physical_stream_20261106/fit/seed{s}_ANCHOR.json"
        anchor=read(path)["anchor"]["val"]
        blocks=score(anchor,roster,eligible)
        references.append(dict(method="Transformer FT+Ccap",seed=s,personal=False,source_sha256=digest(path),
                               **{m:macro(blocks)[m] for m in METRICS}))
    d26=ROOT.parent/"s03_contrastive_history_20261106"
    for s in SEEDS:
        for arm in ("HH","CH","CC"):
            path=d26/f"formal/seed{s}/READOUT_{arm}/SCORE_{'near' if arm=='HH' else 'native'}.json"
            value=read(path)["Ccap"];blocks=blockset(value["segments"],roster,eligible)
            compare({m:macro(blocks)[m] for m in METRICS},value["phases"]["val"],"D26 same support")
            references.append(dict(method=f"D26 {arm}",seed=s,personal=False,source_sha256=digest(path),
                                   **{m:macro(blocks)[m] for m in METRICS}))
    d28=ROOT.parent/"s03_matched_raw_ridge_20261107"
    end28=read(d28/"formal/COMPLETED.json");audit28=read(d28/"COORDINATOR_RESULT_REVIEW.json")
    need(audit28["passed"] and audit28["completed_sha256"]==digest(d28/"formal/COMPLETED.json")
         and audit28["exact_original_support"],"Independently reviewed fixed D28 evidence")
    for arm in ("G","I"):
        references.append(dict(method="D28 "+arm,seed=None,personal=arm=="I",
                               source_sha256=digest(d28/"formal/COMPLETED.json"),**end28["scores"][arm]))
    actual=index(summary["same_support_comparators"]["seed_rows"],("method","seed"))
    need(actual.keys()==index(references,("method","seed")).keys(),"All same-support comparator rows")
    for row in references:
        compare(row,actual[row["method"],str(row["seed"]) if row["seed"] is not None else ""],"Same-support comparator")
    reference_means=[dict(method=m,**{k:mean(r[k] for r in references if r["method"]==m) for k in METRICS})
                     for m in dict.fromkeys(r["method"] for r in references)]
    reported_means=index(summary["same_support_comparators"]["means"],("method",))
    need(reported_means.keys()==index(reference_means,("method",)).keys(),"All comparator means")
    for row in reference_means:
        compare(row,reported_means[row["method"],],"Whole-method comparator means")
    delta=[]
    for stage in ("initial","personal"):
        for s in SEEDS:
            row=next(r for r in seed_scores if r["seed"]==s and r["stage"]==stage)
            anchor=next(r for r in references if r["seed"]==s and r["method"]=="Transformer FT+Ccap")
            delta.append(dict(stage=stage,seed=s,**{"delta_"+m:row[m]-anchor[m] for m in METRICS}))
    result=dict(created_UTC=datetime.now(timezone.utc).isoformat(),saved_evidence_audit_passed=True,
                protocol_sha256=GATE,resource_protocol_sha256=RESOURCE_GATE,
                formal_completed_sha256=digest(formal/"COMPLETED.json"),overall=overall,
                same_support_delta_to_Transformer=delta,actual_totals=totals,actual_workers=workers,
                checked_counts=dict(CHECKS),source_sha256=digest(Path(__file__)),evidence_sha256=HASHES,
                new_signal_payload_reads=0,weight_tensors_decoded=0,new_model_forwards=0,new_fits=0,
                new_optimizer_updates=0,raw_to_prediction_replay_performed=False,scientific_goal_achieved=False,
                limitations=["Repeated development VAL; all 49 people TRAIN-seen, five stories.",
                             "Shared initial has no population global FT; personal and shared methods are distinct.",
                             "Saved-block aggregation and weight-byte integrity do not independently replay EEG predictions.",
                             "Wall/CPU costs and temporal permissions differ across method families.",
                             "Independent numerical replay, rendered-figure review and scientific interpretation remain separate.",
                             "This audit never establishes independent generalization or SOTA."],
                audit_CPU_seconds=time.process_time()-cpu,audit_wall_seconds=time.monotonic()-wall)
    destination=ROOT/"COORDINATOR_SAVED_EVIDENCE_AUDIT.json"
    with destination.open("x") as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k!="evidence_sha256"},indent=2))


def self_test():
    # Synthetic adversarial checks only; no formal files, torch or model execution.
    inc=dict(epoch=-1,r=.5,MSE=1.)
    hist=[dict(epoch=0,r=.5,MSE=1.,stale=1),dict(epoch=1,r=.4,MSE=2.,stale=2)]
    need(select(hist,inc,2)[0]["epoch"]==-1,"Tie keeps original incumbent")
    tied=[dict(epoch=0,r=.5,MSE=1.,stale=0),dict(epoch=1,r=.5,MSE=1.,stale=1)]
    need(select(tied,None,None)[0]["epoch"]==0,"Initial ties keep earliest")
    g,v=geometry([{"common_samples":10},{"common_samples":9}],[{"common_samples":8200}],
                 {"frame":4,"hop":3,"batch":4})
    need(g["windows"]==5 and g["updates"]==2 and g["tail"]==1 and g["unique_record_sample_times"]==17,
         "Record-local grid, partial last batch")
    need(v==dict(unique=8200,calls=3,presented=8580),"Both edge halos and tail support")
    x=exposure(g,v,0,True)
    need(x["optimizer_updates"]==0 and x["validation_passes"]==1 and x["training_distinct_record_sample_times"]==0,
         "Minus1 selects no personal updates but retains baseline inference")
    roster={"a":dict(subject="p",source_family="f",story="s"),"b":dict(subject="q",source_family="g",story="t")}
    rows=[dict(record="a",start=0,subject="p",source_family="f",samples=3,r=.2,MSE=1.),
          dict(record="a",start=4,subject="p",source_family="f",samples=4,r=.4,MSE=2.),
          dict(record="b",start=0,subject="q",source_family="g",samples=2,r=.8,MSE=3.)]
    mask={("a",0):3,("a",4):4,("b",0):2}
    bs=blockset(rows,roster,mask)
    compare(dict(r=.55,MSE=2.25),macro(bs),"Participant macro, not pooled blocks/samples")
    rejected=0
    bad_hist=hist+[dict(epoch=2,r=.6,MSE=1.,stale=0)]
    cases=[lambda:select(bad_hist,inc,2),lambda:blockset(rows+rows[:1],roster,mask),
           lambda:blockset(rows[:-1],roster,mask),
           lambda:blockset([dict(rows[0],samples=4)]+rows[1:],roster,mask),
           lambda:blockset([dict(rows[0],subject="q")]+rows[1:],roster,mask),
           lambda:blockset([dict(rows[0],r=float("nan"))]+rows[1:],roster,mask),
           lambda:compare(.5,.50001,"Metric corruption"),
           lambda:index([{"seed":1},{"seed":1}],("seed",))]
    for case in cases:
        try:
            case()
        except AssertionError:
            rejected+=1
        else:
            raise AssertionError("Corruption was silently accepted")
    need(rejected==8,"All intended corruption cases rejected")
    print(json.dumps(dict(self_test_passed=True,adversarial_rejections=rejected,
                         formal_file_reads=0,signal_payload_reads=0,model_forwards=0,
                         new_fits=0,new_optimizer_updates=0,source_sha256=digest(Path(__file__)))))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    self_test() if args.self_test else main()
