"""D27 fit and worker accounting from scalar receipts only."""
import json
from collections import defaultdict
from math import ceil, fsum

PARAMETERS = 150433
CORE, HALO = 4096, 124


def read(path):
    return json.loads(path.read_text())


def need(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b):
    return abs(a-b) <= 1e-11 * max(1., abs(b))


def train_geometry(rows, specs):
    frame, hop, batch = (specs[k] for k in ("frame", "hop", "batch"))
    counts = [(r["common_samples"]-frame)//hop+1 for r in rows]
    need(all(n > 0 for n in counts), "Complete record-local training grid")
    windows = sum(counts)
    unique = sum((n-1)*hop+frame for n in counts)
    return dict(records=len(rows), windows=windows, updates=ceil(windows/batch), tail=windows % batch or batch,
                unique_record_sample_times=unique, target_sample_presentations=windows*frame,
                repeated_target_sample_presentations=windows*frame-unique)


def eval_geometry(rows):
    chunks = [(max(0, s-HALO), min(n, min(n, s+CORE)+HALO))
              for row in rows for n in (row["common_samples"],) for s in range(0, n, CORE)]
    return dict(unique=sum(r["common_samples"] for r in rows), calls=len(chunks),
                presented=sum(b-a for a,b in chunks))


def exposure(geometry, validation, epochs, has_baseline):
    passes = epochs+int(has_baseline)
    presentations = geometry["target_sample_presentations"]*epochs
    distinct = geometry["unique_record_sample_times"] if epochs else 0
    return dict(training_epochs=epochs, optimizer_updates=geometry["updates"]*epochs,
                training_window_presentations=geometry["windows"]*epochs,
                training_EEG_and_target_sample_time_presentations_each=presentations,
                training_distinct_record_sample_times=distinct,
                training_repeated_sample_time_presentations=presentations-distinct,
                validation_passes=passes, validation_calls=validation["calls"]*passes,
                validation_EEG_sample_time_presentations=validation["presented"]*passes,
                validation_distinct_record_sample_times=validation["unique"],
                validation_halo_repeat_sample_times=(validation["presented"]-validation["unique"])*passes,
                validation_total_repeat_sample_times=validation["presented"]*passes-validation["unique"])


def fit(folder, seed, stage, subject, train, val, specs, recorded_geometry, initial_sha):
    completed = read(folder/"COMPLETED.json")
    selection = read(folder/"SELECTED_VAL.json")
    personal = stage == "personal"
    need(completed["passed"] and completed["seed"] == seed and completed["stage"] == stage
         and completed["subject"] == subject and completed["initial_sha256"] == initial_sha
         and completed["parameters"] == PARAMETERS and completed["old_test_reads"] == 0
         and completed["population_global_FT"] is False, f"Fixed fit identity {folder}")
    geometry, validation = train_geometry(train, specs), eval_geometry(val)
    need(geometry == completed["geometry"] == recorded_geometry, f"Exact frozen training geometry {folder}")
    n = completed["complete_epochs"]
    need(0 < n <= specs["epochs"] and (personal or n == 201), f"Actual completed epochs {folder}")
    need(len(list(folder.glob("EPOCH_[0-9][0-9][0-9].json"))) == n, f"Every formal epoch receipt {folder}")
    baseline = read(folder/"EPOCH_MINUS1.json") if personal else None
    if personal:
        need(baseline["epoch"] == -1 and baseline["initial_sha256"] == initial_sha
             and baseline["updates"] == 0, f"Shared initial incumbent {folder}")
    best = baseline if personal else dict(epoch=None,r=-float("inf"))
    stale = 0
    history = []
    for epoch in range(n):
        row = read(folder/f"EPOCH_{epoch:03}.json")
        if row["r"] > best["r"]:
            best, stale = row, 0
        else:
            stale += 1
        need(row["epoch"] == epoch and row["updates"] == (epoch+1)*geometry["updates"]
             and row["windows"] == geometry["windows"] and row["stale"] == stale
             and row["VAL_model_calls"] == validation["calls"]
             and row["VAL_model_input_sample_times"] == validation["presented"], f"Frozen epoch work {folder}")
        if personal and epoch < n-1:
            need(stale < specs["patience"], f"First personal stopping boundary {folder}")
        history.append(row)
    need(not personal or n == specs["epochs"] or stale >= specs["patience"], f"Actual personal stopping rule {folder}")
    selected = completed["selected_epoch"]
    full_cost = exposure(geometry, validation, n, personal)
    prefix = exposure(geometry, validation, selected+1, personal)
    need(selection["VAL_model_calls"] == validation["calls"]
         and selection["VAL_model_input_sample_times"] == validation["presented"]
         and selection["VAL_unique_sample_times"] == validation["unique"], f"Selected whole-record inference {folder}")
    need(best["epoch"] == selected and same(best["r"], selection["r"]) and same(best["MSE"], selection["MSE"])
         and same(best["r"], completed["selected_r"]), f"Original early-tie choice {folder}")
    need(completed["selected_prefix_updates"] == prefix["optimizer_updates"]
         and completed["actual_updates"] == full_cost["optimizer_updates"]
         and completed["windows"] == full_cost["training_window_presentations"]
         and completed["VAL_model_calls"] == full_cost["validation_calls"]
         and completed["VAL_model_input_sample_times"] == full_cost["validation_EEG_sample_time_presentations"],
         f"Actual and selected prefix receipt {folder}")
    ids = dict(seed=seed, stage=stage, subject=subject)
    costs = dict(**ids, selected_epoch=selected, selected_r=selection["r"], selected_MSE=selection["MSE"],
                 initial_sha256=initial_sha, selected_sha256=completed["selected_sha256"],
                 parameters=completed["parameters"], checkpoint_selected_bytes=(folder/"SELECTED.pt").stat().st_size,
                 checkpoint_final_bytes=(folder/"FINAL.pt").stat().st_size,
                 peak_cuda_reserved=completed["peak_cuda_reserved"], process=completed["process"],
                 committed_CPU_seconds=completed["CPU_seconds"], committed_elapsed_seconds=completed["elapsed_seconds"],
                 stopped="epoch_cap" if n == specs["epochs"] else "personal_patience",
                 training_geometry=geometry, actual=full_cost, used_prefix=prefix)
    return selection, costs, [dict(**ids,**row) for row in ([baseline] if personal else [])+history]


def worker_accounting(root, formal, expected, completed_fits):
    attempts, workers, failures = [], [], []
    passed, terminal_workers = set(), set()
    fit_process = {(r['seed'],r['stage'],r['subject']):r['process'] for r in completed_fits}
    for a in sorted(formal.glob("ATTEMPT_*")):
        rows = sorted(a.glob("*_PROCESS.json"))
        attempts.append(dict(attempt=a.name, process=read(a/"PROCESS.json"), spawned=len(rows)))
        for path in rows:
            worker = read(path)
            command = worker["argv"]
            seed = int(command[command.index("--seed")+1])
            stage = command[3]
            subject = command[command.index("--subject")+1] if "--subject" in command else None
            key = seed, stage, subject
            need(key in expected, f"Known worker case {path}")
            if all(worker[k] == fit_process[key][k] for k in ("pid","uid","startticks")):
                terminal_workers.add(key)
            prefix = path.name.removesuffix("_PROCESS.json")
            outcome = a/(prefix+"_EXIT.json")
            outcome = read(outcome) if outcome.is_file() else None
            if outcome is not None:
                need(outcome["pid"] == worker["pid"], f"Worker exit identity {path}")
            success = bool(outcome and outcome["returncode"] == 0 and outcome["complete"])
            if success:
                passed.add(key)
            logfile = a/(prefix+".log")
            timing = None
            last_update = None
            if logfile.is_file():
                for line in logfile.read_text().splitlines():
                    if not line.startswith("{"):
                        continue
                    entry = json.loads(line)
                    if entry.get("event") == "worker_timing":
                        need(timing is None and entry["process"]["pid"] == worker["pid"], f"One worker timing {path}")
                        timing = entry
                    if "epoch" in entry and "updates" in entry:
                        last_update = entry["updates"]
            workers.append(dict(attempt=a.name, seed=seed, stage=stage, subject=subject, success=success,
                                process=worker, exit=outcome, timing=timing,
                                last_logged_cumulative_updates=last_update,
                                uncommitted_extra_updates=0 if success else None))
        if (a/"FAILURE.json").is_file():
            failures.append(dict(path=str((a/"FAILURE.json").relative_to(root)), receipt=read(a/"FAILURE.json")))
    for path in sorted(root.glob("FAILURE_*.json")):
        failures.append(dict(path=path.name, receipt=read(path)))
    need(terminal_workers == expected, "Every completed fit has its exact registered worker")
    no_clean_exit = sorted(expected-passed, key=lambda x:(x[0],x[1],x[2] or ""))
    return dict(attempts=attempts, workers=workers, failures=failures,
                completed_fit_identity_count=len(terminal_workers), successful_case_count=len(passed),
                completed_cases_without_clean_exit=no_clean_exit, worker_attempt_count=len(workers),
                elapsed_sum_not_parallel_wall=True,
                actual_uncommitted_failure_work=None if failures or any(not r["success"] for r in workers) else 0,
                observed_worker_wall_seconds_sum=fsum(r["exit"]["worker_wall_seconds_observed"] for r in workers if r["exit"]),
                reported_worker_CPU_seconds_sum=fsum(r["timing"]["CPU_seconds"] for r in workers if r["timing"]),
                workers_without_CPU_timing=sum(r["timing"] is None for r in workers))


def budget(costs, inventory, terminal):
    totals = {}
    for stage in ("initial","personal"):
        group = [r for r in costs if r["stage"] == stage]
        data = dict(fits=len(group), committed_CPU_seconds=fsum(r["committed_CPU_seconds"] for r in group),
                    committed_elapsed_seconds_sum=fsum(r["committed_elapsed_seconds"] for r in group),
                    max_cuda_reserved=max(r["peak_cuda_reserved"] for r in group),
                    selected_checkpoint_bytes=sum(r["checkpoint_selected_bytes"] for r in group))
        for bucket in ("actual","used_prefix"):
            data[bucket] = {key:sum(r[bucket][key] for r in group) for key in group[0][bucket]}
        totals[stage] = data
    seed_budget = []
    for seed, native in sorted(terminal.items()):
        group = [r for r in costs if r["seed"] == seed]
        a, = [r for r in group if r["stage"] == "initial"]
        personal = [r for r in group if r["stage"] == "personal"]
        need({r["subject"] for r in personal} == set(inventory["subjects"]), "Complete 49-person branches")
        p_updates = sum(r["actual"]["optimizer_updates"] for r in personal)
        need(a["actual"]["optimizer_updates"] == native["initial_updates"] and p_updates == native["personal_updates"],
             "Actual update totals agree with formal terminal")
        seed_budget.append(dict(seed=seed, actual_initial_updates_once=a["actual"]["optimizer_updates"],
                                selected_initial_prefix_updates_once=a["used_prefix"]["optimizer_updates"],
                                actual_personal_updates=p_updates,
                                selected_personal_prefix_updates=sum(r["used_prefix"]["optimizer_updates"] for r in personal),
                                initial_equal_person_allocation_updates=a["actual"]["optimizer_updates"]/49,
                                used_initial_prefix_equal_person_allocation_updates=a["used_prefix"]["optimizer_updates"]/49,
                                epoch_minus1_selected=sum(r["selected_epoch"] == -1 for r in personal)))
    return dict(totals=totals, seed_budget=seed_budget,
                distinct_times_note="Sums across fits are per-fit counts, not union sizes. Initial and personal overlap.",
                initial_sharing_note="Three initial fits charged once each; per-person allocation is accounting only.")
