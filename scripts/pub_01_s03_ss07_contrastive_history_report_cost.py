"""D26 accounting from immutable stage receipts and process logs only."""
import json
from collections import defaultdict
from math import fsum


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    return abs(actual - expected) <= 1e-11 * max(1., abs(expected))


def unique_grid(rows):
    return sum((row["windows"] - 1) * 96 + 192 for row in rows)


def contrastive_cost(history, phase):
    coverage = [row[phase + "_coverage"] for row in history]
    all_slots = sum(row["history_slot_presentations"] for row in coverage)
    all_current = sum(row["current_windows"] for row in coverage)
    if phase == "train":
        unique = coverage[-1]["cumulative"]
        n_current = unique["unique_current_windows"]
        n_eeg = unique["unique_EEG_samples"]
        n_audio = unique["unique_audio_windows"]
        n_slots = [unique[f"unique_slot{i}_samples"] for i in (1, 2)]
    else:
        require(all(row == coverage[0] for row in coverage), "Fixed VAL contrastive coverage")
        one = coverage[0]
        n_current = one["unique_current_windows"]
        n_eeg = one["unique_all_EEG_sample_times"]
        n_audio = one["unique_audio_story_time_windows"]
        n_slots = [one[f"unique_slot{i}_EEG_sample_times"] for i in (1, 2)]
    return dict(current_EEG_windows=all_current, memory_EEG_slots=all_slots,
                audio_target_windows=all_current, unique_current_EEG_windows=n_current,
                repeated_current_EEG_windows=all_current-n_current, unique_EEG_sample_times=n_eeg,
                unique_slot1_EEG_sample_times=n_slots[0], unique_slot2_EEG_sample_times=n_slots[1],
                repeated_EEG_sample_time_presentations=192*(all_current+all_slots)-n_eeg,
                unique_audio_story_time_windows=n_audio, repeated_audio_target_windows=all_current-n_audio)


def fixed_grid_cost(current, active, physical, passes, cached=False):
    n_windows = current*passes
    n_slots = 0 if cached else 2*active*passes
    return dict(current_encoder_windows=0 if cached else n_windows, memory_encoder_slots=n_slots,
                target_or_head_windows=n_windows, unique_supervised_windows=current,
                repeated_supervised_windows=n_windows-current,
                unique_EEG_sample_times=None if cached else physical,
                repeated_EEG_sample_time_presentations=None if cached else 192*(n_windows+n_slots)-physical)


def epoch_ledger(folder, seed, branch, stage, updates, windows):
    receipt = read(folder / "COMPLETED.json")
    history = receipt["history"]
    require(receipt["passed"] and receipt["seed"] == seed, f"Stage complete {folder}")
    require(receipt.get("arm", receipt.get("role")) == branch, f"Stage route {folder}")
    require(len(history) == receipt["actual_epochs"] > 0, f"Epoch count {folder}")
    best = min(range(len(history)), key=lambda i: history[i]["val"])
    require(best == receipt["selected_epoch"] and close(history[best]["val"], receipt["selected_val"]),
            f"Best objective epoch {folder}")
    require(receipt["selected_prefix_updates"] == (best+1)*updates
            and receipt["selected_prefix_windows"] == (best+1)*windows
            and receipt["updates"] == len(history)*updates, f"Selected and total budget {folder}")
    rows = []
    best_val, stale = float("inf"), 0
    for epoch, row in enumerate(history):
        if row["val"] < best_val:
            best_val, stale = row["val"], 0
        else:
            stale += 1
        require(row["epoch"] == epoch and row["total_updates"] == (epoch+1)*updates
                and row["epoch_windows"] == windows and row["epoch_updates"] == updates and row["stale"] == stale,
                f"Epoch identity and work {folder}/{epoch}")
        require(read(folder / f"EPOCH_{epoch:03}.json") == row, f"Immutable epoch receipt {folder}/{epoch}")
        rows.append(dict(seed=seed, branch=branch, stage=stage, **row))
    return receipt, rows


def worker_ledger(formal, seeds, arms):
    expected = {(seed, phase, role) for seed in seeds
                for phase in ("cl", "head", "ft", "readout")
                for role in (("H", "C") if phase in ("cl", "head") else arms)}
    successes = set()
    workers, failures, attempts = [], [], []
    for folder in sorted(formal.glob("ATTEMPT_*")):
        files = sorted(folder.glob("*_PROCESS.json"))
        attempts.append(dict(attempt=folder.name, process=read(folder / "PROCESS.json"), started_workers=len(files)))
        for file in files:
            start = read(file)
            key = tuple(start[k] for k in ("seed", "stage", "role"))
            require(key in expected, f"Unregistered worker {file}")
            base = file.name.removesuffix("_PROCESS.json")
            exit_path = folder / (base + "_EXIT.json")
            stop = read(exit_path) if exit_path.is_file() else None
            log = folder / (base + ".log")
            timings, committed = [], {}
            if log.is_file():
                for line in log.read_text().splitlines():
                    if not line.startswith("{"):
                        continue
                    event = json.loads(line)
                    if event.get("event") == "worker_timing":
                        timings.append(event)
                    if event.get("event") == "epoch":
                        branch = event["path"].rsplit("/", 1)[-1]
                        committed[branch] = event["total_updates"]
            require(len(timings) <= 1, f"Multiple worker timing events {log}")
            ok = stop is not None and stop["returncode"] == 0 and stop["terminal_receipt"]
            if stop is not None:
                require(stop["pid"] == start["pid"], f"Worker exit identity {file}")
            if ok:
                successes.add(key)
            workers.append(dict(attempt=folder.name, seed=key[0], stage=key[1], role=key[2], passed=ok,
                                process=start, exit=stop, timing=timings[0] if timings else None,
                                final_logged_cumulative_updates_by_branch=committed,
                                exact_extra_uncommitted_updates=None if not ok else 0,
                                log=str(log.relative_to(formal))))
    for file in sorted(formal.glob("**/*.json")):
        if "FAILURE" in file.name or "ABORT" in file.name:
            failures.append(dict(path=str(file.relative_to(formal)), details=read(file)))
    require(successes == expected, "All 30 scheduled worker roles have clean exit evidence")
    intervals = [(row["process"]["launched_monotonic_seconds"], row["exit"]["worker_wall_seconds_observed"])
                 for row in workers if row["exit"] is not None]
    # Across a reboot, monotonic times are not comparable. Publish spans per attempt.
    for attempt in attempts:
        group = [row for row in workers if row["attempt"] == attempt["attempt"] and row["exit"] is not None]
        if group:
            starts = [r["process"]["launched_monotonic_seconds"] for r in group]
            ends = [t+r["exit"]["worker_wall_seconds_observed"] for t,r in zip(starts,group)]
            attempt["observed_first_launch_to_last_exit_seconds"] = max(ends)-min(starts)
    return dict(attempts=attempts, workers=workers, failures=failures,
                sum_observed_worker_wall_seconds=fsum(duration for _,duration in intervals),
                retry_accounting="Logical committed stage totals; failed or resumed workers may have executed extra work. "
                                 "Logged per-branch updates are cumulative and cannot be added across retries.",
                actual_worker_wall_is_not_matrix_wall=True,
                summed_worker_CPU_seconds=fsum(row["timing"]["worker_CPU_seconds"] for row in workers if row["timing"]),
                workers_without_timing=sum(row["timing"] is None for row in workers))


def summarize_cost(formal, inventory, terminal, seeds, arms):
    physical = {phase: unique_grid(inventory["phases"][phase]) for phase in ("train", "val")}
    base_windows = {phase: inventory["counts"][phase]["supervised_windows"] for phase in physical}
    active = {phase: inventory["counts"][phase]["supervised_history_windows"] for phase in physical}
    stage_rows, epoch_rows, cl_cost, supervised = [], [], {}, {}
    for seed in seeds:
        seed_root = formal / f"seed{seed}"
        for stage, branches, updates, windows in (
            ("CL", ("H", "C"), 820, 104960), ("HEAD", arms, 3945, 252454), ("FT", arms, 3945, 252454)
        ):
            receipts = []
            for branch in branches:
                folder = seed_root / f"{stage}_{branch}"
                receipt, rows = epoch_ledger(folder, seed, branch, stage, updates, windows)
                receipts.append(receipt)
                epoch_rows.extend(rows)
                history = receipt["history"]
                selected = receipt["selected_epoch"]
                stage_rows.append(dict(seed=seed, branch=branch, stage=stage, epochs=len(history),
                                       committed_updates=receipt["updates"], selection_epoch=selected,
                                       selected_prefix_updates=receipt["selected_prefix_updates"],
                                       used_model_epoch=4 if stage == "FT" else selected,
                                       used_prefix_updates=receipt["updates"] if stage == "FT" else receipt["selected_prefix_updates"],
                                       elapsed_seconds=receipt["elapsed_seconds"], CPU_seconds=receipt["CPU_seconds"],
                                       cuda_peak_reserved=receipt["cuda_peak_reserved"], stopping=receipt["stopping"],
                                       parameters=receipt.get("parameters", receipt.get("head_parameters"))))
                if stage == "CL":
                    cl_cost[seed, branch] = dict(
                        full_TRAIN=contrastive_cost(history, "train"), full_VAL=contrastive_cost(history, "val"),
                        selected_prefix_TRAIN=contrastive_cost(history[:selected+1], "train"),
                        selected_prefix_VAL=contrastive_cost(history[:selected+1], "val"))
                else:
                    item = dict()
                    supervised[seed, branch, stage] = item
                    for phase in physical:
                        passes = len(history) + int(stage == "FT" and phase == "val")
                        prefix = (5 if stage == "FT" else selected+1) + int(stage == "FT" and phase == "val")
                        config = base_windows[phase], active[phase], physical[phase]
                        item[phase] = dict(full=fixed_grid_cost(*config, passes, cached=stage == "HEAD"),
                                           used_prefix=fixed_grid_cost(*config, prefix, cached=stage == "HEAD"))
                    if stage == "HEAD":
                        caches = [read(folder / file) for file in receipt["cache_receipts"]]
                        require(caches and {r["arm"] for r in caches} == {branch}, "Head cache identities")
                        for cache in caches:
                            for phase in physical:
                                require(cache[phase]["current_encoder_windows"] == base_windows[phase]
                                        and cache[phase]["history_encoder_windows"] == 2*active[phase],
                                        "Exact observed cache cost")
                        item.update(cache_attempts=caches, head_optimize=receipt["head_optimize"],
                                    head_cached_VAL=receipt["head_cached_VAL"])
                    else:
                        require(len(history) == 5 and receipt["final_epoch"] == 4, "Exactly five FT epochs")
                        baseline = read(folder / "EPOCH_MINUS1.json")
                        require(baseline["epoch"] == -1 and baseline["abs_difference"] <= 1e-4,
                                "Actual cached head to FT objective bridge")
                        epoch_rows.append(dict(seed=seed, branch=branch, stage=stage, **baseline))
            n = receipts[0]["actual_epochs"]
            require(all(r["actual_epochs"] == n for r in receipts), f"Paired stage execution {seed}/{stage}")
            if stage in ("CL", "HEAD"):
                cap, patience = (100, 15) if stage == "CL" else (250, 10)
                require(n <= cap and (n == cap or all(r["history"][-1]["stale"] >= patience for r in receipts)),
                        f"Paired stop rule {seed}/{stage}")
                for epoch in range(n-1):
                    require(not all(r["history"][epoch]["stale"] >= patience for r in receipts),
                            f"First paired stop {seed}/{stage}/{epoch}")
    readouts = []
    for row in terminal["evaluations"]:
        require(row["TRAIN_fit"]["TRAIN_fit_windows"] == 250750 and row["new_neural_updates"] == 0,
                "Exact fixed readout fit geometry")
        history_factor = 6 if row["arm"] == "HH" else 2
        require(row["encoder_current_windows"] == 282954
                and row["encoder_slots"] == 2*active["train"] + history_factor*active["val"], "Actual readout paths")
        readouts.append(dict(seed=row["seed"], arm=row["arm"], train_fit=row["TRAIN_fit"],
                             current_EEG_windows=row["encoder_current_windows"], memory_EEG_slots=row["encoder_slots"],
                             TRAIN= fixed_grid_cost(base_windows["train"], active["train"], physical["train"], 1),
                             VAL_unique_EEG_sample_times=physical["val"],
                             VAL_repeated_EEG_sample_time_presentations=192*(base_windows["val"]+history_factor*active["val"])-physical["val"],
                             ridge_fits=1, updates=0, CPU_seconds=row["CPU_seconds"], elapsed_seconds=row["elapsed_seconds"]))
    require(len(readouts) == 9, "Nine unique Ccap solves")
    stages = {(row["seed"], row["branch"], row["stage"]): row for row in stage_rows}
    allocations = []
    for seed in seeds:
        for arm in arms:
            pre = stages[seed, arm[0], "CL"]
            head = stages[seed, arm, "HEAD"]
            ft = stages[seed, arm, "FT"]
            share = 1 if arm[0] == "H" else .5
            allocations.append(dict(seed=seed, arm=arm, pretraining_branch=arm[0],
                                    pretraining_job_share=share, charged_CL_updates=pre["committed_updates"]*share,
                                    full_standalone_CL_updates=pre["committed_updates"],
                                    actual_selected_CL_prefix_updates=pre["used_prefix_updates"],
                                    selected_head_prefix_updates=head["used_prefix_updates"], used_FT_updates=ft["used_prefix_updates"],
                                    committed_head_updates=head["committed_updates"], committed_FT_updates=ft["committed_updates"]))
    require(sum(row["charged_CL_updates"] for row in allocations)
            == sum(row["committed_updates"] for row in stage_rows if row["stage"] == "CL"), "Charge shared CL once")
    return dict(stages=stage_rows, readouts=readouts, branch_allocations=allocations,
                CL_coverage=[dict(seed=seed, branch=branch, **value) for (seed, branch), value in cl_cost.items()],
                supervised_coverage=[dict(seed=seed, branch=branch, stage=stage, **value)
                                     for (seed, branch, stage), value in supervised.items()],
                committed_updates_by_stage={stage:sum(r["committed_updates"] for r in stage_rows if r["stage"] == stage)
                                            for stage in ("CL", "HEAD", "FT")},
                failure_and_worker_ledger=worker_ledger(formal, seeds, arms)), epoch_rows
