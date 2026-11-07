"""Continue only the three unstarted D24 GPU0 jobs after an audit-context collision."""
import fcntl
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_source_family_crossfit_20261106")
sys.path.insert(0, str(ROOT / "input_v04"))
from pub_01_s03_ss07_crossfit_meta import digest, read
from pub_01_s03_ss07_crossfit_queue import GPU, GATE, lane
from pub_01_s03_ss07_crossfit_upstream import exclusive, identity, require

ORIGINAL = ROOT / "queue"
OUTPUT = ROOT / "upstream_recovery_v01"
TERMINAL = ROOT / "UPSTREAM_RECOVERY_TERMINAL.json"
ORIGINAL_GATE_SHA = "0ad69424fb76f827b268b63f7d45cbd954ae514c7f687d13a870e66e52322482"
ORIGINAL_QUEUE_ID = {"pid": 1346529, "uid": 1012, "startticks": 788549137}
RECOVERY = ((2026, 2), (1024, 2), (1024, 1))


def require_identity(actual):
    require(all(actual[k] == value for k, value in ORIGINAL_QUEUE_ID.items()), "Original queue identity")


def verify_old_queue():
    require_identity(read(ORIGINAL / "PROCESS.json"))
    failure = read(ORIGINAL / f"{GPU[0]}_FAILURE.json")
    require_identity(failure["process"])
    require(failure["gpu"] == GPU[0] and failure["reason"] == f"GPU already busy: {GPU[0]}",
            "Recovery must follow the observed GPU0 admission failure only")
    require(not (ORIGINAL / f"{GPU[1]}_FAILURE.json").exists(), "The original GPU1 lane failed")
    return failure


def original_terminal():
    deadline = time.monotonic()+24*3600
    terminal = ORIGINAL / "TERMINAL.json"
    while not terminal.is_file():
        require(time.monotonic() <= deadline, "Original GPU1 observation timeout; do not restart")
        verify_old_queue()
        pid = Path(f"/proc/{ORIGINAL_QUEUE_ID['pid']}")
        try:
            fields = (pid/"stat").read_text().rsplit(")", 1)[1].split()
            require_identity(dict(pid=ORIGINAL_QUEUE_ID["pid"], uid=pid.stat().st_uid,
                                  startticks=int(fields[19])))
            require(fields[0] not in ("Z", "X"), "Original queue exited without a terminal receipt")
        except FileNotFoundError:
            if not terminal.is_file():
                raise
        time.sleep(30)
    ended = read(terminal)
    require_identity(ended["process"])
    require(not ended["passed"] and ended["failures"] == [verify_old_queue()],
            "Preserve the original queue's observed GPU0 failure")
    return terminal


def verify_jobs(original):
    candidate_path = ROOT / "FOLD_AND_BUDGET_CANDIDATE_V2.json"
    candidate = read(candidate_path)
    gate = read(GATE)
    require(digest(candidate_path) == gate["candidate_sha256"], "Candidate pin")
    assignment = Path(candidate["train_manifest"])
    require(digest(assignment) == candidate["train_manifest_sha256"], "Record assignment pin")
    manifest = read(assignment)["records"]
    checked = {}
    for spec in candidate["jobs"]:
        seed, fold = spec["seed"], spec["model_fold"]
        key = f"seed{seed}_fold{fold}"
        run = OUTPUT if (seed, fold) in RECOVERY else ORIGINAL
        process = read(run/f"{key}.json")
        formal = ROOT/"formal"/key
        done, frozen = formal/"COMPLETED.json", formal/"UPSTREAM_FROZEN.json"
        receipt, ridge = read(done), read(frozen)
        require(not (formal/"FAILURE.json").exists() and process["returncode"] == 0
                and (process["seed"],process["fold"]) == (seed,fold), "Actual worker completion")
        require(process["pid"] == receipt["process"]["pid"] and receipt["seed"] == seed
                and receipt["fold"] == fold and receipt["stage"] == "formal", "Formal worker identity")
        fit = {row["record"] for row in manifest if row["OOF_model_fold"] != fold}
        require(set(receipt["fit_records"]) == set(ridge["fit_records"]) == fit
                and set(receipt["contrastive"]["record_presentations"]) <= fit, "TRAIN record fence")
        require(receipt["original_VAL_reads"] == receipt["OOF_label_reads"] == receipt["old_test_reads"] == 0,
                "Upstream phase fence")
        require(receipt["normalization_samples"] == spec["fit_raw_channel_normalization_samples"]
                and ridge["Ccap"]["TRAIN_physical_points"] == spec["affine_physical_points"]
                and ridge["Ccap"]["TRAIN_fit_windows"] == spec["Ccap"]["fit_windows"]
                and receipt["contrastive"]["updates"] == spec["contrastive"]["updates"],
                "Normalization, Ccap and contrastive counts")
        for stage in ("head","FT"):
            for field in ("updates","target_windows"):
                require(sum(row[field] for row in receipt[stage]) == spec[stage][field],
                        "Supervised budget")
        for lane_name in ("IS", "OOF"):
            rows = read(formal/f"{lane_name}_ANCHOR.json")
            expected = [row["record"] for row in manifest if row[f"{lane_name}_model_fold"] == fold]
            require([row["record"] for row in rows] == expected
                    and sum(row["windows"] for row in rows) == spec[f"{lane_name}_output_windows"],
                    "Output routing and coverage")
        checked[key] = dict(queue=str(run), seed=seed, fold=fold, pid=process["pid"],
                            completed_sha256=digest(done), worker_returncode=process["returncode"],
                            wall_seconds=receipt["wall_seconds"], CPU_seconds=receipt["total_CPU_seconds"])
    require(len(checked) == 9, "Exact terminal matrix")
    return dict(passed=True, jobs=checked, process=identity(), recovery_jobs=RECOVERY,
                original_queue_terminal=str(original), original_queue_terminal_sha256=digest(original),
                original_queue_passed=False, frozen_upstream_gate_sha256=ORIGINAL_GATE_SHA,
                recovery_source_sha256=digest(Path(__file__)), automatic_retry=False)


def main():
    OUTPUT.mkdir()
    exclusive(OUTPUT/"PROCESS.json", identity())
    try:
        require(digest(GATE) == ORIGINAL_GATE_SHA, "Unchanged upstream gate")
        gate = read(GATE)
        require(all(digest(Path(path)) == sha for path, sha in gate["sources"].items()), "Unchanged upstream sources")
        require(tuple(map(tuple, gate["lanes"][GPU[0]][2:])) == RECOVERY,
                "Only original unstarted GPU0 suffix")
        old = verify_old_queue()
        for seed,fold in RECOVERY:
            key = f"seed{seed}_fold{fold}"
            require(not (ROOT/"formal"/key).exists() and not (ORIGINAL/f"{key}.log").exists()
                    and not (ORIGINAL/f"{key}.json").exists(), "A recovery job already started")
        exclusive(OUTPUT/"PLAN.json", dict(gpu=GPU[0], jobs=RECOVERY, original_failure=old,
                                          gate_sha256=ORIGINAL_GATE_SHA, source_sha256=digest(Path(__file__))))
        with (ORIGINAL / f"{GPU[0]}.lock").open("r") as original_lock:
            fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(lane(OUTPUT, GPU[0], RECOVERY) == 3, "Exact recovery lane")
        terminal = original_terminal()
        exclusive(TERMINAL, verify_jobs(terminal))
    except BaseException as error:
        exclusive(OUTPUT/"FAILURE.json", dict(process=identity(), message=str(error),
                                               traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
