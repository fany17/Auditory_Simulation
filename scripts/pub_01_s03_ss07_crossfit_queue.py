"""Launch the already-engineered D24 upstream matrix on two isolated L40s."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pub_01_s03_ss07_crossfit_meta import BASE, OLD, ROOT, digest, read
from pub_01_s03_ss07_crossfit_upstream import (
    FEATURES, GIB, NATIVE_CFG, exclusive, identity, imported, require,
)

GPU = ("GPU-be77d5f7-abf2-5e41-2a1c-9510e4a832d9",
       "GPU-cd053687-ccd7-87d3-3f4c-a3beffadfcba")
# Distribute the original fixed per-seed epoch counts across the longer and shorter lanes.
LANES = (
    [(1024, 0), (31415, 1), (2026, 2), (1024, 2), (1024, 1)],
    [(2026, 0), (31415, 2), (2026, 1), (31415, 0)],
)
WORKER = Path(__file__).with_name("pub_01_s03_ss07_crossfit_upstream.py")
CANDIDATE = ROOT / "FOLD_AND_BUDGET_CANDIDATE_V2.json"
GATE = ROOT / "UPSTREAM_FORMAL_GATE_V04.json"
ENGINEERING = ROOT / "engineer_v03/seed1024_fold0/COMPLETED.json"


def prepare():
    eng = read(ENGINEERING)
    canary = eng["real_TRAIN_readout_canary"]
    require(canary["first4_equal"] and canary["saved_roundtrip"] and canary["affine"]["a"] > 0,
            "Engineering readout gate")
    require(eng["original_VAL_reads"] == 0 and eng["old_test_reads"] == 0, "Engineering scope")
    modules = imported()
    source_paths = {Path(module.__file__).resolve() for module in modules}
    # Bind transitive local modules, including the actual shared objective and model constructors.
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and (path.name.startswith("pub_01_")
                                        or path == WORKER.resolve()):
                source_paths.add(path)
    source_paths.update([Path(__file__), WORKER, NATIVE_CFG, FEATURES, ENGINEERING,
                         BASE / "BASELINE_PROTOCOL.json", BASE / "INCREMENTAL_PROTOCOL.json",
                         CANDIDATE, Path(read(CANDIDATE)["train_manifest"])])
    jobs = {(job["seed"], job["model_fold"]) for job in read(CANDIDATE)["jobs"]}
    require(jobs == {pair for lane in LANES for pair in lane} and sum(map(len, LANES)) == len(jobs),
            "Exact disjoint schedule")
    require(shutil.disk_usage(ROOT).free >= 128*GIB, "Initial space supports the 64 GiB cap/reserve")
    result = dict(
        passed=True, engineer=str(ENGINEERING), worker_sha256=digest(WORKER),
        candidate_sha256=digest(CANDIDATE),
        sources={str(path.resolve()): digest(path) for path in sorted(source_paths)},
        lanes={gpu: pairs for gpu, pairs in zip(GPU, LANES)},
        cap_bytes=64*GIB, reserve_bytes=64*GIB, automatic_retry=False,
        process_identity=identity(),
    )
    exclusive(GATE, result)


def lane(run_root, gpu, jobs):
    with (run_root / f"{gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed, fold in jobs:
            resident = subprocess.check_output(
                ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True).strip()
            require(resident.isdecimal() and int(resident) <= 256, f"GPU already busy: {gpu}")
            stem = run_root / f"seed{seed}_fold{fold}"
            env = os.environ | {"CUDA_VISIBLE_DEVICES": gpu, "OPENBLAS_NUM_THREADS": "8",
                                "OMP_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
            with stem.with_suffix(".log").open("x") as log:
                child = subprocess.Popen([sys.executable, "-B", str(WORKER), "--stage", "formal",
                                          "--seed", str(seed), "--fold", str(fold)],
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
                result = child.wait()
            exclusive(stem.with_suffix(".json"), dict(seed=seed, fold=fold, gpu=gpu,
                                                      pid=child.pid, returncode=result))
            require(result == 0 and (ROOT / f"formal/seed{seed}_fold{fold}/COMPLETED.json").is_file(),
                    f"Worker failed; inspect {stem}.log")
    return len(jobs)


def run():
    gate = read(GATE)
    require(gate["passed"] and all(digest(Path(p)) == h for p, h in gate["sources"].items()),
            "Frozen formal gate drift")
    root = ROOT / "queue"
    root.mkdir()
    exclusive(root / "PROCESS.json", identity())
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(lane, root, gpu, jobs) for gpu, jobs in gate["lanes"].items()]
        for gpu, task in zip(gate["lanes"], tasks):
            try:
                task.result()
            except Exception as error:
                message = {"gpu": gpu, "reason": str(error), "process": identity()}
                exclusive(root / f"{gpu}_FAILURE.json", message)
                failures.append(message)
    exclusive(root / "TERMINAL.json", dict(passed=not failures, failures=failures, process=identity()))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    if parser.parse_args().stage == "freeze":
        prepare()
    else:
        run()
