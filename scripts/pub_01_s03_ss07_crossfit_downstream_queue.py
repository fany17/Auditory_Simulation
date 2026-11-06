"""Wait on the D24 audited upstream jobs, then run the paired downstream matrix once."""
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pub_01_s03_ss07_crossfit_meta import ROOT, digest, read

WORKER = Path(__file__).with_name("pub_01_s03_ss07_crossfit_downstream.py")
WORKER_SHA = "afffe133a001cfaa3553d62cd271fd622e9c8af5be53dfb69d3d28a93ca96f41"
OUTPUT = ROOT / "downstream_queue_v02"
UPSTREAM = ROOT / "queue"
AUDIT = ROOT / "coordinator_upstream_audit_v02"
GATE = ROOT / "DOWNSTREAM_FORMAL_GATE_V05.json"
LANES = {"GPU-be77d5f7-abf2-5e41-2a1c-9510e4a832d9": (1024, 31415),
         "GPU-cd053687-ccd7-87d3-3f4c-a3beffadfcba": (2026,)}


def save(path, item):
    with path.open("x") as stream:
        json.dump(item, stream, indent=2)
        stream.write("\n")


def verify_process(directory):
    identity = read(directory / "PROCESS.json")
    process = Path(f"/proc/{identity['pid']}")
    fields = (process/"stat").read_text().rsplit(")", 1)[1].split()
    if int(fields[19]) != identity["startticks"] or process.stat().st_uid != identity["uid"]:
        raise RuntimeError(f"Lost exact dependency process for {directory}")


def await_dependencies():
    deadline = time.monotonic()+24*3600
    needs = {UPSTREAM: "TERMINAL.json", AUDIT: "COMPLETED.json"}
    while needs:
        for directory, filename in list(needs.items()):
            terminal = directory/filename
            if list(directory.glob("*FAILURE.json")):
                raise RuntimeError(f"Dependency failure: {directory}")
            if terminal.is_file():
                if not read(terminal)["passed"]:
                    raise RuntimeError(f"Failed dependency: {terminal}")
                del needs[directory]
            else:
                try:
                    verify_process(directory)
                except FileNotFoundError:
                    if not terminal.is_file():
                        raise
        if needs:
            if time.monotonic() > deadline:
                raise TimeoutError("Observation timeout. No dependency is restarted.")
            time.sleep(60)


def run_lane(gpu, seeds):
    for seed in seeds:
        used = subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip()
        if not used.isdecimal() or int(used) > 256:
            raise RuntimeError(f"Selected GPU is busy: {gpu}")
        env = os.environ | {"CUDA_VISIBLE_DEVICES": gpu, "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                            "OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "8"}
        with (OUTPUT/f"seed{seed}.log").open("x") as log:
            subprocess.run([sys.executable, "-B", str(WORKER), "--stage", "formal", "--seed", str(seed)],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = read(ROOT/f"downstream_formal/seed{seed}/COMPLETED.json")
        if not result["passed"] or result["seed"] != seed or len(result["results"]) != 4:
            raise RuntimeError(f"Incomplete downstream seed: {seed}")
    return list(seeds)


def main():
    OUTPUT.mkdir()
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    identity = dict(pid=os.getpid(), uid=os.getuid(), startticks=int(fields[19]),
                    source_sha256=digest(Path(__file__)))
    save(OUTPUT/"PROCESS.json", identity)
    try:
        if digest(WORKER) != WORKER_SHA:
            raise RuntimeError("Unreviewed downstream worker")
        await_dependencies()
        if digest(WORKER) != WORKER_SHA:
            raise RuntimeError("Downstream worker drift during dependency wait")
        with (OUTPUT/"FREEZE.log").open("x") as log:
            subprocess.run([sys.executable, "-B", str(WORKER), "--stage", "freeze"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        save(OUTPUT/"START.json", dict(process=identity, lanes=LANES, gate_sha256=digest(GATE)))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(run_lane, gpu, seeds) for gpu, seeds in LANES.items()]
            complete = sorted(seed for job in jobs for seed in job.result())
        save(OUTPUT/"TERMINAL.json", dict(passed=True, seeds=complete, fits=12, process=identity))
    except BaseException as error:
        save(OUTPUT/"FAILURE.json", dict(process=identity, message=str(error),
                                          traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
