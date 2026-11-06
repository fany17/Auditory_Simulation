"""Wait for the approved D26 to D27 handoff without using either GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
D26 = BASE / "s03_contrastive_history_20261106" / "formal"
D27 = BASE / "s03_seawave_seen_reference_20261106"
EXPECTED_D26 = (1622219, 1012, 790259158)
D26_PROTOCOL = "8c249eabeb9edb9a5360006f1d3dabb9552e2a22ff649b9bae72801f75311b4d"
PROTOCOL = "8442767e6fb1a220bab129d2c2163321e760610102fdefb986f43a06079f7476"
REVIEW = "e5cfcb770962a322e8f25304621478db63a0fdd1835445c0907064400cef033d"
QUEUE_SOURCE = "461c824ebff77a54eb8baf87c7fd9e5f94ed6eec9f021551bf70ce160c905d4a"
D26_ROWS = {(seed, arm) for seed in (1024, 2026, 31415) for arm in ("HH", "CH", "CC")}
GPU_UUIDS = ("GPU-be77d5f7-abf2-5e41-2a1c-9510e4a832d9", "GPU-cd053687-ccd7-87d3-3f4c-a3beffadfcba")
V04 = D27 / "input_v04" / "pub_01_s03_ss07_seawave_seen_queue.py"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    process = Path(f"/proc/{pid}")
    try:
        fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
        return (pid, process.stat().st_uid, int(fields[19])), fields[0]
    except FileNotFoundError:
        return None, None


def live(key):
    actual, state = identity(key[0])
    return actual == key and state not in ("Z", "X")


def key(record):
    return tuple(record[k] for k in ("pid", "uid", "startticks"))


def preconditions():
    gate = D27 / "PROTOCOL_V04.json"
    review = D27 / "COORDINATOR_EXECUTION_REVIEW.json"
    if digest(D26.parent / "PROTOCOL_V05.json") != D26_PROTOCOL:
        raise RuntimeError("D26 V05 protocol changed")
    if digest(gate) != PROTOCOL or digest(review) != REVIEW:
        raise RuntimeError("D27 V04 protocol or approval changed")
    sources = read(gate)["sources"]
    if (sources.get(str(V04)) != QUEUE_SOURCE
        or not all(digest(Path(path)) == sha for path, sha in sources.items())):
        raise RuntimeError("Approved V04 sources changed")
    approval = read(review)
    if not approval["passed"] or approval["protocol_sha256"] != PROTOCOL:
        raise RuntimeError("D27 V04 approval incomplete")
    if any((D27 / name).exists() for name in ("formal", "FORMAL_CONSOLE.log", "D26_WAIT_ADMISSION.json")):
        raise RuntimeError("D27 already has a formal path, console or admission receipt")
    if key(read(D26 / "ATTEMPT_000" / "PROCESS.json")) != EXPECTED_D26:
        raise RuntimeError("Original D26 identity mismatch")


def admission():
    terminal = D26 / "COMPLETED.json"
    if live(EXPECTED_D26):
        return "WAIT_D26", {}
    if not terminal.is_file():
        raise RuntimeError("Original D26 exited without its full completion receipt")
    receipt = read(terminal)
    rows = receipt["evaluations"]
    if (not receipt["passed"] or receipt["protocol_sha256"] != D26_PROTOCOL
        or key(receipt["process"]) != EXPECTED_D26 or len(rows) != 9
        or receipt["old_test_reads"] != 0
        or {(r["seed"], r["arm"]) for r in rows
            if r["passed"] and r["epoch"] == 4 and r["old_test_reads"] == 0} != D26_ROWS):
        raise RuntimeError("D26 completion must contain nine unique passing V05 seed/arm evaluations")
    for row in rows:
        saved = D26 / f'seed{row["seed"]}' / f'READOUT_{row["arm"]}' / "COMPLETED.json"
        if read(saved) != row:
            raise RuntimeError("D26 terminal row differs from its saved readout")
    identities = []
    for attempt in sorted(D26.glob("ATTEMPT_*")):
        for path in (attempt / "PROCESS.json", *sorted(attempt.glob("*_PROCESS.json"))):
            identities.append(key(read(path)))
    if any(live(identity) for identity in identities):
        return "WAIT_D26_PROCESSES", {}
    free = {}
    for gpu in GPU_UUIDS:
        command = ["nvidia-smi", "-i", gpu, "--query-gpu=memory.used",
                   "--format=csv,noheader,nounits"]
        free[gpu] = int(subprocess.check_output(command, text=True, timeout=15).strip())
    if any(value > 256 for value in free.values()):
        return "WAIT_GPU", {"GPU_memory_MiB": free}
    return "READY", {"D26_terminal_sha256": digest(terminal), "D26_exited_processes": identities,
                     "GPU_memory_MiB": free}


def publish(path, payload):
    process, _ = identity(os.getpid())
    with path.open("x") as stream:
        json.dump(dict(payload, time_utc=datetime.now(timezone.utc).isoformat(), process=process), stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(check_only):
    if check_only:
        preconditions()
        state, evidence = admission()
        print(json.dumps(dict(state=state, evidence=evidence)))
        return
    publish(D27 / "D26_WAIT_STARTED.json", dict(expected_d26=EXPECTED_D26, approved_protocol_sha256=PROTOCOL,
                                               approved_review_sha256=REVIEW, queue_source_sha256=QUEUE_SOURCE,
                                               waiter_source_sha256=digest(Path(__file__)), upstream_protocol_sha256=D26_PROTOCOL))
    try:
        preconditions()
        while True:
            state, evidence = admission()
            if state == "READY":
                # Repeat the free-card check before yielding to the approved queue.
                time.sleep(5)
                state, evidence = admission()
                if state == "READY":
                    break
            time.sleep(60)
        preconditions()
        command = [sys.executable, "-B", str(V04)]
        publish(D27 / "D26_WAIT_ADMISSION.json", dict(evidence, argv=command, protocol_sha256=PROTOCOL))
        with (D27 / "FORMAL_CONSOLE.log").open("x") as output:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            os.chdir(D27)
            os.execv(sys.executable, command)
    except BaseException as error:
        publish(D27 / "D26_WAIT_FAILURE.json", dict(message=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    main(parser.parse_args().check_only)

