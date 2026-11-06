"""Run the reviewed D27 metadata report after the original V04 queue exits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
ROOT = BASE/"s03_seawave_seen_reference_20261106"
FORMAL = ROOT/"formal"
SOURCE = ROOT/"report_source_v02"
OUTPUT = ROOT/"report01"
UPSTREAM = (1748466, 1012, 791000426)
SEEDS = {1024, 2026, 31415}
PROTOCOL = "8442767e6fb1a220bab129d2c2163321e760610102fdefb986f43a06079f7476"
SOURCE_REVIEW = "9c0831201ccf120d69ffccc503e91cc2712bf81536704856217d68303e0560ec"
HELPER_DIRECTORY = BASE/"s03_contrastive_history_20261106"/"report_source_v03"
CPUS = {40, 41, 42, 43}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def key(record):
    return tuple(record[k] for k in ("pid", "uid", "startticks"))


def identity(pid):
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory/"stat").read_text().rsplit(")", 1)[1].split()
        return (pid, directory.stat().st_uid, int(fields[19])), fields[0]
    except FileNotFoundError:
        return None, None


def live(process):
    current, state = identity(process[0])
    return current == process and state not in ("Z", "X")


def publish(name, payload):
    with (ROOT/name).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify():
    handoff = read(ROOT/"D26_WAIT_STARTED.json")
    need(tuple(handoff["process"]) == UPSTREAM and handoff["approved_protocol_sha256"] == PROTOCOL,
         "The original D27 handoff receipt changed")
    review_path = ROOT/"COORDINATOR_REPORT_SOURCE_REVIEW_V02.json"
    need(digest(review_path) == SOURCE_REVIEW and digest(ROOT/"PROTOCOL_V04.json") == PROTOCOL,
         "The exact D27 V04 protocol or report review changed")
    review = read(review_path)
    need(review["passed"] and review["protocol_sha256"] == PROTOCOL, "Reviewed D27 reporting source")
    for directory, pins in ((SOURCE, review["sources"]), (HELPER_DIRECTORY, review["reviewed_common_support_helpers"])):
        need(all(digest(directory/name) == fingerprint for name,fingerprint in pins.items()), "Reviewed source changed")
    need(not OUTPUT.exists(), "D27 report output already exists")
    return review


def complete_terminal(terminal):
    return (terminal["passed"] and terminal["protocol_sha256"] == PROTOCOL
            and key(terminal["process"]) == UPSTREAM and terminal["old_test_reads"] == 0
            and terminal["initial_fits"] == 3 and terminal["personal_fits"] == 147
            and len(terminal["results"]) == 3 and {r["seed"] for r in terminal["results"]} == SEEDS)


def terminal_path():
    path = FORMAL/"COMPLETED.json"
    need(path.is_file() and complete_terminal(read(path)), "Original D27 exited without the complete V04 terminal")
    for attempt in sorted(FORMAL.glob("ATTEMPT_*")):
        for receipt in (attempt/"PROCESS.json", *sorted(attempt.glob("*_PROCESS.json"))):
            need(not live(key(read(receipt))), "An exact D27 training process is still active")
    return path


def run(check_only):
    review = verify()
    if check_only:
        if live(UPSTREAM):
            print("WAIT_D27")
        else:
            terminal_path()
            print("READY")
        return
    need(os.sched_getaffinity(0) == CPUS and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
         "D27 report requires its four CPU cores and no GPU")
    trigger = Path(__file__)
    source_hash = digest(trigger)
    process, _ = identity(os.getpid())
    publish("REPORT_WAIT_STARTED.json", dict(process=process, upstream=UPSTREAM, CPUs=sorted(CPUS),
             protocol_sha256=PROTOCOL, trigger_sha256=source_hash, source_review_sha256=SOURCE_REVIEW,
             report_sources=review["sources"], common_support_helpers=review["reviewed_common_support_helpers"]))
    try:
        while live(UPSTREAM):
            time.sleep(60)
        verify()
        need(digest(trigger) == source_hash, "Report trigger changed after start")
        terminal = terminal_path()
        command = [sys.executable, "-B", str(SOURCE/"pub_01_s03_ss07_seawave_seen_report.py"),
                   "--output", str(OUTPUT)]
        subprocess.run(command, check=True)
        artifact = OUTPUT/"REPORT_ARTIFACTS.json"
        receipt = read(artifact)
        need(receipt["passed"] and receipt["protocol_sha256"] == PROTOCOL
             and receipt["formal_completed_sha256"] == digest(terminal), "Expected result report receipt")
        publish("REPORT_WAIT_COMPLETED.json", dict(process=process, formal_completed_sha256=digest(terminal),
                report_artifacts_sha256=digest(artifact)))
    except BaseException as error:
        publish("REPORT_WAIT_FAILURE.json", dict(process=process, error=str(error), automatically_retried=False))
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check-only", action="store_true")
    run(cli.parse_args().check_only)
