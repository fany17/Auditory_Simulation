"""Start the D26 metadata report once the frozen formal queue completes."""
import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_contrastive_history_20261106")
UPSTREAM = (1622219, 1012, 790259158)
PROTOCOL = "8c249eabeb9edb9a5360006f1d3dabb9552e2a22ff649b9bae72801f75311b4d"
SOURCES = {
    "D28_MATCHED_RAW_RIDGE_COMPLETED.json": "9be6b1dcc502318c4d6a1c7df734818af3deb5a64c76a2df739e2464599229f5",
    "HISTORICAL_REFERENCES.json": "5c95ce13af5eecf960a74d9c935e626a232ad048ccf31b1e4f7a6de5bf51b1bc",
    "pub_01_s03_ss07_contrastive_history_report_cost.py": "d07c8ab6bfb5fbae8292432da8ca4abac276846d8dadc780bf792c906709dbfd",
    "pub_01_s03_ss07_contrastive_history_report_figures.py": "9650a02253958d217ed0d6c5af8c9b6ec7a252f5050181e74459cac837f093f9",
    "pub_01_s03_ss07_contrastive_history_report.py": "550913f9c880c44687fbac280675db0285c42ee9bfeef18ee2b7a1619a149a08",
}
SOURCE = ROOT / "report_source_v03"
OUTPUT = ROOT / "report01"


def digest(file):
    return sha256(file.read_bytes()).hexdigest()


def identity(pid):
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return (pid, directory.stat().st_uid, int(fields[19])), fields[0]
    except FileNotFoundError:
        return None, None


def publish(name, value):
    with (ROOT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify():
    process = json.loads((ROOT / "formal/ATTEMPT_000/PROCESS.json").read_text())
    if tuple(process[k] for k in ("pid", "uid", "startticks")) != UPSTREAM:
        raise ValueError("Original formal queue identity drift")
    if digest(ROOT / "PROTOCOL_V05.json") != PROTOCOL or any(digest(SOURCE / p) != h for p, h in SOURCES.items()):
        raise ValueError("Protocol or isolated report source changed")
    if OUTPUT.exists():
        raise ValueError("Report output is already present")


def waiting():
    actual, state = identity(UPSTREAM[0])
    return actual == UPSTREAM and state not in ("Z", "X")


def run(check_only):
    verify()
    if check_only:
        print("WAIT_D26" if waiting() else "VERIFY_TERMINAL_ON_LAUNCH")
        return
    process, _ = identity(os.getpid())
    publish("REPORT_WAIT_STARTED.json", dict(process=process, upstream=UPSTREAM,
            upstream_protocol_sha256=PROTOCOL, sources=SOURCES, trigger_sha256=digest(Path(__file__))))
    try:
        while waiting():
            time.sleep(60)
        verify()
        subprocess.run([sys.executable, "-B", str(SOURCE / "pub_01_s03_ss07_contrastive_history_report.py"),
                        "--output", str(OUTPUT)], check=True)
        publish("REPORT_WAIT_COMPLETED.json", dict(process=process,
                report_artifacts_sha256=digest(OUTPUT / "REPORT_ARTIFACTS.json")))
    except BaseException as error:
        publish("REPORT_WAIT_FAILURE.json", dict(process=process, message=str(error)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    run(parser.parse_args().check_only)
