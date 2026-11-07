"""Bind the reviewed scalar report to the actual replacement D27 scheduler."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from d27_resource_process import FORMAL, JOB_COUNT, RESOURCE, ROOT, SEEDS, TRAIN_PY, V04
from d27_resource_process import identity, key, live, need, publish, read, sha, verify_resource, verify_retirement

BASE = ROOT.parent
SOURCE_REVIEW = "9c0831201ccf120d69ffccc503e91cc2712bf81536704856217d68303e0560ec"
REPORT_SOURCE = ROOT / "report_source_v02"
HELPER = BASE / "s03_contrastive_history_20261106/report_source_v03"
OUTPUT = ROOT / "report01"
CPUS = {40, 41, 42, 43}


def verify_report():
    review = ROOT / "COORDINATOR_REPORT_SOURCE_REVIEW_V02.json"
    need(sha(review) == SOURCE_REVIEW, "Reviewed report receipt changed")
    evidence = read(review)
    need(evidence["passed"] and evidence["protocol_sha256"] == V04, "Exact report approval")
    for directory, pins in ((REPORT_SOURCE, evidence["sources"]), (HELPER, evidence["reviewed_common_support_helpers"])):
        need(all(sha(directory / path) == expected for path, expected in pins.items()), "Reviewed report source drift")
    need(not OUTPUT.exists(), "D27 report is already present")


def finish_allowed(upstream, fingerprint):
    path = FORMAL / "COMPLETED.json"
    data = read(path)
    need(data["passed"] and key(data["process"]) == upstream and data["protocol_sha256"] == V04
         and data["old_test_reads"] == 0 and data["initial_fits"] == 3 and data["personal_fits"] == 147
         and len(data["results"]) == 3 and {r["seed"] for r in data["results"]} == set(SEEDS), "Exact replacement V04 terminal")
    receipts = [p for a in FORMAL.glob("ATTEMPT_*") for p in a.glob("*_PROCESS.json")]
    need(len(receipts) == JOB_COUNT, "Exactly 150 formal registrations")
    for p in receipts:
        process = read(p)
        outcome = read(p.with_name(p.name.replace("_PROCESS.json", "_EXIT.json")))
        need(not live(key(process)) and outcome["pid"] == process["pid"]
             and outcome["complete"] and outcome["returncode"] == 0, "Authentic exit of every formal worker")
    resource_end = read(RESOURCE / "TERMINAL.json")
    need(tuple(resource_end["process"]) == upstream and resource_end["resource_protocol_sha256"] == fingerprint
         and resource_end["formal_completed_sha256"] == sha(path), "Exact completed resource transition")
    verify_retirement()
    return path


def run(upstream):
    fingerprint = verify_resource()
    verify_report()
    need(live(upstream) and key(read(FORMAL / "ATTEMPT_001/PROCESS.json")) == upstream, "Actual replacement scheduler")
    os.sched_setaffinity(0, CPUS)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Metadata report has no GPU")
    started = dict(process=key(identity(os.getpid())), upstream=upstream, resource_protocol_sha256=fingerprint,
                   V04_science_sha256=V04, CPUs=sorted(CPUS), report_source_review_sha256=SOURCE_REVIEW)
    publish(RESOURCE / "REPORT_WAITER_STARTED.json", started)
    try:
        while live(upstream):
            time.sleep(45)
        need(verify_resource() == fingerprint, "Resource protocol drift")
        verify_report()
        final = finish_allowed(upstream, fingerprint)
        subprocess.run([TRAIN_PY, "-B", str(REPORT_SOURCE / "pub_01_s03_ss07_seawave_seen_report.py"),
                        "--output", str(OUTPUT)], check=True)
        artifact = OUTPUT / "REPORT_ARTIFACTS.json"
        need(read(artifact)["formal_completed_sha256"] == sha(final), "Report bound to completed terminal")
        publish(RESOURCE / "REPORT_WAITER_COMPLETED.json",
                dict(**started, formal_completed_sha256=sha(final), report_artifacts_sha256=sha(artifact)))
    except BaseException as exc:
        publish(RESOURCE / "REPORT_WAITER_FAILURE.json", dict(**started, error=str(exc)))
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--upstream", nargs=3, type=int, required=True)
    run(tuple(cli.parse_args().upstream))
