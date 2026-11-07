"""Process and immutable-receipt helpers for the D27 operational handoff."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path("/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_seawave_seen_reference_20261106")
FORMAL = ROOT / "formal"
RESOURCE = ROOT / "resource_concurrency_v01"
ORIGINAL = (1748466, 1012, 791000426)
REPORT = (1812007, 1012, 791424876)
ORIGINAL_CHILDREN = {1024: (1879079, 1012, 794756993), 2026: (1879085, 1012, 794756996)}
V04 = "8442767e6fb1a220bab129d2c2163321e760610102fdefb986f43a06079f7476"
V04_REVIEW = "e5cfcb770962a322e8f25304621478db63a0fdd1835445c0907064400cef033d"
SYSTEM_PY = "/opt/miniconda3/bin/python3"
TRAIN_PY = "/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python"
TRAIN = ROOT / "input_v04/pub_01_s03_ss07_seawave_seen_train.py"
GPU = ("GPU-be77d5f7-abf2-5e41-2a1c-9510e4a832d9", "GPU-cd053687-ccd7-87d3-3f4c-a3beffadfcba")
SEEDS = (1024, 2026, 31415)
JOB_COUNT = 150


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, payload):
    # An exclusive receipt claims its name even if the publisher is interrupted.
    with path.open("x") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    with directory(path.parent) as fd:
        os.fsync(fd)


class directory:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        return self.fd

    def __exit__(self, *_):
        os.close(self.fd)


def key(record):
    return tuple(record[k] for k in ("pid", "uid", "startticks"))


def name(job):
    stage, seed, person = job
    return f"{stage}_{seed}" + (f"_{person}" if person else "")


def folder(job):
    stage, seed, person = job
    base = FORMAL / f"seed{seed}"
    return base / "initial" if stage == "initial" else base / "personal" / person


def trainer_argv(job):
    stage, seed, person = job
    argv = [TRAIN_PY, "-B", str(TRAIN), stage, "--seed", str(seed)]
    return argv + (["--subject", person] if person else [])


def read_fd(fd, relative):
    with os.fdopen(os.open(relative, os.O_RDONLY | os.O_CLOEXEC, dir_fd=fd), "rb") as f:
        return f.read()


def parse_stat(raw, uid):
    prefix, fields = raw.rsplit(b")", 1)
    f = fields.split()
    return dict(pid=int(prefix.split(b" ", 1)[0]), uid=uid, startticks=int(f[19]),
                state=f[0].decode(), ppid=int(f[1]), readable_startcode=int(f[23]) > 0,
                wait_status_field52=int(f[49]))


class Process:
    def __init__(self, process):
        self.expected = process
        self.fd = os.open(f"/proc/{process[0]}", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            self.stat()
        except BaseException:
            self.close()
            raise

    def close(self):
        os.close(self.fd)

    def stat(self):
        status = parse_stat(read_fd(self.fd, "stat"), os.fstat(self.fd).st_uid)
        need(key(status) == self.expected, "Pinned process identity drift")
        return status

    def signal(self, sig):
        self.stat()
        signal.pidfd_send_signal(self.fd, sig)

    def argv(self):
        return [os.fsdecode(item) for item in read_fd(self.fd, "cmdline").split(b"\0") if item]

    def env(self):
        return dict(item.split(b"=", 1) for item in read_fd(self.fd, "environ").split(b"\0") if item)

    def quiescent_zombie(self):
        if self.stat()["state"] != "Z":
            return False
        fd = os.open("task", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=self.fd)
        try:
            return os.listdir(fd) == [str(self.expected[0])]
        finally:
            os.close(fd)

    def stopped(self):
        try:
            fd = os.open("task", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=self.fd)
            try:
                threads = os.listdir(fd)
                return bool(threads) and all(read_fd(fd, f"{tid}/stat").rsplit(b")", 1)[1].split()[0] == b"T"
                                             for tid in threads)
            finally:
                os.close(fd)
        except FileNotFoundError:
            return False


def identity(pid):
    with directory(f"/proc/{pid}") as fd:
        return parse_stat(read_fd(fd, "stat"), os.fstat(fd).st_uid)


def live(process):
    try:
        state = identity(process[0])
        return key(state) == process and state["state"] not in ("Z", "X")
    except FileNotFoundError:
        return False


def verify_retirement():
    retired = read(RESOURCE / "OLD_QUEUE_RETIRED.json")
    superseded = read(RESOURCE / "OLD_REPORT_SUPERSEDED.json")
    replacement = read(RESOURCE / "REPORT_WAITER_STARTED.json")
    need(tuple(retired["old_process"]) == ORIGINAL and retired["observed_exited"] and not live(ORIGINAL),
         "Exact original scheduler retirement")
    need(tuple(superseded["old_process"]) == REPORT and superseded["observed_exited"] and not live(REPORT)
         and superseded["replacement_process"] == replacement["process"], "Exact old reporter supersession")


def until(test, seconds, message):
    deadline = time.monotonic() + seconds
    while not test():
        need(time.monotonic() < deadline, message)
        time.sleep(.1)


def verify_resource(require_review=True):
    manifest = RESOURCE / "RESOURCE_PROTOCOL_V01.json"
    config = read(manifest)
    fingerprint = sha(manifest)
    need(config["V04_science_sha256"] == V04 and config["maximum_workers_per_GPU"] == 2,
         "Frozen operational scope")
    for file, expected in config["sources"].items():
        need(sha(RESOURCE / file) == expected, f"Resource source drift {file}")
    need(sha(ROOT / "PROTOCOL_V04.json") == V04 and sha(ROOT / "COORDINATOR_EXECUTION_REVIEW.json") == V04_REVIEW,
         "Original V04 or review changed")
    science = read(ROOT / "PROTOCOL_V04.json")
    need(sha(ROOT / "INVENTORY.json") == science["inventory_sha256"]
         and all(sha(Path(file)) == expected for file, expected in science["sources"].items()),
         "Original V04 frozen scientific sources")
    if require_review:
        review = read(RESOURCE / "RESOURCE_EXECUTION_REVIEW.json")
        need(review["passed"] and review["resource_protocol_sha256"] == fingerprint
             and review["V04_science_sha256"] == V04, "Independent operational source review")
    return fingerprint
