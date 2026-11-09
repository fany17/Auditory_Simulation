"""D27 resource-only scheduler migration, with an irreversible release boundary."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from d27_resource_process import (FORMAL, GPU, JOB_COUNT, ORIGINAL, ORIGINAL_CHILDREN, REPORT, RESOURCE, ROOT,
                                  SEEDS, SYSTEM_PY, TRAIN_PY, Process, folder, identity, key, live, name,
                                  need, publish, read, sha, trainer_argv, until, verify_resource, verify_retirement)

OLD_ATTEMPT = FORMAL / "ATTEMPT_000"
ATTEMPT = FORMAL / "ATTEMPT_001"
SUPERVISORS = ATTEMPT / "supervisors"
PILOT = ("initial", 31415, None)
GIB_MIB = 1024
MIN_FREE = 16 * GIB_MIB


def initial(seed):
    return "initial", seed, None


def confirmed_old(parent, children, reporter, stopped):
    p = parent.stat()
    need(p["state"] not in ("X", "Z") and p["readable_startcode"], "Original parent is readable and alive")
    need(p["state"] == "T" and parent.stopped() if stopped else p["state"] != "T", "Original queue stop state")
    need(parent.argv() == [TRAIN_PY, "-B", str(ROOT / "input_v04/pub_01_s03_ss07_seawave_seen_queue.py")],
         "Original queue executable")
    need(key(read(OLD_ATTEMPT / "PROCESS.json")) == ORIGINAL and
         tuple(read(ROOT / "REPORT_WAIT_STARTED.json")["process"]) == REPORT, "Original registration")
    child_files = sorted(OLD_ATTEMPT.glob("*_PROCESS.json"))
    need({p.name for p in child_files} == {f"{name(initial(seed))}_PROCESS.json" for seed in ORIGINAL_CHILDREN}
         and not list(OLD_ATTEMPT.glob("*_EXIT.json")), "Original queue has exactly its first two jobs")
    actual_children = {int(pid) for pid in parent_children(parent)}
    need(actual_children == {handle.expected[0] for handle in children.values()}, "Exact original kernel child set")
    for job, handle in children.items():
        record = read(OLD_ATTEMPT / f"{name(job)}_PROCESS.json")
        state = handle.stat()
        need(key(record) == handle.expected and state["ppid"] == ORIGINAL[0] and state["state"] not in ("T", "Z", "X")
             and state["readable_startcode"] and handle.argv() == trainer_argv(job) == record["argv"],
             f"Original worker identity and command {name(job)}")
        need(record["gpu"] in GPU and handle.env().get(b"CUDA_VISIBLE_DEVICES") == record["gpu"].encode(),
             f"Exact original GPU binding {name(job)}")
        need(not (folder(job) / "COMPLETED.json").exists(), "This transition adopts two active initial fits")
    rpt = reporter.stat()
    argv = reporter.argv()
    need(rpt["state"] not in ("T", "Z", "X") and argv[:2] == [TRAIN_PY, "-B"]
         and (ROOT / argv[2]).resolve() == ROOT / "report_trigger_v01/pub_01_s03_ss07_seawave_seen_report_wait.py",
         "Original report waiter identity and source")
    need(not folder(PILOT).exists() and not (FORMAL / "COMPLETED.json").exists(), "Pilot and terminal must be new")


def parent_children(parent):
    # /proc task children is per-thread. Collect all children, including a dispatch from another thread.
    from d27_resource_process import read_fd
    fd = os.open("task", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC, dir_fd=parent.fd)
    try:
        return [pid for tid in os.listdir(fd) for pid in read_fd(fd, f"{tid}/children").split()]
    finally:
        os.close(fd)


def inventory_jobs():
    subjects = read(ROOT / "INVENTORY.json")["subjects"]
    jobs = [initial(seed) for seed in SEEDS]
    jobs.extend(("personal", seed, subject) for seed in SEEDS for subject in subjects)
    need(len(set(subjects)) == 49 and len(set(jobs)) == JOB_COUNT, "Original 3+147 matrix")
    return jobs


def child_environment(children):
    first, second = [handle.env() for handle in children.values()]
    first.pop(b"CUDA_VISIBLE_DEVICES")
    second.pop(b"CUDA_VISIBLE_DEVICES")
    need(first == second and first[b"OMP_NUM_THREADS"] == first[b"OPENBLAS_NUM_THREADS"] == b"4",
         "Common original runtime environment")
    return first


def observed_devices():
    args = ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"]
    memory = {uuid: (int(used), int(total)) for uuid, used, total in csv_rows(args)}
    args = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    clients = {uuid: set() for uuid in GPU}
    for uuid, pid in csv_rows(args):
        if uuid in clients:
            clients[uuid].add(int(pid))
    need(set(GPU) <= memory.keys(), "Both original GPU UUIDs visible")
    return memory, clients


def csv_rows(argv):
    text = subprocess.check_output(argv, text=True, timeout=10)
    return [tuple(value.strip() for value in line.split(",")) for line in text.splitlines() if line.strip()]


def choose_gpu(assignments, registered, devices):
    memory, clients = devices
    need(all(assignments.count(gpu) <= 2 for gpu in GPU), "Operational occupancy cap")
    choices = [gpu for gpu in GPU if assignments.count(gpu) < 2 and clients[gpu] <= registered[gpu]
               and memory[gpu][1]-memory[gpu][0] >= MIN_FREE]
    return min(choices, key=lambda g: (assignments.count(g), memory[g][0], GPU.index(g))) if choices else None


def assignment_state(old, active):
    assignments, registered = [], {gpu: set() for gpu in GPU}
    for job, handle in old.items():
        gpu = read(OLD_ATTEMPT / f"{name(job)}_PROCESS.json")["gpu"]
        assignments.append(gpu)
        registered[gpu].add(handle.expected[0])
    for job, (_, gpu) in active.items():
        assignments.append(gpu)
        path = ATTEMPT / f"{name(job)}_PROCESS.json"
        if path.is_file():
            registered[gpu].add(read(path)["pid"])
    return assignments, registered


def timing_record(log, process):
    matches = []
    for line in log.read_text().splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if row.get("event") == "worker_timing":
                matches.append(row)
    need(len(matches) == 1 and key(matches[0]["process"]) == process, "One actual final worker timing line")
    return matches[0]


def old_exit(parent, job, handle):
    row = handle.stat()
    need(parent.stopped() and handle.quiescent_zombie() and row["ppid"] == ORIGINAL[0],
         "Original stopped parent retains fully terminated zombie")
    need(row == handle.stat(), "Stable exact zombie evidence")
    raw = row["wait_status_field52"]
    code = os.waitstatus_to_exitcode(raw)
    prefix = name(job)
    original = read(OLD_ATTEMPT / f"{prefix}_PROCESS.json")
    terminal_path = folder(job) / "COMPLETED.json"
    done, diagnostic, timing = False, None, None
    try:
        receipt = read(terminal_path)
        timing = timing_record(OLD_ATTEMPT / f"{prefix}.log", handle.expected)
        done = bool(receipt["passed"] and key(receipt["process"]) == handle.expected
                    and receipt["stage"] == job[0] and receipt["seed"] == job[1] and receipt["subject"] == job[2]
                    and receipt["complete_epochs"] == 201 and receipt["actual_updates"] == 201 * 23600
                    and receipt["old_test_reads"] == 0)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        diagnostic = str(exc)
    measured = time.monotonic()
    evidence = dict(proc_stat_field=52, proc_state=row, observation_monotonic=measured,
                    observed_with_stopped_parent=ORIGINAL, field_reader_uid=os.getuid(), actual_returncode=code,
                    worker_timing=timing, completed_receipt_sha256=sha(terminal_path) if done else None,
                    check_error=diagnostic)
    proof = RESOURCE / f"{prefix}_KERNEL_EXIT.json"
    publish(proof, evidence)
    publish(OLD_ATTEMPT / f"{prefix}_EXIT.json",
            dict(pid=handle.expected[0], returncode=code, complete=done,
                 worker_wall_seconds_observed=measured-original["started_monotonic"],
                 exit_source="Linux proc stat field 52 of exact unreaped zombie",
                 independent_exit_receipt=str(proof)))
    return code == 0 and done


def prepare_worker(job, gpu, environment):
    label = name(job)
    need(not folder(job).exists() and not (ATTEMPT / f"{label}_PROCESS.json").exists(),
         f"Never start an existing case {label}")
    rd, wr = os.pipe()
    try:
        args = [SYSTEM_PY, "-B", str(RESOURCE / "d27_resource_worker.py"), job[0],
                "--seed", str(job[1]), "--gpu", gpu, "--release-fd", str(rd)]
        if job[2]:
            args += ["--subject", job[2]]
        with (SUPERVISORS / f"{label}.log").open("x") as log:
            supervisor = subprocess.Popen(args, pass_fds=(rd,), start_new_session=True,
                env=environment | {b"CUDA_VISIBLE_DEVICES": gpu.encode()}, stdout=log, stderr=subprocess.STDOUT)
        publish(SUPERVISORS / f"{label}.json",
                dict(supervisor=key(identity(supervisor.pid)), case=job, gpu=gpu, training_argv=trainer_argv(job)))
        return supervisor, wr
    except BaseException:
        os.close(wr)
        raise
    finally:
        os.close(rd)


def release_worker(fd):
    try:
        need(os.write(fd, b"1") == 1, "Released exactly one supervisor")
    finally:
        os.close(fd)


def start_reporter(upstream):
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                        "MKL_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1",
                        "MPLCONFIGDIR": str(RESOURCE / "matplotlib_cache")}
    cmd = [SYSTEM_PY, "-B", str(RESOURCE / "d27_resource_report_wait.py"), "--upstream", *map(str, upstream)]
    with (RESOURCE / "REPORT_WAITER_CONSOLE.log").open("x") as log:
        child = subprocess.Popen(cmd, env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    path = RESOURCE / "REPORT_WAITER_STARTED.json"
    until(lambda: path.exists() or child.poll() is not None, 15, "Replacement reporter did not register")
    row = read(path)
    need(tuple(row["process"]) == key(identity(child.pid)) and tuple(row["upstream"]) == upstream
         and child.poll() is None, "Replacement reporter live and bound")
    return row


def supersede_reporter(reporter, replacement):
    proof = ROOT / "REPORT_WAIT_STARTED.json"
    reporter.signal(signal.SIGTERM)
    until(lambda: not live(REPORT), 10, "Old report waiter failed to exit")
    publish(RESOURCE / "OLD_REPORT_SUPERSEDED.json",
            dict(old_process=REPORT, old_started_sha256=sha(proof), observed_exited=True,
                 replacement_process=replacement["process"], reason="Scheduler identity changed",
                 signal_sent="SIGTERM", old_exit_code=None))


def retire_parent(parent):
    need(parent.stopped(), "Original parent still stopped")
    need(all((OLD_ATTEMPT / f"{name(initial(seed))}_EXIT.json").is_file() for seed in ORIGINAL_CHILDREN),
         "Both genuine original worker exits preserved before parent retirement")
    parent.signal(signal.SIGKILL)
    until(lambda: not live(ORIGINAL), 10, "Old parent failed to exit")
    publish(RESOURCE / "OLD_QUEUE_RETIRED.json",
            dict(old_process=ORIGINAL, observed_exited=True, signal_sent="SIGKILL",
                 exit_code=None, reason="Both original children ended with independent kernel exit evidence"))


def done(job, status):
    return job in status and status[job]


def run_schedule(parent, old, active, pending, environment, completion):
    errors = []
    retired = False
    while old or active or (pending and not errors):
        if not retired:
            need(parent.stopped(), "Old parent no longer paused. Stop further dispatch.")
        for job, handle in list(old.items()):
            if handle.quiescent_zombie():
                completion[job] = old_exit(parent, job, handle)
                if not completion[job]:
                    errors.append(f"Original worker failed: {name(job)}")
                del old[job]
        if not old and not retired:
            retire_parent(parent)
            retired = True
        for job, (supervisor, _) in list(active.items()):
            code = supervisor.poll()
            if code is not None:
                outcome = ATTEMPT / f"{name(job)}_EXIT.json"
                value = read(outcome) if outcome.exists() else {}
                completion[job] = bool(code == 0 and value.get("complete") and value["returncode"] == 0)
                if not completion[job]:
                    errors.append(f"New worker or supervisor failed: {name(job)}; supervisor code {code}")
                del active[job]
        if not errors and pending:
            job = next((j for j in pending if j[0] == "initial" or done(initial(j[1]), completion)), None)
            if job is not None:
                assignments, registered = assignment_state(old, active)
                gpu = choose_gpu(assignments, registered, observed_devices())
                if gpu is not None:
                    supervisor, gate = prepare_worker(job, gpu, environment)
                    release_worker(gate)
                    active[job] = supervisor, gpu
                    pending.remove(job)
        if old or active or (pending and not errors):
            need(bool(old or active) or any(j[0] == "initial" or done(initial(j[1]), completion) for j in pending),
                 "No real initial source for pending personal workers")
            time.sleep(2)
    need(not errors, " | ".join(errors))
    return completion


def finalize():
    fingerprint = verify_resource()
    ready = read(RESOURCE / "READY_FINALIZE.json")
    me = key(identity(os.getpid()))
    need(tuple(ready["process"]) == me == key(read(ATTEMPT / "PROCESS.json"))
         and ready["resource_protocol_sha256"] == fingerprint and ready["successful_cases"] == JOB_COUNT,
         "Only the actual replacement may finalize the frozen science")
    try:
        verify_retirement()
        sys.path.insert(0, str(ROOT / "input_v04"))
        import pub_01_s03_ss07_seawave_seen_queue as science
        science.finish()
        final = FORMAL / "COMPLETED.json"
        need(key(read(final)["process"]) == me, "Terminal has actual replacement identity")
        publish(RESOURCE / "TERMINAL.json", dict(process=me, resource_protocol_sha256=fingerprint,
                formal_completed_sha256=sha(final), preserved_V04=True))
    except BaseException as exc:
        publish(RESOURCE / "FINALIZE_FAILURE.json", dict(process=me, error=str(exc)))
        raise


def migration(preflight):
    fingerprint = verify_resource(require_review=not preflight)
    need(sys.executable == SYSTEM_PY and hasattr(signal, "pidfd_send_signal"), "Verified coordination interpreter")
    need(not ATTEMPT.exists() and not (RESOURCE / "LEASE.json").exists(), "Exclusive fresh migration")
    with ExitStack() as stack:
        def pin(process):
            handle = Process(process)
            stack.callback(handle.close)
            return handle
        parent, reporter = pin(ORIGINAL), pin(REPORT)
        children = {initial(seed): pin(process) for seed,process in ORIGINAL_CHILDREN.items()}
        confirmed_old(parent, children, reporter, stopped=False)
        jobs = inventory_jobs()
        environment = child_environment(children)
        assignments, registered = assignment_state(children, {})
        devices = observed_devices()
        gpu = choose_gpu(assignments, registered, devices)
        need(gpu is not None, "Two original jobs and room for exactly one 31415 initial")
        if preflight:
            print(json.dumps(dict(state="READY_FOR_REVIEW", scientific_protocol_verified=True,
                  original_parent=ORIGINAL, original_children=list(ORIGINAL_CHILDREN.values()), selected_pilot_gpu=gpu,
                  original_reports=REPORT, signals_sent=0, models_launched=0)))
            return
        owner = key(identity(os.getpid()))
        publish(RESOURCE / "LEASE.json", dict(process=owner, resource_protocol_sha256=fingerprint,
                                              original_parent=ORIGINAL, children=list(ORIGINAL_CHILDREN.values())))
        stopped, committed, gate = False, False, None
        try:
            stopped = True
            parent.signal(signal.SIGSTOP)
            until(parent.stopped, 10, "Old queue did not stop completely")
            confirmed_old(parent, children, reporter, stopped=True)
            publish(RESOURCE / "ORIGINAL_PAUSED.json", dict(process=ORIGINAL, children=list(ORIGINAL_CHILDREN.values())))
            ATTEMPT.mkdir()
            SUPERVISORS.mkdir()
            publish(ATTEMPT / "PROCESS.json", dict(pid=owner[0], uid=owner[1], startticks=owner[2],
                    resource_protocol_sha256=fingerprint, science="V04", original_attempt="ATTEMPT_000"))
            assignments, registered = assignment_state(children, {})
            devices = observed_devices()
            gpu = choose_gpu(assignments, registered, devices)
            need(gpu is not None, "Pilot GPU availability changed after stop")
            publish(RESOURCE / "PILOT_GPU_ADMISSION.json", dict(gpu=gpu, memory_MiB=devices[0],
                observed_GPU_PIDs={k:sorted(v) for k,v in devices[1].items()}, maximum_workers_per_GPU=2))
            supervisor, gate = prepare_worker(PILOT, gpu, environment)
            confirmed_old(parent, children, reporter, stopped=True)
            publish(RESOURCE / "PILOT_RELEASE_INTENT.json", dict(case=PILOT, gpu=gpu,
                    supervisor=key(identity(supervisor.pid)), process=owner))
            # Commit before releasing the pipe. Even a crash during write cannot cause a duplicate.
            committed = True
            release_fd, gate = gate, None
            release_worker(release_fd)
            publish(RESOURCE / "PILOT_RELEASED.json", dict(case=PILOT, gpu=gpu, process=owner))
            replacement = start_reporter(owner)
            supersede_reporter(reporter, replacement)
            active = {PILOT:(supervisor,gpu)}
            pending = [job for job in jobs if job not in children and job != PILOT]
            completion = run_schedule(parent, children.copy(), active, pending, environment, {})
            need(set(completion) == set(jobs) and all(completion.values()), "Exactly 150 successful registered fits")
            publish(RESOURCE / "READY_FINALIZE.json", dict(process=owner, successful_cases=len(completion),
                    resource_protocol_sha256=fingerprint))
            os.execve(TRAIN_PY, [TRAIN_PY, "-B", __file__, "--finalize"], os.environ | {"CUDA_VISIBLE_DEVICES": ""})
        except BaseException as exc:
            if gate is not None:
                os.close(gate)
            committed = committed or (RESOURCE / "PILOT_RELEASE_INTENT.json").exists()
            resumed, rollback_error = False, None
            if stopped and not committed and live(ORIGINAL):
                try:
                    parent.signal(signal.SIGCONT)
                    until(lambda: not parent.stopped(), 10, "Original queue remained stopped after rollback")
                    resumed = True
                    publish(RESOURCE / "ROLLBACK.json", dict(old_process=ORIGINAL, action="SIGCONT before pilot release",
                            observed_alive=live(ORIGINAL), replacement_launch_released=False))
                except BaseException as err:
                    rollback_error = str(err)
            publish(RESOURCE / "SCHEDULER_FAILURE.json", dict(process=owner, error=str(exc), committed=committed,
                    old_parent_resumed_before_release=resumed, rollback_error=rollback_error, automatic_retry=False))
            raise


def cancel(signum, _frame):
    raise RuntimeError(f"Replacement coordinator received signal {signum}")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, cancel)
    cli = argparse.ArgumentParser(description=__doc__)
    flags = cli.add_mutually_exclusive_group(required=True)
    flags.add_argument("--preflight", action="store_true")
    flags.add_argument("--migrate", action="store_true")
    flags.add_argument("--finalize", action="store_true")
    args = cli.parse_args()
    if args.finalize:
        finalize()
    else:
        migration(args.preflight)
