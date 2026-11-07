"""Keep ownership of each new V04 worker's true exit receipt."""
import argparse
import os
import subprocess
import sys
import time

from d27_resource_process import (FORMAL, GPU, RESOURCE, SEEDS, folder, identity, key, name, need, publish,
                                  read, trainer_argv)

ATTEMPT = FORMAL / "ATTEMPT_001"
# Keep training blocked until its original executable and exact identity are registered.
GATED_EXEC = "import os,sys;fd=int(sys.argv[1]);token=os.read(fd,1);os.close(fd);(os.execv(sys.argv[2],sys.argv[2:]) if token==b'1' else sys.exit(74))"


def supervise(job, gpu, release_fd):
    with os.fdopen(release_fd, "rb", buffering=0) as gate:
        need(gate.read(1) == b"1", "Training case was never released")
    label = name(job)
    need(gpu in GPU and os.environ.get("CUDA_VISIBLE_DEVICES") == gpu, "Assigned original GPU environment")
    need(not folder(job).exists(), "Case already running, complete, or partially written")
    if job[0] == "personal":
        need(read(folder(("initial", job[1], None)) / "COMPLETED.json")["passed"], "Own initial is complete")
    started = time.monotonic()
    argv = trainer_argv(job)
    with (ATTEMPT / f"{label}.log").open("x") as log:
        rd, wr = os.pipe()
        try:
            child = subprocess.Popen([sys.executable, "-B", "-c", GATED_EXEC, str(rd), *argv],
                                     pass_fds=(rd,), stdout=log, stderr=subprocess.STDOUT)
            os.close(rd)
            rd = None
            info = identity(child.pid)
            need(info["ppid"] == os.getpid(), "Actual train process is owned by this supervisor")
            publish(ATTEMPT / f"{label}_PROCESS.json",
                    dict(pid=info["pid"], uid=info["uid"], startticks=info["startticks"],
                         status=info["state"], argv=argv, gpu=gpu, started_monotonic=started,
                         supervisor=key(identity(os.getpid())), origin="V04 exec through gated supervisor"))
            need(os.write(wr, b"1") == 1, "Registered training release")
        finally:
            if rd is not None:
                os.close(rd)
            os.close(wr)
        code = child.wait()
    completed = folder(job) / "COMPLETED.json"
    diagnostic = None
    try:
        receipt = read(completed)
        done = bool(receipt.get("passed") and key(receipt["process"]) == key(info))
    except (OSError, ValueError, KeyError) as exc:
        done, diagnostic = False, str(exc)
    publish(ATTEMPT / f"{label}_EXIT.json",
            dict(pid=child.pid, returncode=code, complete=done, worker_wall_seconds_observed=time.monotonic()-started,
                 exit_source="actual Popen.wait from the training worker's supervisor", completion_check_error=diagnostic))
    need(code == 0 and done, f"Actual training failure {label}: {code}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("initial", "personal"))
    cli.add_argument("--seed", type=int, choices=SEEDS, required=True)
    cli.add_argument("--subject")
    cli.add_argument("--gpu", required=True)
    cli.add_argument("--release-fd", type=int, required=True)
    args = cli.parse_args()
    need(bool(args.subject) == (args.stage == "personal"), "Valid case key")
    supervise((args.stage, args.seed, args.subject), args.gpu, args.release_fd)
