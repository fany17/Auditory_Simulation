"""One-shot D16 review after the identified formal queue completes; 2203 only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_token_normalization_20261102')
PYTHON = '/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python'
SOURCES = {
    'pub_01_s03_ss07_token_norm_independent_review.py': '1bd999ec8d1e09bb3f7141f4ee9f8ab54ad8594f8bcf333c1d1c0ccf814e15f2',
    'pub_01_s03_ss07_full_ft_history_independent_review.py': '0113a30704b0fe2f13ae0e23aa53b2527afa29cfefc3c0c784789f384229cc64',
}


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    with (ROOT / name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    start, cpu = time.time(), time.process_time()
    queue = read(ROOT / 'QUEUE_STARTED.json')
    assert queue['pid'] == args.queue_pid
    deadline = queue['time'] + 21600
    assert sha(ROOT / 'PROTOCOL.json') == args.protocol_sha
    for name, digest in SOURCES.items():
        assert sha(ROOT / 'coordinator_input' / name) == digest
    assert not (ROOT / 'coordinator_review').exists()
    while not (ROOT / 'QUEUE_COMPLETED.json').exists():
        assert not (ROOT / 'QUEUE_FAILURE.json').exists(), 'Formal queue failed; no retry'
        assert time.time() < deadline, 'Frozen queue deadline exceeded'
        proc = Path(f'/proc/{args.queue_pid}')
        try:
            fields = (proc / 'stat').read_text().split()
            valid = (fields[2] != 'Z' and fields[21] == args.queue_start_ticks
                     and proc.stat().st_uid == args.queue_uid
                     and str(ROOT / 'input/pub_01_s03_ss07_token_norm_queue.py').encode()
                     in (proc / 'cmdline').read_bytes())
        except FileNotFoundError:
            valid = False
        if not valid:
            # The queue may finish between reading its receipt and /proc.
            assert (ROOT / 'QUEUE_COMPLETED.json').exists(), 'Queue identity lost without completion'
            break
        time.sleep(10)
    completed = read(ROOT / 'QUEUE_COMPLETED.json')
    assert len(completed['workers']) == 9 and all(x['returncode'] == 0 for x in completed['workers'])
    assert sha(ROOT / 'PROTOCOL.json') == args.protocol_sha
    for name, digest in SOURCES.items():
        assert sha(ROOT / 'coordinator_input' / name) == digest
    write('COORDINATOR_REVIEW_RUN_STARTED.json', dict(
        time=time.time(), runner_pid=os.getpid(), protocol_sha256=args.protocol_sha,
        source_pins=SOURCES, max_NN_windows=631239, optimizer_updates=0, fit=0))
    code = subprocess.run([
        PYTHON, str(ROOT / 'coordinator_input/pub_01_s03_ss07_token_norm_independent_review.py'),
        '--protocol-sha', args.protocol_sha], cwd=ROOT, check=False).returncode
    assert code == 0, ('Independent reviewer failed', code)
    result = read(ROOT / 'coordinator_review/REVIEW.json')
    assert result['passed']
    write('COORDINATOR_REVIEW_RUN_COMPLETED.json', dict(
        time=time.time(), runner_pid=os.getpid(), returncode=code,
        review_sha256=sha(ROOT / 'coordinator_review/REVIEW.json'), review_passed=True,
        runner_only_CPU_seconds=time.process_time() - cpu,
        wait_and_child_wall_seconds=time.time() - start,
        child_cost_record='coordinator_review/REVIEW.json', extra_NN_beyond_child=0,
        goal_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--queue-pid', type=int, required=True)
    parser.add_argument('--queue-start-ticks', required=True)
    parser.add_argument('--queue-uid', type=int, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except BaseException:
        write('COORDINATOR_REVIEW_RUN_FAILURE.json', dict(
            time=time.time(), runner_pid=os.getpid(), traceback=traceback.format_exc(), automatic_retry=False))
        raise
