"""Execute the fixed independent D26 CPU checks once their final inputs exist."""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_contrastive_history_20261106')
OUT = ROOT / 'coordinator_final_replay_v01'
SOURCE = OUT / 'COORDINATOR_FINAL_REPLAY.py'
SOURCE_SHA = 'c3f748355121532ec4bf77bc9b990bd3803f25d105d185c903d0031b08e448a9'
PLAN_SHA = 'b0c4c28eb43b972d7b83807d9431f97e8dc7d066e5becd337f34bc522e40aa10'
GATE_SHA = '8c249eabeb9edb9a5360006f1d3dabb9552e2a22ff649b9bae72801f75311b4d'
ORIGINAL_QUEUE = (1622219, 1012, 790259158)
CASES = [(s, a) for s in (1024, 2026, 31415) for a in ('HH', 'CH', 'CC')]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def identity(pid):
    p = Path('/proc') / str(pid)
    try:
        f = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return (pid, p.stat().st_uid, int(f[19])), f[0]
    except FileNotFoundError:
        return None, None


def live_original():
    key, state = identity(ORIGINAL_QUEUE[0])
    return key == ORIGINAL_QUEUE and state not in ('Z', 'X')


def publish(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def wait_receipt(path):
    while True:
        if path.exists():
            try:
                value = read(path)
            except json.JSONDecodeError:
                if not live_original():
                    raise RuntimeError('Original D26 exited with incomplete JSON; no restart')
            else:
                if not value['passed']:
                    raise RuntimeError('D26 terminal receipt is not passing')
                return value
        if not live_original():
            raise RuntimeError('Original D26 ended before this complete input; no restart')
        time.sleep(30)


def verify_sources():
    assert sha(SOURCE) == SOURCE_SHA and sha(OUT / 'COORDINATOR_FINAL_REPLAY_PLAN.md') == PLAN_SHA
    assert sha(ROOT / 'PROTOCOL_V05.json') == GATE_SHA
    e = read(OUT / 'PREFLIGHT.json')
    assert e['passed'] and e['source_sha256'] == SOURCE_SHA and e['plan_sha256'] == PLAN_SHA


def main():
    verify_sources()
    os.sched_setaffinity(0, set(range(44, 48)))
    process, _ = identity(os.getpid())
    publish('QUEUE_STARTED.json', dict(process=process, original_D26=ORIGINAL_QUEUE,
            source_sha256=SOURCE_SHA, plan_sha256=PLAN_SHA, queue_sha256=sha(Path(__file__)),
            cases=CASES, started_utc=datetime.now(timezone.utc).isoformat(), automatic_training_restart=False))
    print('D26_CPU_REPLAY_WAIT', json.dumps(process), flush=True)
    try:
        env = os.environ | dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        results = []
        for seed, arm in CASES:
            terminal = ROOT / 'formal' / f'seed{seed}' / f'READOUT_{arm}' / 'COMPLETED.json'
            value = wait_receipt(terminal)
            assert value['seed'] == seed and value['arm'] == arm and value['epoch'] == 4
            verify_sources()
            result = OUT / f'REPLAY_{seed}_{arm}.json'
            assert not result.exists() and not (OUT / f'FAILURE_{seed}_{arm}.json').exists()
            before = time.monotonic()
            command = [sys.executable, '-B', str(SOURCE), '--seed', str(seed), '--arm', arm]
            with (OUT / f'REPLAY_{seed}_{arm}.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                child_identity, _ = identity(child.pid)
                publish(f'PROCESS_{seed}_{arm}.json', dict(process=child_identity, argv=command))
                code = child.wait()
            publish(f'EXIT_{seed}_{arm}.json', dict(returncode=code, observed_spawn_to_exit_seconds=time.monotonic()-before))
            assert code == 0 and result.exists(), (seed, arm, code)
            receipt = read(result)
            assert receipt['passed'] and receipt['source_sha256'] == SOURCE_SHA
            results.append(receipt)
            print('REPLAY_PASSED', seed, arm, flush=True)
        terminal = wait_receipt(ROOT / 'formal/COMPLETED.json')
        assert terminal['protocol_sha256'] == GATE_SHA
        assert {(r['seed'], r['arm']) for r in terminal['evaluations']} == set(CASES)
        counts = {k: sum(r['actual_calls'][k] for r in results) for k in results[0]['actual_calls']}
        assert counts == dict(backbone_windows=1290, native_unused_projection_windows=1290,
                             final_projection_windows=600, head_windows=600,
                             manual_memory_query_windows=465, library_memory_query_windows=465)
        publish('COMPLETED.json', dict(passed=True, process=process, source_sha256=SOURCE_SHA,
                protocol_sha256=GATE_SHA, cases=CASES, actual_calls=counts,
                CPU_seconds=sum(r['CPU_seconds'] for r in results),
                worker_wall_seconds_sum=sum(r['wall_seconds'] for r in results),
                case_receipt_sha256={f'{s}_{a}': sha(OUT / f'REPLAY_{s}_{a}.json') for s, a in CASES},
                new_fits=0, optimizer_updates=0, old_test_reads=0, GPU=0,
                scope='Fixed first VAL record/block; all nine final models and original HH interventions. Not whole-cohort raw replay.'))
    except BaseException:
        publish('QUEUE_FAILURE.json', dict(process=process, traceback=traceback.format_exc(),
                D26_not_restarted=True, completed_cases=[str(p.name) for p in OUT.glob('REPLAY_*.json')]))
        raise


if __name__ == '__main__':
    main()
