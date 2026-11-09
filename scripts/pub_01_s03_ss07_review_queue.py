"""Read-only review queue waiting for the frozen evaluation, never retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss07_baseline import ROOT, read, dump, update, sha


def checked_protocol():
    protocol = read(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json')
    assert protocol['frozen'] and protocol['independent_code_review_completed']
    assert protocol['executor_protocol_sha256'] == sha(ROOT/'EVALUATION_EXECUTOR_PROTOCOL.json')
    for path, digest in dict(protocol['sources'], **protocol['proofs']).items(): assert sha(path) == digest
    return protocol


def upstream():
    pid = read(ROOT/'FINAL_EVAL_QUEUE_LEASE.json')['pid']
    command = Path(f'/proc/{pid}/cmdline')
    assert command.exists() and b'pub_01_s03_ss07_eval_queue.py' in command.read_bytes(), 'evaluation ended without complete summary'
    return pid


def main():
    protocol = checked_protocol()
    dump(ROOT/'INDEPENDENT_REVIEW_QUEUE_LEASE.json', {'pid': os.getpid(), 'time': time.time(),
        'protocol_sha256': sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json'), 'no_automatic_retries': True})
    deadline = time.time()+14*86400
    while not (ROOT/'final_summary/COMPLETED.json').exists():
        if (ROOT/'FINAL_EVAL_QUEUE_FAILURE.json').exists(): raise RuntimeError('upstream evaluation failure, preserve and review')
        if time.time() > deadline: raise RuntimeError('evaluation wait expired')
        checked_protocol(); pid = upstream()
        update(ROOT/'INDEPENDENT_REVIEW_QUEUE_STATUS.json', {'stage': 'WAIT_COMPLETE_FIXED_EVALUATION',
            'actual_live_upstream_pid': pid, 'time': time.time(), 'test_or_DTU_payload_read': False})
        time.sleep(30)
    from pub_01_s03_ss07_review_metrics import gated
    gated(); results = []
    for script, args in [('review_raw', ['--device', 'cuda:0']), ('review_metrics', [])]:
        checked_protocol()
        update(ROOT/'INDEPENDENT_REVIEW_QUEUE_STATUS.json', {'stage': script.upper(), 'time': time.time()})
        command = [sys.executable, str(ROOT/f'input/pub_01_s03_ss07_{script}.py')] + args
        environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        with (ROOT/f'{script}.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            update(ROOT/'INDEPENDENT_REVIEW_QUEUE_STATUS.json', {'stage': script.upper(),
                'child_pid': process.pid, 'time': time.time()})
            code = process.wait()
        results.append({'script': script, 'exit': code})
        # Metrics remain useful if raw replay fails, but either failure blocks
        # the combined technical pass. Never retry or modify scientific data.
    assert all(r['exit'] == 0 for r in results), results
    assert read(ROOT/'independent_raw_review/RAW_INPUT_REVIEW.json')['passed']
    assert read(ROOT/'independent_review/METRICS_REVIEW.json')['passed']
    dump(ROOT/'INDEPENDENT_REVIEW_COMPLETED.json', {'technical_review_passed': True, 'children': results,
        'protocol_sha256': sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json'), 'time': time.time(),
        'scientific_goal_complete': False, 'scientific_acceptance_and_interpretation_pending': True,
        'no_training_model_selection_or_score_threshold_changes': True})
    update(ROOT/'INDEPENDENT_REVIEW_QUEUE_STATUS.json', {'stage': 'TECHNICAL_REVIEWS_PASSED_SCIENCE_PENDING',
        'time': time.time(), 'scientific_goal_complete': False})


if __name__ == '__main__':
    try: main()
    except Exception as error:
        update(ROOT/'INDEPENDENT_REVIEW_QUEUE_FAILURE.json', {'error': repr(error), 'time': time.time(),
            'preserve_all_evidence_no_automatic_retries': True})
        raise
