"""Execute the frozen HDPS subject list on server2203, in isolated processes."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(base, workers):
    if not str(base).startswith('/home/fanyu/auditory_simulation_m6a/'):
        raise ValueError('server2203 project path required')
    split = json.loads((base / 'hdps_split.json').read_text())
    batch = base / 'raw_batch01'
    batch.mkdir(exist_ok=False)
    protocol = {'subjects': list(split), 'workers': workers, 'input': 'published1024Hz BDF',
        'split': str(base / 'hdps_split.json'), 'author_commit': '42a5e1ec9649f4a673bfc018d377bbf38e289ed4',
        'preprocessing': 'GPL brain_pipe0.0.4 / Challenge2024 with100Hz and40Hz lowpass',
        'source_script': 'pub_01_s03_hdps_100hz_v6.py', 'no_test_selection': True}
    (batch / 'protocol_frozen.json').write_text(json.dumps(protocol, indent=2))
    outcomes = []
    def process(subject):
        started = time.time()
        logfile = batch / (subject + '.log')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4')
        with logfile.open('w') as stream:
            result = subprocess.run([sys.executable, str(base / 'pub_01_s03_hdps_100hz_v6.py'),
                '--base', str(base), '--mode', 'full', '--run', 'author01_' + subject,
                '--subject', subject], env=env, stdout=stream, stderr=subprocess.STDOUT)
        output = base / ('preprocessing100_author01_' + subject)
        return {'subject': subject, 'exit_code': result.returncode, 'seconds': time.time() - started,
            'output': str(output), 'log': str(logfile),
            'failed_records': json.loads((output / 'failures.json').read_text()) if (output / 'failures.json').exists() else ['process did not reach loader'],
            'record_count': len(json.loads((output / 'output_check.json').read_text())) if (output / 'output_check.json').exists() else 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(process, subject) for subject in split]
        for future in concurrent.futures.as_completed(jobs):
            row = future.result()
            outcomes.append(row)
            (batch / 'outcomes.json').write_text(json.dumps(outcomes, indent=2))
            print(json.dumps(row), flush=True)
    (batch / 'completed.json').write_text(json.dumps({'subjects': len(outcomes),
        'records': sum(r['record_count'] for r in outcomes), 'all_processes_ok': all(r['exit_code'] == 0 for r in outcomes),
        'failed_records': sum(len(r['failed_records']) for r in outcomes)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    main(args.base, args.workers)
