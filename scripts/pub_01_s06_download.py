"""Resumable public-only SparrKULee download on server2203. No hashes.

The official inventory is frozen input. Restricted rows are never requested.
Existing complete-sized files are reused but size is not an integrity proof.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

ROOT = Path('/home/fanyu/auditory_simulation_m6a')


def download(row, raw):
    target = (raw / row['relative_path']).resolve()
    if raw not in target.parents:
        raise ValueError('Unsafe relative path')
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = int(row['bytes'])
    partial = target.with_name(target.name + '.partial')
    if target.exists():
        if target.stat().st_size == expected:
            return dict(row, status='EXISTING_SIZE_MATCH', actual_bytes=expected, error='')
        return dict(row, status='CONFLICT_PRESERVED', actual_bytes=target.stat().st_size, error='Existing target size mismatch')
    last_error = ''
    for attempt in range(3):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            if start > expected:
                raise ValueError('Partial oversized; retained for review')
            if start < expected:
                headers = {'User-Agent': 'PUB-01-noncommercial-research-audit'}
                if start:
                    headers['Range'] = f'bytes={start}-'
                request = urllib.request.Request(row['source_url'], headers=headers)
                with urllib.request.urlopen(request, timeout=90) as response:
                    if start and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-')):
                        raise ValueError('Resume unsupported; partial retained, no overwrite')
                    with partial.open('ab' if start else 'wb') as f:
                        while block := response.read(1024 * 1024):
                            f.write(block)
            actual = partial.stat().st_size
            if actual != expected:
                raise ValueError(f'Size mismatch: {actual} vs {expected}')
            partial.rename(target)
            return dict(row, status='DOWNLOADED_SIZE_MATCH', actual_bytes=actual, error='')
        except Exception as exc:
            last_error = f'{type(exc).__name__}: {exc}'
            time.sleep(2 * (attempt + 1))
    return dict(row, status='FAILED_PARTIAL_RETAINED', actual_bytes=partial.stat().st_size if partial.exists() else 0, error=last_error)


def priority(row):
    path = row['relative_path']
    # Fetch core provenance then derivatives for early QC, without waiting for
    # thousands of tiny sidecars to finish. Remaining public scope is retained.
    core = '/' not in path or path.endswith('/.save_metadata.json')
    return (0 if core else 1 if path.startswith('derivatives/') else 2 if int(row['bytes']) < 200000 else 3, int(row['bytes']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise RuntimeError('Server only')
    out = Path(args.output).resolve()
    if ROOT not in out.parents:
        raise ValueError('Outside server project')
    out.mkdir(parents=True, exist_ok=True)
    raw = out / 'raw'
    raw.mkdir(exist_ok=True)
    lock = out / 'download.lock'
    # Lock is never deleted; flock releases automatically when the process exits.
    import fcntl
    with lock.open('a') as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with open(args.inventory) as f:
            rows = list(csv.DictReader(f))
        selected = sorted([r for r in rows if r['restricted'] == 'False'], key=priority)
        expected = sum(int(r['bytes']) for r in selected)
        status = dict(task='PUB-01-S06', dataset='SparrKULee', version='3.1', pid=os.getpid(),
                      expected_public_files=len(selected), expected_public_bytes=expected,
                      restricted_excluded=sum(r['restricted'] != 'False' for r in rows),
                      completed_files=0, completed_bytes=0, failed_files=0, state='RUNNING',
                      raw_server_path=str(raw), integrity='Size only; full format QC separate; no cryptographic verification')
        fields = list(rows[0]) + ['status', 'actual_bytes', 'error']
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        with (out / f'download_inventory_{run_id}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            def save():
                status['updated_utc'] = datetime.now(timezone.utc).isoformat()
                tmp = out / 'download_status.json.tmp'
                tmp.write_text(json.dumps(status, indent=2))
                tmp.replace(out / 'download_status.json')
            save()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(download, row, raw) for row in selected]
                for future in as_completed(futures):
                    result = future.result()
                    writer.writerow(result)
                    f.flush()
                    if result['status'] in ['EXISTING_SIZE_MATCH', 'DOWNLOADED_SIZE_MATCH']:
                        status['completed_files'] += 1
                        status['completed_bytes'] += int(result['actual_bytes'])
                    else:
                        status['failed_files'] += 1
                    save()
                    print(status['completed_files'], status['completed_bytes'], result['status'], result['relative_path'], flush=True)
            status['state'] = 'COMPLETE_SIZE_CHECK_ONLY' if not status['failed_files'] else 'PARTIAL_WITH_FAILURES'
            save()


if __name__ == '__main__':
    main()
