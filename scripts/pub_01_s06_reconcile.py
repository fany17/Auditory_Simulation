"""Freeze current public expected/download/format status by file, no hashes.

Read-only inventory reconciliation; new output snapshot each invocation.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a')


def kind(path):
    for ext in ['.bdf.gz', '.npz.gz', '.npy.gz', '.data_dict', '.npy', '.json', '.tsv', '.apr', '.apx', '.xml', '.docx']:
        if path.endswith(ext):
            return ext
    return Path(path).suffix or 'TEXT_OTHER'


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output', required=True); args=p.parse_args()
    out=Path(args.output).resolve()
    if ROOT not in out.parents:
        raise ValueError('Server project only')
    out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'pub_01/s06/sparrkulee_v3.1'
    with (ROOT/'pub_01/s01/audit_20260919_v1/sparrkulee_official_inventory.csv').open() as f:
        official=list(csv.DictReader(f))
    log=ROOT/'pub_01/s06/sparrkulee_format_qc_20260919_v1/format_qc.jsonl'
    latest={}
    with log.open() as f:
        for line in f:
            try:
                row=json.loads(line)
            except json.JSONDecodeError:
                # Running writer may have an incomplete final line; not counted.
                continue
            latest[row['relative_path']]=row
    rows=[]; formats=defaultdict(Counter)
    for row in official:
        rel=row['relative_path']; path=base/'raw'/rel; restricted=row['restricted']=='True'
        expected=int(row['bytes']); actual=path.stat().st_size if path.is_file() else 0
        partial=path.with_name(path.name+'.partial')
        state='RESTRICTED_NOT_REQUESTED' if restricted else 'DOWNLOADED_SIZE_MATCH' if path.is_file() and actual==expected else 'SIZE_MISMATCH' if path.is_file() else 'PARTIAL' if partial.exists() else 'MISSING'
        qc=latest.get(rel,{})
        entry=dict(row,actual_bytes=actual,partial_bytes=partial.stat().st_size if partial.exists() else 0,
                   download_state=state,format_readability=qc.get('status','NOT_CHECKED'),
                   decoded_format=qc.get('format','UNKNOWN'),quality_status=qc.get('quality_status','SEE_DEDICATED_QC_OR_UNKNOWN'),
                   error=qc.get('error',''),source_object_exclusion=rel.endswith('/podcast_35-1.data_dict'))
        rows.append(entry)
        if not restricted:
            count=formats[kind(rel)];count['expected']+=1;count['expected_bytes']+=expected
            count['downloaded_size_match']+=state=='DOWNLOADED_SIZE_MATCH'
            count['format_checked']+=bool(qc)
            count['format_hold']+=qc.get('status')=='HOLD'
    with (out/'download_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary_rows=[dict(format=k,**dict(v)) for k,v in sorted(formats.items())]
    with (out/'format_coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary_rows[0]));w.writeheader();w.writerows(summary_rows)
    public=[r for r in rows if r['restricted']=='False']
    process=json.loads((base/'download_status.json').read_text())
    try:
        os.kill(int(process['pid']),0);alive=True
    except ProcessLookupError:
        alive=False
    status=dict(task='PUB-01-S06',dataset='SparrKULee',version='3.1',
                 timestamp_utc=datetime.now(timezone.utc).isoformat(),expected_public_files=len(public),
                 expected_public_bytes=sum(int(r['bytes']) for r in public),restricted_excluded=len(rows)-len(public),
                 downloaded_size_match=sum(r['download_state']=='DOWNLOADED_SIZE_MATCH' for r in public),
                 downloaded_bytes=sum(r['actual_bytes'] for r in public if r['download_state']=='DOWNLOADED_SIZE_MATCH'),
                 missing_or_partial=sum(r['download_state']!='DOWNLOADED_SIZE_MATCH' for r in public),
                 format_checked=sum(r['format_readability']!='NOT_CHECKED' for r in public),
                 format_hold=sum(r['format_readability']=='HOLD' for r in public),
                 download_pid=process['pid'],download_process_alive=alive,
                 state='FULL_PUBLIC_SIZE_RECONCILED' if all(r['download_state']=='DOWNLOADED_SIZE_MATCH' for r in public) else 'PARTIAL_ACTIVE' if alive else 'PARTIAL_PROCESS_STOPPED',
                 integrity='Bytes/decoder checks only; no digest proof',benchmark_admission='HOLD_S01')
    (out/'download_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))


if __name__=='__main__':
    main()
