"""Validate final lightweight S06 outputs without reopening signals."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from pub_01_s06_final_handoff import ROOT, require_complete, require_known_hold


def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args()
    root=Path(args.folder).resolve()
    assert ROOT in root.parents
    output=root/'validation.json'
    if output.exists():raise FileExistsError('Preserve prior validation')
    inventory=rows(root/'download_inventory.csv')
    assert len(inventory)==4715
    assert len({(r['dataset'],r['relative_path']) for r in inventory})==4715
    spar=[r for r in inventory if r['dataset']=='SparrKULee']
    ds=[r for r in inventory if r['dataset']=='ds004703']
    assert len(spar)==4338 and len(ds)==377
    require_known_hold(spar)
    status=json.loads((root/'download_status.json').read_text())
    require_complete(status['SparrKULee'])
    assert sum(r['format_readability']=='ALL_SAMPLES_QC' for r in ds)==281
    assert sum(r['format_readability']=='EXCLUDED_NOT_EXECUTED' for r in ds)==1
    assert sum(r['format_readability'] in ['PASS_FORMAT','PASS_TEXT_ONLY'] for r in ds)==95
    rec=rows(root/'recording_qc.csv')
    counts=dict(Counter(r['record_kind'] for r in rec))
    assert counts==dict(SOURCE_EDF=11,PUBLISHED_EEG_VIRTUAL_VIEW=666,RAW_BDF_FORMAT_QC_NOT_NEW_DERIVATIVE=660)
    derived=[r for r in rec if r['record_kind']=='PUBLISHED_EEG_VIRTUAL_VIEW']
    assert len({r['eeg_path'] for r in derived})==666
    assert all(r['status']=='PASS_VIRTUAL_INITIAL_QC' and int(r['nonfinite'])==0 and float(r['numerical_max_difference'])==0 for r in derived)
    assert sum(r['tail_flag']=='LARGE_TAIL_REVIEW' for r in derived)==26
    excluded=rows(root/'exclusion_inventory.csv')
    assert sum(r['action']=='NOT_REQUESTED' for r in excluded)==196
    assert sum(r['action']=='EXCLUDE_ORPHAN' for r in excluded)==1
    assert sum(r['action']=='EXCLUDE_EXECUTION_AND_DATA_USE' for r in excluded)==1
    manifest=json.loads((root/'cleaning_manifest.json').read_text())
    assert manifest['raw_preserved'] is True
    assert manifest['benchmark_admission']=='HOLD_S01'
    assert manifest['physical_sync']=='UNKNOWN' and manifest['split']=='UNASSIGNED'
    summary=dict(task='PUB-01-S06',validation='PASS',inventory_rows=len(inventory),recording_rows=counts,
                 exclusions_rows=len(excluded),source_corruptions_preserved=1,restricted_not_requested=196,
                 scope='Lightweight schema/count/uniqueness/numeric evidence checks, not new signal scans or scientific admission')
    output.write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
