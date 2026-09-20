"""Combine completed server QC evidence, refusing partial download/format scope.

No signal reads, transformations, hashes, or overwrites. Input snapshots stay intact.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a')
S06 = ROOT / 'pub_01/s06'


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def require_complete(status):
    assert status['expected_public_files'] == status['downloaded_size_match'] == status['format_checked'] == 4142
    assert status['expected_public_bytes'] == status['downloaded_bytes'] == 135967030049
    assert status['missing_or_partial'] == 0
    assert status['restricted_excluded'] == 196
    # One independently confirmed, explicitly excluded orphan source object.
    assert status['format_hold'] == 1


def require_known_hold(inventory):
    holds=[r for r in inventory if r['format_readability']=='HOLD']
    assert len(holds)==1
    assert holds[0]['relative_path']=='derivatives/preprocessed_stimuli/podcast_35-1.data_dict'
    return holds


def decoded_format(row):
    return row.get('format','FAILED_FORMAT')


def tag_dataset(row, dataset):
    assert row.get('dataset',dataset)==dataset
    return dict(row, dataset=dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reconcile', required=True)
    p.add_argument('--channels', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rec, cp, out = [Path(v).resolve() for v in [args.reconcile, args.channels, args.output]]
    assert all(ROOT in path.parents for path in [rec, cp, out])
    status = json.loads((rec/'download_status.json').read_text())
    require_complete(status)
    channels = read(cp/'channel_provenance.csv')
    assert len(channels) == 666
    assert sum(r['status']=='PASS_HEADER_PROVENANCE' for r in channels) == 617
    assert sum(r['source_access']=='RESTRICTED' for r in channels) == 49
    assert not any(r['error']=='RAW_DOWNLOAD_PENDING' for r in channels)
    ds_inv = read(ROOT/'pub_01/review/ds_inventory_20260919_v1/inventory_reconciliation.csv')
    remaining_dir=S06/'ds_remaining_formats_20260920_v1'
    remaining={r['relative_path']:r for r in read(remaining_dir/'remaining_format_inventory.csv')}
    assert len(remaining)==96 and not any(r['status'].startswith('HOLD') for r in remaining.values())
    assert len(ds_inv)==377 and all(r['status']=='SIZE_MATCH' for r in ds_inv)
    assert sum(int(r['actual_bytes']) for r in ds_inv)==14173350514
    inventory = [tag_dataset(r,'SparrKULee') for r in read(rec/'download_inventory.csv')]
    holds = require_known_hold(inventory)
    for r in ds_inv:
        inventory.append(dict(dataset='ds004703', relative_path=r['path'], bytes=r['expected_bytes'],
                              actual_bytes=r['actual_bytes'], restricted=False,
                              download_state='EXISTING_SIZE_MATCH', category=r['category'],
                              format_readability='ALL_SAMPLES_QC' if r['category'] in ['audio','neural'] else remaining[r['path']]['status'],
                              source_scope='Live public listing vs existing v1.1.0; immutable-version identity not established'))
    ds = S06/'ds004703_qc_20260919_v2'
    eeg = S06/'sparrkulee_virtual_20260919_v4'
    recordings = [dict(dataset='ds004703', record_kind='SOURCE_EDF', **r) for r in read(ds/'recording_qc.csv')]
    for r in read(eeg/'cleaning_index.csv'):
        recordings.append(dict(dataset='SparrKULee', record_kind='PUBLISHED_EEG_VIRTUAL_VIEW', **r))
    latest = {}
    with (S06/'sparrkulee_format_qc_20260919_v1/format_qc.jsonl').open() as f:
        for line in f:
            r = json.loads(line); latest[r['relative_path']] = r
    raw_qc = [r for r in latest.values() if decoded_format(r)=='BDF_GZIP']
    assert len(raw_qc)==660 and all(r['all_samples_decoded'] for r in raw_qc)
    for r in raw_qc:
        recordings.append(dict(dataset='SparrKULee', record_kind='RAW_BDF_FORMAT_QC_NOT_NEW_DERIVATIVE',
                               **{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()}))
    exclusions = [dict(dataset='ds004703', action='EXCLUDE', **r) for r in read(ds/'exclusion_inventory.csv')]
    for r in remaining.values():
        if r['status']=='EXCLUDED_NOT_EXECUTED':
            exclusions.append(dict(dataset='ds004703',item=r['relative_path'],kind='legacy_bytecode',action='EXCLUDE_EXECUTION_AND_DATA_USE',reason=r['scope']))
    for r in inventory:
        if str(r['restricted'])=='True':
            exclusions.append(dict(dataset='SparrKULee', item=r['relative_path'], kind='access', action='NOT_REQUESTED', reason='OFFICIAL_RESTRICTED'))
    exclusions.append(dict(dataset='SparrKULee',item=holds[0]['relative_path'],kind='source_object',action='EXCLUDE_ORPHAN',reason='Independent refetch identical bytes and same truncated pickle; no EEG/envelope references'))
    for r in read(eeg/'cleaning_index.csv'):
        if r['tail_flag']=='LARGE_TAIL_REVIEW':
            exclusions.append(dict(dataset='SparrKULee',item=r['eeg_path'],kind='tail',action='REVIEW_FLAG_NOT_EXCLUSION',reason='Common-prefix tail exceeds frozen 192-sample advisory'))
    for r in read(cp/'official_bad_recordings.csv'):
        exclusions.append(dict(dataset='SparrKULee',item=r['source_recording'],kind='official_bad',action='EXCLUDE_RAW_ALIGNMENT',reason=r['official_reason']))
    for r in raw_qc:
        if r['flat_channels']:
            exclusions.append(dict(dataset='SparrKULee',item=r['relative_path'],kind='raw_flat_channels',action='FLAG_RAW_NOT_CONSUMED',reason=json.dumps(r['flat_channels'])))
        if r['quality_status']!='CALIBRATION_VALID':
            exclusions.append(dict(dataset='SparrKULee',item=r['relative_path'],kind='raw_calibration',action='HOLD_RAW_NOT_CONSUMED',reason=r['quality_status']))
    out.mkdir(parents=True,exist_ok=False)
    save(out/'download_inventory.csv', inventory)
    save(out/'recording_qc.csv', recordings)
    save(out/'exclusion_inventory.csv', exclusions)
    download = dict(task='PUB-01-S06',status='PUBLIC_SCOPE_SIZE_RECONCILED_WITH_SOURCE_EXCLUSION',
                    SparrKULee=status,ds004703=dict(files=377,bytes=14173350514,source='Existing assets reused; live inventory only',
                                                  all_sample_readability_files=281,remaining_format_checked=96,
                                                  remaining_statuses=dict(Counter(r['status'] for r in remaining.values())),
                                                  anatomy_not_used_in_benchmark=27,other_metadata=69))
    (out/'download_status.json').write_text(json.dumps(download,indent=2))
    manifest = dict(task='PUB-01-S06',status='READY_FOR_REVIEW',benchmark_admission='HOLD_S01',
                    physical_sync='UNKNOWN',split='UNASSIGNED',raw_preserved=True,
                    datasets=[json.loads((d/'cleaning_manifest.json').read_text()) for d in [ds,eeg]],
                    raw_format_counts=dict(Counter(decoded_format(r) for r in latest.values())),
                    raw_bdf_quality_counts=dict(Counter(r['quality_status'] for r in raw_qc)),
                    recording_rows=dict(Counter(r['record_kind'] for r in recordings)),
                    evidence=dict(reconcile=str(rec),channel_provenance=str(cp),
                                  stimulus_roles=str(S06/'raw_stimulus_roles_20260920_v2'),
                                  ds_remaining_formats=str(remaining_dir),
                                  independent_raw_stimulus=str(S06/'review_raw_stimulus_complete_20260920_v1')),
                    limitations=['196 restricted not requested; 49 derivative raw headers inaccessible plus one public-raw stimulation file restricted',
                                 '26 large-tail flags; historical publisher generator revision unknown',
                                 'ds004703 Catalan playback provenance and immutable identity unresolved',
                                 'Virtual views only: no new filtering, normalization, resampling, ICA or model training',
                                 'Raw format/size PASS is not all-channel quality or physiological alignment PASS'])
    (out/'cleaning_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(output=str(out),recording_rows=manifest['recording_rows'],inventory_rows=len(inventory),exclusions=len(exclusions))))


if __name__=='__main__':main()
