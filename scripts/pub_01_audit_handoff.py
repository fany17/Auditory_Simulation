"""Materialize current S01 audit tables from verified lightweight S06 evidence.

No signal reads or new preprocessing; no train/test assignment. Keeps initial
failed audit files unchanged and writes a new dated canonical audit snapshot.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a')


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); args = p.parse_args()
    out = Path(args.output).resolve()
    if ROOT not in out.parents:
        raise ValueError('Server project only')
    out.mkdir(parents=True, exist_ok=False)
    ds = ROOT / 'pub_01/s06/ds004703_qc_20260919_v2'
    eeg = ROOT / 'pub_01/s06/sparrkulee_virtual_20260919_v4'
    stim = ROOT / 'pub_01/s06/sparrkulee_stimulus_audit_20260919_v2'
    dp, ep = read(ds / 'audio_neural_pairing_audit.csv'), read(eeg / 'cleaning_index.csv')
    rec = {r['recording']:r for r in read(ds / 'recording_qc.csv')}
    sr = {r['story']:r for r in read(stim / 'stimulus_provenance.csv')}
    status = json.loads((ROOT / 'pub_01/s06/sparrkulee_v3.1/download_status.json').read_text())
    format_status=json.loads((ROOT/'pub_01/s06/sparrkulee_format_qc_20260919_v1/status.json').read_text())
    format_complete=format_status['files_checked']==4142
    dataset = [
        dict(dataset='ds004703', version='1.1.0', source='https://openneuro.org/datasets/ds004703/versions/1.1.0',
             license='CC0 metadata + noncommercial/no-reidentification README', subjects=len({r['subject'] for r in dp}),
             recordings=len(rec), recording_scope='existing EDF recordings', audio_assets=270,
             audio_asset_definition='all bundled WAV assets including controls/questions/duplicates; not distinct story count',
             server_raw=str(ROOT/'data/ds004703/v1.1.0'),
             access='PUBLIC', download='EXISTING_RECONCILED_377_SIZE_ONLY', initial_qc='PASS_WITH_LIMITATION',
             benchmark_admission='HOLD', unresolved='immutable snapshot identity; physical sync; grouped split; Catalan provenance'),
        dict(dataset='SparrKULee', version='3.1', source='https://doi.org/10.48804/K3VSND',
             license='CC-BY-NC-4.0', subjects=len({r['subject'] for r in ep}), recordings=len(ep),
             recording_scope='independent published preprocessed EEG derivatives; raw inventory separate', audio_assets=73,
             audio_asset_definition='published envelope NPY count; 73 data_dict objects audited separately; raw stimulus/trigger/noise NPZ inventory separate',
             server_raw=str(ROOT/'pub_01/s06/sparrkulee_v3.1/raw'), access='4142 PUBLIC / 196 RESTRICTED_NOT_ACCESSED',
             download=status['state'], initial_qc='666_VIRTUAL_DERIVATIVES_PASS_WITH_LIMITATION; '+('full format scope checked, one source exclusion' if format_complete else 'full format QC pending'),
             benchmark_admission='HOLD', unresolved='historical preprocessing revision; physical sync; grouped split; 26 large tails; orphan corrupted data_dict')]
    save(out/'dataset_manifest.csv',dataset)
    licenses = [
        dict(dataset='SparrKULee', version='3.1', license='CC-BY-NC-4.0', noncommercial_use='PERMITTED_PUBLIC_SCOPE',
             commercial_use='FORBIDDEN', redistribution='Attribution + license/source + modification notices; noncommercial only; no restricted access bypass',
             derivatives='Same noncommercial boundary; no unconditional commercial artifact release', restricted_access='HOLD_USER_GRANTED_ACCESS_REQUIRED',
             evidence='https://rdr.kuleuven.be/api/datasets/:persistentId/?persistentId=doi:10.48804/K3VSND; https://creativecommons.org/licenses/by-nc/4.0/legalcode.en'),
        dict(dataset='ds004703', version='1.1.0', license='CC0 metadata; README additional restrictions retained', noncommercial_use='CURRENT_PROJECT_AUDIT_ALLOWED_CONSERVATIVELY',
             commercial_use='FORBIDDEN_BY_README_PROJECT_BOUNDARY', redistribution='HOLD: conflicting license signals not adjudicated; no unconditional redistribution approval',
             derivatives='Noncommercial only in this task; future distribution gate still HOLD', restricted_access='Do not reidentify; no private clinical data',
             evidence='https://raw.githubusercontent.com/OpenNeuroDatasets/ds004703/master/README; https://openneuro.org/datasets/ds004703/versions/1.1.0')]
    save(out/'license_use_boundary.csv',licenses)
    groups, timings, pairs, exclusions = [],[],[],[]
    for row in dp:
        key=row['recording']+'__trial-'+row['trial']
        groups.append(dict(dataset='ds004703', sample_key=key, subject=row['subject'], session=row['session'],
                           recording=row['recording'], story=row['story'], block=row['block'], eligible=row['eligible'],
                           split='UNASSIGNED', leakage_guard='No adjacent-time random split; subject/session/story keys retained; train-only fitted transforms'))
        timings.append(dict(dataset='ds004703', sample_key=key, neural_rate_hz=rec[row['recording']]['sample_rate_hz'],
                            audio_rate_hz='SEE_AUDIO_QC_PER_PATH', neural_offset_s=row['offset_s'], duration_s=row['audio_duration_s'],
                            alignment_rule='template word/POS/phone+duration match; constant onset offset',
                            residual_s=row['offset_spread_s'], physical_sync='UNKNOWN', source=str(ds/'audio_neural_pairing_audit.csv')))
        pairs.append(dict(dataset='ds004703', sample_key=key, neural_source=rec[row['recording']]['server_raw_path'],
                          audio_or_envelope=row['audio_path'], story=row['story'], pairing='METADATA_ONLY',
                          status='PASS_WITH_LIMITATION' if row['eligible']=='True' else 'EXCLUDED', reason=row['reason']))
        if row['eligible']!='True':
            exclusions.append(dict(dataset='ds004703', item=key, reason=row['reason'], action='EXCLUDE_SEGMENT'))
    for row in ep:
        key=Path(row['eeg_path']).name
        groups.append(dict(dataset='SparrKULee',sample_key=key,subject=row['subject'],session=row['session'],
                           recording=row['source_recording'].removesuffix('.gz'),story=row['story'],block='UNKNOWN',eligible=row['status'],
                           split='UNASSIGNED',leakage_guard='No adjacent-time random split; repeated story and subject grouping; train-only fitted transforms'))
        timings.append(dict(dataset='SparrKULee',sample_key=key,neural_rate_hz=64,audio_rate_hz='ENVELOPE_64; raw audio rate separately audited',
                            neural_offset_s=0,duration_s=float(row['samples'])/64,
                            alignment_rule='publisher common prefix consumption; no re-normalization/split',residual_s='UNKNOWN',
                            physical_sync='UNKNOWN',source=str(eeg/'cleaning_index.csv')))
        provenance = sr.get(row['story'],{})
        pairs.append(dict(dataset='SparrKULee',sample_key=key,neural_source=row['eeg_path'],audio_or_envelope=row['envelope_path'],
                          story=row['story'],pairing='PUBLISHER_MAPPING_AND_COMMON_OVERLAP',status='PASS_WITH_LIMITATION',
                          reason=row['tail_flag']+'; data_dict='+provenance.get('status','UNKNOWN')+'; historical_generator_UNKNOWN'))
        if row['tail_flag']=='LARGE_TAIL_REVIEW':
            exclusions.append(dict(dataset='SparrKULee',item=key,reason='large tail: '+row['eeg_tail_samples']+' EEG samples; preserved advisory, not silently removed',action='REVIEW_FLAG'))
    exclusions.append(dict(dataset='SparrKULee',item='podcast_35-1.data_dict',reason='official served object truncated; identical independent refetch; no exact-story EEG references/envelope',action='EXCLUDE_ORPHAN_OBJECT'))
    save(out/'group_split_keys.csv',groups); save(out/'timing_event_inventory.csv',timings)
    save(out/'audio_neural_pairing_audit.csv',pairs); save(out/'exclusion_inventory.csv',exclusions)
    summary=dict(task='PUB-01-S01',status='REVIEW',dataset_count=2,group_rows=len(groups),pair_rows=len(pairs),
                 split_assignments='NONE',physical_sync='UNKNOWN',
                 exclusions_scope='segment/external-object/large-tail flags; full channel exclusions remain in ds v2 channel_qc',
                 initial_failed_audit='S01 root original 438 HOLD table retained; current snapshot supersedes duplicate-path heuristic only',
                 scope='Audit snapshot generated from actual S06 evidence, not completed benchmark admission')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary))


if __name__=='__main__':
    main()
