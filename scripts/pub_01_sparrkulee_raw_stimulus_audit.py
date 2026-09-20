"""Actual NPZ sample rates and waveform-to-publisher-data_dict provenance.

Consumes preserved expanded NPZs from full format QC. Never infers raw audio
rate from the data_dict stimulus_sr (which was overwritten to envelope rate).
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from pub_01_s06_format_qc import NumericUnpickler, ROOT


def classify_role(path, comparison):
    name = Path(path).name.removesuffix('.gz')
    if comparison == 'EXACT':
        return 'PUBLISHED_STIMULUS_AUDIO', 'Exact waveform match to publisher stimulus_data; does not prove speech content or playback'
    if name.startswith('t_') or name == 'triggers.npz':
        return 'TRIGGER_BY_FILENAME', 'Filename only; trigger function not independently established'
    return 'AUXILIARY_ROLE_UNVERIFIED', 'No exact published stimulus_data match; noise/swn/name alone does not establish role'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--metadata-only-source', help='Revise roles from prior complete CSV without reopening waveforms')
    args=p.parse_args()
    out=Path(args.output).resolve()
    if ROOT not in out.parents:raise ValueError('Server project only')
    out.mkdir(parents=True,exist_ok=False)
    if args.metadata_only_source:
        source=Path(args.metadata_only_source).resolve()
        if ROOT not in source.parents:raise ValueError('Server project only')
        with source.open() as f:rows=list(csv.DictReader(f))
        for row in rows:
            row['role'],row['role_basis']=classify_role(row['source_npz_gzip'],row['data_dict_comparison'])
        save_results(out,rows,metadata_source=str(source))
        return
    raw=ROOT/'pub_01/s06/sparrkulee_v3.1/raw'
    expanded=ROOT/'pub_01/s06/sparrkulee_format_qc_20260919_v1/expanded_archives'
    completed={}
    with (expanded.parent/'format_qc.jsonl').open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            completed[r['relative_path']]=r
    with (ROOT/'pub_01/s01/audit_20260919_v1/sparrkulee_official_inventory.csv').open() as f:
        official=[r for r in csv.DictReader(f) if r['restricted']=='False' and r['relative_path'].endswith('.npz.gz')]
    rows=[]
    for item in official:
        name=Path(item['relative_path']).name.removesuffix('.gz');path=expanded/name
        role='TRIGGER' if name.startswith('t_') else 'NOISE' if name.startswith('noise_') else 'AUDIO'
        row=dict(source_npz_gzip=item['relative_path'],expanded_server_path=str(path),role=role,
                 sample_rate_hz='UNKNOWN',samples='UNKNOWN',duration_s='UNKNOWN',status='PENDING_FORMAT_QC',
                 data_dict_comparison='NOT_APPLICABLE',max_abs_difference='UNKNOWN',error='')
        if completed.get(item['relative_path'],{}).get('status')!='PASS_FORMAT_READABILITY' or not path.is_file():
            rows.append(row);continue
        try:
            with np.load(path,allow_pickle=False) as z:
                audio=z['audio'];rate=float(np.asarray(z['fs']).item())
                if not np.isfinite(rate) or rate<=0 or audio.dtype.hasobject:raise ValueError('Invalid actual NPZ rate/dtype')
                row.update(sample_rate_hz=rate,samples=len(audio),duration_s=len(audio)/rate,status='PASS_RAW_NPZ_METADATA')
                if role=='AUDIO':
                    dp=raw/'derivatives/preprocessed_stimuli'/(Path(name).stem+'.data_dict')
                    if not dp.is_file():row['data_dict_comparison']='NO_MATCHING_PUBLISHED_DATA_DICT'
                    else:
                        with dp.open('rb') as f:data=NumericUnpickler(f).load()
                        saved=np.asarray(data['stimulus_data'])
                        if saved.shape!=audio.shape:
                            row.update(data_dict_comparison='SHAPE_MISMATCH',status='HOLD')
                        else:
                            difference=0.0
                            for start in range(0,len(audio),1048576):
                                difference=max(difference,float(np.max(np.abs(audio[start:start+1048576].astype(np.float64)-saved[start:start+1048576]))))
                            row.update(data_dict_comparison='EXACT' if difference==0 else 'VALUE_MISMATCH',max_abs_difference=difference)
                            if difference:row['status']='HOLD'
                        del data,saved
                del audio
        except Exception as exc:row.update(status='HOLD',error=type(exc).__name__+': '+str(exc))
        rows.append(row)
    for row in rows:
        row['role'],row['role_basis']=classify_role(row['source_npz_gzip'],row['data_dict_comparison'])
    save_results(out,rows)


def save_results(out, rows, metadata_source=None):
    from collections import Counter
    with (out/'raw_stimulus_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(expected_npz=len(rows),available=sum(r['status']!='PENDING_FORMAT_QC' for r in rows),
                 hold=sum(r['status']=='HOLD' for r in rows),exact_data_dict_audio_matches=sum(r['data_dict_comparison']=='EXACT' for r in rows),
                 rate_basis='NPZ fs scalar, never the overwritten data_dict stimulus_sr',physical_sync='UNKNOWN',
                 roles=dict(Counter(r['role'] for r in rows)),metadata_only_source=metadata_source,
                 role_limitation='Filename roles provisional; exact publisher match is not proof of natural speech/playback')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
