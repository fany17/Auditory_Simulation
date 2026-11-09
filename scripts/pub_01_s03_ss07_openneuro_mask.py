"""Build prediction-independent layout from pinned raw lengths and actual audio targets.

This is provisional until full EEG readability/preprocessing is checked; not an inference gate.
"""
import hashlib,json,re
from pathlib import Path
import numpy as np
from pub_01_s03_ss07_openneuro_transport import ROOT
from pub_01_s03_ss07_openneuro_metrics import mechanical_mask,synthetic_check


def main():
    source=json.loads((ROOT/'ANNEX_CATALOG_V2.json').read_text())
    audio=json.loads((ROOT/'AUDIO_PREPARE_COMPLETED.json').read_text())
    assert audio['passed'] and len(audio['audios'])==20
    rows=[]
    for entry in source['objects']:
        if not entry['path'].endswith('.eeg'):continue
        subject,run=re.match(r'(sub-\d+)/eeg/sub-\d+_task-listening_run-(\d+)_eeg.eeg',entry['path']).groups()
        raw_frames,remain=divmod(entry['declared_bytes'],128*4);assert remain==0
        target_path=ROOT/'targets'/f'audio{run}.npy'
        target=np.load(target_path,allow_pickle=False)
        record_length=min((raw_frames+7)//8,len(target))
        row=dict(subject=subject,run=run,record=Path(entry['path']).stem,
            raw_frames=raw_frames,raw_sha256=entry['payload_sha256'],
            target_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
            EEG_readability_status='NOT_YET_FULL_ACQUIRED',**mechanical_mask(target[:record_length]))
        rows.append(row)
    assert len(rows)==380 and len({r['subject'] for r in rows})==19
    j=dict(status='PROVISIONAL_LAYOUT_NOT_FINAL_PREDICTION_FREEZE',prediction_independent=True,
        constant_rule='Exact max==min on covered target points',rows=rows,
        records=len(rows),people=19,eligible_blocks=sum(sum(b['eligible'] for b in r['blocks']) for r in rows),
        windows=sum(r['windows'] for r in rows),processed_EEG_bytes=sum(64*r['length']*4+128 for r in rows),
        three_seed_cache_bytes=sum(r['windows']*(1536+192)*4 for r in rows)*3+6*128,
        no_NN=True,EEG_unit_status='UNRESOLVED')
    with (ROOT/'PROVISIONAL_DATA_MASK.json').open('x') as f:json.dump(j,f,indent=2,allow_nan=False)
    check=synthetic_check()
    with (ROOT/'STATISTICS_SYNTHETIC_CHECK.json').open('x') as f:json.dump(check,f,indent=2)
    print(json.dumps({k:v for k,v in j.items() if k!='rows'}))
    print(json.dumps(check))


if __name__=='__main__':main()
