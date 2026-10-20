"""Codex server-only source-group audit and independent synthetic reruns."""
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path('/home/fanyu/auditory_simulation_m6a')
OUT=ROOT/'pub_01/s03_history_memory_20261019/codex_run01'


def dump(path,obj):
    with path.open('x') as f:json.dump(obj,f,indent=2)


def main():
    dump(OUT/'started.json',{'pid':os.getpid(),'utc_epoch':time.time(),'task':'PUB-01-S03_SS05','stage':'CODEX_SERVER_EXECUTION_STARTED','PR3_merged_commit':'8d21d0a0bc3b22ec9fe2d27c7bd5522beb379c6f'})
    prior=json.loads((ROOT/'pub_01/s03_architecture_20261019/cache01/manifest.json').read_text())
    records=[r for r in prior['records'] if r['phase']=='train']
    with (ROOT/'pub_01/s02_acoustic_baseline_20260924/split_manifest.csv').open() as f:mapping={r['source_recording']:r for r in csv.DictReader(f)}
    groups=defaultdict(list)
    for r in records:
        m=mapping[r['record']];assert m['source_family'] and m['subject']==r['subject'] and m['raw_story']==r['story']
        groups[m['source_family']].append(dict(r,source_family=m['source_family'],family_mapping_basis=m['family_mapping_basis'],old_phase='SS03_train'))
    names=sorted(groups);assert len(names)>=6,'insufficient family groups for prospective3way split'
    rng=np.random.default_rng(20261019);names=[names[i] for i in rng.permutation(len(names))]
    ntest=max(2,len(names)//5);nval=max(2,len(names)//5)
    phase_names={'holdout':names[:ntest],'validation':names[ntest:ntest+nval],'train':names[ntest+nval:]}
    selected=[]
    for phase,families in phase_names.items():
        for family in families:
            for row in groups[family]:selected.append(dict(row,phase=phase))
    assert not(set(phase_names['train'])&set(phase_names['validation']) or set(phase_names['train'])&set(phase_names['holdout']) or set(phase_names['validation'])&set(phase_names['holdout']))
    summary={p:{'families':phase_names[p],'records':sum(r['phase']==p for r in selected),'subjects':len({r['subject'] for r in selected if r['phase']==p})} for p in phase_names}
    split={'task':'PUB-01-S03_SS05','source':'SS03_train_only; old exposed test payloads and scores never loaded',
        'type':'prospective_INTERNAL_holdout; source was previously used for SS03 training, not external independent data',
        'algorithm':'sorted real source_family IDs; numpy RNG20261019 permutation; floor20percent at least2 families each for holdout and validation; remainder train',
        'freeze_before_model_fitting':True,'source_family_groups':len(names),'phases':summary,'records':selected,
        'old_normalization_reuse':'FORBIDDEN_FOR_SS05_TRAIN_ONLY_CLAIM; old stats include new validation/holdout; refit on newtrain separately',
        'physical_sync':'UNKNOWN','excluded_old_SS03_test_records':len(prior['records'])-len(records)}
    dump(OUT/'split_frozen.json',split);print(json.dumps({'real_metadata_split':summary,'total_families':len(names)}),flush=True)
    stage=OUT/'synthetic01';stage.mkdir(exist_ok=False)
    commands=[('unit_tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_pub_01_s03_ss05_predictive_memory.py','-v']),
        ('cue160',[sys.executable,'pub_01_s03_ss05_synthetic_probe.py','--steps','160','--output',str(stage/'cue160.json')]),
        ('key160',[sys.executable,'pub_01_s03_ss05_keyed_probe.py','--steps','160','--output',str(stage/'key160.json')]),
        ('gru600',[sys.executable,'pub_01_s03_ss05_keyed_probe.py','--steps','600','--variants','gru','--output',str(stage/'gru600.json')])]
    for name,command in commands:
        print(json.dumps({'stage':name,'utc_epoch':time.time()}),flush=True)
        with (stage/(name+'.log')).open('x') as log:subprocess.run(command,cwd=Path(__file__).parent,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    dump(OUT/'intake_completed.json',{'status':'ACTUAL_METADATA_AND_SYNTHETIC_RERUN_COMPLETE','public_EEG_scores':False,'next':'newtrain normalization, continuous loader, fivegroup trainval pilot'})


if __name__=='__main__':main()
