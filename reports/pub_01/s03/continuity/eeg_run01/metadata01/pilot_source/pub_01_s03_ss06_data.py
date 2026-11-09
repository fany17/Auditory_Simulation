"""SS06 server-only metadata freeze and newly fitted continuous EEG cache."""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import numpy as np

PUBLIC = Path('/home/fanyu/auditory_simulation_m6a')
ROOT = PUBLIC/'pub_01/s03_continuity_20261023/eeg_run01'
LENGTH = 3840
WARMUP = 3840
LAG = 8


def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2)


def read(path):
    return json.loads(Path(path).read_text())


def freeze():
    old = read(PUBLIC/'pub_01/s03_history_memory_20261019/codex_run01/split_frozen.json')
    earlier = read(PUBLIC/'pub_01/s03_architecture_20261019/cache01/manifest.json')
    old_train = [r for r in old['records'] if r['phase']=='train']
    index = {r['record']: r for r in earlier['records']}
    with (PUBLIC/'pub_01/s02_acoustic_baseline_20260924/split_manifest.csv').open() as f:
        mapping = {r['source_recording']: r for r in csv.DictReader(f)}
    names = sorted({r['source_family'] for r in old_train})
    assert len(names)==23 and len(old_train)==353
    order = np.random.default_rng(20261023).permutation(names).tolist()
    groups = {'audit_holdout':order[:5], 'validation':order[5:10], 'train':order[10:]}
    rows = []
    for r in old_train:
        m = mapping[r['record']]
        assert m['subject']==r['subject'] and m['raw_story']==r['story'] and m['source_family']==r['source_family']
        assert index[r['record']]['phase']=='train'
        phase = next(p for p, families in groups.items() if r['source_family'] in families)
        rows.append(dict(r, phase=phase, old_SS05_phase='train', old_SS03_phase='train'))
    summary = {p:{'families':fs, 'records':sum(r['phase']==p for r in rows),
                  'subjects':len({r['subject'] for r in rows if r['phase']==p}),
                  'full_60s_score_blocks':sum(max(0,(r['common_samples']-LAG-WARMUP)//LENGTH) for r in rows if r['phase']==p)}
               for p, fs in groups.items()}
    dump(ROOT/'SPLIT_PROVENANCE.json', {'task':'PUB-01-S03_SS06', 'created_epoch':time.time(),
         'kind':'prospective INTERNAL / EXPLORATORY audit, not independent external confirmation',
         'history':'Every selected source was used in SS03 and SS05 training/development; old validation/test groups excluded.',
         'algorithm':'sorted SS05-train 23 source families; numpy RNG20261023 permutation; first5 audit, next5 validation, remaining13 train',
         'records':rows, 'phases':summary, 'frozen_before_any_SS06_fitting_or_audit_payload':True,
         'lag_samples':LAG, 'lag_basis':'fixed previously train/validation-selected 8 samples; no SS06 lag search',
         'coordinates':'x[8+s:8+s+3840], envelope[s:s+3840]; EEG response is 125ms later than sound',
         'warmup_samples':WARMUP, 'score_samples':LENGTH,
         'tail_rule':'complete60s only after initial60s warmup; incomplete tail excluded equally in all modes',
         'training_stream':'first60s warmup then first two adjacent60s updates per record, batch1; randomized record order per seed/epoch; retain detached states between adjacent updates; reset at record/end-of-selected-span',
         'normalization':'refit new train EEG channels and scalar envelope mean/std; old files untouched',
         'physical_sync':'UNKNOWN', 'person_overlap':'allowed; source-family holdout, not unseen-person generalization'})
    print(json.dumps(summary), flush=True)


def oriented(row):
    for key in ('eeg_path','envelope_path'):
        assert str(Path(row[key])).startswith(str(PUBLIC)+'/')
    a=np.load(row['eeg_path'], mmap_mode='r')
    assert a.ndim==2 and a.shape.count(64)==1
    return a.T if a.shape[0]==64 else a


def fit():
    rows=[r for r in read(ROOT/'SPLIT_PROVENANCE.json')['records'] if r['phase']=='train']
    count=0; mean=np.zeros(65); m2=np.zeros(65)
    for ri,r in enumerate(rows):
        x=oriented(r); y=np.load(r['envelope_path'], mmap_mode='r').reshape(-1); n=r['common_samples']
        assert min(len(x),len(y))>=n
        for first in range(0,n,65536):
            last=min(n,first+65536)
            z=np.column_stack((x[first:last],y[first:last])).astype(np.float64)
            assert np.isfinite(z).all()
            k=len(z); cm=z.mean(0); delta=cm-mean; total=count+k
            m2+=((z-cm)**2).sum(0)+delta**2*(count*k/total)
            mean+=delta*(k/total); count=total
        if ri%30==0:print(json.dumps({'normalization_train_records':ri+1}),flush=True)
    std=np.sqrt(m2/count); assert (std>0).all() and np.isfinite(std).all()
    target=ROOT/'newtrain_EEG_target_normalization.npz'
    assert not target.exists(); np.savez(target,mean=mean,std=std)
    dump(ROOT/'normalization_contract.json',{'fit_phase':'SS06_train_only','fit_records':[r['record'] for r in rows],
         'fit_samples':count,'EEG_channels':64,'envelope_mean':float(mean[-1]),'envelope_std':float(std[-1]),
         'old_statistics_used':False,'holdout_or_validation_fit':False})


def prepare(stage):
    if stage=='trainval':fit(); phases={'train','validation'}
    else:
        assert (ROOT/'FORMAL_PROTOCOL.json').exists() and (ROOT/'all_training_complete.json').exists()
        dump(ROOT/'audit_opened.json',{'epoch':time.time(),'all_training_precedes_audit':True})
        phases={'audit_holdout'}
    out=ROOT/('cache_'+stage);out.mkdir(exist_ok=False)
    scale=np.load(ROOT/'newtrain_EEG_target_normalization.npz'); rows=[]
    for i,row in enumerate(read(ROOT/'SPLIT_PROVENANCE.json')['records']):
        if row['phase'] not in phases:continue
        n=row['common_samples']; x=oriented(row); y=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        assert min(len(x),len(y))>=n
        xp=out/f'r{i:03d}_x.npy';yp=out/f'r{i:03d}_y.npy'
        xn=np.lib.format.open_memmap(xp,mode='w+',dtype='float32',shape=(n-LAG,64))
        yn=np.lib.format.open_memmap(yp,mode='w+',dtype='float32',shape=(n-LAG,1))
        for first in range(0,n-LAG,65536):
            last=min(n-LAG,first+65536)
            xx=(np.asarray(x[first+LAG:last+LAG],np.float64)-scale['mean'][:64])/scale['std'][:64]
            yy=(np.asarray(y[first:last],np.float64)-scale['mean'][-1])/scale['std'][-1]
            assert np.isfinite(xx).all() and np.isfinite(yy).all()
            xn[first:last]=xx;yn[first:last,0]=yy
        xn.flush();yn.flush()
        blocks=max(0,(n-LAG-WARMUP)//LENGTH)
        rows.append(dict(row,record_id=i,cache_x=str(xp),cache_y=str(yp),samples=n-LAG,score_blocks=blocks))
        if len(rows)%30==0:print(json.dumps({'cached_records':len(rows),'stage':stage}),flush=True)
    dump(out/'manifest.json',{'records':rows,'stage':stage,'score_samples':LENGTH,'warmup_samples':WARMUP,
        'excluded':[r['record'] for r in rows if not r['score_blocks']], 'sample_order_preserved':True})
    print(json.dumps({'stage':stage,'records':len(rows),'blocks':sum(r['score_blocks'] for r in rows)}),flush=True)


class Records:
    def __init__(self,phase):
        self.rows=[r for r in read(ROOT/('cache_audit' if phase=='audit_holdout' else 'cache_trainval')/'manifest.json')['records'] if r['phase']==phase and r['score_blocks']]
    def arrays(self,r):
        return np.load(r['cache_x'],mmap_mode='r'),np.load(r['cache_y'],mmap_mode='r')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','trainval','audit']);a=p.parse_args()
    if a.stage=='freeze':freeze()
    else:prepare(a.stage)
