"""New source-family split, newtrain-only scaling and continuous raw EEG cache."""
import argparse
import json
from pathlib import Path
import numpy as np

PUBLIC=Path('/home/fanyu/auditory_simulation_m6a')
ROOT=PUBLIC/'pub_01/s03_history_memory_20261019/codex_run01'


def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2)


def split():return json.loads((ROOT/'split_frozen.json').read_text())


def oriented(record):
    assert record['eeg_path'].startswith(str(PUBLIC)+'/') and record['envelope_path'].startswith(str(PUBLIC)+'/')
    raw=np.load(record['eeg_path'],mmap_mode='r');assert raw.ndim==2 and raw.shape.count(64)==1
    return raw.T if raw.shape[0]==64 else raw


def fit_normalization():
    rows=[r for r in split()['records'] if r['phase']=='train'];count=0;mean=np.zeros(64);m2=np.zeros(64)
    for ri,r in enumerate(rows):
        eeg=oriented(r);assert len(eeg)>=r['common_samples']
        for first in range(0,r['common_samples'],65536):
            chunk=np.asarray(eeg[first:min(first+65536,r['common_samples'])],dtype=np.float64)
            assert np.isfinite(chunk).all();n=len(chunk);cm=chunk.mean(0);cs=((chunk-cm)**2).sum(0)
            delta=cm-mean;total=count+n;m2+=cs+delta*delta*(count*n/total);mean+=delta*(n/total);count=total
        if ri%40==0:print(json.dumps({'newtrain_norm_records':ri+1,'total':len(rows)}),flush=True)
    std=np.sqrt(m2/count);assert np.isfinite(std).all() and (std>0).all()
    np.savez(ROOT/'newtrain_normalization.npz',mean=mean,std=std)
    old=np.load(PUBLIC/'pub_01/s03_strong_baselines_20261015/transformer_available01/prepared/train_only_normalization.npz')
    dump(ROOT/'normalization_contract.json',{'fit_phase':'SS05_train_only','fit_record_count':len(rows),'fit_sample_count':count,
        'fit_record_ids':[r['record'] for r in rows],'old_normalization_used':False,
        'reason':'SS02 originaltrain stats included newly assigned validation/holdout; new stats prevent SS05 normalization leakage; original unchanged',
        'new_old_mean_difference_L2':float(np.linalg.norm(mean-old['mean'])),
        'new_old_std_relative_L2':float(np.linalg.norm(std-old['std'])/np.linalg.norm(old['std']))})


def prepare(stage):
    source=split()
    if stage=='trainval':
        if (ROOT/'newtrain_normalization.npz').exists():raise RuntimeError('normalization write-once')
        fit_normalization();phases={'train','validation'}
    else:
        if not (ROOT/'FORMAL_PROTOCOL.json').exists():raise RuntimeError('holdout signal reading requires prior formal freeze')
        phases={'holdout'}
    out=ROOT/('cache_'+stage);out.mkdir(exist_ok=False);scale=np.load(ROOT/'newtrain_normalization.npz')
    records=[];segments=[];excluded=[]
    for i,row in enumerate(source['records']):
        if row['phase'] not in phases:continue
        r=dict(row,record_id=i);eeg=oriented(r);env=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)
        length=r['common_samples'];assert min(len(eeg),len(env))>=length
        xp=out/f'r{i:03d}_eeg.npy';yp=out/f'r{i:03d}_envelope.npy'
        x=np.lib.format.open_memmap(xp,mode='w+',dtype='float32',shape=(length,64))
        y=np.lib.format.open_memmap(yp,mode='w+',dtype='float32',shape=(length,))
        for start in range(0,length,65536):
            a=(np.asarray(eeg[start:start+65536],dtype=np.float32)-scale['mean'])/scale['std']
            a=a[:length-start].astype(np.float32);b=np.asarray(env[start:start+len(a)],dtype=np.float32)
            assert np.isfinite(a).all() and np.isfinite(b).all();x[start:start+len(a)]=a;y[start:start+len(a)]=b
        x.flush();y.flush();nwindow=length//192;nsegment=max(0,(nwindow-40)//20)
        r.update(cache_eeg=str(xp),cache_envelope=str(yp),window_count=nwindow,segments=nsegment)
        records.append(r)
        if not nsegment:excluded.append({'record':r['record'],'reason':'less_than180s_support_for120s_history_and60s_target'})
        for seg in range(nsegment):segments.append({'record_id':i,'phase':r['phase'],'subject':r['subject'],
            'source_family':r['source_family'],'start_window':40+20*seg,'windows':20})
        if len(records)%40==0:print(json.dumps({'cached_phase':stage,'records':len(records)}),flush=True)
    manifest={'records':records,'segments':segments,'excluded_records':excluded,'window_samples':192,'history_windows':20,
        'prior_gap_windows':20,'score_windows':20,'first_target_window':40,'target_lag_formula':'env[t-lag:t-lag+192]; positive lag means EEG response after sound',
        'history_is_contiguous_within_same_record':True,'boundary_rule':'each query starts a fresh bounded60s past context; no state handed across query/record',
        'normalization':'newtrain_normalization.npz exclusively SS05_train fit','stage':stage}
    dump(out/'manifest.json',manifest);dump(out/'completed.json',{'ready':True,'scientific_holdout_metrics_computed':False})
    print(json.dumps({'stage':stage,'records':len(records),'segments':len(segments),'by_phase':{p:sum(s['phase']==p for s in segments) for p in phases},'exclusions':len(excluded)}),flush=True)


class ContinuousPool:
    def __init__(self,phase,device='cuda:0'):
        import torch
        stage='holdout' if phase=='holdout' else 'trainval'
        m=json.loads((ROOT/f'cache_{stage}/manifest.json').read_text());self.phase=phase;self.device=device
        self.records={r['record_id']:r for r in m['records'] if r['phase']==phase}
        self.segments=[s for s in m['segments'] if s['phase']==phase]
        self.x={};self.y={}
        for rid,r in self.records.items():
            self.x[rid]=torch.from_numpy(np.load(r['cache_eeg']).copy()).to(device)
            self.y[rid]=torch.from_numpy(np.load(r['cache_envelope']).copy()).to(device)
        self.byrecord={rid:[i for i,s in enumerate(self.segments) if s['record_id']==rid] for rid in self.records}

    def batch(self,ids,lag):
        import torch
        blocks=[];targets=[]
        for i in ids:
            s=self.segments[int(i)];rid=s['record_id'];start=s['start_window']*192
            # 40 windows = exactly20 continuous prior +20 current windows.
            blocks.append(self.x[rid][start-20*192:start+20*192].reshape(40,192,64))
            targets.append(self.y[rid][start-lag:start+20*192-lag].reshape(20,192,1))
        return torch.stack(blocks),torch.stack(targets)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['trainval','holdout'],required=True);prepare(p.parse_args().stage)
