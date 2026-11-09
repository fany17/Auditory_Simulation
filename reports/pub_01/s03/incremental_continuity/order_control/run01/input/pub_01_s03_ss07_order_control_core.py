"""D14: the only intervention is full global-window training permutation."""
import sys,os,time,resource,shutil,json,hashlib
from pathlib import Path
import numpy as np
import torch
BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
D12=BASE/'s03_full_ft_history_20261102';D13=BASE/'s03_full_ft_error_decomposition_20261102'
ROOT=BASE/'s03_full_ft_order_control_20261102'
sys.path.insert(0,str(D12/'input'))
import pub_01_s03_ss07_full_ft_history_core as frozen
SEEDS=frozen.SEEDS;ARMS=frozen.ARMS;Conditional=frozen.Conditional;read=frozen.read;sha=frozen.sha
def dump(path,value):
    p=Path(path);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**2
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<2*1024**3
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/sata5/fanyu/auditory_simulation_m6a').free)>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<128*1024**2
def permutation(seed,epoch,n):
    # NumPy-owned local stream cannot consume initialization/torch/Adam RNG.
    order=np.random.default_rng(np.random.SeedSequence([seed,20261102,epoch])).permutation(n)
    assert np.array_equal(np.sort(order),np.arange(n))
    return order
class Data(frozen.Data):
    def batch_indices(self,indices,arm,device,older=False):
        ids_all=np.asarray(indices,dtype=np.int64);assert ids_all.ndim==1 and len(ids_all)>0
        assert ids_all.min()>=0 and ids_all.max()<self.n and len(np.unique(ids_all))==len(ids_all)
        ok=self.local[ids_all]>=9;ids=ids_all[ok];current=self.z[ids].copy()
        if arm=='C_match':history=np.stack((current,current),1)
        else:
            lags=(9,7) if older else (5,3);ii=np.stack((ids-lags[0],ids-lags[1]),1)
            assert np.all(ii>=0) and np.all(ii<ids[:,None])
            assert np.all(self.record_id[ii]==self.record_id[ids,None])
            assert np.all(self.local[ii]==self.local[ids,None]-np.array(lags)[None,:])
            history=self.prior[ii,:,None].transpose(0,1,3,2).copy() if arm=='P_prior' else self.z[ii].copy()
        return (torch.from_numpy(current).to(device),torch.from_numpy(history).to(device),
            torch.from_numpy(self.anchor[ids_all].copy()).to(device),
            torch.from_numpy(np.asarray(self.y[ids_all],dtype='f4').astype('f8').reshape(len(ids_all),192)).to(device),ok)
    def batch(self,first,last,arm,device,older=False):
        return self.batch_indices(np.arange(first,last),arm,device,older)
@torch.no_grad()
def predict(model,data,device,older=False):
    # Inference retains D12's global64 order. Permutation applies only to updates.
    model.eval();out=data.anchor.copy();windows=0
    for first in range(0,data.n,64):
        last=min(first+64,data.n);x,h,_,_,ok=data.batch(first,last,model.arm,device,older)
        if len(x):out[first:last][ok]+=.1*model(x,h).cpu().numpy().astype('f8');windows+=len(x)
    return out,windows
def train_score(data,pred):
    assert data.phase=='train' and pred.shape==(252454,192) and np.isfinite(pred).all()
    mask=read(D13/'SUPPORT_MASK.json')['train'];blocks=[];window_sse=0.
    for row in data.rows:
        o,w=row['offset'],row['windows'];p=pred[o:o+w]
        y=np.asarray(data.y[o:o+w],dtype='f4').astype('f8').reshape(w,192)
        window_sse+=float(np.sum((p-y)**2))
        wave,keep=frozen.metric.fuse(p,row['common_samples'])
        truth=np.asarray(np.load(row['envelope_path'],mmap_mode='r'),dtype='f4').astype('f8').reshape(-1)
        for b in mask['blocks']:
            if b['record']!=row['record'] or not b['eligible']:continue
            s,t=b['start'],b['stop'];k=keep[s:t];a=wave[s:t][k];y=truth[s:t][k]
            assert int(k.sum())==b['covered']
            blocks.append(dict(record=row['record'],subject=row['subject'],start=s,r=frozen.metric.corr(a,y),MSE=float(np.mean((a-y)**2))))
    people=[]
    for subject in sorted({b['subject'] for b in blocks}):
        bb=[b for b in blocks if b['subject']==subject]
        people.append(dict(subject=subject,blocks=len(bb),r=float(np.mean([b['r'] for b in bb])),MSE=float(np.mean([b['MSE'] for b in bb]))))
    assert len(blocks)==6104 and len(people)==70
    return dict(window_MSE=window_sse/(data.n*192),physical60s_r=float(np.mean([p['r'] for p in people])),physical60s_MSE=float(np.mean([p['MSE'] for p in people])),people=people,blocks=blocks,windows=data.n,target_precision='record/cache float32 then float64',parameters_fixed=True)
