"""Fixed full-FT conditional residual. TRAIN/VAL only; no encoder construction."""
import hashlib,json,os,resource,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
D8=BASE/'s03_finetuned_readout_transfer_20261101'
OLD=BASE/'s03_strong_baselines_20261015/transformer_available01'
ROOT=BASE/'s03_full_ft_history_20261102'
sys.path.insert(0,str(BASE/'s03_current_readout_internal_audit_20261101/input'))
import pub_01_s03_ss07_generalization_core as metric
ARMS=('H_full','C_match','P_prior');SEEDS=(1024,2026,31415)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def dump(p,j):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(j,f,indent=2,allow_nan=False)
def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**2,'8GiB RSS'
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<2*1024**3,'2GiB allocated GPU'
    assert shutil.disk_usage(ROOT).free>20*1024**3,'20GiB home reserve'
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<128*1024**2,'128MiB run cap'
def norm(x,st):
    x=(x-st['mean'])/st['std'];x[...,st['constant']]=0;return x

class Conditional(nn.Module):
    def __init__(self,seed,arm):
        super().__init__();assert arm in ARMS;self.arm=arm
        # Each component owns an RNG stream, independent of the prior's C=1.
        def make(index,build):
            with torch.random.fork_rng(devices=[]):torch.manual_seed(seed+1000*index);return build()
        self.token=nn.Sequential(make(11,lambda:nn.Conv1d(1 if arm=='P_prior' else 8,16,9,padding=4)),nn.GELU(),
            make(12,lambda:nn.Conv1d(16,16,9,padding=4)),nn.GELU(),nn.Flatten(),make(13,lambda:nn.Linear(3072,32)),nn.GELU())
        self.gru=make(2,lambda:nn.GRU(32,32,batch_first=True))
        self.current=make(3,lambda:nn.Sequential(nn.Flatten(),nn.Linear(1536,32),nn.GELU()))
        self.film=make(4,lambda:nn.Linear(32,64));self.output=make(5,lambda:nn.Linear(32,192))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,current,history,return_state=False):
        b=len(current);tokens=self.token(history.reshape(b*2,history.shape[2],192)).reshape(b,2,32)
        seq,_=self.gru(tokens);h=seq[:,-1];q=self.current(current);g,beta=self.film(h).chunk(2,1)
        out=self.output(q*(1+.1*torch.tanh(g))+.1*beta)
        return (out,h,q) if return_state else out

class Data:
    def __init__(self,seed,phase):
        assert phase in ('train','val');self.phase=phase;self.seed=seed
        protocol=read(D8/'PROTOCOL.json');self.rows=protocol['records'][phase];n=sum(r['windows'] for r in self.rows)
        self.mask=protocol['mask'] if phase=='val' else None
        folder=D8/f'seed{seed}';self.cal=read(folder/'P_FT_BASELINE.json')['calibration']
        with np.load(folder/'TRAIN_TRANSFORMS.npz') as f:self.stats=[{k:f[f'{k}_{i}'].copy() for k in ['mean','std','constant']} for i in range(3)]
        with np.load(folder/'Ccap_lambda0.01_COEFFICIENTS.npz') as f:co={k:f[k].copy() for k in f.files}
        z=np.load(folder/f'{phase}_latent.npy',mmap_mode='r');p=np.load(folder/f'{phase}_P.npy',mmap_mode='r')
        assert z.shape==(n,8,192) and z.dtype==np.float32 and p.shape==(n,192)
        self.z=np.empty(z.shape,dtype='f4');self.anchor=np.empty((n,192),dtype='f8');self.local=np.empty(n,dtype='i4');self.record_id=np.empty(n,dtype='i4')
        for ri,row in enumerate(self.rows):
            o,w=row['offset'],row['windows'];assert w==(row['common_samples']-192)//96+1
            self.local[o:o+w]=np.arange(w);self.record_id[o:o+w]=ri
            for a in range(o,o+w,1024):
                b=min(a+1024,o+w);raw=np.asarray(z[a:b],dtype='f8').reshape(b-a,1536);zz=norm(raw,self.stats[0])
                self.z[a:b]=zz.reshape(-1,8,192).astype('f4')
                base=self.cal['a']*np.asarray(p[a:b],dtype='f8')+self.cal['b'];self.anchor[a:b]=base
                start=max(a,o+4)
                if start<b:
                    off=start-a;c=zz[off:];pp=norm(base[off:],self.stats[1]);qq=norm(c*c,self.stats[2])
                    features=np.concatenate((c/np.sqrt(1536),pp/np.sqrt(192),qq/np.sqrt(1536)),axis=1)
                    self.anchor[start:b]+=features@co['coef']+co['bias']
            guard()
        assert np.isfinite(self.z).all() and np.isfinite(self.anchor).all()
        self.prior=norm(self.anchor,self.stats[1]).astype('f4')
        self.y=np.load(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy',mmap_mode='r')
        assert len(self.y)==n;self.n=n
    def batch(self,first,last,arm,device,older=False):
        indices=np.arange(first,last);ok=self.local[indices]>=9;ids=indices[ok]
        current=self.z[ids].copy()
        if arm=='C_match':history=np.stack((current,current),1)
        else:
            lags=(9,7) if older else (5,3);ii=np.stack((ids-lags[0],ids-lags[1]),1)
            assert np.all(self.record_id[ii]==self.record_id[ids,None]) and np.all(ii>=0)
            if arm=='P_prior':history=self.prior[ii,:,None].transpose(0,1,3,2).copy()
            else:history=self.z[ii].copy()
        x=torch.from_numpy(current).to(device);h=torch.from_numpy(history).to(device)
        a=torch.from_numpy(self.anchor[first:last].copy()).to(device)
        y=torch.from_numpy(np.array(self.y[first:last],dtype='f4').astype('f8').reshape(last-first,192)).to(device)
        return x,h,a,y,ok
    def score(self,pred):
        assert self.phase=='val' and pred.shape==(self.n,192)
        records=[]
        for row in self.rows:
            a=row['offset'];b=a+row['windows'];target=np.load(row['envelope_path'],mmap_mode='r')
            blocks=[x for x in self.mask['blocks'] if x['record']==row['record']]
            record=metric.measure(pred[a:b],target,row,blocks)
            wave,keep=metric.fuse(pred[a:b],row['common_samples']);truth=np.asarray(target,dtype='f8').reshape(-1)
            def region(first,last):
                kk=keep[first:last];pp=wave[first:last][kk];tt=truth[first:last][kk]
                return dict(samples=len(pp),MSE=float(np.mean((pp-tt)**2)) if len(pp) else None,
                    r=metric.corr(pp,tt) if len(pp)>1 and np.ptp(pp)>0 and np.ptp(tt)>0 else None)
            n=row['common_samples'];tail=(n//3840)*3840
            record['boundary_tail']=dict(prefix=region(0,864),first_learning_overlap=region(864,960),tail=region(tail,n),
                last_single_window=region((row['windows']-1)*96+96,min(n,(row['windows']-1)*96+192)),
                maximum_window_abs=float(np.max(np.abs(pred[a:b]))),maximum_fused_abs=float(np.max(np.abs(wave[keep]))))
            records.append(record)
        result=metric.aggregate(records);assert result['phases']['val']['people']==49 and result['phases']['val']['blocks']==752
        bb=[b for r in records for b in r['segments'] if b['start']>=3840];subjects=sorted({b['subject'] for b in bb})
        result['fixed_warm_auxiliary']=dict(blocks=len(bb),people=len(subjects),r=float(np.mean([np.mean([b['r'] for b in bb if b['subject']==s]) for s in subjects])),
            rule='Original60s blocks with start>=3840; all contributing windows are beyond i>=9; auxiliary only')
        result['boundary_tail_records']=[dict(record=r['record'],**r['boundary_tail']) for r in records]
        return result

@torch.no_grad()
def predict(model,data,device,older=False):
    model.eval();out=data.anchor.copy();batches=0;windows=0
    for first in range(0,data.n,64):
        last=min(first+64,data.n);x,h,_,_,ok=data.batch(first,last,model.arm,device,older)
        if len(x):out[first:last][ok]+=.1*model(x,h).cpu().numpy().astype('f8');windows+=len(x);batches+=1
    return out,dict(batches=batches,effective_residual_windows=windows,encoder_head_windows=0)
