"""D15: D14 inputs/indexing/model, only float64 centered-cosine objective changes."""
import sys,os,time,json,hashlib,resource,shutil,stat
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');D14=BASE/'s03_full_ft_order_control_20261102'
ROOT=BASE/'s03_full_ft_objective_control_20261102'
sys.path.insert(0,str(D14/'input'))
import pub_01_s03_ss07_order_control_core as previous
Data=previous.Data;Conditional=previous.Conditional;predict=previous.predict;permutation=previous.permutation
SEEDS=previous.SEEDS;ARMS=previous.ARMS;read=previous.read;sha=previous.sha
def dump(path,value):
    p=Path(path);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
INFLIGHT_RESERVE_BYTES=28*1024**2 #9*(checkpoint+one full statistic <=3MiB)+1MiB logs/status
def directory_bytes(root=ROOT,paths=None):
    total=0
    for path in root.rglob('*') if paths is None else paths:
        try:info=path.stat()
        except FileNotFoundError:continue # Concurrent atomic replace already removed this temporary path.
        if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total
def guard_checks():
    class Stable:
        def stat(self):return type('Info',(),dict(st_mode=stat.S_IFREG,st_size=17))()
    class Missing:
        def stat(self):raise FileNotFoundError('simulated atomic replacement')
    class Denied:
        def stat(self):raise PermissionError('other OSError must propagate')
    assert directory_bytes(paths=[Stable(),Missing()])==17
    try:directory_bytes(paths=[Denied()])
    except PermissionError:pass
    else:raise AssertionError('Do not suppress non-disappearance IO errors')
    return dict(passed=True,disappeared_path_ignored=True,other_OSError_propagates=True,inflight_reserve_bytes=INFLIGHT_RESERVE_BYTES,NN=0,optimizer_updates=0)
def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**2
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<2*1024**3
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/sata5/fanyu/auditory_simulation_m6a').free)>=20*1024**3
    assert directory_bytes()+INFLIGHT_RESERVE_BYTES<160*1024**2,'Measured bytes plus concurrent write reserve exceed160MiB'
def window_losses(pred,y):
    assert pred.dtype==y.dtype==torch.float64 and pred.shape==y.shape and pred.ndim==2 and pred.shape[1]==192
    pc=pred-pred.mean(-1,keepdim=True);yc=y-y.mean(-1,keepdim=True)
    return -F.cosine_similarity(pc,yc,dim=-1,eps=1e-6)
def objective(pred,y):return window_losses(pred,y).mean()
def module_gradient_summary(model):
    names=['token','gru','film','current','output'];values=[]
    for name in names:
        gg=[p.grad.detach() for p in getattr(model,name).parameters() if p.grad is not None]
        assert gg
        values += [sum(torch.count_nonzero(g).double() for g in gg),sum(g.double().square().sum() for g in gg).sqrt()]
    numbers=torch.stack(values).cpu().tolist()
    return {name:dict(nonzero_scalars=int(numbers[2*i]),L2_norm=numbers[2*i+1]) for i,name in enumerate(names)}
def target_counts(y):
    a=np.asarray(y,dtype='f4').astype('f8').reshape(-1,192);c=a-a.mean(1,keepdims=True);n=np.linalg.norm(c,axis=1)
    return dict(windows=len(a),exact_raw_constant=int(np.sum(np.ptp(a,axis=1)==0)),exact_centered_zero=int(np.sum(n==0)),centered_norm_le_eps=int(np.sum(n<=1e-6)))
def train_score(data,pred):
    result=previous.train_score(data,pred);loss_sum=0.;pairs=0;overlap_sse=0.;records=[];counts=dict(windows=0,exact_raw_constant=0,exact_centered_zero=0,centered_norm_le_eps=0)
    max_window=0.;max_fused=0.;pred_sum=0.;pred_squared_sum=0.;pred_points=0
    for row in data.rows:
        o,w=row['offset'],row['windows'];p=pred[o:o+w];y=np.asarray(data.y[o:o+w],dtype='f4').astype('f8').reshape(w,192)
        with torch.no_grad():loss_sum+=float(window_losses(torch.from_numpy(p),torch.from_numpy(y)).sum())
        cc=target_counts(y)
        for k in counts:counts[k]+=cc[k]
        pc=p-p.mean(1,keepdims=True);norm=np.linalg.norm(pc,axis=1)
        pair=p[:-1,96:]-p[1:,:96];sse=float(np.sum(pair*pair));pairs+=w-1;overlap_sse+=sse
        wave,keep=previous.frozen.metric.fuse(p,row['common_samples'])
        maximum=float(np.max(abs(p)));fused=float(np.max(abs(wave[keep])))
        max_window=max(max_window,maximum);max_fused=max(max_fused,fused);pred_sum+=float(p.sum());pred_squared_sum+=float(np.sum(p*p));pred_points+=p.size
        records.append(dict(record=row['record'],subject=row['subject'],windows=w,maximum_window_abs=maximum,maximum_fused_abs=fused,window_prediction_mean=float(p.mean()),window_prediction_RMS=float(np.sqrt(np.mean(p*p))),prediction_centered_norm_le_eps=int(np.sum(norm<=1e-6)),overlap_MSE=sse/((w-1)*96),**{f'target_{k}':v for k,v in cc.items() if k!='windows'}))
    assert counts['windows']==data.n
    result.update(negative_cosine=loss_sum/data.n,objective_dtype='float64, original centered per-window formula, separate norm floors eps1e-6',target_window_counts=counts,maximum_window_abs=max_window,maximum_fused_abs=max_fused,window_prediction_mean=pred_sum/pred_points,window_prediction_RMS=float(np.sqrt(pred_squared_sum/pred_points)),overlap_MSE=overlap_sse/(pairs*96),overlap_pairs=pairs,amplitude_overlap_records=records)
    return result
