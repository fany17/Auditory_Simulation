"""D17 fixed-window scalar diagnostics. No dataset-wide loader or learned transform."""
import os,sys,json,hashlib,time,resource,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT=BASE/'s03_history_sensitivity_20261102';D16=BASE/'s03_full_ft_token_normalization_20261102'
D8=BASE/'s03_finetuned_readout_transfer_20261101'
sys.path.insert(0,str(D16/'input'))
import pub_01_s03_ss07_token_norm_core as frozen
CHECKPOINTS=((2026,'selected',5,'best_validation.pt'),(1024,'last',19,'last.pt'),(2026,'last',19,'last.pt'),(31415,'last',19,'last.pt'))
LOCALS=(0,8,9,10,11,12,13,14)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<=8*1024**2
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<=2*1024**3
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/sata5/fanyu/auditory_simulation_m6a').free)>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<8*1024**2
def admission(device):
    index=device.split(':')[1]
    r=subprocess.run(['nvidia-smi','--id='+index,'--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=int(r.stdout.strip())
    r=subprocess.run(['nvidia-smi','--id='+index,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids=[int(x.strip()) for x in r.stdout.splitlines() if x.strip().isdigit()]
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    return dict(device=device,free_GPU_MiB=free,compute_pids=pids,MemAvailable_bytes=mem,admitted=not pids and free>=4096 and mem>=16*1024**3)
def metadata():
    rows=read(D8/'PROTOCOL.json')['records']['train'][:8];out=[]
    assert len(rows)==8
    for r in rows:
        assert r['windows']>14
        for local in LOCALS:
            out.append(dict(record=r['record'],subject=r['subject'],story=r['story'],record_offset=r['offset'],local=local,global_index=r['offset']+local,effective=local>=9,near_local=[local-5,local-3] if local>=9 else None,older_local=[local-9,local-7] if local>=9 else None))
    assert len(out)==64 and sum(x['effective'] for x in out)==48
    return out
def normalized(x,st):
    x=(x-st['mean'])/st['std'];x[...,st['constant']]=0;return x
def fixed_inputs(seed,device):
    """Only 48 current positions and their predeclared histories; no full TRAIN scan."""
    m=metadata();valid=[x for x in m if x['effective']];ids=np.array([x['global_index'] for x in valid])
    folder=D8/f'seed{seed}'
    with np.load(folder/'TRAIN_TRANSFORMS.npz') as f:stats=[{k:f[f'{k}_{i}'].copy() for k in ('mean','std','constant')} for i in range(3)]
    with np.load(folder/'Ccap_lambda0.01_COEFFICIENTS.npz') as f:coef={k:f[k].copy() for k in f.files}
    cal=read(folder/'P_FT_BASELINE.json')['calibration']
    z=np.load(folder/'train_latent.npy',mmap_mode='r');p=np.load(folder/'train_P.npy',mmap_mode='r')
    assert z.shape==(252454,8,192) and z.dtype==np.float32 and p.shape==(252454,192)
    zz=normalized(np.array(z[ids],dtype='f8').reshape(48,1536),stats[0])
    base=cal['a']*np.array(p[ids],dtype='f8')+cal['b']
    features=np.concatenate((zz/np.sqrt(1536),normalized(base,stats[1])/np.sqrt(192),normalized(zz*zz,stats[2])/np.sqrt(1536)),axis=1)
    anchor=base+(features@coef['coef']+coef['bias'])
    current=torch.from_numpy(zz.reshape(48,8,192).astype('f4')).to(device)
    histories={}
    for condition,lags in [('near',(5,3)),('older',(9,7))]:
        ii=np.stack([ids-lags[0],ids-lags[1]],1)
        for n,row in enumerate(valid):assert np.all(ii[n]>=row['record_offset']) and np.all(ii[n]+192//96<=ids[n])
        hh=normalized(np.array(z[ii],dtype='f8').reshape(48,2,1536),stats[0]).reshape(48,2,8,192).astype('f4')
        histories[condition]=torch.from_numpy(hh).to(device)
    del z,p
    assert all(torch.isfinite(x).all() for x in (current,histories['near'],histories['older'])) and np.isfinite(anchor).all()
    return current,torch.from_numpy(anchor).to(device),histories,m
def node_stats(x,token_axis=False):
    a=np.asarray(x,dtype='f8');assert a.shape[0]==48 and np.isfinite(a).all()
    centered=a-a.mean(axis=0,keepdims=True)
    result=dict(shape=list(a.shape),RMS=float(np.sqrt(np.mean(a*a))),max_abs=float(np.max(abs(a))),across_windows_centered_RMS=float(np.sqrt(np.mean(centered*centered))),centering_axis='axis0 over48 windows, each other coordinate fixed; then RMS over all coordinates')
    if token_axis:
        assert a.shape[1]==2
        result['per_token_position_across_windows_centered_RMS']=[float(np.sqrt(np.mean(centered[:,i]**2))) for i in range(2)]
    return result
def difference_stats(near,older):
    a=np.asarray(near,dtype='f8');b=np.asarray(older,dtype='f8');assert a.shape==b.shape and a.shape[0]==48
    d=a-b;rms=float(np.sqrt(np.mean(d*d)));pooled=float(np.sqrt((np.mean(a*a)+np.mean(b*b))/2))
    ca=a-a.mean(axis=0,keepdims=True);cb=b-b.mean(axis=0,keepdims=True)
    centered=float(np.sqrt((np.mean(ca*ca)+np.mean(cb*cb))/2))
    window=np.sqrt(np.mean(d.reshape(48,-1)**2,axis=1))
    return dict(difference_RMS=rms,difference_max_abs=float(np.max(abs(d))),pooled_global_RMS=pooled,relative_difference_RMS=rms/pooled if pooled!=0 else None,pooled_across_window_centered_RMS=centered,relative_to_across_window_centered_RMS=rms/centered if centered!=0 else None,denominator_zero_returns_null=True,exactly_equal_element_fraction=float(np.mean(a==b)),exactly_equal_window_count=int(np.count_nonzero(np.all(a.reshape(48,-1)==b.reshape(48,-1),axis=1))),per_window_difference_RMS_min=float(np.min(window)),per_window_difference_RMS_median=float(np.median(window)),per_window_difference_RMS_max=float(np.max(window)))
def grad_stats(g):
    a=g.detach().double().cpu().numpy();return dict(nonzero_scalars=int(np.count_nonzero(a)),scalars=int(a.size),L2_norm=float(np.linalg.norm(a)),max_abs=float(np.max(abs(a))),finite=bool(np.isfinite(a).all()))
def axis_tests():
    # Coordinate-constant across48 but different between tokens must center to0.
    a=np.broadcast_to(np.arange(8,dtype='f8').reshape(1,2,4),(48,2,4)).copy()
    assert node_stats(a,True)['across_windows_centered_RMS']==0
    z=np.zeros_like(a);s=difference_stats(z,z)
    assert s['relative_difference_RMS'] is None and s['relative_to_across_window_centered_RMS'] is None and s['exactly_equal_element_fraction']==1
    b=a.copy();b[:,1]+=2
    s=difference_stats(a,b);assert s['difference_RMS']==np.sqrt(2) and s['per_window_difference_RMS_min']==s['per_window_difference_RMS_max']
    assert node_stats(a+7,True)['across_windows_centered_RMS']==0
    return dict(passed=True,new_NN=0,backward=0,axis0_centering_and_zero_denominator_and_known_difference=True)
