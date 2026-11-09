"""D4 finite TRAIN/VAL ridge probe; scientific payloads execute on2203 only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import time
import traceback
import numpy as np
import scipy.linalg as la
import torch
from torch import nn

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT=BASE/'s03_conditional_history_probe_20261101'
P=BASE/'s03_incremental_continuity_20261101'
OLD=BASE/'s03_strong_baselines_20261015/transformer_available01'
WARM=BASE/'s03_incremental_warm_20261101'
SEEDS=(1024,2026,31415)
ARMS=('C','Ccap','CS','CF','CO','CF48')
LAMBDAS=(.0001,.01,1.)
DIMS=(1536,192,48,1536,1536,1536)
EDGES=np.cumsum((0,)+DIMS)
SLICES=[slice(int(a),int(b)) for a,b in zip(EDGES[:-1],EDGES[1:])]
START=time.time()

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def status(**v):
    t=ROOT/'STATUS.writing';t.write_text(json.dumps(dict(time=time.time(),elapsed=time.time()-START,**v)))
    os.replace(t,ROOT/'STATUS.json');print(json.dumps(v),flush=True)
def guard():
    assert time.time()-START<7200,'120minute cap; no automatic retry'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3,'8GiB RSS cap'
    assert shutil.disk_usage(ROOT).free>=20*1024**3,'20GiB free reserve'
    size=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert size<=2*1024**3,'2GiB new derived output cap'

def rows(phase):
    assert phase in ('train','val');out=[];offset=0
    for r in read(OLD/'envelope_readout_seed1024/manifest_frozen.json')['records']:
        if r['phase']!=phase:continue
        n=r['readout_windows'];assert r['phase_offset']==offset and n==(r['common_samples']-192)//96+1 and n>=4
        out.append(dict(r,offset=offset,windows=n));offset+=n
    assert offset=={'train':252454,'val':30500}[phase]
    return out

def corr(a,b):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert a.max()!=a.min() and b.max()!=b.min(),'constant series'
    x=a-a.mean();y=b-b.mean();v=float(x@y/np.sqrt((x@x)*(y@y)))
    assert np.isfinite(v);return v

def fused(pred,row):
    n=row['common_samples'];count=row['windows'];w=np.hanning(192)
    total=np.zeros(n);weight=np.zeros(n)
    for i in range(count):
        s=i*96;total[s:s+192]+=pred[i]*w;weight[s:s+192]+=w
    keep=weight>1e-8;total[keep]/=weight[keep]
    return total,keep

def score(pred,y,rlist,mask):
    assert pred.shape==(len(y),192) and np.isfinite(pred).all()
    a=pred-pred.mean(1,keepdims=True);b=np.asarray(y[:,0],np.float64);b=b-b.mean(1,keepdims=True)
    assert np.all(np.ptp(pred,axis=1)>0) and np.all(np.ptp(b,axis=1)>0)
    rs=(a*b).sum(1)/np.sqrt((a*a).sum(1)*(b*b).sum(1));assert np.isfinite(rs).all()
    eligible={(r['record'],r['start']):r for r in mask['blocks']};blocks=[];windows=[]
    for row in rlist:
        o,n=row['offset'],row['windows'];total,keep=fused(pred[o:o+n],row)
        truth=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        windows.append({'record':row['record'],'subject':row['subject'],'story':row['story'],'windows':n,'r3s':float(rs[o:o+n].mean())})
        for s in range(0,row['common_samples']-3840+1,3840):
            k=keep[s:s+3840];e=eligible[(row['record'],s)];assert int(k.sum())==e['samples']
            if e['eligible']:blocks.append({'record':row['record'],'subject':row['subject'],'story':row['story'],'start':s,'samples':int(k.sum()),'r':corr(total[s:s+3840][k],np.asarray(truth[s:s+3840],np.float64)[k])})
    assert len(blocks)==mask['eligible_blocks']==752
    people=[{'subject':s,'r':float(np.mean([r['r'] for r in blocks if r['subject']==s]))} for s in sorted({r['subject'] for r in blocks})]
    assert len(people)==49
    return dict(r3s=float(rs.mean()),r60s=float(np.mean([r['r'] for r in people])),people=people,blocks=blocks,records=windows)

def history_blocks(z,p,row):
    """All sources remain inside THIS record; no labels passed to features."""
    o,n=row['offset'],row['windows'];a=np.asarray(z[o:o+n],np.float64)
    assert a.shape==(n,8,192) and np.isfinite(a).all()
    current=a[4:].reshape(n-4,1536);near=a[2:n-2];far=a[:n-4]
    summary=near.reshape(n-4,8,6,32).mean(3).reshape(n-4,48)
    return [current,np.asarray(p[o+4:o+n],np.float64),summary,near.reshape(n-4,1536),far.reshape(n-4,1536)]

def stats_finish(s,t,n):
    mean=s/n;var=t/n-mean**2
    assert var.min()>=-1e-9*max(1.,float((t/n).max()))
    constant=var<=0;std=np.sqrt(np.maximum(var,0));std[constant]=1
    return {'mean':mean,'std':std,'constant':constant}

def normalize(x,st):
    v=(x-st['mean'])/st['std'];v[:,st['constant']]=0;return v

def features(raw,stats):
    base=[normalize(x,s) for x,s in zip(raw,stats[:5])]
    q=normalize(base[0]**2,stats[5]);base.append(q)
    return np.concatenate([x/np.sqrt(d) for x,d in zip(base,DIMS)],axis=1)

def arm_indices(arm):
    extra={'C':None,'Ccap':5,'CS':2,'CF':3,'CO':4,'CF48':None}[arm]
    ids=list(range(1728))
    if extra is not None:ids.extend(range(int(EDGES[extra]),int(EDGES[extra+1])))
    return np.array(ids)

def arm_moments(arm,G,H,m,pca):
    if arm!='CF48':
        ids=arm_indices(arm);return G[np.ix_(ids,ids)],H[ids],m[ids]
    # PCA scores=(standardized full @ loading)/sqrt(eigenvalue)/sqrt(48).
    # Existing full block is standardized full /sqrt(1536).
    T=pca*np.sqrt(1536/48);s=SLICES[3]
    out=np.zeros((1776,1776));out[:1728,:1728]=G[:1728,:1728]
    out[:1728,1728:]=G[:1728,s]@T;out[1728:,:1728]=out[:1728,1728:].T
    out[1728:,1728:]=T.T@G[s,s]@T
    return out,np.concatenate([H[:1728],T.T@H[s]],0),np.concatenate([m[:1728],m[s]@T])

def arm_features(arm,x,pca):
    if arm!='CF48':return x[:,arm_indices(arm)]
    return np.concatenate([x[:,:1728],x[:,SLICES[3]]@(pca*np.sqrt(1536/48))],1)

def ridge_eigen(G,H,lam):
    values,vectors=la.eigh((G+G.T)*.5,check_finite=True,driver='evr')
    assert values.min()>-1e-8
    B=vectors@((vectors.T@H)/(values[:,None]+lam))
    return B,float(np.sum(values/(values+lam)))

def math_check():
    rng=np.random.default_rng(19024);x=rng.normal(size=(47,9));y=rng.normal(size=(47,5));lam=.01
    xm=x.mean(0);ym=y.mean(0);xc=x-xm;yc=y-ym
    B,_=ridge_eigen(xc.T@xc/47,xc.T@yc/47,lam);bias=ym-xm@B
    aug=np.column_stack([x,np.ones(47)]);pen=np.column_stack([np.eye(9)*np.sqrt(47*lam),np.zeros(9)])
    direct=la.lstsq(np.vstack([aug,pen]),np.vstack([y,np.zeros((9,5))]))[0]
    err=float(np.max(np.abs(np.vstack([B,bias])-direct)));assert err<1e-12
    # Index canary: i-2 ends at current start, i-4 entirely earlier; two records isolated.
    for tag in (0,10000):
        z=(tag+np.arange(10))[:,None,None]*np.ones((10,8,192));p=np.zeros((10,192))
        raw=history_blocks(z,p,{'offset':0,'windows':10})
        assert np.array_equal(raw[0][:,0],tag+np.arange(4,10))
        assert np.array_equal(raw[3][:,0],tag+np.arange(2,8))
        assert np.array_equal(raw[4][:,0],tag+np.arange(0,6))
    # CF48 moment transform agrees with explicitly materialized features.
    q=rng.normal(size=(11,6384));pc=rng.normal(size=(1536,48))*.001
    g=q.T@q/11;h=q.T@y[:11]/11;m=q.mean(0)
    gg,hh,mm=arm_moments('CF48',g,h,m,pc);f=arm_features('CF48',q,pc)
    assert np.max(abs(gg-f.T@f/11))<1e-12 and np.max(abs(hh-f.T@y[:11]/11))<1e-12 and np.max(abs(mm-f.mean(0)))<1e-12
    return dict(passed=True,ridge_augmented_lstsq_max_abs=err,no_future_record_index_canaries=True,PCA_moment_transform=True)

def prepare():
    assert not (ROOT/'PROTOCOL.json').exists()
    mask=read(P/'block_eligibility/val.json');assert mask['prediction_independent'] and mask['eligible_blocks']==752
    rr={p:rows(p) for p in ('train','val')}
    assert [r['record'] for r in rr['val']]==mask['records']
    paths=[Path(__file__),ROOT/'PLAN.md',OLD/'envelope_readout_seed1024/manifest_frozen.json',P/'block_eligibility/val.json',P/'INCREMENTAL_PROTOCOL.json']
    inputs={}
    total_cache=0
    for seed in SEEDS:
        folder=P/f'baseline/head_seed{seed}';m=read(folder/'cache_manifest.json')
        assert m['only_train_val'] and m['encoder_sha256']==sha(P/f'baseline/pretrain_seed{seed}/best_validation.pt')
        paths.extend([folder/'cache_manifest.json',folder/'best_validation.pt',P/f'baseline/pretrain_seed{seed}/best_validation.pt',WARM/f'fits/frozen_p_seed{seed}/INCUMBENT_VALIDATION.json'])
        for phase in ('train','val'):
            z=folder/f'{phase}_latent.npy';n={'train':252454,'val':30500}[phase]
            v=np.load(z,mmap_mode='r');assert v.shape==(n,8,192) and v.dtype==np.float32
            for r in rr[phase]:
                c=m['records'][r['record']];assert c['phase']==phase and c['offset']==r['offset'] and c['windows']==r['windows']
            inputs[str(z)]={'bytes':z.stat().st_size,'shape':list(v.shape),'mtime_ns':z.stat().st_mtime_ns}
            total_cache+=z.stat().st_size
    assert total_cache+2*1024**3<=25*1024**3
    dump(ROOT/'ENGINEERING_CHECK.json',math_check())
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,scientific_status='TRAIN_VAL_DEVELOPMENT_ONLY',seeds=SEEDS,arms=ARMS,lambdas=LAMBDAS,max_solutions=54,
        authorization='Coordinator explicitly approved PLAN at4f59651; implement/freeze/run54, code review proceeds in parallel, no new user gate',
        sources={str(p):sha(p) for p in paths},inputs=inputs,records=rr,mask=mask,initial_existing_cache_bytes=total_cache,
        no_test_DTU=True,no_encoder_forward=True,new_NN_fits=False,common_local_mask='i>=4; i<4 baseline fallback for every arm',
        formula='centered float64 G=Xc.T@Xc/N,H=Xc.T@Rc/N; B=(G+lambda I)^-1H; unpenalized intercept',
        maximum_threads=8,maximum_RAM_GiB=8,maximum_disk_GiB=2,maximum_wall_minutes=120,automatic_retry=False))
    print('PREPARED_NOT_FITTED',flush=True)

@torch.no_grad()
def p_predictions(seed,phase,folder,device):
    z=np.load(P/f'baseline/head_seed{seed}/{phase}_latent.npy',mmap_mode='r')
    head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01)).to(device).eval()
    cp=torch.load(P/f'baseline/head_seed{seed}/best_validation.pt',map_location=device,weights_only=True)
    assert cp['seed']==seed;head.load_state_dict(cp['head'])
    out=np.lib.format.open_memmap(folder/f'{phase}_P.npy',mode='w+',dtype=np.float32,shape=(len(z),192))
    for start in range(0,len(z),64):
        x=torch.from_numpy(np.array(z[start:start+64],copy=True)).to(device)
        value=head(x)[:,0].cpu().numpy();assert np.isfinite(value).all();out[start:start+len(value)]=value
    out.flush();return np.load(folder/f'{phase}_P.npy',mmap_mode='r')

def calibration(p,rr):
    n=0;sx=sy=sxx=sxy=0.
    for row in rr:
        o,w=row['offset'],row['windows'];value,k=fused(np.asarray(p[o:o+w],np.float64),row)
        target=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        x=value[k];y=np.asarray(target[:len(k)],np.float64)[k];assert np.isfinite(y).all()
        n+=len(x);sx+=x.sum();sy+=y.sum();sxx+=x@x;sxy+=x@y
    a=float((sxy-sx*sy/n)/(sxx-sx*sx/n));b=float(sy/n-a*sx/n)
    assert a>0 and np.isfinite(a) and np.isfinite(b),'positive TRAIN calibration required; no clipping'
    return a,b,n

def seed_run(seed,protocol,device):
    begin=time.time();folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);guard()
    trainrows=protocol['records']['train'];valrows=protocol['records']['val'];mask=protocol['mask']
    status(stage='original_P_forward',seed=seed)
    p=p_predictions(seed,'train',folder,device);pv=p_predictions(seed,'val',folder,device)
    y=np.load(OLD/'envelope_readout_seed1024/train_envelope.npy',mmap_mode='r')
    yv=np.load(OLD/'envelope_readout_seed1024/val_envelope.npy',mmap_mode='r')
    z=np.load(P/f'baseline/head_seed{seed}/train_latent.npy',mmap_mode='r')
    zv=np.load(P/f'baseline/head_seed{seed}/val_latent.npy',mmap_mode='r')
    a,b,caln=calibration(p,trainrows);calp=a*np.asarray(p,np.float64)+b;calv=a*np.asarray(pv,np.float64)+b
    baseline=score(np.asarray(pv,np.float64),yv,valrows,mask);calbase=score(calv,yv,valrows,mask)
    inv=max(abs(baseline['r3s']-calbase['r3s']),abs(baseline['r60s']-calbase['r60s']),max(abs(x['r']-v['r']) for x,v in zip(baseline['blocks'],calbase['blocks'])))
    assert inv<=1e-10
    incumbent=read(WARM/f'fits/frozen_p_seed{seed}/INCUMBENT_VALIDATION.json')
    assert abs(baseline['r60s']-incumbent['continuous']['macro_r'])<=1e-10
    assert abs(baseline['r3s']+incumbent['negative_Pearson'])<=1e-6
    dump(folder/'P_CHECK.json',dict(a=a,b=b,TRAIN_physical_points=caln,affine_Pearson_max_abs=inv,original_P_VAL60s_abs=abs(baseline['r60s']-incumbent['continuous']['macro_r']),baseline=baseline))
    status(stage='TRAIN_statistics',seed=seed)
    sums=[np.zeros(d) for d in DIMS[:5]];squares=[np.zeros(d) for d in DIMS[:5]];n=0
    for row in trainrows:
        raw=history_blocks(z,calp,row);n+=len(raw[0])
        for j,x in enumerate(raw):sums[j]+=x.sum(0);squares[j]+=(x*x).sum(0)
        guard()
    assert n==250750;stats=[stats_finish(s,t,n) for s,t in zip(sums,squares)]
    qs=np.zeros(1536);qt=qs.copy();cov=np.zeros((1536,1536))
    for row in trainrows:
        raw=history_blocks(z,calp,row);current=normalize(raw[0],stats[0]);near=normalize(raw[3],stats[3])
        q=current**2;qs+=q.sum(0);qt+=(q*q).sum(0);cov+=near.T@near
        guard()
    stats.append(stats_finish(qs,qt,n))
    eig,vec=la.eigh(cov/n,subset_by_index=(1536-48,1535),driver='evr');eig=eig[::-1];vec=vec[:,::-1]
    vec*=np.where(vec[np.argmax(abs(vec),axis=0),np.arange(48)]<0,-1,1)[None,:]
    pc=np.zeros_like(vec);good=eig>0;pc[:,good]=vec[:,good]/np.sqrt(eig[good])
    np.savez(folder/'TRAIN_TRANSFORMS.npz',pca=pc,pca_eigenvalues=eig,**{f'{k}_{j}':v for j,s in enumerate(stats) for k,v in s.items()})
    del cov,vec
    status(stage='TRAIN_joint_Gram',seed=seed)
    G=np.zeros((6384,6384));H=np.zeros((6384,192));xs=np.zeros(6384);rs=np.zeros(192)
    for row in trainrows:
        o,w=row['offset'],row['windows'];x=features(history_blocks(z,calp,row),stats)
        r=np.asarray(y[o+4:o+w,0],np.float64)-calp[o+4:o+w]
        assert np.isfinite(x).all() and np.isfinite(r).all()
        G+=x.T@x;H+=x.T@r;xs+=x.sum(0);rs+=r.sum(0);guard()
    xm=xs/n;rm=rs/n;G=G/n-np.outer(xm,xm);H=H/n-np.outer(xm,rm)
    np.savez(folder/'TRAIN_MOMENTS.npz',G=G,H=H,x_mean=xm,residual_mean=rm,n=n)
    # VAL features are ~1.55GiB in float64; retaining only one shared matrix fits8GiB.
    xv=np.zeros((30304,6384));vindices=[];cursor=0
    for row in valrows:
        o,w=row['offset'],row['windows'];part=features(history_blocks(zv,calv,row),stats)
        xv[cursor:cursor+len(part)]=part;cursor+=len(part);vindices.extend(range(o+4,o+w))
    assert cursor==30304;vindices=np.asarray(vindices);results=[];checks=[]
    for arm in ARMS:
        guard();status(stage='ridge_solve',seed=seed,arm=arm,solutions_done=len(results))
        ga,ha,ma=arm_moments(arm,G,H,xm,pc);values,vectors=la.eigh((ga+ga.T)*.5,driver='evr');assert values.min()>-1e-8
        vh=vectors.T@ha;xa=arm_features(arm,xv,pc)
        for lam in LAMBDAS:
            t=time.time();coef=vectors@(vh/(values[:,None]+lam));bias=rm-ma@coef
            err=float(np.max(abs((ga+lam*np.eye(len(ga)))@coef-ha)));assert err<=1e-7*max(1.,float(abs(ha).max()))
            pred=calv.copy();pred[vindices]+=xa@coef+bias;assert np.array_equal(pred[np.setdiff1d(np.arange(30500),vindices)],calv[np.setdiff1d(np.arange(30500),vindices)])
            metrics=score(pred,yv,valrows,mask)
            tag=f'{arm}_lambda{lam:g}';np.savez(folder/f'{tag}_COEFFICIENTS.npz',coef=coef,bias=bias)
            dump(folder/f'{tag}_METRICS.json',dict(seed=seed,arm=arm,lam=lam,d=len(ga),parameters=int(coef.size+bias.size),effective_df=float(np.sum(values/(values+lam))),normal_equation_max_abs=err,seconds=time.time()-t,**metrics))
            results.append(dict(seed=seed,arm=arm,lam=lam,d=len(ga),parameters=int(coef.size+bias.size),r3s=metrics['r3s'],r60s=metrics['r60s'],effective_df=float(np.sum(values/(values+lam))),solve_score_seconds=time.time()-t))
            checks.append(err);guard()
        del ga,ha,vectors,xa,vh
    dump(folder/'COMPLETED.json',dict(seed=seed,solutions=len(results),results=results,seconds=time.time()-begin,
        constant_coordinates=[int(s['constant'].sum()) for s in stats],PCA_eigenvalues=list(map(float,eig)),PCA_positive_components=int(good.sum()),
        maximum_normal_equation_abs=max(checks),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_bytes=torch.cuda.max_memory_allocated(device)))
    return results,baseline

def csvfile(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(device):
    protocol=read(ROOT/'PROTOCOL.json');assert protocol['frozen'] and protocol['max_solutions']==54
    for p,d in protocol['sources'].items():assert sha(p)==d,p
    for p,v in protocol['inputs'].items():assert Path(p).stat().st_size==v['bytes'] and Path(p).stat().st_mtime_ns==v['mtime_ns']
    assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    precision=read(P/'INCREMENTAL_PROTOCOL.json')['precision']
    torch.backends.cuda.matmul.allow_tf32=precision['TF32_matmul'];torch.backends.cudnn.allow_tf32=precision['TF32_cudnn']
    torch.set_num_threads(8);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min cap')));signal.alarm(7200)
    (ROOT/'STARTED.json').open('x').write(json.dumps(dict(pid=os.getpid(),time=time.time(),protocol_sha=sha(ROOT/'PROTOCOL.json'))))
    results=[];baseline={}
    for seed in SEEDS:
        rr,bb=seed_run(seed,protocol,device);results+=rr;baseline[str(seed)]=bb
    assert len(results)==54
    selected=[];base_mean=float(np.mean([x['r60s'] for x in baseline.values()]))
    for arm in ARMS:
        candidates=[dict(lam=lam,mean=float(np.mean([r['r60s'] for r in results if r['arm']==arm and r['lam']==lam]))) for lam in LAMBDAS]
        best=max(candidates,key=lambda x:(x['mean'],x['lam']));use=best['mean']>base_mean
        selected.append(dict(arm=arm,selected_lambda=best['lam'] if use else None,incumbent_P_selected=not use,mean_VAL60s=best['mean'] if use else base_mean,all_candidates=candidates))
    csvfile(ROOT/'RESULTS.csv',results)
    dump(ROOT/'SUMMARY.json',dict(protocol_sha=sha(ROOT/'PROTOCOL.json'),all_solutions=results,baseline=baseline,selection=selected,selection_rule='shared lambda across seeds, strict improvement over P else P',
        scientific_status='TRAIN_VAL_DEVELOPMENT_ONLY',goal_complete=False))
    dump(ROOT/'COMPLETED.json',dict(solutions=54,seeds=3,wall_seconds=time.time()-START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(device),derived_disk_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),test_DTU_reads=False,encoder_forward=False,new_NN_fits=False))
    status(stage='COMPLETED',solutions=54,goal_complete=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['math','prepare','run']);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    assert str(ROOT).startswith('/home/fanyu/') and os.uname().sysname=='Linux'
    try:
        if args.action=='math':print(json.dumps(math_check()))
        elif args.action=='prepare':prepare()
        else:run(args.device)
    except Exception:
        if args.action=='run':dump(ROOT/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
