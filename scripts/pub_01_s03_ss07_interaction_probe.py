"""D5 finite bilinear probe, frozen strong-current anchor; CPU2203 only."""
import argparse
import csv
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
import pub_01_s03_ss07_conditional_probe as d4

ROOT=d4.BASE/'s03_history_interaction_probe_20261101'
SOURCE=d4.ROOT
SEEDS=d4.SEEDS
ARMS=('Add','CC','Near','Older')
LAMBDAS=d4.LAMBDAS
START=time.time();CPU_START=time.process_time();COST=[]

def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def status(**v):
    tmp=ROOT/'STATUS.writing';tmp.write_text(json.dumps(dict(time=time.time(),elapsed=time.time()-START,**v)));os.replace(tmp,ROOT/'STATUS.json');print(json.dumps(v),flush=True)
def clock():return time.time(),time.process_time()
def account(stage,seed,started,**v):
    COST.append(dict(stage=stage,seed=seed,wall_seconds=time.time()-started[0],CPU_seconds=time.process_time()-started[1],**v))
def guard():
    assert time.time()-START<900 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert all(0<int(os.environ[k])<=8 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=512*1024**2

def loadstats(seed):
    v=np.load(SOURCE/f'seed{seed}/TRAIN_TRANSFORMS.npz',allow_pickle=False)
    return [{k:v[f'{k}_{j}'] for k in ('mean','std','constant')} for j in range(6)]

def anchor(z,p,rr,seed,stats):
    cp=read(SOURCE/f'seed{seed}/P_CHECK.json');coef=np.load(SOURCE/f'seed{seed}/Ccap_lambda0.01_COEFFICIENTS.npz',allow_pickle=False)
    out=cp['a']*np.asarray(p,np.float64)+cp['b']
    for row in rr:
        o,n=row['offset'],row['windows'];current=np.asarray(z[o+4:o+n],np.float64).reshape(n-4,1536)
        c=d4.normalize(current,stats[0]);pr=d4.normalize(out[o+4:o+n],stats[1]);q=d4.normalize(c*c,stats[5])
        x=np.concatenate([c/np.sqrt(1536),pr/np.sqrt(192),q/np.sqrt(1536)],1)
        out[o+4:o+n]+=x@coef['coef']+coef['bias'];guard()
    assert np.isfinite(out).all();return out

def factors(z,row,st,basis):
    o,n=row['offset'],row['windows'];current=np.asarray(z[o:o+n],np.float64).reshape(n,1536)
    q=d4.normalize(current,st)@basis
    return np.concatenate([q[4:],q[2:n-2,16:],q[:n-4,16:]],1)

def products(a):
    u=a[:,:16]
    return [(u[:,:,None]*a[:,s][:,None,:]).reshape(len(a),256) for s in (slice(16,32),slice(32,48),slice(48,64))]

def features(raw,st,ps):
    a=d4.normalize(raw,st);jj=products(a)
    return np.concatenate([a[:,:32]/np.sqrt(32),a[:,32:48]/4,a[:,48:64]/4]+[d4.normalize(j,s)/16 for j,s in zip(jj,ps)],1)

def indices(arm):
    ids=list(range(64))
    if arm!='Add':
        k={'CC':0,'Near':1,'Older':2}[arm];ids+=list(range(64+k*256,64+(k+1)*256))
    return np.array(ids)

def math_check():
    rng=np.random.default_rng(22101);u=rng.normal(size=(7,16));v=rng.normal(size=(7,16));B=rng.normal(size=(16,16,5))
    explicit=np.stack([sum(u[i,j]*v[i,k]*B[j,k] for j in range(16) for k in range(16)) for i in range(7)])
    flattened=(u[:,:,None]*v[:,None,:]).reshape(7,256)@B.reshape(256,5)
    error=float(abs(explicit-flattened).max());assert error<1e-12
    for tag in (0,1000):
        z=(tag+np.arange(10))[:,None,None]*np.ones((10,8,192));st={'mean':np.zeros(1536),'std':np.ones(1536),'constant':np.zeros(1536,dtype=bool)}
        basis=np.zeros((1536,32));basis[0,:]=1
        r=factors(z,{'offset':0,'windows':10},st,basis)
        assert np.array_equal(r[:,0],tag+np.arange(4,10)) and np.array_equal(r[:,32],tag+np.arange(2,8)) and np.array_equal(r[:,48],tag+np.arange(6))
    x=rng.normal(size=(71,320));y=rng.normal(size=(71,3));xm=x.mean(0);ym=y.mean(0);xc=x-xm;yc=y-ym
    G=xc.T@xc/71;H=xc.T@yc/71;lam=.01;bb=la.solve(G+lam*np.eye(320),H,assume_a='pos');bias=ym-xm@bb
    aug=np.column_stack([x,np.ones(71)]);pen=np.column_stack([np.eye(320)*np.sqrt(71*lam),np.zeros(320)])
    direct=la.lstsq(np.vstack([aug,pen]),np.vstack([y,np.zeros((320,3))]))[0]
    err=float(abs(np.vstack([bb,bias])-direct).max());assert err<1e-12
    assert all(np.array_equal(indices(arm)[:64],np.arange(64)) for arm in ARMS)
    return dict(passed=True,bilinear_equivalence_max_abs=error,ridge_augmented_lstsq_max_abs=err,record_time_support_canaries=True,identical_additive_columns=True)

def prepare():
    assert not (ROOT/'PROTOCOL.json').exists();guard();source=read(SOURCE/'PROTOCOL.json');rows=source['records']
    paths=[Path(__file__),Path(d4.__file__),ROOT/'PLAN.md',SOURCE/'PROTOCOL.json',SOURCE/'COORDINATOR_RESULT_REVIEW.json',SOURCE/'SUMMARY.json']
    inputs={};lengths=[]
    for seed in SEEDS:
        folder=SOURCE/f'seed{seed}';stats=loadstats(seed)
        paths.extend(folder/p for p in ('TRAIN_TRANSFORMS.npz','TRAIN_MOMENTS.npz','P_CHECK.json','Ccap_lambda0.01_COEFFICIENTS.npz','Ccap_lambda0.01_METRICS.json'))
        for phase in ('train','val'):
            for p in [folder/f'{phase}_P.npy',d4.P/f'baseline/head_seed{seed}/{phase}_latent.npy']:
                a=np.load(p,mmap_mode='r');s=p.stat();inputs[str(p)]=dict(shape=list(a.shape),dtype=str(a.dtype),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    for phase,rr in rows.items():
        pp=d4.OLD/f'envelope_readout_seed1024/{phase}_envelope.npy';a=np.load(pp,mmap_mode='r');assert a.shape==({'train':252454,'val':30500}[phase],1,192)
        for p in [pp]+[Path(r['envelope_path']) for r in rr]:
            if str(p) in inputs:continue
            a=np.load(p,mmap_mode='r');s=p.stat();inputs[str(p)]=dict(shape=list(a.shape),dtype=str(a.dtype),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
        for r in rr:
            n=int(np.prod(inputs[r['envelope_path']]['shape']));assert n>=r['common_samples']
            lengths.append(dict(record=r['record'],phase=phase,target_samples=n,common_samples=r['common_samples']))
    dump(ROOT/'ENGINEERING_CHECK.json',math_check())
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,authorization='Coordinator approved D5 PLAN; source reviewed in parallel without extra user gate',seeds=SEEDS,arms=ARMS,lambdas=LAMBDAS,max_solutions=36,
        anchor='D4 Ccap fixed lambda=.01, no refit; pure current anchor; CC still has common historical additive effects',
        source_pins={str(p):d4.sha(p) for p in paths},inputs=inputs,record_lengths=lengths,records=rows,mask=source['mask'],
        PCA='32 current TRAIN covariance eigenvectors, same transform all times; u first16, v second16',
        old_cache_bytes=source['initial_existing_cache_bytes'],CPU_threads=8,RAM_GiB=8,GPU=False,wall_minutes=15,disk_MiB=512,
        previous_D4_validation_selected_anchor=True,test_DTU=False,new_NN_fits=False,automatic_retry=False))
    print('D5_PREPARED_NOT_FITTED',flush=True)

def seed_run(seed,protocol):
    started=clock();folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);rr=protocol['records'];stats=loadstats(seed);guard()
    status(seed=seed,stage='TRAIN_PCA_from_frozen_moments');t=clock()
    with np.load(SOURCE/f'seed{seed}/TRAIN_MOMENTS.npz',allow_pickle=False) as m:
        assert int(m['n'])==250750;full=m['G'];cov=full[:1536,:1536].copy()*1536;del full
    eig,vec=la.eigh(cov,subset_by_index=(1504,1535),driver='evr');eig=eig[::-1];vec=vec[:,::-1]
    vec*=np.where(vec[np.argmax(abs(vec),axis=0),np.arange(32)]<0,-1,1)[None,:]
    good=eig>0;basis=np.zeros_like(vec);basis[:,good]=vec[:,good]/np.sqrt(eig[good]);del cov,vec
    account('PCA_from_frozen_TRAIN_covariance',seed,t,components=32)
    z={p:np.load(d4.P/f'baseline/head_seed{seed}/{p}_latent.npy',mmap_mode='r') for p in ('train','val')}
    yy={p:np.load(d4.OLD/f'envelope_readout_seed1024/{p}_envelope.npy',mmap_mode='r') for p in ('train','val')}
    aa={}
    for phase in ('train','val'):
        t=clock();pp=np.load(SOURCE/f'seed{seed}/{phase}_P.npy',mmap_mode='r');aa[phase]=anchor(z[phase],pp,rr[phase],seed,stats)
        account('frozen_Ccap_reconstruction_'+phase,seed,t,windows=len(pp))
    t=clock();baseline=d4.score(aa['val'],yy['val'],rr['val'],protocol['mask']);reference=read(SOURCE/f'seed{seed}/Ccap_lambda0.01_METRICS.json')
    err=max(abs(baseline['r3s']-reference['r3s']),abs(baseline['r60s']-reference['r60s']),max(abs(a['r']-b['r']) for a,b in zip(baseline['blocks'],reference['blocks'])))
    assert err<=1e-10;dump(folder/'ANCHOR_CHECK.json',dict(max_metric_abs=err,baseline=baseline));account('anchor_VAL_score_check',seed,t,windows=30500,blocks=752)
    status(seed=seed,stage='TRAIN_factor_statistics');t=clock();s=np.zeros(64);sq=s.copy();n=0
    for row in rr['train']:
        raw=factors(z['train'],row,stats[0],basis);s+=raw.sum(0);sq+=(raw*raw).sum(0);n+=len(raw);guard()
    assert n==250750;st=d4.stats_finish(s,sq,n);account('TRAIN_factor_statistics',seed,t,windows=n)
    t=clock();ss=[np.zeros(256) for _ in range(3)];tt=[np.zeros(256) for _ in range(3)]
    for row in rr['train']:
        for j,x in enumerate(products(d4.normalize(factors(z['train'],row,stats[0],basis),st))):ss[j]+=x.sum(0);tt[j]+=(x*x).sum(0)
        guard()
    ps=[d4.stats_finish(s,t,n) for s,t in zip(ss,tt)];account('TRAIN_product_statistics',seed,t,windows=n)
    np.savez(folder/'TRAIN_TRANSFORMS.npz',basis=basis,eigenvalues=eig,**{f'main_{k}':v for k,v in st.items()},**{f'product{j}_{k}':v for j,p in enumerate(ps) for k,v in p.items()})
    status(seed=seed,stage='TRAIN_Gram');t=clock();G=np.zeros((832,832));H=np.zeros((832,192));xs=np.zeros(832);rs=np.zeros(192)
    for row in rr['train']:
        o,w=row['offset'],row['windows'];x=features(factors(z['train'],row,stats[0],basis),st,ps);r=np.asarray(yy['train'][o+4:o+w,0],np.float64)-aa['train'][o+4:o+w]
        assert np.isfinite(x).all() and np.isfinite(r).all();G+=x.T@x;H+=x.T@r;xs+=x.sum(0);rs+=r.sum(0);guard()
    xm=xs/n;rm=rs/n;G=G/n-np.outer(xm,xm);H=H/n-np.outer(xm,rm);np.savez(folder/'TRAIN_MOMENTS.npz',G=G,H=H,x_mean=xm,residual_mean=rm,n=n)
    account('TRAIN_Gram_crossmoments',seed,t,windows=n)
    t=clock();xv=np.zeros((30304,832));vi=[];cursor=0
    for row in rr['val']:
        x=features(factors(z['val'],row,stats[0],basis),st,ps);xv[cursor:cursor+len(x)]=x;cursor+=len(x);vi.extend(range(row['offset']+4,row['offset']+row['windows']))
    assert cursor==30304;vi=np.asarray(vi);fallback=np.setdiff1d(np.arange(30500),vi);account('VAL_features',seed,t,windows=30304)
    results=[]
    for arm in ARMS:
        status(seed=seed,stage='ridge',arm=arm,solutions_done=len(results));ids=indices(arm);g=G[np.ix_(ids,ids)];h=H[ids];ma=xm[ids];t=clock()
        values,vectors=la.eigh((g+g.T)*.5,driver='evr');assert values.min()>-1e-8;vh=vectors.T@h;account('shared_eigen',seed,t,arm=arm,d=len(ids))
        for lam in LAMBDAS:
            t=clock();coef=vectors@(vh/(values[:,None]+lam));bias=rm-ma@coef
            residual=float(abs((g+lam*np.eye(len(g)))@coef-h).max());assert residual<1e-8
            pred=aa['val'].copy();pred[vi]+=xv[:,ids]@coef+bias;assert np.array_equal(pred[fallback],aa['val'][fallback]);account('coefficients_and_VAL_prediction',seed,t,arm=arm,lam=lam)
            t=clock();metrics=d4.score(pred,yy['val'],rr['val'],protocol['mask']);account('VAL_score',seed,t,arm=arm,lam=lam,windows=30500,blocks=752)
            tag=f'{arm}_lambda{lam:g}';np.savez(folder/f'{tag}_COEFFICIENTS.npz',coef=coef,bias=bias)
            result=dict(arm=arm,seed=seed,lam=lam,d=len(ids),parameters=int(coef.size+bias.size),effective_df=float(np.sum(values/(values+lam))),normal_equation_max_abs=residual,r3s=metrics['r3s'],r60s=metrics['r60s'])
            dump(folder/f'{tag}_METRICS.json',dict(**result,people=metrics['people'],blocks=metrics['blocks'],records=metrics['records']));results.append(result);guard()
    dump(folder/'COMPLETED.json',dict(seed=seed,solutions=12,wall_seconds=time.time()-started[0],CPU_seconds=time.process_time()-started[1],constant_main=int(st['constant'].sum()),constant_products=[int(p['constant'].sum()) for p in ps],positive_PCA_components=int(good.sum()),PCA_eigenvalues=list(map(float,eig)),results=results))
    return results,baseline

def run():
    protocol=read(ROOT/'PROTOCOL.json');assert protocol['frozen'] and protocol['max_solutions']==36;guard()
    for p,h in protocol['source_pins'].items():assert d4.sha(p)==h,p
    for p,v in protocol['inputs'].items():
        s=Path(p).stat();assert s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'],p
    assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('15min cap')));signal.alarm(900)
    dump(ROOT/'STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha=d4.sha(Path(__file__)),protocol_sha=d4.sha(ROOT/'PROTOCOL.json')))
    results=[];baseline={}
    for seed in SEEDS:
        a,b=seed_run(seed,protocol);results+=a;baseline[str(seed)]=b
    assert len(results)==36;selection=[];mean=float(np.mean([b['r60s'] for b in baseline.values()]))
    for arm in ARMS:
        values=[dict(lam=lam,mean=float(np.mean([r['r60s'] for r in results if r['arm']==arm and r['lam']==lam]))) for lam in LAMBDAS]
        best=max(values,key=lambda x:(x['mean'],x['lam']));use=best['mean']>mean
        selection.append(dict(arm=arm,selected_lambda=best['lam'] if use else None,anchor_selected=not use,mean_VAL60s=best['mean'] if use else mean,all_candidates=values))
    for p,v in protocol['inputs'].items():
        s=Path(p).stat();assert s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'],p
    with (ROOT/'RESULTS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    dump(ROOT/'SUMMARY.json',dict(all_solutions=results,baseline=baseline,selection=selection,scientific_status='REPEATED_SOURCE_VAL_DEVELOPMENT_ONLY',goal_complete=False))
    dump(ROOT/'COST.json',dict(stages=COST,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,encoder_head_forwards=0,new_NN_updates=0))
    dump(ROOT/'COMPLETED.json',dict(solutions=36,seeds=3,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),derived_disk_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),inputs_unchanged=True,test_DTU=False))
    status(stage='COMPLETED',solutions=36,goal_complete=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['math','prepare','run']);args=parser.parse_args()
    assert os.uname().sysname=='Linux' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    try:
        if args.action=='math':print(json.dumps(math_check()))
        elif args.action=='prepare':prepare()
        else:run()
    except Exception:
        if args.action=='run':dump(ROOT/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
