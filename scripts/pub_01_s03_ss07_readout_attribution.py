"""D6:27 new constrained ridge models +18 immutable D4 models;2203 CPU."""
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
import pub_01_s03_ss07_readout_core as core

ROOT=d4.BASE/'s03_readout_attribution_20261101';SOURCE=d4.ROOT
SEEDS=d4.SEEDS;LAMBDAS=d4.LAMBDAS;ARMS=('Ponly','S32','U32','C','Ccap')
START=time.time();CPU_START=time.process_time();COST=[]
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def clock():return time.time(),time.process_time()
def account(stage,seed,t,**v):COST.append(dict(stage=stage,seed=seed,wall_seconds=time.time()-t[0],CPU_seconds=time.process_time()-t[1],**v))
def status(**v):
    p=ROOT/'STATUS.writing';p.write_text(json.dumps(dict(time=time.time(),elapsed=time.time()-START,**v)));os.replace(p,ROOT/'STATUS.json');print(json.dumps(v),flush=True)
def guard():
    assert time.time()-START<1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert not torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert all(0<int(os.environ[k])<=8 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=512*1024**2

def prepare():
    assert not (ROOT/'PROTOCOL.json').exists();guard();old=read(SOURCE/'PROTOCOL.json');rr=old['records']['val']
    pins=[Path(__file__),Path(core.__file__),Path(d4.__file__),ROOT/'PLAN.md',SOURCE/'PROTOCOL.json',SOURCE/'SUMMARY.json',SOURCE/'COORDINATOR_RESULT_REVIEW.json']
    inputs={};lengths=[]
    for seed in SEEDS:
        folder=SOURCE/f'seed{seed}'
        pins.extend(folder/p for p in ('TRAIN_MOMENTS.npz','TRAIN_TRANSFORMS.npz','P_CHECK.json'))
        for arm in ('C','Ccap'):
            for lam in LAMBDAS:
                for ending in ('COEFFICIENTS.npz','METRICS.json'):pins.append(folder/f'{arm}_lambda{lam:g}_{ending}')
        for p in (folder/'val_P.npy',d4.P/f'baseline/head_seed{seed}/val_latent.npy'):
            a=np.load(p,mmap_mode='r');st=p.stat();inputs[str(p)]=dict(shape=list(a.shape),dtype=str(a.dtype),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    target=d4.OLD/'envelope_readout_seed1024/val_envelope.npy'
    assert np.load(target,mmap_mode='r').shape==(30500,1,192)
    for p in [target]+[Path(r['envelope_path']) for r in rr]:
        if str(p) in inputs:continue
        a=np.load(p,mmap_mode='r');s=p.stat();inputs[str(p)]=dict(shape=list(a.shape),dtype=str(a.dtype),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    for row in rr:
        n=int(np.prod(inputs[row['envelope_path']]['shape']));assert n>=row['common_samples'];lengths.append(dict(record=row['record'],target_samples=n,common_samples=row['common_samples']))
    dump(ROOT/'ENGINEERING_CHECK.json',core.math_check())
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,authorization='Coordinator explicitly approved D6 plus required P-only before any D6 scores',seeds=SEEDS,arms=ARMS,lambdas=LAMBDAS,new_model_lambda_fits=27,reused_D4_model_lambda_fits=18,total_candidate_scores=45,baseline_scores=3,
        source_pins={str(p):d4.sha(p) for p in pins},inputs=inputs,record_lengths=lengths,records=rr,mask=old['mask'],
        offsets=list(range(-15,17)),S32_parameters=37312,U32_parameters=84160,Ponly_parameters=37056,
        normalization='Exact D4 per-coordinate TRAIN statistics and block scaling; sharing is in these coordinates',
        regularizer='Full expanded Frobenius, S32 theta=sqrt(m_k)*w; common dense P and192 free intercepts',
        train_payload_read=False,new_normalization_PCA=False,new_NN_fits=False,test_DTU=False,
        CPU_threads=8,GPU=False,RAM_GiB=8,wall_minutes=30,derived_disk_MiB=512,automatic_retry=False))
    print('D6_PREPARED_NOT_FITTED',flush=True)

def val_features(seed):
    cp=read(SOURCE/f'seed{seed}/P_CHECK.json');p=np.load(SOURCE/f'seed{seed}/val_P.npy',mmap_mode='r')
    baseline=cp['a']*np.asarray(p,np.float64)+cp['b'];assert cp['a']>0
    st=np.load(SOURCE/f'seed{seed}/TRAIN_TRANSFORMS.npz',allow_pickle=False)
    stats=[{k:st[f'{k}_{j}'] for k in ('mean','std','constant')} for j in range(6)]
    z=np.load(d4.P/f'baseline/head_seed{seed}/val_latent.npy',mmap_mode='r');assert z.shape==(30500,8,192)
    normalized=d4.normalize(np.asarray(z,np.float64).reshape(30500,1536),stats[0]);q=d4.normalize(normalized**2,stats[5])/np.sqrt(1536)
    xp=d4.normalize(baseline,stats[1])/np.sqrt(192);xz=normalized/np.sqrt(1536)
    return baseline,xz,xp,q

def save_score(seed,arm,lam,role,bz,bp,bias,quad,df,normal,parameters,features,protocol,folder,reused=None):
    baseline,xz,xp,xq,vi,y=features;t=clock();pred=baseline.copy()
    pred[vi]+=xz[vi]@bz+xp[vi]@bp+bias
    if quad is not None:pred[vi]+=xq[vi]@quad
    fallback=np.setdiff1d(np.arange(30500),vi);assert np.array_equal(pred[fallback],baseline[fallback])
    account('prediction',seed,t,arm=arm,lam=lam,windows=30500,history_inputs=0)
    t=clock();metrics=d4.score(pred,y,protocol['records'],protocol['mask']);account('primary_score',seed,t,arm=arm,lam=lam,windows=30500,blocks=752)
    reference_difference=None
    if reused is not None:
        reference_difference=max(abs(metrics['r3s']-reused['r3s']),abs(metrics['r60s']-reused['r60s']),max(abs(a['r']-b['r']) for a,b in zip(metrics['blocks'],reused['blocks'])))
        assert reference_difference<=1e-10
    t=clock();aux=core.secondary(pred,y,protocol['records'],protocol['mask']);account('secondary_score',seed,t,arm=arm,lam=lam,overlap_pairs=30451,blocks=752)
    tag=f'{arm}_lambda{lam:g}';result=dict(seed=seed,arm=arm,lam=lam,role=role,parameters=parameters,effective_df_total=df,normal_equation_max_abs=normal,reused_primary_max_abs=reference_difference,
        r3s=metrics['r3s'],r60s=metrics['r60s'],window_target_MSE=aux['window_target_MSE'],fused_macro_MSE=aux['macro_MSE']['fused_MSE'],weighted_single_macro_MSE=aux['macro_MSE']['weighted_single_MSE'],
        disagreement_macro=aux['macro_MSE']['disagreement'],overlap_MSE=aux['overlap_MSE'],overlap_NRMSE=aux['overlap_NRMSE'],Hann_identity_max_abs=aux['Hann_identity_max_abs'])
    dump(folder/f'{tag}_METRICS.json',dict(**result,people=metrics['people'],blocks=metrics['blocks'],records=metrics['records'],secondary=aux))
    if role=='NEW':np.savez(folder/f'{tag}_COEFFICIENTS.npz',BZ=bz,BP=bp,bias=bias)
    guard();return result

def seed_run(seed,protocol):
    started=clock();folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);t=clock();status(seed=seed,stage='VAL_features_and_frozen_reuse')
    baseline,xz,xp,xq=val_features(seed);rr=protocol['records'];vi=np.array([i for r in rr for i in range(r['offset']+4,r['offset']+r['windows'])]);assert len(vi)==30304
    y=np.load(d4.OLD/'envelope_readout_seed1024/val_envelope.npy',mmap_mode='r');features=(baseline,xz,xp,xq,vi,y);account('VAL_features',seed,t,windows=30500)
    t=clock();base=d4.score(baseline,y,rr,protocol['mask']);aux=core.secondary(baseline,y,rr,protocol['mask']);dump(folder/'P_BASELINE.json',dict(**base,secondary=aux));account('P_baseline_scores',seed,t,windows=30500,blocks=752,overlap_pairs=30451)
    results=[]
    # Immutable existing models and original score gate precede any NEW fit.
    for arm in ('C','Ccap'):
        for lam in LAMBDAS:
            path=SOURCE/f'seed{seed}/{arm}_lambda{lam:g}_COEFFICIENTS.npz';coef=np.load(path,allow_pickle=False);ref=read(SOURCE/f'seed{seed}/{arm}_lambda{lam:g}_METRICS.json')
            b=coef['coef'];quad=b[1728:] if arm=='Ccap' else None
            results.append(save_score(seed,arm,lam,'REUSED_D4',b[:1536],b[1536:1728],coef['bias'],quad,ref['effective_df']*192,ref['normal_equation_max_abs'],ref['parameters'],features,protocol,folder,reused=ref))
    t=clock()
    with np.load(SOURCE/f'seed{seed}/TRAIN_MOMENTS.npz',allow_pickle=False) as old:
        assert int(old['n'])==250750;full=old['G'];G=full[:1728,:1728].copy();del full
        H=old['H'][:1728].copy();xm=old['x_mean'][:1728].copy();rm=old['residual_mean'].copy()
    account('frozen_TRAIN_moments_load',seed,t,TRAIN_payload_windows=0);guard()
    status(seed=seed,stage='Ponly_ridge');t=clock();gp=G[1536:,1536:];hp=H[1536:];eig,Q=la.eigh((gp+gp.T)*.5,driver='evr');qh=Q.T@hp;account('Ponly_shared_eigen',seed,t)
    for lam in LAMBDAS:
        t=clock();bp=Q@(qh/(eig[:,None]+lam));bz=np.zeros((1536,192));bias=rm-xm[1536:]@bp
        normal=float(abs((gp+lam*np.eye(192))@bp-hp).max());df=192*float(np.sum(eig/(eig+lam)));account('Ponly_model_lambda_solve',seed,t,lam=lam,joint_outputs=192)
        results.append(save_score(seed,'Ponly',lam,'NEW',bz,bp,bias,None,df,normal,37056,features,protocol,folder))
    status(seed=seed,stage='S32_shared_Schur');t=clock();setup=core.shared_setup(G,H);account('S32_shared_setup',seed,t)
    for lam in LAMBDAS:
        t=clock();sol=core.shared_solve(setup,lam);bias=rm-xm[:1536]@sol['BZ']-xm[1536:]@sol['BP'];assert sol['normal_equation_max_abs']<1e-7
        account('S32_Schur_and_P_backsubstitution',seed,t,lam=lam,Schur_dimension=256,P_outputs=192)
        results.append(save_score(seed,'S32',lam,'NEW',sol['BZ'],sol['BP'],bias,None,sol['effective_df_total'],sol['normal_equation_max_abs'],37312,features,protocol,folder))
    del setup;guard();status(seed=seed,stage='U32_local_ridge');t=clock();solutions=core.unshared_solve(G,H,LAMBDAS);account('U32_all_lambda_local_solve',seed,t,per_output_systems=576,model_lambda_fits=3)
    for lam in LAMBDAS:
        sol=solutions[lam];bias=rm-xm[:1536]@sol['BZ']-xm[1536:]@sol['BP'];assert sol['normal_equation_max_abs']<1e-7
        results.append(save_score(seed,'U32',lam,'NEW',sol['BZ'],sol['BP'],bias,None,sol['effective_df_total'],sol['normal_equation_max_abs'],84160,features,protocol,folder))
    assert len(results)==15
    dump(folder/'COMPLETED.json',dict(seed=seed,new_models=9,reused_models=6,scores=15,wall_seconds=time.time()-started[0],CPU_seconds=time.process_time()-started[1],results=results))
    return results,base

def run():
    p=read(ROOT/'PROTOCOL.json');assert p['frozen'] and p['new_model_lambda_fits']==27 and read(ROOT/'ENGINEERING_CHECK.json')['passed'];guard()
    for path,h in p['source_pins'].items():assert d4.sha(path)==h,path
    for path,v in p['inputs'].items():
        st=Path(path).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],path
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min cap')));signal.alarm(1800)
    dump(ROOT/'STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha=d4.sha(Path(__file__)),core_sha=d4.sha(Path(core.__file__)),protocol_sha=d4.sha(ROOT/'PROTOCOL.json')))
    results=[];baseline={}
    for seed in SEEDS:
        rows,base=seed_run(seed,p);results+=rows;baseline[str(seed)]=base
    assert len(results)==45 and sum(r['role']=='NEW' for r in results)==27
    bm=float(np.mean([b['r60s'] for b in baseline.values()]));selection=[];prior=read(SOURCE/'SUMMARY.json')
    for arm in ARMS:
        candidates=[dict(lam=lam,mean=float(np.mean([r['r60s'] for r in results if r['arm']==arm and r['lam']==lam]))) for lam in LAMBDAS]
        best=max(candidates,key=lambda x:(x['mean'],x['lam']));use=best['mean']>bm;selected=best['lam'] if use else None
        if arm in ('C','Ccap'):assert selected==next(r for r in prior['selection'] if r['arm']==arm)['selected_lambda']
        selection.append(dict(arm=arm,selected_lambda=selected,P_baseline_selected=not use,mean_VAL60s=best['mean'] if use else bm,all_candidates=candidates))
    for path,v in p['inputs'].items():
        st=Path(path).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],path
    with (ROOT/'RESULTS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    dump(ROOT/'SUMMARY.json',dict(all_solutions=results,baseline=baseline,selection=selection,scientific_status='REPEATED_SOURCE_VAL_DEVELOPMENT_ONLY',goal_complete=False))
    dump(ROOT/'COST.json',dict(stages=COST,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,TRAIN_payload_windows=0,encoder_head_forwards=0,new_NN_updates=0,new_model_lambda_fits=27,reused_models=18,full_VAL_scores=48))
    dump(ROOT/'COMPLETED.json',dict(new_model_lambda_fits=27,reused_models=18,candidate_scores=45,baseline_scores=3,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_initialized=torch.cuda.is_initialized(),derived_disk_bytes=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file()),inputs_unchanged=True,test_DTU=False))
    status(stage='COMPLETED',new_models=27,reused_models=18,goal_complete=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['math','prepare','run']);args=parser.parse_args()
    assert os.uname().sysname=='Linux' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    try:
        if args.action=='math':print(json.dumps(core.math_check()))
        elif args.action=='prepare':prepare()
        else:run()
    except Exception:
        if args.action=='run':dump(ROOT/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
