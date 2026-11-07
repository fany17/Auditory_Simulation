"""D9 six bounded frozen-feature physical-objective solves on2203."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');D8=BASE/'s03_finetuned_readout_transfer_20261101';ROOT=BASE/'s03_physical_objective_20261101'
sys.path.insert(0,str(D8/'input'))
import pub_01_s03_ss07_ft_readout as d8
import pub_01_s03_ss07_physical_core as core
metric=d8.metric;d4=d8.d4;SEEDS=(1024,2026,31415);ROLES=('W','F');LAMBDA=.01/192
START=time.time();CPU_START=time.process_time()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def update(p,v):
    p=Path(p);assert p.is_relative_to(ROOT);tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));os.replace(tmp,p)
def guard(arrays=False):
    assert time.time()-START<5400,'90min per worker'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=256*1024**2
    assert all(0<int(os.environ[k])<=8 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<=16*1024**3
    if arrays:assert sum(p.stat().st_size for r in BASE.glob('s03_*_20261101') for p in r.rglob('*.npy'))<=25*1024**3
def assert_protocol():
    p=read(ROOT/'PROTOCOL.json');assert p['frozen'] and p['solves']==6
    for path,h in p['source_pins'].items():assert d4.sha(path)==h,path
    for path,v in p['inputs'].items():
        st=Path(path).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],path
    return p
def geometry(nfit,records,T):
    h=np.hanning(192);alpha=h[96:]/(h[96:]+h[:96]);rows=[]
    for j,a in enumerate(alpha):
        b=1-a;q=np.array([b,a]);left=np.array([b,0.]);tail=np.array([0.,float(j<95)])
        F=((nfit-records)*np.outer(q,q)+records*np.outer(left,left)+records*np.outer(tail,tail))/T
        W=np.diag([nfit*b/T,((nfit-records)*a+(records if j<95 else 0.))/T])
        rows.append(dict(F_bias_block=F.tolist(),W_bias_block=W.tolist()))
    return rows
def frozen(seed):
    f=D8/f'seed{seed}';st=np.load(f/'TRAIN_TRANSFORMS.npz',allow_pickle=False)
    stats=[{k:st[f'{k}_{i}'] for k in ('mean','std','constant')} for i in range(3)]
    co=np.load(f/'Ccap_lambda0.01_COEFFICIENTS.npz',allow_pickle=False)
    return stats,dict(coef=co['coef'].copy(),bias=co['bias'].copy()),read(f/'P_FT_BASELINE.json')['calibration']
def math_check():
    rng=np.random.default_rng(99101);n=7;device='cpu';dtype=torch.float64
    x=torch.zeros((n,3264),dtype=dtype);x[4:,:3]=torch.tensor(rng.normal(size=(3,3)),dtype=dtype)
    active=torch.tensor([0.]*4+[1.]*3,dtype=dtype);base=torch.tensor(rng.normal(scale=.1,size=(n,192)),dtype=dtype)
    target=rng.normal(size=(n-1)*96+192);left=torch.arange(3,n-1);right=left+1;tail=torch.tensor([n-1]);first=torch.tensor([0])
    yp=torch.tensor(np.stack([target[i*96:i*96+96] for i in right]),dtype=dtype)
    yt=torch.tensor(target[(n-1)*96+96:(n-1)*96+191][None,:],dtype=dtype)
    op=core.PhysicalQuadratic(x,active,base,yp,yt,left,right,tail,first,geometry(3,1,382),LAMBDA)
    theta=torch.zeros((3265,190),dtype=dtype);theta[:3]=torch.tensor(rng.normal(scale=.02,size=(3,190)));theta[-1]=torch.tensor(rng.normal(scale=.01,size=190))
    value=op.objectives(theta);pred=(base+op.A(theta)).numpy();h=np.hanning(192);weight=np.zeros(len(target));total=weight.copy();error=weight.copy()
    for i,v in enumerate(pred):
        s=i*96;weight[s:s+192]+=h;total[s:s+192]+=h*v;error[s:s+192]+=h*(v-target[s:s+192])**2
    keep=np.arange(385,(n-1)*96+191);assert len(keep)==op.T
    W=float(np.mean(error[keep]/weight[keep]));F=float(np.mean((total[keep]/weight[keep]-target[keep])**2));direct=max(abs(W-value['W_data']),abs(F-value['F_data']));assert direct<1e-12
    d=theta.clone();d[:3]=torch.tensor(rng.normal(size=(3,190)));d[-1]=torch.tensor(rng.normal(size=190))
    e=theta.clone();e[:3]=torch.tensor(rng.normal(size=(3,190)));e[-1]=torch.tensor(rng.normal(size=190))
    symmetry=[];grad=[]
    for role in ROLES:
        hd=op.hessian(d,role);he=op.hessian(e,role);sy=abs(float((d*he).sum()-(e*hd).sum()));assert sy<1e-10;symmetry.append(sy)
        analytic=2*float(((op.hessian(theta,role)-op.rhs(role))*d).sum());eps=1e-5
        fd=(op.objectives(theta+eps*d)[role+'_total']-op.objectives(theta-eps*d)[role+'_total'])/(2*eps);err=abs(analytic-fd);assert err<1e-7;grad.append(err)
    q=torch.tensor(rng.normal(size=(9,9)),dtype=dtype);H=q.T@q+.01*torch.eye(9,dtype=dtype);rhs=torch.tensor(rng.normal(size=9),dtype=dtype)
    got,proof=core.pcg(lambda v:H@v,rhs,torch.zeros(9,dtype=dtype),lambda v:v/H.diag(),lambda *_:None,tolerance=1e-10)
    err=float(abs(got-torch.linalg.solve(H,rhs)).max());assert proof['converged'] and err<1e-8
    assert torch.count_nonzero(op.A(theta)[:,[0,191]])==0 and torch.count_nonzero(op.A(theta)[:4])==0
    return dict(passed=True,scope='synthetic changed operator/PCG only; no real arrays',direct_physical_W_F_max_abs=direct,Hessian_symmetry_max_abs=max(symmetry),directional_gradient_max_abs=max(grad),PCG_direct_max_abs=err,PCG_hessian_actions=proof['hessian_actions'],fixed_endpoints_and_first4=True)
def prepare():
    assert not (ROOT/'PROTOCOL.json').exists();approval=read(ROOT/'COORDINATOR_PLAN_REVIEW.json');assert approval['status']=='PLAN_APPROVED' and approval['approved_for_execution']
    for key,name in [('plan_sha256','PLAN.md'),('inventory_sha256','INPUT_INVENTORY.json'),('geometry_sha256','GEOMETRY_PLAN_CHECK.json')]:assert approval[key]==d4.sha(ROOT/name)
    inv=read(ROOT/'INPUT_INVENTORY.json');old=read(D8/'PROTOCOL.json');pins=[Path(__file__),Path(core.__file__),ROOT/'PLAN.md',ROOT/'INPUT_INVENTORY.json',ROOT/'GEOMETRY_PLAN_CHECK.json',ROOT/'COORDINATOR_PLAN_REVIEW.json']
    pins.extend(Path(path) for path in inv['source_pins']);pins.extend(Path(path) for path in old['source_pins']);pins.extend(Path(m.__file__) for m in (d8,d4,metric))
    for path,h in inv['source_pins'].items():assert d4.sha(path)==h,path
    torch.set_num_threads(8);guard(True);dump(ROOT/'MATH_CHECK.json',math_check())
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,seeds=SEEDS,roles=ROLES,solves=6,lambda_scalar=LAMBDA,maximum_CG_updates=200,maximum_Hessian_actions=214,tolerance=1e-8,records=old['records'],mask=old['mask'],source_pins={str(path):d4.sha(path) for path in dict.fromkeys(pins)},inputs=old['inputs']|inv['inputs'],counts=inv['counts'],no_new_NPY=True,encoder_head_forwards=0,old_test_DTU=False,scientific_status='REPEATED_SOURCE_TRAIN_VAL_DEVELOPMENT_ONLY',goal_complete=False))
    print('D9_FROZEN_NO_REAL_NEW_SOLVES',flush=True)
def score(seed,co,p):
    stats,_,cal=frozen(seed);z=np.load(D8/f'seed{seed}/val_latent.npy',mmap_mode='r');raw=np.load(D8/f'seed{seed}/val_P.npy',mmap_mode='r');base=cal['a']*np.asarray(raw,np.float64)+cal['b'];records=[]
    for row in p['records']['val']:
        o,w=row['offset'],row['windows'];pred=base[o:o+w].copy();pred[4:]+=d8.row_features(z,base,row,stats)@co['coef']+co['bias']
        target=np.load(row['envelope_path'],mmap_mode='r');bb=[b for b in p['mask']['blocks'] if b['record']==row['record']]
        records.append(metric.measure(pred,target,row,bb));guard()
    out=metric.aggregate(records);assert out['phases']['val']['windows']==30500 and len(out['people'])==49 and len(out['segments'])==752
    return out
def check_real():
    p=assert_protocol();assert not torch.cuda.is_initialized();checks=[];windows=0
    # Targets are common to all seeds and read only once for this check.
    y=np.load(d8.OLD/'envelope_readout_seed1024/train_envelope.npy',mmap_mode='r')
    for row in p['records']['train']:
        o,w=row['offset'],row['windows'];part=np.asarray(y[o:o+w,0]);assert np.array_equal(part[:-1,96:],part[1:,:96])
        raw=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        for i in range(w):assert np.array_equal(part[i],np.asarray(raw[i*96:i*96+192],np.float32))
        windows+=w;guard()
    assert windows==252454
    g=geometry(250750,426,24112044);old=read(ROOT/'GEOMETRY_PLAN_CHECK.json')['blocks']
    difference=max(float(np.max(abs(np.asarray(a[role+'_bias_block'])-np.asarray(b[role+'_bias_block'])))) for a,b in zip(g,old) for role in ROLES);assert difference<1e-14
    for seed in SEEDS:
        stats,co,cal=frozen(seed);z=np.load(D8/f'seed{seed}/val_latent.npy',mmap_mode='r');raw=np.load(D8/f'seed{seed}/val_P.npy',mmap_mode='r');base=cal['a']*np.asarray(raw,np.float64)+cal['b']
        ref=read(D8/f'seed{seed}/Ccap_lambda0.01_METRICS.json')['metrics'];bykey={(b['record'],b['start']):b['r'] for b in ref['segments']}
        for row in (p['records']['val'][0],p['records']['val'][-1]):
            o,w=row['offset'],row['windows'];pred=base[o:o+w].copy();pred[4:]+=d8.row_features(z,base,row,stats)@co['coef']+co['bias']
            st=[stats[0],stats[1],None,None,None,stats[2]];cf=dict(BZ=co['coef'][:1536],BP=co['coef'][1536:1728],BQ=co['coef'][1728:],bias=co['bias'])
            other=metric.predict(np.asarray(z[o:o+w]),np.asarray(raw[o:o+w]),cal,st,cf);wave=float(abs(pred-other).max());assert wave<=1e-8
            target=np.load(row['envelope_path'],mmap_mode='r');bb=[b for b in p['mask']['blocks'] if b['record']==row['record']];m=metric.measure(pred,target,row,bb)
            err=max(abs(b['r']-bykey[(b['record'],b['start'])]) for b in m['segments']);assert err<=1e-10
            checks.append(dict(seed=seed,record=row['record'],D8_prediction_max_abs=wave,D8_block_r_max_abs=err))
    dump(ROOT/'REAL_INPUT_CHECK.json',dict(passed=True,TRAIN_windows_exact_duplicate_and_raw_float32_target=windows,physical_targets=24112044,geometry_max_abs=difference,initial_VAL_cases=checks,encoder_head_forward=0,new_solves=0))
    print('D9_REAL_CHANGED_PATH_CHECK_PASSED',flush=True)
def tensors(seed,p,device):
    t=time.time();stats,co,cal=frozen(seed);rows=p['records']['train'];N=252454;z=np.load(D8/f'seed{seed}/train_latent.npy',mmap_mode='r');raw=np.load(D8/f'seed{seed}/train_P.npy',mmap_mode='r');base=cal['a']*np.asarray(raw,np.float64)+cal['b']
    x=torch.zeros((N,3264),dtype=torch.float64,device=device);active=torch.zeros(N,dtype=torch.float64,device=device);left=[];tail=[];first=[];cursor=0
    for row in rows:
        o,w=row['offset'],row['windows'];part=d8.row_features(z,base,row,stats);x[o+4:o+w]=torch.from_numpy(part).to(device);active[o+4:o+w]=1
        first.append(cursor);left.extend(range(o+3,o+w-1));tail.append(o+w-1);cursor+=w-4;guard()
    assert cursor==250750 and len(first)==426
    y=np.load(d8.OLD/'envelope_readout_seed1024/train_envelope.npy',mmap_mode='r');ln=np.array(left);rn=ln+1;tn=np.array(tail)
    yp=torch.as_tensor(np.array(y[rn,0,:96],np.float64),device=device);yt=torch.as_tensor(np.array(y[tn,0,96:191],np.float64),device=device)
    ids=lambda v:torch.as_tensor(v,dtype=torch.long,device=device)
    op=core.PhysicalQuadratic(x,active,torch.as_tensor(base,device=device),yp,yt,ids(ln),ids(rn),ids(tn),ids(first),geometry(250750,426,24112044),LAMBDA)
    assert op.T==p['counts']['common_physical_target_points']
    saved=np.load(D8/f'seed{seed}/TRAIN_MOMENTS.npz',allow_pickle=False);moment=np.diag(saved['G']).copy()+saved['x_mean']**2
    theta=torch.as_tensor(np.vstack((co['coef'][:,1:191],co['bias'][None,1:191])),device=device)
    return op,theta,torch.as_tensor(moment,device=device),co,time.time()-t
def worker(seed,device):
    p=assert_protocol();assert read(ROOT/'REAL_INPUT_CHECK.json')['passed'];torch.set_num_threads(8);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90min worker cap')));signal.alarm(5400)
    folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);current={};stage='build';results=[]
    dump(folder/'STARTED.json',dict(seed=seed,pid=os.getpid(),device=device,protocol_sha=d4.sha(ROOT/'PROTOCOL.json'),time=time.time()))
    try:
        op,start,moment,anchor,build_seconds=tensors(seed,p,device);dump(folder/'INITIAL_TRAIN_OBJECTIVES.json',op.objectives(start));guard(True)
        for role in ROLES:
            stage=role;t=time.time();cpu=time.process_time();before=op.counts.copy();rhs=op.rhs(role);precondition=op.preconditioner(moment,role);history=[]
            def progress(row,theta):
                current['theta']=theta;history.append(row);update(folder/f'{role}_STATUS.json',dict(seed=seed,role=role,wall_seconds=time.time()-t,**row));guard()
            result,proof=core.pcg(lambda v:op.hessian(v,role),rhs,start,precondition,progress)
            assert proof['hessian_actions']==op.counts['Hessian']-before['Hessian']<=214
            train=op.objectives(result);co=dict(coef=anchor['coef'].copy(),bias=anchor['bias'].copy());co['coef'][:,1:191]=result[:-1].cpu().numpy();co['bias'][1:191]=result[-1].cpu().numpy()
            assert np.array_equal(co['coef'][:,[0,191]],anchor['coef'][:,[0,191]]) and np.array_equal(co['bias'][[0,191]],anchor['bias'][[0,191]])
            np.savez(folder/f'{role}_COEFFICIENTS.npz',**co);m=score(seed,co,p)
            summary=dict(seed=seed,role=role,**{k:v for k,v in proof.items() if k!='history'},r=m['phases']['val']['r'],MSE=m['phases']['val']['MSE'],train_objectives=train,wall_seconds=time.time()-t,CPU_seconds=time.process_time()-cpu,operator_counts={k:op.counts[k]-before[k] for k in op.counts})
            dump(folder/f'{role}_SOLVER.json',dict(**summary,history=history));dump(folder/f'{role}_METRICS.json',dict(**summary,metrics=m));results.append(summary);guard(True)
        dump(folder/'COMPLETED.json',dict(seed=seed,solves=2,results=results,build_seconds=build_seconds,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),encoder_head_forward=0,goal_complete=False))
    except Exception:
        if 'theta' in current:np.savez(folder/'FAILURE_LAST_FREE_PARAMETERS.npz',theta=current['theta'].cpu().numpy())
        dump(folder/'FAILURE.json',dict(stage=stage,time=time.time(),traceback=traceback.format_exc(),automatic_retry=False));raise
def queue():
    p=assert_protocol();assert read(ROOT/'REAL_INPUT_CHECK.json')['passed'];assert not (ROOT/'QUEUE_STARTED.json').exists()
    dump(ROOT/'QUEUE_STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha=d4.sha(Path(__file__)),protocol_sha=d4.sha(ROOT/'PROTOCOL.json')))
    pending=list(SEEDS);active={};done=[];failures=[];launches=[]
    while pending or active:
        for slot in (0,1):
            if slot in active or not pending:continue
            seed=pending.pop(0);log=(ROOT/f'seed{seed}.log').open('xb');argv=[sys.executable,str(Path(__file__).resolve()),'worker','--seed',str(seed),'--device',f'cuda:{slot}'];child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy());log.close();active[slot]=(seed,child)
            launches.append(dict(seed=seed,device=f'cuda:{slot}',pid=child.pid,argv=argv,time=time.time()));update(ROOT/'WORKER_LAUNCHES.json',launches)
        for slot,(seed,child) in list(active.items()):
            code=child.poll()
            if code is None:continue
            (done if code==0 else failures).append(dict(seed=seed,exit=code));del active[slot]
        update(ROOT/'STATUS.json',dict(stage='RUNNING' if pending or active else 'FINISHED',pending=pending,active={str(k):dict(seed=v[0],pid=v[1].pid) for k,v in active.items()},done=done,failures=failures,elapsed=time.time()-START))
        assert time.time()-START<10800,'180min matrix cap'
        if pending or active:time.sleep(5)
    if failures:dump(ROOT/'FAILURE.json',dict(failed_workers=failures,completed_workers=done,automatic_retry=False));return
    results=[r for seed in SEEDS for r in read(ROOT/f'seed{seed}/COMPLETED.json')['results']];assert len(results)==6
    summary=dict(results=results,mean_r={role:float(np.mean([r['r'] for r in results if r['role']==role])) for role in ROLES},paired_F_minus_W=[next(r['r'] for r in results if r['seed']==seed and r['role']=='F')-next(r['r'] for r in results if r['seed']==seed and r['role']=='W') for seed in SEEDS],goal_complete=False)
    dump(ROOT/'SUMMARY.json',summary);dump(ROOT/'COMPLETED.json',dict(solves=6,wall_seconds=time.time()-START,workers=[read(ROOT/f'seed{seed}/COMPLETED.json') for seed in SEEDS],all_converged=all(r['converged'] for r in results),derived_disk_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),free_disk_bytes=shutil.disk_usage(ROOT).free,encoder_head_forward=0,new_NPY=0,old_test_DTU=False,goal_complete=False))
    update(ROOT/'STATUS.json',dict(stage='COMPLETED',solves=6,goal_complete=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','check','queue','worker']);parser.add_argument('--seed',type=int);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    assert os.uname().sysname=='Linux'
    if args.action=='prepare':prepare()
    elif args.action=='check':check_real()
    elif args.action=='queue':queue()
    else:assert args.seed in SEEDS;worker(args.seed,args.device)
