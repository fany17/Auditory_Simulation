"""D8 bounded frozen-FT TRAIN/VAL cache +27 ridge fits; server2203 only."""
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
import scipy.linalg as la
import torch

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
P=BASE/'s03_incremental_continuity_20261101'
OLD=BASE/'s03_strong_baselines_20261015/transformer_available01'
ROOT=BASE/'s03_finetuned_readout_transfer_20261101'
for path in (P/'input',OLD/'input',BASE/'s03_conditional_history_probe_20261101/input',BASE/'s03_current_readout_internal_audit_20261101/input'):
    sys.path.append(str(path))
import pub_01_s03_ss07_finetune as ft
import pub_01_s03_ss07_conditional_probe as d4
import pub_01_s03_ss07_generalization_core as metric

SEEDS=(1024,2026,31415);ARMS=('Ponly','C','Ccap');LAMBDAS=(.0001,.01,1.)
START=time.time();CPU_START=time.process_time();COST=[]
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def clock():return time.time(),time.process_time()
def account(stage,seed,t,**v):COST.append(dict(stage=stage,seed=seed,wall_seconds=time.time()-t[0],CPU_seconds=time.process_time()-t[1],**v))
def status(**v):
    p=ROOT/'STATUS.writing';p.write_text(json.dumps(dict(time=time.time(),elapsed=time.time()-START,**v)));os.replace(p,ROOT/'STATUS.json');print(json.dumps(v),flush=True)
def npy_bytes():return sum(p.stat().st_size for root in BASE.glob('s03_*_20261101') for p in root.rglob('*.npy'))
def guard(arrays=False):
    assert time.time()-START<7200,'120min wall cap'
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3,'8GiB RSS cap'
    assert shutil.disk_usage(ROOT).free>=20*1024**3,'20GiB free reserve'
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=7*1024**3,'7GiB new root cap'
    assert all(0<int(os.environ[k])<=8 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<=8*1024**3
    if arrays:assert npy_bytes()<=25*1024**3,'25GiB all SS07 NPY cap'
def inputs_unchanged(protocol):
    for path,v in protocol['inputs'].items():
        st=Path(path).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],path
def prepare():
    assert not (ROOT/'PROTOCOL.json').exists() and not torch.cuda.is_initialized();guard(True)
    inv=read(ROOT/'INPUT_INVENTORY.json');assert inv['no_new_cache_fit_or_forward']
    approval=read(ROOT/'COORDINATOR_PLAN_REVIEW.json');assert approval['status']=='PLAN_APPROVED'
    assert approval['plan_sha256']==d4.sha(ROOT/'PLAN.md') and approval['inventory_sha256']==d4.sha(ROOT/'INPUT_INVENTORY.json')
    native=ft.native()  # Import classes only, no model/data instantiated.
    imported={name:str(Path(module.__file__).resolve()) for name,module in list(sys.modules.items()) if getattr(module,'__file__',None) and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(BASE)}
    assert Path(ft.__file__).resolve()==P/'input/pub_01_s03_ss07_finetune.py'
    assert Path(d4.__file__).resolve()==BASE/'s03_conditional_history_probe_20261101/input/pub_01_s03_ss07_conditional_probe.py'
    assert Path(metric.__file__).resolve()==BASE/'s03_current_readout_internal_audit_20261101/input/pub_01_s03_ss07_generalization_core.py'
    records={phase:d4.rows(phase) for phase in ('train','val')};mask=read(P/'block_eligibility/val.json')
    assert mask['prediction_independent'] and mask['eligible_blocks']==752 and [r['record'] for r in records['val']]==mask['records']
    pins=[Path(__file__),ROOT/'PLAN.md',ROOT/'INPUT_INVENTORY.json',ROOT/'COORDINATOR_PLAN_REVIEW.json',Path(native.__file__),OLD/'training_seed1024_attempt04/config_frozen.json']+[Path(p) for p in imported.values()]+[Path(p) for p in inv['source_pins']]
    for path,h in inv['source_pins'].items():assert d4.sha(path)==h,path
    inputs_unchanged(inv)
    dump(ROOT/'ENGINEERING_CHECK.json',dict(passed=True,scope='metadata/source/mask/budget only, inherited D4 ridge and D7 metrics unchanged; FT raw bridges occur inside bounded run',no_new_forward_or_fit=True))
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,seeds=SEEDS,arms=ARMS,lambdas=LAMBDAS,new_ridge_fits=27,baseline_role='P_FT*',records=records,mask=mask,inputs=inv['inputs'],source_pins={str(p):d4.sha(p) for p in dict.fromkeys(pins)},actual_imported_project_modules=imported,checkpoints={str(v['seed']):v for v in inv['seeds']},precision=inv['precision'],budget=inv['budget'],raw_wave_absolute=1e-6,historical_VAL_loss_absolute=1e-7,positive_affine_Pearson_absolute=1e-10,runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,CUDA=torch.version.cuda),fit_mask='i>=4 common; first4 calibrated FT baseline fallback',selection='shared lambda by mean3seed VAL60s; larger lambda tie; strict improvement else all3 baseline',scientific_status='REPEATED_SOURCE_TRAIN_VAL_DEVELOPMENT_ONLY',test_DTU=False,NN_updates=0,automatic_retry=False))
    print('D8_PREPARED_NO_NEW_FORWARD_FIT',flush=True)

def gpu_driver_mib():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    values=[int(line.split(',')[1]) for line in out.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    return max(values) if values else None

@torch.no_grad()
def encode_phase(seed,phase,model,folder,p,device):
    t=clock();data=ft.RawWindows(phase);n=len(data)
    zfile=folder/f'{phase}_latent.npy';pfile=folder/f'{phase}_P.npy'
    assert not zfile.exists() and not pfile.exists()
    zcache=np.lib.format.open_memmap(zfile,mode='w+',dtype=np.float32,shape=(n,8,192))
    pcache=np.lib.format.open_memmap(pfile,mode='w+',dtype=np.float32,shape=(n,192))
    guard(True);model.eval();loss_total=0.;checks=[];strides=None;extra=0
    for first in range(0,n,64):
        last=min(n,first+64);x,y=data.batch(first,last,device)
        latent=model.encoder(x).transpose(1,2);pred=model.head(latent)
        assert latent.shape==(last-first,8,192) and pred.shape==(last-first,1,192)
        assert torch.isfinite(latent).all() and torch.isfinite(pred).all()
        if strides is None:strides=dict(input=list(x.stride()),latent_view=list(latent.stride()),pred=list(pred.stride()),parameters_require_grad=all(v.requires_grad for v in model.parameters()))
        if phase=='val':
            loss_total+=float(ft.objective(pred,y))*(last-first)
            if first==0 or last==n:
                original=model(x);difference=float((pred-original).abs().max());assert difference<=p['raw_wave_absolute'],'FT original Decoder path wave bridge'
                checks.append(dict(first=first,last=last,original_units_max_abs=difference));extra+=last-first
        zcache[first:last]=latent.cpu().numpy();pcache[first:last]=pred[:,0].cpu().numpy()
        if first%32000==0:status(stage='FT_cache',seed=seed,phase=phase,windows_done=last,windows_total=n);guard()
    zcache.flush();pcache.flush();del zcache,pcache
    result=dict(seed=seed,phase=phase,encoder_head_windows=n,global64_batches=(n+63)//64,extra_original_Decoder_windows=extra,original_Decoder_wave_checks=checks,strides=strides)
    if phase=='val':
        loss=loss_total/n;reference=p['checkpoints'][str(seed)]['best_validation_negative_Pearson'];difference=abs(loss-reference)
        result.update(historical_validation_negative_Pearson=reference,reencoded_validation_negative_Pearson=loss,loss_absolute_difference=difference,passed=difference<=p['historical_VAL_loss_absolute'])
        dump(folder/'FT_VAL_BRIDGE.json',result)
        assert result['passed'],'FT selected full VAL scalar loss bridge failed; no TRAIN forward or fit'
    else:dump(folder/'FT_TRAIN_CACHE.json',result)
    account('FT_raw_encoder_head_cache',seed,t,phase=phase,primary_windows=n,extra_bridge_windows=extra)
    guard(True)

def cache_seed(seed,folder,p,device):
    torch.set_num_threads(2);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=p['precision']['TF32_matmul'];torch.backends.cudnn.allow_tf32=p['precision']['TF32_cudnn']
    native=ft.native();original=native.TransformerCLIP(ft.cfg());model=ft.Decoder(original.eeg);del original
    cp=torch.load(p['checkpoints'][str(seed)]['checkpoint'],map_location='cpu',weights_only=True)
    assert cp['seed']==seed and cp['epoch']==4 and abs(cp['validation_loss']-p['checkpoints'][str(seed)]['best_validation_negative_Pearson'])<=1e-12
    model.load_state_dict(cp['model'],strict=True);del cp;model=model.to(device).eval()
    assert all(v.requires_grad for v in model.parameters())
    encode_phase(seed,'val',model,folder,p,device)
    encode_phase(seed,'train',model,folder,p,device)
    memory=gpu_driver_mib();dump(folder/'CACHE_COMPLETED.json',dict(seed=seed,checkpoint_sha=p['checkpoints'][str(seed)]['checkpoint_sha256'],GPU_driver_MiB=memory,new_NN_updates=0,original_FT_unchanged=True))
    del model;torch.cuda.empty_cache();torch.set_num_threads(8)

def raw_current(z,baseline,row):
    o,w=row['offset'],row['windows']
    return np.asarray(z[o+4:o+w],np.float64).reshape(w-4,1536),baseline[o+4:o+w]
def row_features(z,baseline,row,stats):
    zz,pp=raw_current(z,baseline,row);zn=d4.normalize(zz,stats[0]);pn=d4.normalize(pp,stats[1]);qn=d4.normalize(zn**2,stats[2])
    return np.concatenate((zn/np.sqrt(1536),pn/np.sqrt(192),qn/np.sqrt(1536)),1)
def score(pred,rows,p):
    records=[]
    for row in rows:
        o,w=row['offset'],row['windows'];target=np.load(row['envelope_path'],mmap_mode='r')
        blocks=[b for b in p['mask']['blocks'] if b['record']==row['record']]
        records.append(metric.measure(pred[o:o+w],target,row,blocks));guard()
    out=metric.aggregate(records);assert len(out['people'])==49 and len(out['segments'])==752 and out['phases']['val']['windows']==30500
    return out

def fit_seed(seed,folder,p):
    t=clock();tr=p['records']['train'];vr=p['records']['val']
    rawp=np.load(folder/'train_P.npy',mmap_mode='r');rawv=np.load(folder/'val_P.npy',mmap_mode='r')
    z=np.load(folder/'train_latent.npy',mmap_mode='r');zv=np.load(folder/'val_latent.npy',mmap_mode='r')
    y=np.load(OLD/'envelope_readout_seed1024/train_envelope.npy',mmap_mode='r')
    a,b,points=d4.calibration(rawp,tr);base=a*np.asarray(rawp,np.float64)+b;val=a*np.asarray(rawv,np.float64)+b
    rawmetrics=score(np.asarray(rawv,np.float64),vr,p);basemetrics=score(val,vr,p)
    rawkeys={(r['record'],r['start']):r['r'] for r in rawmetrics['segments']}
    delta=max(abs(r['r']-rawkeys[(r['record'],r['start'])]) for r in basemetrics['segments']);assert delta<=p['positive_affine_Pearson_absolute']
    dump(folder/'P_FT_BASELINE.json',dict(seed=seed,role='P_FT',calibration=dict(a=a,b=b,TRAIN_physical_points=points),positive_affine_block_r_max_abs=delta,**basemetrics))
    account('TRAIN_calibration_and_FT_baseline',seed,t,TRAIN_physical_points=points,VAL_scores=2)
    status(stage='FT_TRAIN_statistics',seed=seed);t=clock();sums=[np.zeros(1536),np.zeros(192)];squares=[v.copy() for v in sums];n=0
    for row in tr:
        raw=raw_current(z,base,row);n+=len(raw[0])
        for j,x in enumerate(raw):sums[j]+=x.sum(0);squares[j]+=(x*x).sum(0)
        guard()
    assert n==250750;stats=[d4.stats_finish(s,q,n) for s,q in zip(sums,squares)]
    qs=np.zeros(1536);qt=qs.copy()
    for row in tr:
        zz,_=raw_current(z,base,row);q=d4.normalize(zz,stats[0])**2;qs+=q.sum(0);qt+=(q*q).sum(0);guard()
    stats.append(d4.stats_finish(qs,qt,n))
    np.savez(folder/'TRAIN_TRANSFORMS.npz',**{f'{k}_{j}':v for j,s in enumerate(stats) for k,v in s.items()})
    account('FT_TRAIN_normalization_and_square_statistics',seed,t,TRAIN_fit_windows=n)
    t=clock();status(stage='FT_TRAIN_Gram',seed=seed);G=np.zeros((3264,3264));H=np.zeros((3264,192));xs=np.zeros(3264);rs=np.zeros(192)
    for row in tr:
        o,w=row['offset'],row['windows'];x=row_features(z,base,row,stats);res=np.asarray(y[o+4:o+w,0],np.float64)-base[o+4:o+w]
        assert np.isfinite(x).all() and np.isfinite(res).all();G+=x.T@x;H+=x.T@res;xs+=x.sum(0);rs+=res.sum(0);guard()
    xm=xs/n;rm=rs/n;G=G/n-np.outer(xm,xm);H=H/n-np.outer(xm,rm)
    np.savez(folder/'TRAIN_MOMENTS.npz',G=G,H=H,x_mean=xm,residual_mean=rm,n=n)
    account('FT_TRAIN_joint_Gram',seed,t,TRAIN_fit_windows=n)
    t=clock();vi=np.array([i for row in vr for i in range(row['offset']+4,row['offset']+row['windows'])]);xv=np.concatenate([row_features(zv,val,row,stats) for row in vr],0)
    assert xv.shape==(30304,3264) and len(vi)==30304
    fallback=np.setdiff1d(np.arange(30500),vi);account('FT_VAL_fixed_features',seed,t,windows=30500);results=[]
    for arm in ARMS:
        ids=np.arange(1536,1728) if arm=='Ponly' else np.arange(1728 if arm=='C' else 3264)
        ga=G[np.ix_(ids,ids)];ha=H[ids];ma=xm[ids];t=clock();eig,Q=la.eigh((ga+ga.T)*.5,driver='evr');assert eig.min()>-1e-8;qh=Q.T@ha;xa=xv[:,ids]
        account('ridge_shared_eigen',seed,t,arm=arm,dimension=len(ids))
        for lam in LAMBDAS:
            status(stage='FT_ridge',seed=seed,arm=arm,lam=lam,solutions_seed=len(results));t=clock()
            coef=Q@(qh/(eig[:,None]+lam));bias=rm-ma@coef
            normal=float(abs((ga+lam*np.eye(len(ids)))@coef-ha).max());assert normal<=1e-7*max(1.,float(abs(ha).max()))
            assert np.isfinite(coef).all() and np.isfinite(bias).all();pred=val.copy();pred[vi]+=xa@coef+bias
            assert np.array_equal(pred[fallback],val[fallback]);tag=f'{arm}_lambda{lam:g}'
            np.savez(folder/f'{tag}_COEFFICIENTS.npz',coef=coef,bias=bias)
            account('ridge_model_lambda_solve_and_prediction',seed,t,arm=arm,lam=lam);t=clock();m=score(pred,vr,p);v=m['phases']['val']
            result=dict(seed=seed,arm=arm,lam=lam,dimension=len(ids),parameters=int(coef.size+bias.size),effective_df=float(np.sum(eig/(eig+lam))),normal_equation_max_abs=normal,r60s=v['r'],r3s=v['r3s'],undefined_3s=v['window_r_undefined'],window_MSE=v['window_MSE'],fused_MSE=v['MSE'],weighted_single_MSE=v['weighted_single_MSE'],disagreement=v['disagreement'],overlap_MSE=v['overlap_MSE'],Hann_identity_max_abs=m['Hann_identity_max_abs'])
            dump(folder/f'{tag}_METRICS.json',dict(**result,metrics=m));results.append(result);account('VAL_full_primary_secondary',seed,t,arm=arm,lam=lam,windows=30500,blocks=752)
        del Q,xa,qh;guard(True)
    dump(folder/'COMPLETED.json',dict(seed=seed,ridge_fits=9,results=results,constant_coordinates=[int(s['constant'].sum()) for s in stats],history_inputs=0,NN_updates=0))
    return results,basemetrics

def run(device):
    p=read(ROOT/'PROTOCOL.json');approval=read(ROOT/'COORDINATOR_PLAN_REVIEW.json')
    assert p['frozen'] and p['new_ridge_fits']==27 and approval['status']=='PLAN_APPROVED'
    assert approval['plan_sha256']==d4.sha(ROOT/'PLAN.md') and approval['inventory_sha256']==d4.sha(ROOT/'INPUT_INVENTORY.json')
    for path,h in p['source_pins'].items():assert d4.sha(path)==h,path
    inputs_unchanged(p);guard(True);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120min cap')));signal.alarm(7200)
    dump(ROOT/'STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha=d4.sha(Path(__file__)),protocol_sha=d4.sha(ROOT/'PROTOCOL.json')))
    results=[];baseline={}
    for seed in SEEDS:
        folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);cache_seed(seed,folder,p,device);rr,bb=fit_seed(seed,folder,p);results+=rr;baseline[str(seed)]=bb
    assert len(results)==27;bm=float(np.mean([v['phases']['val']['r'] for v in baseline.values()]));selection=[]
    for arm in ARMS:
        candidates=[dict(lam=lam,mean=float(np.mean([r['r60s'] for r in results if r['arm']==arm and r['lam']==lam]))) for lam in LAMBDAS]
        best=max(candidates,key=lambda x:(x['mean'],x['lam']));use=best['mean']>bm
        selection.append(dict(arm=arm,selected_lambda=best['lam'] if use else None,P_FT_baseline_selected=not use,mean_VAL60s=best['mean'] if use else bm,all_candidates=candidates))
    inputs_unchanged(p);guard(True)
    with (ROOT/'RESULTS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    dump(ROOT/'SUMMARY.json',dict(baseline=baseline,all_solutions=results,selection=selection,scientific_status='REPEATED_SOURCE_TRAIN_VAL_DEVELOPMENT_ONLY',goal_complete=False))
    dump(ROOT/'COST.json',dict(stages=COST,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,primary_encoder_head_windows=848862,extra_bridge_encoder_head_windows=300,new_ridge_fits=27,NN_updates=0,raw_VAL_scalar_bridges=3,raw_VAL_batch_wave_bridges=6,inherited_training_costs_not_zero=True,test_DTU=False))
    dump(ROOT/'COMPLETED.json',dict(new_ridge_fits=27,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),new_root_bytes=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file()),all_SS07_npy_bytes=npy_bytes(),free_disk_bytes=shutil.disk_usage(ROOT).free,inputs_unchanged=True,test_DTU=False,goal_complete=False))
    status(stage='COMPLETED',ridge_fits=27,goal_complete=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    assert os.uname().sysname=='Linux' and str(ROOT).startswith('/home/fanyu/')
    try:
        if args.action=='prepare':prepare()
        else:run(args.device)
    except Exception:
        if args.action=='run':dump(ROOT/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
