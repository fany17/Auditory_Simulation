"""D7 fixed internal audit: frozen cached latents, head-only P bridge, no fit."""
import argparse
import csv
import hashlib
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
import pub_01_s03_ss07_conditional_probe as d4
import pub_01_s03_ss07_generalization_core as core
import pub_01_s03_ss07_eval_common as original_eval
import pub_01_s03_ss07_train as original_train

BASE=d4.BASE;ORIGINAL=d4.P;D4=d4.ROOT;D6=BASE/'s03_readout_attribution_20261101'
ROOT=BASE/'s03_current_readout_internal_audit_20261101';SEEDS=(1024,2026,31415)
START=time.time();CPU_START=time.process_time();COST=[]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def clock():return time.time(),time.process_time()
def account(stage,seed,t,**v):COST.append(dict(stage=stage,seed=seed,wall_seconds=time.time()-t[0],CPU_seconds=time.process_time()-t[1],**v))
def status(**v):
    p=ROOT/'STATUS.writing';p.write_text(json.dumps(dict(time=time.time(),elapsed=time.time()-START,**v)));os.replace(p,ROOT/'STATUS.json');print(json.dumps(v),flush=True)
def array_roots():return sorted(BASE.glob('s03_*_20261101'))
def array_bytes():return sum(p.stat().st_size for r in array_roots() for p in r.rglob('*.npy'))
def guard():
    assert time.time()-START<1800 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert all(0<int(os.environ[k])<=8 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=512*1024**2
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<=1024**3
def fingerprint(p):
    a=np.load(p,mmap_mode='r',allow_pickle=False);s=Path(p).stat()
    return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=s.st_size,mtime_ns=s.st_mtime_ns)

def role_paths(seed):
    return {role:(D6/f'seed{seed}/{role}_lambda{core.LAMBDAS[role]:g}_COEFFICIENTS.npz' if role in ('Ponly','S32','U32') else D4/f'seed{seed}/{role}_lambda{core.LAMBDAS[role]:g}_COEFFICIENTS.npz') for role in core.ROLES if role!='P'}
def load_frozen(seed):
    f=np.load(D4/f'seed{seed}/TRAIN_TRANSFORMS.npz',allow_pickle=False)
    stats=[{k:f[f'{k}_{j}'] for k in ('mean','std','constant')} for j in range(6)]
    cal=read(D4/f'seed{seed}/P_CHECK.json');assert cal['a']>0
    coefs={}
    for role,p in role_paths(seed).items():
        c=np.load(p,allow_pickle=False)
        if role in ('C','Ccap'):
            v=c['coef'];co=dict(BZ=v[:1536],BP=v[1536:1728],bias=c['bias'],BQ=v[1728:] if role=='Ccap' else None)
        else:co=dict(BZ=c['BZ'],BP=c['BP'],bias=c['bias'],BQ=None)
        assert co['BZ'].shape==(1536,192) and co['BP'].shape==(192,192) and co['bias'].shape==(192,)
        assert all(np.isfinite(v).all() for v in co.values() if v is not None)
        if role=='Ponly':assert not co['BZ'].any()
        if role in ('S32','U32'):
            for t in range(192):
                ids=np.array([c*192+s for c in range(8) for s in range(max(0,t-15),min(192,t+17))])
                assert not co['BZ'][np.setdiff1d(np.arange(1536),ids),t].any()
            if role=='S32':
                for c in range(8):
                    for k in range(-15,17):
                        ts=np.arange(max(0,-k),min(192,192-k));v=co['BZ'][c*192+ts+k,ts];assert np.ptp(v)==0
        coefs[role]=co
    return cal,stats,coefs

def prepare():
    assert not (ROOT/'PROTOCOL.json').exists();assert not torch.cuda.is_initialized();guard()
    mask=read(ORIGINAL/'SPARR_TEST_MASK.json');rows=mask['records'];assert len(rows)==144
    metadata=read(ORIGINAL/'FINAL_CHECKPOINT_MANIFEST.json');inventory=read(ROOT/'INPUT_INVENTORY.json');assert not inventory['errors']
    expected={'test_seen':dict(records=69,people=69,windows=36573,blocks=872),'test_heldout':dict(records=75,people=13,windows=48119,blocks=1172)}
    maskmap={r['record']:r['blocks'] for r in mask['record_masks']}
    for phase,e in expected.items():
        rr=[r for r in rows if r['phase']==phase];assert len(rr)==e['records'] and len({r['subject'] for r in rr})==e['people']
        assert sum((r['common_samples']-192)//96+1 for r in rr)==e['windows']
        assert sum(b['eligible'] for r in rr for b in maskmap[r['record']])==e['blocks']
    assert Path(original_eval.__file__).resolve()==(ORIGINAL/'input/pub_01_s03_ss07_eval_common.py').resolve()
    assert Path(original_train.__file__).resolve()==(ORIGINAL/'input/pub_01_s03_ss07_train.py').resolve()
    imported={name:str(Path(module.__file__).resolve()) for name,module in list(sys.modules.items()) if getattr(module,'__file__',None) and str(module.__file__).endswith('.py') and Path(module.__file__).resolve().is_relative_to(BASE)}
    completed=read(ORIGINAL/'INDEPENDENT_REVIEW_COMPLETED_V2.json');assert completed['technical_review_passed'] and all(c['exit']==0 for c in completed['children'])
    assert read(ORIGINAL/'independent_raw_review/RAW_INPUT_REVIEW.json')['passed']
    pins=[Path(__file__),Path(core.__file__),Path(d4.__file__),Path(original_eval.__file__),Path(original_train.__file__),ROOT/'PLAN.md',ROOT/'INPUT_INVENTORY.json',ROOT/'COORDINATOR_PLAN_REVIEW.json',ROOT/'COORDINATOR_REVIEW_SCOPE.json',
        ORIGINAL/'SPARR_TEST_MASK.json',ORIGINAL/'FINAL_CHECKPOINT_MANIFEST.json',ORIGINAL/'EVALUATION_READY.json',ORIGINAL/'EVALUATION_EXECUTOR_PROTOCOL.json',ORIGINAL/'INCREMENTAL_PROTOCOL.json',
        ORIGINAL/'INDEPENDENT_REVIEW_COMPLETED_V2.json',ORIGINAL/'INDEPENDENT_REVIEW_PROTOCOL.json',ORIGINAL/'INDEPENDENT_REVIEW_PROTOCOL_V2.json',ORIGINAL/'independent_raw_review/RAW_INPUT_REVIEW.json',ORIGINAL/'independent_review/METRICS_REVIEW.json',
        D4/'PROTOCOL.json',D4/'COORDINATOR_RESULT_REVIEW.json',D6/'PROTOCOL.json',D6/'SUMMARY.json',D6/'COORDINATOR_RESULT_REVIEW.json']
    pins.extend(Path(path) for path in imported.values())
    for name in ['pub_01_s03_ss07_history.py','pub_01_s03_ss07_incremental.py','pub_01_s03_ss07_baseline.py','pub_01_s03_ss07_finetune.py']:
        pins.append(ORIGINAL/'input'/name)
    inputs={};cache_paths={};head_paths={}
    for seed in SEEDS:
        cachefile=ORIGINAL/f'final_latent/seed{seed}/cache_manifest.json';cache=read(cachefile);assert cache['complete'] and not cache['read_failures'] and len(cache['records'])==144
        assert cache['checkpoint_manifest_sha256']==sha(ORIGINAL/'FINAL_CHECKPOINT_MANIFEST.json')
        assert cache['encoder_sha256']==metadata['encoders'][str(seed)]['sha256'];cache_paths[str(seed)]=cache['records'];head=Path(metadata['entries'][f'native_P:{seed}']['path']);head_paths[str(seed)]=str(head)
        assert sha(head)==metadata['entries'][f'native_P:{seed}']['sha256']
        saved=torch.load(head,map_location='cpu',weights_only=True)
        assert saved['seed']==seed and saved['epoch']==metadata['entries'][f'native_P:{seed}']['selected_epoch']
        pins.extend([cachefile,head,D4/f'seed{seed}/TRAIN_TRANSFORMS.npz',D4/f'seed{seed}/P_CHECK.json',D6/f'seed{seed}/P_BASELINE.json']);pins.extend(role_paths(seed).values())
        load_frozen(seed)
        for role in core.ROLES:
            if role!='P':assert next(r for r in read(D6/'SUMMARY.json')['selection'] if r['arm']==role)['selected_lambda']==core.LAMBDAS[role]
        for row in rows:
            for p in [Path(cache['records'][row['record']]),ORIGINAL/f'final_results/seed{seed}/predictions/internal/native_P/{row["record"]}.npy']:
                inputs[str(p)]=fingerprint(p)
        for p in [D4/f'seed{seed}/val_P.npy',ORIGINAL/f'baseline/head_seed{seed}/val_latent.npy']:
            inputs[str(p)]=fingerprint(p)
        for role in core.ROLES:
            if role!='P':pins.append(D6/f'seed{seed}/{role}_lambda{core.LAMBDAS[role]:g}_METRICS.json')
    valproto=read(D4/'PROTOCOL.json');valrows=valproto['records']['val']
    for row in rows+valrows:
        path=Path(row['envelope_path'])
        if str(path) not in inputs:inputs[str(path)]=fingerprint(path)
        assert int(np.prod(inputs[str(path)]['shape']))>=row['common_samples']
    counts=[dict(root=str(r),npy_count=len(list(r.rglob('*.npy'))),bytes=sum(p.stat().st_size for p in r.rglob('*.npy'))) for r in array_roots()]
    existing=sum(v['bytes'] for v in counts);planned=3*84692*192*4+432*128;assert existing+planned<=25*1024**3
    dump(ROOT/'INPUT_INVENTORY_SUPPLEMENT.json',dict(roots=counts,existing_npy_bytes=existing,proposed_P_window_bytes_including_headers=planned,combined_proposed_npy_bytes=existing+planned,
        correction='Original inventory counted four explicit roots. This supplement enumerates ALL seven dated SS07 roots including warm/diagnostics; no value reads or inference here.',parameter_sources='3 original head checkpoints and15 ridge coefficient files; not18 ridge models'))
    dump(ROOT/'MATH_CHECK.json',core.math_check())
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,scientific_status='POST-HOC INTERNAL EXPLORATORY',seeds=SEEDS,roles=core.ROLES,lambda_by_role=core.LAMBDAS,
        role_split_scores=36,fit_or_selection=False,source_pins={str(p):sha(p) for p in dict.fromkeys(pins)},inputs=inputs,records=rows,masks=maskmap,expected=expected,
        val_records=valrows,val_mask=valproto['mask'],cache_paths=cache_paths,head_paths=head_paths,rawP_bridge_absolute=1e-6,calibrated_P_block_r_bridge_absolute=1e-10,
        raw_encoder_forward=False,raw_EEG_read=False,head_window_recipe='Original NativeHead/predict_windows/global phase-aligned64 record splits/no_grad/original precision/grad flags',
        selected_sources='3 original heads+15 selected ridge coefficient files',array_roots=[str(r) for r in array_roots()],maximum_array_bytes=25*1024**3,
        actual_imported_project_modules={name:dict(path=path,sha256=sha(path)) for name,path in imported.items()},
        runtime=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,CUDA=torch.version.cuda),
        CPU_threads=8,RAM_GiB=8,GPU_GiB=1,wall_minutes=30,new_disk_MiB=512,DTU=False,new_download=False,automatic_retry=False,
        primary_comparisons=[['Ccap','P'],['Ccap','Ponly']],secondary_comparisons=[['Ccap','C'],['S32','Ponly'],['U32','S32'],['C','U32']]))
    print('D7_PREPARED_NOT_TESTED',flush=True)

def check_val():
    assert not torch.cuda.is_initialized();p=read(ROOT/'PROTOCOL.json')
    for f,h in p['source_pins'].items():assert sha(f)==h,f
    rows=p['val_records'];maskmap={r['record']:[b for b in p['val_mask']['blocks'] if b['record']==r['record']] for r in rows};cells=[]
    for seed in SEEDS:
        cal,stats,coefs=load_frozen(seed);z=np.load(ORIGINAL/f'baseline/head_seed{seed}/val_latent.npy',mmap_mode='r');rawp=np.load(D4/f'seed{seed}/val_P.npy',mmap_mode='r')
        records={role:[] for role in core.ROLES}
        for row in rows:
            o,w=row['offset'],row['windows'];zz=np.array(z[o:o+w],copy=True);pp=np.array(rawp[o:o+w],copy=True);target=np.load(row['envelope_path'],mmap_mode='r')
            for role in core.ROLES:
                pred=core.predict(zz,pp,cal,stats,coefs.get(role));records[role].append(core.measure(pred,target,row,maskmap[row['record']]))
            guard()
        for role in core.ROLES:
            actual=core.aggregate(records[role]);ref=read(D6/f'seed{seed}/P_BASELINE.json') if role=='P' else read(D6/f'seed{seed}/{role}_lambda{core.LAMBDAS[role]:g}_METRICS.json')
            rm={(r['record'],r['start']):r for r in ref['blocks']};assert len(rm)==len(actual['segments'])==752
            error=max(abs(actual['phases']['val']['r']-ref['r60s']),abs(actual['phases']['val']['r3s']-ref['r3s']),max(abs(b['r']-rm[(b['record'],b['start'])]['r']) for b in actual['segments']))
            auxref=ref['secondary']['macro_MSE'];secondary=max(abs(actual['phases']['val'][a]-auxref[b]) for a,b in [('MSE','fused_MSE'),('weighted_single_MSE','weighted_single_MSE'),('disagreement','disagreement')])
            assert error<=1e-10 and secondary<=1e-10
            cells.append(dict(seed=seed,role=role,people=49,blocks=752,windows=30500,primary_max_abs=error,secondary_max_abs=secondary))
        print('VAL_BRIDGE',seed,flush=True)
    dump(ROOT/'ENGINEERING_CHECK.json',dict(passed=True,cells=cells,cell_count=18,source_sha=sha(Path(__file__)),core_sha=sha(Path(core.__file__)),protocol_sha=sha(ROOT/'PROTOCOL.json'),
        wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,no_head_or_encoder_forward=True,no_test_values_or_predictions=True))
    print('D7_ALL18_FIXED_VAL_BRIDGES_PASSED',flush=True)

def assert_inputs(p):
    for path,v in p['inputs'].items():
        s=Path(path).stat();assert s.st_size==v['bytes'] and s.st_mtime_ns==v['mtime_ns'],path
def gpu_observation():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    values=[int(line.split(',')[1]) for line in output.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
    return max(values) if values else None

def seed_run(seed,p,device):
    folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);(folder/'P_windows').mkdir();cal,stats,coefs=load_frozen(seed);started=clock();status(seed=seed,stage='raw_P_head_bridge_all144')
    precision=read(ORIGINAL/'INCREMENTAL_PROTOCOL.json')['precision'];torch.backends.cuda.matmul.allow_tf32=precision['TF32_matmul'];torch.backends.cudnn.allow_tf32=precision['TF32_cudnn']
    torch.set_num_threads(2);head=original_train.NativeHead().to(device).eval();assert all(q.requires_grad for q in head.parameters())
    saved=torch.load(p['head_paths'][str(seed)],map_location='cpu',weights_only=True);assert saved['seed']==seed;head.head.load_state_dict(saved['head'],strict=True)
    t=clock();bridges=[];baseline_records=[]
    for row in p['records']:
        record=row['record'];z=np.load(p['cache_paths'][str(seed)][record],mmap_mode='r');x=torch.from_numpy(np.array(z,copy=True)).to(device)
        pred=original_eval.predict_windows(head,x,record,global_offset=row['phase_offset']);assert pred.dtype==np.float32
        raw_fused,covered=original_eval.fuse(pred,row['common_samples']);old=np.load(ORIGINAL/f'final_results/seed{seed}/predictions/internal/native_P/{record}.npy',mmap_mode='r')
        wave=float(np.max(abs(raw_fused-old)));assert wave<=p['rawP_bridge_absolute'],'original unscaled P bridge failure'
        target=np.load(row['envelope_path'],mmap_mode='r').reshape(-1);predcal=core.predict(np.asarray(z),pred,cal,stats)
        metrics=core.measure(predcal,target,row,p['masks'][record]);rmax=0.
        for b in metrics['segments']:
            s=b['start'];keep=covered[s:s+3840];old_r=core.corr(np.asarray(old[s:s+3840],np.float64)[keep],np.asarray(target[s:s+3840],np.float64)[keep]);rmax=max(rmax,abs(old_r-b['r']))
        assert rmax<=p['calibrated_P_block_r_bridge_absolute']
        with (folder/'P_windows'/f'{record}.npy').open('xb') as f:np.save(f,pred,allow_pickle=False)
        baseline_records.append(metrics);bridges.append(dict(record=record,phase=row['phase'],windows=len(pred),raw_original_units_max_abs=wave,P_affine_block_r_max_abs=rmax))
        del x;guard()
    assert len(bridges)==144
    account('head_only_full_original_P_bridge',seed,t,records=144,head_windows=84692,encoder_forwards=0)
    driver_memory=gpu_observation();assert driver_memory is None or driver_memory<=1024
    dump(folder/'P_BRIDGE_PASSED.json',dict(records=bridges,original_units_max_abs=max(r['raw_original_units_max_abs'] for r in bridges),P_affine_block_r_max_abs=max(r['P_affine_block_r_max_abs'] for r in bridges),GPU_driver_MiB=driver_memory))
    baseline=core.aggregate(baseline_records);dump(folder/'P_METRICS.json',dict(seed=seed,role='P',**baseline));allmetrics={'P':baseline};del head
    for role in core.ROLES[1:]:
        t=clock();status(seed=seed,stage='frozen_readout',role=role);records=[]
        for row in p['records']:
            record=row['record'];z=np.load(p['cache_paths'][str(seed)][record],mmap_mode='r');rawp=np.load(folder/'P_windows'/f'{record}.npy',mmap_mode='r');target=np.load(row['envelope_path'],mmap_mode='r')
            pred=core.predict(np.asarray(z),np.asarray(rawp),cal,stats,coefs[role]);records.append(core.measure(pred,target,row,p['masks'][record]));guard()
        metrics=core.aggregate(records);dump(folder/f'{role}_METRICS.json',dict(seed=seed,role=role,**metrics));allmetrics[role]=metrics
        account('fixed_readout_and_score',seed,t,role=role,records=144,windows=84692,head_encoder_forwards=0)
    for role,m in allmetrics.items():
        for phase,e in p['expected'].items():assert all(m['phases'][phase][k]==v for k,v in e.items())
    dump(folder/'COMPLETED.json',dict(seed=seed,roles=6,role_split_scores=12,wall_seconds=time.time()-started[0],CPU_seconds=time.process_time()-started[1],fit_or_selection=False))
    return allmetrics

def run(device):
    p=read(ROOT/'PROTOCOL.json');proof=read(ROOT/'ENGINEERING_CHECK.json');approval=read(ROOT/'COORDINATOR_CODE_REVIEW.json')
    assert p['frozen'] and proof['passed'] and proof['cell_count']==18
    assert approval['approved_for_execution'] and approval['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
    assert proof['protocol_sha']==sha(ROOT/'PROTOCOL.json') and proof['source_sha']==sha(Path(__file__)) and proof['core_sha']==sha(Path(core.__file__))
    for path,h in p['source_pins'].items():assert sha(path)==h,path
    assert_inputs(p);assert array_bytes()+3*84692*192*4+432*128<=p['maximum_array_bytes'];guard()
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30min cap')));signal.alarm(1800)
    dump(ROOT/'STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha=sha(Path(__file__)),core_sha=sha(Path(core.__file__)),protocol_sha=sha(ROOT/'PROTOCOL.json'),review_sha=sha(ROOT/'COORDINATOR_CODE_REVIEW.json')))
    results={};summary=[]
    for seed in SEEDS:
        results[seed]=seed_run(seed,p,device)
        for role,m in results[seed].items():
            for phase,x in m['phases'].items():summary.append(dict(seed=seed,role=role,phase=phase,**x,weighted_r=m['weighted']['r']))
    assert len(summary)==36;assert_inputs(p);assert array_bytes()<=p['maximum_array_bytes']
    with (ROOT/'RESULTS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    compact={str(seed):{role:dict(phases=m['phases'],weighted=m['weighted'],Hann_identity_max_abs=m['Hann_identity_max_abs']) for role,m in rr.items()} for seed,rr in results.items()}
    dump(ROOT/'SUMMARY.json',dict(seeds=SEEDS,roles=core.ROLES,results=compact,scientific_status='POST-HOC INTERNAL EXPLORATORY',fit_or_selection=False,goal_complete=False))
    dump(ROOT/'COST.json',dict(stages=COST,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,head_windows=254076,encoder_forwards=0,new_NN_updates=0,role_split_scores=36,record_role_scores=2592))
    dump(ROOT/'COMPLETED.json',dict(role_split_scores=36,seeds=3,records_per_seed=144,fit_or_selection=False,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),new_derived_bytes=sum(q.stat().st_size for q in ROOT.rglob('*') if q.is_file()),
        all_SS07_npy_bytes=array_bytes(),inputs_unchanged=True,DTU=False,new_download=False))
    status(stage='COMPLETED',role_split_scores=36,goal_complete=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','check','run']);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    assert os.uname().sysname=='Linux'
    try:
        if args.action=='prepare':prepare()
        elif args.action=='check':check_val()
        else:run(args.device)
    except Exception:
        if args.action=='run':dump(ROOT/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
