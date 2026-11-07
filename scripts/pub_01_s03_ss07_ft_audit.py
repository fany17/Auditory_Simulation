"""D10 immutable strong FT readouts on exposed internal tests; no fitting."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');P=BASE/'s03_incremental_continuity_20261101';D8=BASE/'s03_finetuned_readout_transfer_20261101';D9=BASE/'s03_physical_objective_20261101';ROOT=BASE/'s03_finetuned_internal_audit_20261101'
sys.path.insert(0,str(D8/'input'))
import pub_01_s03_ss07_ft_readout as d8
import pub_01_s03_ss07_eval_common as original
import pub_01_s03_ss07_eval_io as original_io
d4=d8.d4;core=d8.metric;ft=d8.ft
SEEDS=(1024,2026,31415);ROLES=('P_FT','Ponly','C','Ccap','W','F');PHASES=('test_seen','test_heldout')
START=time.time();CPU_START=time.process_time();COST=[]
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def status(**v):
    tmp=ROOT/'STATUS.writing';tmp.write_text(json.dumps(dict(time=time.time(),elapsed=time.time()-START,**v),allow_nan=False));os.replace(tmp,ROOT/'STATUS.json');print(json.dumps(v),flush=True)
def account(stage,seed,t,**v):COST.append(dict(stage=stage,seed=seed,wall_seconds=time.time()-t[0],CPU_seconds=time.process_time()-t[1],**v))
def clock():return time.time(),time.process_time()
def all_npy():return sum(p.stat().st_size for root in BASE.glob('s03_*_20261101') for p in root.rglob('*.npy'))
def guard(arrays=False):
    assert time.time()-START<3600
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*1024**3
    assert shutil.disk_usage(ROOT).free>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<=2*1024**3
    assert all(0<int(os.environ[k])<=8 for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'))
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<=2*1024**3
    if arrays:assert all_npy()<=25*1024**3
def assert_inputs(p):
    for name,h in p['source_pins'].items():assert d4.sha(name)==h,name
    for name,v in p['inputs'].items():
        st=Path(name).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],name
def coef_paths(seed):return {role:(D9/f'seed{seed}/{role}_COEFFICIENTS.npz' if role in ('W','F') else D8/f'seed{seed}/{role}_lambda{(.0001 if role=="C" else .01):g}_COEFFICIENTS.npz') for role in ROLES[1:]}
def frozen(seed):
    s=np.load(D8/f'seed{seed}/TRAIN_TRANSFORMS.npz',allow_pickle=False);three=[{k:s[f'{k}_{i}'] for k in ('mean','std','constant')} for i in range(3)]
    stats=[three[0],three[1],None,None,None,three[2]];cal=read(D8/f'seed{seed}/P_FT_BASELINE.json')['calibration'];coefs={}
    assert cal['a']>0
    for role,path in coef_paths(seed).items():
        f=np.load(path,allow_pickle=False);b=f['coef'];bias=f['bias'];assert bias.shape==(192,)
        if role=='Ponly':assert b.shape==(192,192);co=dict(BZ=np.zeros((1536,192)),BP=b,bias=bias,BQ=None)
        else:
            assert b.shape==((1728 if role=='C' else 3264),192);co=dict(BZ=b[:1536],BP=b[1536:1728],bias=bias,BQ=b[1728:] if role!='C' else None)
        assert all(np.isfinite(v).all() for v in co.values() if v is not None);coefs[role]=co
    return cal,stats,coefs

class Capture(nn.Module):
    def __init__(self,decoder,zout,pout,verify_decoder=False):
        super().__init__();self.decoder=decoder;self.zout=zout;self.pout=pout;self.cursor=0;self.verify=verify_decoder;self.checks=[];self.signature=None;self.batches=0;self.extra=0
    def forward(self,x):
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled() and x.dtype==torch.float32
        first=self.cursor;last=first+len(x);assert last<=len(self.zout)
        z=self.decoder.encoder(x).transpose(1,2);p=self.decoder.head(z)
        assert z.shape==(len(x),8,192) and p.shape==(len(x),1,192) and torch.isfinite(z).all() and torch.isfinite(p).all()
        if self.signature is None:self.signature=dict(input_stride=list(x.stride()),latent_view_stride=list(z.stride()),pred_stride=list(p.stride()),params_require_grad=all(v.requires_grad for v in self.parameters()),eval=not self.training,grad_enabled=torch.is_grad_enabled(),inference_mode=torch.is_inference_mode_enabled(),dtype=str(x.dtype))
        if self.verify and (first==0 or last==len(self.zout)):
            other=self.decoder(x);diff=float((p-other).abs().max());assert diff<=1e-6
            self.checks.append(dict(first=first,last=last,raw_units_max_abs=diff));self.extra+=len(x)
        self.zout[first:last]=z.cpu().numpy();self.pout[first:last]=p[:,0].cpu().numpy();self.cursor=last;self.batches+=1
        if self.batches%100==0:guard()
        return p

def scatter_check():
    class ToyEncoder(nn.Module):
        def forward(self,x):return x[:,:,:8]
    class ToyHead(nn.Module):
        def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(1.))
        def forward(self,z):return z.sum(1,keepdim=True)*self.scale
    class ToyDecoder(nn.Module):
        def __init__(self):super().__init__();self.encoder=ToyEncoder();self.head=ToyHead()
        def forward(self,x):return self.head(self.encoder(x).transpose(1,2))
    zs=np.empty((87,8,192),np.float32);ps=np.empty((87,192),np.float32);model=Capture(ToyDecoder(),zs,ps,True)
    x=np.arange(1,88,dtype=np.float32)[:,None,None]*np.ones((87,192,64),np.float32)
    rows=[('a',x[:5]),('b',x[5:73]),('c',x[73:])];out=original.predict_raw_phase(model,iter(rows),'cpu')
    assert model.cursor==87 and model.batches==2 and model.extra==87
    assert np.array_equal(zs[:,0,0],np.arange(1,88,dtype=np.float32)) and np.array_equal(np.concatenate(list(out.values())),ps)
    assert [v['last']-v['first'] for v in model.checks]==[64,23]
    return dict(passed=True,scope='synthetic global64 mixed-record capture/scatter only',main_windows=87,batches=2,tail=23,raw_Decoder_max_abs=max(c['raw_units_max_abs'] for c in model.checks),no_real_data=True)
def prepare():
    assert not (ROOT/'PROTOCOL.json').exists();approval=read(ROOT/'COORDINATOR_PLAN_REVIEW.json');assert approval['status']=='PLAN_APPROVED' and approval['approved_for_execution']
    assert approval['plan_sha256']==d4.sha(ROOT/'PLAN.md') and approval['inventory_sha256']==d4.sha(ROOT/'INPUT_INVENTORY.json')
    inv=read(ROOT/'INPUT_INVENTORY.json');mask=read(P/'SPARR_TEST_MASK.json');manifest=read(P/'FINAL_CHECKPOINT_MANIFEST.json')
    pins=[Path(__file__),ROOT/'PLAN.md',ROOT/'INPUT_INVENTORY.json',ROOT/'COORDINATOR_PLAN_REVIEW.json',Path(d8.__file__),Path(core.__file__),Path(original.__file__),Path(original_io.__file__),Path(ft.__file__)]+[Path(path) for path in inv['source_pins']]
    for old in (D8,D9):pins.extend(Path(path) for path in read(old/'PROTOCOL.json')['source_pins'])
    for path,h in inv['source_pins'].items():assert d4.sha(path)==h,path
    guard(True);torch.set_num_threads(8);dump(ROOT/'MATH_CHECK.json',scatter_check())
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,seeds=SEEDS,roles=ROLES,role_split_scores=36,records=mask['records'],masks={r['record']:r['blocks'] for r in mask['record_masks']},counts=inv['counts'],budget=inv['budget'],source_pins={str(path):d4.sha(path) for path in dict.fromkeys(pins)},inputs=inv['inputs'],checkpoints={str(seed):manifest['entries'][f'P_FT:{seed}'] for seed in SEEDS},precision=read(P/'INCREMENTAL_PROTOCOL.json')['precision'],primary_pairs=[['F','W'],['Ccap','P_FT'],['Ccap','Ponly']],secondary_pairs=[['F','Ccap'],['W','Ccap'],['C','Ponly']],VAL_cache_prediction_tolerance=1e-8,VAL_block_r_tolerance=1e-10,VAL_capture_Z_P_tolerance=1e-6,raw_original_FT_fused_tolerance=1e-6,original_FT_block_r_tolerance=1e-10,fit_selection=False,old_test_label='POST-HOC INTERNAL EXPLORATORY',DTU=False,new_download=False,automatic_retry=False))
    print('D10_FROZEN_NOT_TESTED',flush=True)
def check_val():
    p=read(ROOT/'PROTOCOL.json');assert_inputs(p);assert not torch.cuda.is_initialized();rows=read(D8/'PROTOCOL.json')['records']['val'];mask=read(D8/'PROTOCOL.json')['mask'];checks=[]
    for seed in SEEDS:
        cal,stats,co=frozen(seed);z=np.load(D8/f'seed{seed}/val_latent.npy',mmap_mode='r');raw=np.load(D8/f'seed{seed}/val_P.npy',mmap_mode='r')
        for role in ROLES:
            ref=read(D8/f'seed{seed}/P_FT_BASELINE.json') if role=='P_FT' else (read(D9/f'seed{seed}/{role}_METRICS.json')['metrics'] if role in ('W','F') else read(D8/f'seed{seed}/{role}_lambda{(.0001 if role=="C" else .01):g}_METRICS.json')['metrics'])
            rb={(b['record'],b['start']):b['r'] for b in ref['segments']}
            for row in (rows[0],rows[-1]):
                o,w=row['offset'],row['windows'];pred=core.predict(np.asarray(z[o:o+w]),np.asarray(raw[o:o+w]),cal,stats,co.get(role));other=cal['a']*np.asarray(raw[o:o+w],np.float64)+cal['b']
                if role!='P_FT':
                    f=np.load(coef_paths(seed)[role]);xp=d4.normalize(other[4:],stats[1])/np.sqrt(192)
                    if role=='Ponly':x=xp
                    else:
                        zz=d4.normalize(np.asarray(z[o+4:o+w],np.float64).reshape(w-4,1536),stats[0]);parts=[zz/np.sqrt(1536),xp]
                        if role!='C':parts.append(d4.normalize(zz**2,stats[5])/np.sqrt(1536))
                        x=np.concatenate(parts,1)
                    other[4:]+=x@f['coef']+f['bias']
                wave=float(abs(pred-other).max());assert wave<=p['VAL_cache_prediction_tolerance']
                target=np.load(row['envelope_path'],mmap_mode='r');bb=[b for b in mask['blocks'] if b['record']==row['record']];m=core.measure(pred,target,row,bb);err=max(abs(b['r']-rb[(b['record'],b['start'])]) for b in m['segments']);assert err<=p['VAL_block_r_tolerance']
                checks.append(dict(seed=seed,role=role,record=row['record'],prediction_max_abs=wave,block_r_max_abs=err))
    assert len(checks)==36;dump(ROOT/'VAL_MAPPING_CHECK.json',dict(passed=True,cases=checks,new_encoder_head_windows=0,new_test_scoring=False));print('D10_ALL36_CACHED_VAL_BRIDGES_PASSED',flush=True)
def context():
    rows=read(d8.OLD/'envelope_readout_seed1024/manifest_frozen.json')['records'];s=np.load(ft.PREPARED/'train_only_normalization.npz',allow_pickle=False)
    return SimpleNamespace(all_rows={r['record']:r for r in rows},mean=s['mean'],std=s['std'])
def decoder(seed,p,device):
    original_model=ft.native().TransformerCLIP(ft.cfg());model=ft.Decoder(original_model.eeg);del original_model
    cp=torch.load(p['checkpoints'][str(seed)]['path'],map_location='cpu',weights_only=True);assert cp['seed']==seed and cp['epoch']==4;model.load_state_dict(cp['model'],strict=True)
    return model.to(device).eval()
def val_capture(seed,model,ctx,p,device,folder):
    t=clock();rr=read(D8/'PROTOCOL.json')['records']['val'];goldz=np.load(D8/f'seed{seed}/val_latent.npy',mmap_mode='r');goldp=np.load(D8/f'seed{seed}/val_P.npy',mmap_mode='r');checks=[]
    for row,start,count,local in ((rr[0],0,64,0),(rr[-1],30464,36,rr[-1]['windows']-36)):
        windows=original_io.SparrRecords.raw_windows(ctx,row)[local:local+count];zs=np.empty((count,8,192),np.float32);ps=np.empty((count,192),np.float32);capture=Capture(model,zs,ps)
        original.predict_raw_phase(capture,iter([(row['record'],windows)]),device)
        dz=float(abs(zs-np.asarray(goldz[start:start+count])).max());dp=float(abs(ps-np.asarray(goldp[start:start+count])).max());assert max(dz,dp)<=p['VAL_capture_Z_P_tolerance']
        checks.append(dict(global_first=start,windows=count,Z_max_abs=dz,P_raw_max_abs=dp,signature=capture.signature))
    dump(folder/'VAL_CAPTURE_CHECK.json',dict(passed=True,encoder_head_windows=100,cases=checks));account('VAL_capture_gold',seed,t,encoder_head_windows=100)
def seed_run(seed,p,device):
    folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);cal,stats,co=frozen(seed);ctx=context();model=decoder(seed,p,device);val_capture(seed,model,ctx,p,device,folder)
    old=read(P/f'final_results/seed{seed}/INTERNAL_P_FT.json');oldr={(b['record'],b['start']):b['r'] for b in old['segments']};bridges=[];baserecords=[];caches={}
    for phase in PHASES:
        t=clock();rr=[r for r in p['records'] if r['phase']==phase];N=p['counts'][phase]['windows'];zp=folder/f'{phase}_latent.npy';pp=folder/f'{phase}_P.npy';assert not zp.exists() and not pp.exists()
        zout=np.lib.format.open_memmap(zp,mode='w+',dtype=np.float32,shape=(N,8,192));pout=np.lib.format.open_memmap(pp,mode='w+',dtype=np.float32,shape=(N,192));guard(True);capture=Capture(model,zout,pout,True);status(stage='raw_FT_global64',seed=seed,phase=phase)
        outputs=original.predict_raw_phase(capture,((r['record'],original_io.SparrRecords.raw_windows(ctx,r)) for r in rr),device);assert capture.cursor==N and capture.batches==p['counts'][phase]['global64_batches']
        zout.flush();pout.flush();phasebridges=[]
        for row in rr:
            o,w=row['phase_offset'],(row['common_samples']-192)//96+1;pred=np.asarray(pout[o:o+w]);assert np.array_equal(pred,outputs[row['record']]);raw_fused,covered=original.fuse(pred,row['common_samples']);ref=np.load(P/f'final_results/seed{seed}/predictions/internal/P_FT/{row["record"]}.npy',mmap_mode='r');wave=float(abs(raw_fused-ref).max());assert wave<=p['raw_original_FT_fused_tolerance']
            target=np.load(row['envelope_path'],mmap_mode='r');actual=core.measure(core.predict(np.asarray(zout[o:o+w]),pred,cal,stats),target,row,p['masks'][row['record']]);err=max(abs(b['r']-oldr[(b['record'],b['start'])]) for b in actual['segments']);assert err<=p['original_FT_block_r_tolerance']
            baserecords.append(actual);phasebridges.append(dict(record=row['record'],raw_original_fused_max_abs=wave,calibrated_block_r_max_abs=err));guard()
        dump(folder/f'{phase}_CACHE_BRIDGE.json',dict(phase=phase,main_encoder_head_windows=N,extra_original_Decoder_windows=capture.extra,batches=capture.batches,signature=capture.signature,Decoder_checks=capture.checks,records=phasebridges));bridges+=phasebridges;caches[phase]=dict(Z=str(zp),P=str(pp));account('FT_phase_encoding_and_raw_bridge',seed,t,phase=phase,main_encoder_head_windows=N,extra_encoder_head_windows=capture.extra,records=len(rr));del zout,pout,outputs;guard(True)
    assert len(bridges)==144;dump(folder/'RAW_FT_ALL144_PASSED.json',dict(records=bridges,maximum_raw_original_units=max(b['raw_original_fused_max_abs'] for b in bridges),maximum_original_block_r=max(b['calibrated_block_r_max_abs'] for b in bridges)))
    baseline=core.aggregate(baserecords);dump(folder/'P_FT_METRICS.json',dict(seed=seed,role='P_FT',**baseline));models={'P_FT':baseline};del model;torch.cuda.empty_cache()
    for role in ROLES[1:]:
        t=clock();status(stage='fixed_readout_and_score',seed=seed,role=role);out=[]
        for phase in PHASES:
            z=np.load(caches[phase]['Z'],mmap_mode='r');raw=np.load(caches[phase]['P'],mmap_mode='r')
            for row in (r for r in p['records'] if r['phase']==phase):
                o,w=row['phase_offset'],(row['common_samples']-192)//96+1;pred=core.predict(np.asarray(z[o:o+w]),np.asarray(raw[o:o+w]),cal,stats,co[role]);target=np.load(row['envelope_path'],mmap_mode='r');out.append(core.measure(pred,target,row,p['masks'][row['record']]));guard()
        m=core.aggregate(out);dump(folder/f'{role}_METRICS.json',dict(seed=seed,role=role,**m));models[role]=m;account('fixed_readout_and_score',seed,t,role=role,records=144,windows=84692)
    for m in models.values():
        for phase,c in p['counts'].items():assert all(m['phases'][phase][k]==c[v] for k,v in [('records','records'),('people','people'),('windows','windows'),('blocks','eligible_blocks')])
    dump(folder/'COMPLETED.json',dict(seed=seed,roles=6,role_split_scores=12,new_fits=0,caches=caches));return models
def run(device):
    p=read(ROOT/'PROTOCOL.json');assert p['frozen'] and read(ROOT/'VAL_MAPPING_CHECK.json')['passed'];assert_inputs(p);guard(True)
    torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device);torch.backends.cuda.matmul.allow_tf32=p['precision']['TF32_matmul'];torch.backends.cudnn.allow_tf32=p['precision']['TF32_cudnn'];torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60min cap')));signal.alarm(3600)
    dump(ROOT/'STARTED.json',dict(pid=os.getpid(),time=time.time(),source_sha=d4.sha(Path(__file__)),protocol_sha=d4.sha(ROOT/'PROTOCOL.json')));allmodels={}
    for seed in SEEDS:allmodels[str(seed)]=seed_run(seed,p,device)
    assert_inputs(p);guard(True);dump(ROOT/'SUMMARY.json',dict(results={seed:{role:dict(phases=m['phases'],weighted=m['weighted']) for role,m in models.items()} for seed,models in allmodels.items()},scientific_status='POST-HOC INTERNAL EXPLORATORY',fit_selection=False,goal_complete=False))
    dump(ROOT/'COST.json',dict(stages=COST,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,main_encoder_head_windows=254076,extra_VAL_capture_windows=300,extra_original_Decoder_windows=636,role_split_scores=36,new_fits=0,DTU=False,inherited_costs_nonzero=True))
    dump(ROOT/'COMPLETED.json',dict(role_split_scores=36,wall_seconds=time.time()-START,CPU_seconds=time.process_time()-CPU_START,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),new_root_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),all_SS07_NPY_bytes=all_npy(),free_disk_bytes=shutil.disk_usage(ROOT).free,old_test_label='POST-HOC INTERNAL EXPLORATORY',new_fit_selection=False,DTU=False,goal_complete=False));status(stage='COMPLETED',role_split_scores=36,goal_complete=False)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','check','run']);parser.add_argument('--device',default='cuda:0');args=parser.parse_args();assert os.uname().sysname=='Linux'
    try:
        if args.action=='prepare':prepare()
        elif args.action=='check':check_val()
        else:run(args.device)
    except Exception:
        if args.action=='run':dump(ROOT/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
