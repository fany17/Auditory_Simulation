"""One fixed CPU diagnostic of independent/formal paths and 40/64 batch geometry."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

START, CPU_START = time.monotonic(), time.process_time()
import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_contrastive_history_20261106')
OUT = ROOT / 'coordinator_cpu_bridge_v01'
REPLAY = ROOT / 'coordinator_final_replay_v02/COORDINATOR_FINAL_REPLAY.py'
REPLAY_SHA = '81734e974351a33a6c1d91c47379ce5efd5d1b75b847a0038a7cac47f9f53705'
replay = None
CAPTURE_NATIVE = False
hooks = dict(formal_prefix_windows=0, all_projection_windows=0, native_projection_windows=0,
             head_windows=0, library_memory_query_windows=0)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def counts():
    c = replay.COUNTS if replay else dict(backbone_windows=0, native_unused_projection_windows=0, manual_memory_query_windows=0)
    return dict(backbone_windows=c['backbone_windows']+hooks['formal_prefix_windows'],
                native_unused_projection_windows=hooks['native_projection_windows'],
                final_projection_windows=hooks['all_projection_windows']-hooks['native_projection_windows'],
                head_windows=hooks['head_windows'], manual_memory_query_windows=c['manual_memory_query_windows'],
                library_memory_query_windows=hooks['library_memory_query_windows'])

def increment(name, n):
    hooks[name] += n

def projection_counter(module, args):
    increment('all_projection_windows',len(args[0]))
    if CAPTURE_NATIVE:
        increment('native_projection_windows',len(args[0]))

def fuse(windows):
    wave, weight = np.zeros(3936), np.zeros(3936)
    h = .5-.5*np.cos(2*np.pi*np.arange(192)/191)
    for i,p in enumerate(windows):
        s=i*96
        wave[s:s+192] += p*h
        weight[s:s+192] += h
    keep=weight[:3840]>1e-8
    assert int(keep.sum())==3839
    return wave[:3840][keep]/weight[:3840][keep], keep

def run():
    global replay, CAPTURE_NATIVE
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    assert os.sched_getaffinity(0)==set(range(44,48)) and sha(REPLAY)==REPLAY_SHA
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1024)
    spec=importlib.util.spec_from_file_location('fixed_replay',REPLAY)
    replay=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    assert sha(ROOT/'PROTOCOL_V05.json')==replay.GATE_SHA
    gate=read(ROOT/'PROTOCOL_V05.json'); inventory=read(ROOT/'INPUTS.json')
    assert sha(ROOT/'INPUTS.json')==gate['inventory_sha256']
    assert all(sha(p)==h for p,h in gate['sources'].items())
    sys.path.insert(0,str(ROOT/'input_v05'))
    import pub_01_s03_ss07_contrastive_history_model as model
    import pub_01_s03_ss07_contrastive_history_data as data
    row=inventory['phases']['val'][0]
    assert row['record']=='sub-001_ses-shortstories01_task-listeningActive_run-02_eeg.bdf.gz' and row['offset']==0
    norm=data.PREPARED/'train_only_normalization.npz'
    for path in (row['eeg_path'],row['envelope_path'],str(norm)):
        st=Path(path).stat()
        assert inventory['inputs'][path]==dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    with np.load(norm,allow_pickle=False) as f:
        mean,scale=f['mean'],f['std']
    raw=np.load(row['eeg_path'],mmap_mode='r',allow_pickle=False)
    raw=raw.T if raw.shape[0]==64 else raw
    x=((np.asarray(raw[:6240],np.float32)-mean)/scale).astype(np.float32)
    y=np.asarray(np.load(row['envelope_path'],mmap_mode='r',allow_pickle=False).reshape(-1)[:3840],np.float64)
    assert x.shape==(6240,64) and y.shape==(3840,) and np.isfinite(x).all() and np.isfinite(y).all()
    folder=ROOT/'formal/seed1024/READOUT_HH'; checkpoint=ROOT/'formal/seed1024/FT_HH/FINAL.pt'
    receipt=read(folder/'COMPLETED.json')
    assert receipt['passed'] and receipt['seed']==1024 and receipt['arm']=='HH' and receipt['lambda_fixed']==.01
    assert receipt['epoch']==4 and receipt['old_test_reads']==0 and sha(checkpoint)==receipt['FT_checkpoint_sha256']
    ft=read(checkpoint.parent/'COMPLETED.json')
    previous=read(REPLAY.parent/'REPLAY_1024_HH.json')
    assert ft['final_epoch']==4 and ft['updates']==19725
    assert previous['execution_completed'] and previous['seed']==1024 and previous['arm']=='HH' and previous['record']==row['record']
    assert previous['FT_checkpoint_sha256']==sha(checkpoint) and previous['readout_sha256']==sha(folder/'READOUT.npz')
    assert previous['score_sha256']['near']==sha(folder/'SCORE_near.json') and previous['normalization_sha256']==sha(norm)
    with np.load(folder/'READOUT.npz',allow_pickle=False) as f:
        fit={k:f[k] for k in f.files}
    net=model.Decoder(model.Encoder(model.native.TransformerCLIP(read(data.NATIVE_CONFIG)).eeg,1024),model.make_head(1024),'H')
    net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    net.eval()
    handles=[net.encoder.backbone.output.register_forward_pre_hook(projection_counter),
             net.head.register_forward_pre_hook(lambda m,a:increment('head_windows',len(a[0]))),
             net.encoder.memory.register_forward_pre_hook(lambda m,a:increment('library_memory_query_windows',len(a[0])))]
    original_prefix=model.original_memory.prefix
    def counted_prefix(backbone,value):
        increment('formal_prefix_windows',len(value))
        return original_prefix(backbone,value)
    results={}
    try:
        with torch.inference_mode():
            for name,n in [('independent40',40),('formal40',40),('formal64',64)]:
                starts=np.arange(n)*96; ready=starts>=864
                current=torch.from_numpy(np.stack([x[s:s+192] for s in starts]))
                slots=tuple(torch.from_numpy(np.stack([x[s+d:s+d+192] for s in starts[ready]])) for d in (-480,-288))
                mask=torch.from_numpy(ready)
                if name=='independent40':
                    CAPTURE_NATIVE=True
                    try:
                        query=replay.prefix(net.encoder.backbone,current)
                        encoded=torch.cat([replay.prefix(net.encoder.backbone,t) for t in slots],1)
                    finally:
                        CAPTURE_NATIVE=False
                    delta,bridge=replay.memory_delta(net.encoder.memory,query[mask],encoded)
                    fused=query.clone(); fused[mask]+=delta
                    z=net.encoder.backbone.output(fused).transpose(1,2)
                else:
                    model.original_memory.prefix=counted_prefix
                    query=net.encoder.query(current)
                    z=net.encoder.project(query,current,mask,slots,'H').transpose(1,2)
                p=net.head(z)[:,0]
                results[name]=dict(query=query[:40].numpy().copy(),z=z[:40].numpy().copy(),p=p[:40].numpy().copy())
    finally:
        model.original_memory.prefix=original_prefix
        for handle in handles:handle.remove()
    expected=dict(backbone_windows=378,native_unused_projection_windows=102,final_projection_windows=144,
                  head_windows=144,manual_memory_query_windows=31,library_memory_query_windows=117)
    assert counts()==expected
    saved=read(folder/'SCORE_near.json')
    scores,formal_comparisons={},{}
    for name,value in results.items():
        affine,ccap=replay.correct(value['z'],value['p'],fit)
        scores[name]={kind:replay.metrics(windows,y) for kind,windows in dict(raw=value['p'].astype(np.float64),affine=affine,Ccap=ccap).items()}
        formal_comparisons[name]={}
        for kind,metrics in scores[name].items():
            block=[b for b in saved[kind]['segments'] if b['record']==row['record'] and b['start']==0]
            assert len(block)==1 and block[0]['samples']==3839
            formal_comparisons[name][kind]=replay.compare_metrics(metrics,block[0])
    pairs=[]
    for a,b in [('independent40','formal40'),('formal40','formal64'),('independent40','formal64')]:
        left,keep=fuse(results[a]['p'].astype(np.float64)); right,_=fuse(results[b]['p'].astype(np.float64))
        delta=left-right; mse_delta=float(np.mean(delta*delta))
        cross=float(2*np.mean(delta*(right-y[keep])))
        algebra=cross+mse_delta
        observed=scores[a]['raw']['MSE']-scores[b]['raw']['MSE']
        roundoff=1e-12*max(1.,scores[a]['raw']['MSE'],scores[b]['raw']['MSE'])
        bound=2*math.sqrt(scores[b]['raw']['MSE']*mse_delta)+mse_delta
        assert abs(algebra-observed)<=roundoff and abs(observed)<=bound+roundoff
        errors={k:dict(max_abs=float(np.max(np.abs(results[a][k]-results[b][k]))),
                       RMS=float(np.sqrt(np.mean((results[a][k].astype(np.float64)-results[b][k])**2)))) for k in ('query','z','p')}
        pairs.append(dict(left=a,right=b,tensor_errors=errors,raw_wave_RMS_delta=math.sqrt(mse_delta),
                          raw_MSE_signed_difference=observed,raw_MSE_algebra_difference=algebra,
                          raw_MSE_linear_cross_term=cross,raw_MSE_squared_delta_term=mse_delta,
                          raw_MSE_algebra_residual=algebra-observed,raw_MSE_Cauchy_bound=bound,
                          original_metric_checks={kind:replay.compare_metrics(scores[a][kind],scores[b][kind]) for kind in scores[a]}))
    with (OUT/'TRACE.npz').open('xb') as f:
        np.savez(f,**{f'{name}_{k}':v[k] for name,v in results.items() for k in ('z','p')})
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<4*1024**3 and not torch.cuda.is_initialized()
    save('RESULT.json',dict(execution_completed=True,algebra_checks_passed=True,source_sha256=sha(__file__),
         original_replay_source_sha256=REPLAY_SHA,protocol_sha256=replay.GATE_SHA,seed=1024,arm='HH',route='near',
         record=row['record'],EEG_samples_read=6240,target_samples_read=3840,scored_windows=40,scored_samples=3839,
         checkpoint_sha256=sha(checkpoint),readout_sha256=sha(folder/'READOUT.npz'),formal_score_sha256=sha(folder/'SCORE_near.json'),
         actual_calls=counts(),manual_library_MHA_max_abs=bridge,pairs=pairs,formal_CUDA_comparisons=formal_comparisons,
         trace_sha256=sha(OUT/'TRACE.npz'),peak_RSS_bytes=peak,CPU_seconds=time.process_time()-CPU_START,
         wall_seconds=time.monotonic()-START,new_fits=0,optimizer_updates=0,old_test_reads=0,GPU=0,
         interpretation='CPU path and batch diagnostic only. Original thresholds unchanged. No TF32 attribution or replacement of strict failures.'))
    print(json.dumps({'execution_completed':True,'actual_calls':counts(),'result_sha256':sha(OUT/'RESULT.json')}))

if __name__=='__main__':
    assert not any((OUT/name).exists() for name in ('RESULT.json','FAILURE.json','TRACE.npz'))
    proc=Path('/proc')/str(os.getpid())
    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
    save('STARTED.json',dict(process=[os.getpid(),proc.stat().st_uid,int(fields[19])],
         source_sha256=sha(__file__),plan_sha256=sha(OUT/'PLAN.md'),exclusive_before_model_forward=True,
         planned_calls=dict(backbone_windows=378,native_unused_projection_windows=102,final_projection_windows=144,
                            head_windows=144,manual_memory_query_windows=31,library_memory_query_windows=117)))
    try:
        with threadpool_limits(limits=4):run()
    except BaseException:
        save('FAILURE.json',dict(traceback=traceback.format_exc(),actual_calls=counts(),
             CPU_seconds=time.process_time()-CPU_START,wall_seconds=time.monotonic()-START,new_fits=0,optimizer_updates=0,GPU=0))
        raise
