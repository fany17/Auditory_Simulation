"""Freeze D15 and test objective math/real gradients/scheduler without fitting."""
import gc
from pub_01_s03_ss07_objective_control_core import *
from pub_01_s03_ss07_objective_control_queue import scheduler_checks,resources
def numpy_loss(p,y):
    u=p-p.mean(1,keepdims=True);v=y-y.mean(1,keepdims=True)
    return -(u*v).sum(1)/(np.maximum(np.linalg.norm(u,axis=1),1e-6)*np.maximum(np.linalg.norm(v,axis=1),1e-6))
def prepare():
    start=time.time();cpu=time.process_time();torch.set_num_threads(2)
    assert not (ROOT/'PROTOCOL.json').exists()
    old=read(D14/'PROTOCOL.json');assert sha(D14/'PROTOCOL.json')=='83b05eafcb126952d48750930106b5fb04c56d49f82d2b0128ba7d2c813d0172'
    assert read(D14/'coordinator_review/REVIEW.json')['passed']
    for path,digest in old['source_pins'].items():assert sha(path)==digest,path
    # No old-last TRAIN reference rerun in D15; only currently used caches and
    # three D14 nonzero checkpoints for bounded engineering are identities.
    inputs={k:v for k,v in old['inputs'].items() if not k.endswith('/last.pt')}
    for arm in ARMS:
        path=D14/f'fits/{arm}_seed1024/last.pt';st=path.stat();inputs[str(path)]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(path))
    for path,item in inputs.items():
        st=Path(path).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'] and sha(path)==item['sha256'],path
    pins=dict(old['source_pins'])
    paths=[D14/'PROTOCOL.json',D14/'coordinator_review/REVIEW.json',ROOT/'PLAN.md',ROOT/'LOSS_SOURCE_AUDIT.md',ROOT/'STORAGE_ESTIMATE.json',ROOT/'STORAGE_DECISION.md',ROOT/'STORAGE_PREFLIGHT.py']+sorted((ROOT/'input').glob('pub_01_s03_ss07_objective_control_*.py'))
    for seed in SEEDS:
        for arm in ARMS:
            folder=D14/f'fits/{arm}_seed{seed}';paths += [folder/'COMPLETED.json',folder/'HISTORY.jsonl']+[folder/f'OFFLINE_TRAIN_epoch{e}.json' for e in [-1,0,4,9,19]]
    for path in paths:pins[str(path)]=sha(path)
    counts={}
    for phase in ['train','val']:
        y=np.load(previous.frozen.OLD/f'envelope_readout_seed1024/{phase}_envelope.npy',mmap_mode='r')
        out=dict(windows=0,exact_raw_constant=0,exact_centered_zero=0,centered_norm_le_eps=0)
        for first in range(0,len(y),8192):
            cc=target_counts(y[first:first+8192])
            for key in out:out[key]+=cc[key]
        counts[phase]=out
    mock=scheduler_checks();size_mock=guard_checks()
    rng=np.random.default_rng(20261102);pp=rng.normal(size=(5,192));yy=rng.normal(size=(5,192))
    p=torch.tensor(pp,dtype=torch.float64);y=torch.tensor(yy,dtype=torch.float64)
    assert np.max(abs(window_losses(p,y).numpy()-numpy_loss(pp,yy)))<1e-12
    assert torch.allclose(window_losses(p+3,y-5),window_losses(p,y),atol=1e-12,rtol=0)
    assert torch.allclose(window_losses(p*2,y*3),window_losses(p,y),atol=1e-12,rtol=0)
    assert torch.allclose(window_losses(-p,y),-window_losses(p,y),atol=1e-12,rtol=0)
    synthetic_constants=[]
    for value in [0.,.1,float(np.float32(.1))]:
        q=p.clone().requires_grad_(True);t=torch.full_like(q,value);l=objective(q,t);l.backward();assert torch.isfinite(q.grad).all()
        centered_zero=bool(torch.count_nonzero(t-t.mean(-1,keepdim=True))==0)
        if centered_zero:assert l.item()==0 and torch.count_nonzero(q.grad)==0
        synthetic_constants.append(dict(value=value,centered_exactly_zero=centered_zero,loss=float(l.detach()),max_abs_gradient=float(q.grad.abs().max())))
    constant=p.new_full(p.shape,.1).requires_grad_(True);l=objective(constant,y);l.backward();assert torch.isfinite(constant.grad).all()
    protocol=dict(frozen=True,time=time.time(),source_pins=pins,inputs=inputs,seeds=SEEDS,arms=ARMS,epochs=20,train_windows=252454,train_effective=248620,val_effective=30059,batch_size=64,last_batch_size=38,updates_per_fit=78900,updates_total=710100,permutations=old['permutations'],max_per_GPU={'cuda:0':5,'cuda:1':4},max_jobs=9,objective='negative mean F.cosine_similarity(pred-pred.mean(-1), y-y.mean(-1), dim=-1, eps=1e-6)',objective_dtype='float64 adaptation of original native formula, no float32 cast; distinct from native float32 arithmetic',only_scientific_change='D14 windowMSE -> per-window centered negative cosine; no mixed loss/LR/architecture changes',optimizer=old['optimizer'],offline_TRAIN_epochs=[-1,0,4,9,19],offline_outputs=['negative_cosine','window_MSE','physical60s_r','physical60s_MSE','amplitude','overlap','constant/eps counts'],reuse_D14='Frozen MSE/physical/VAL results only; no extra old TRAIN NN to obtain old3s loss',target_window_counts=counts,constant_policy='All windows retained in batch/epoch denominator incl fallback and constant targets; epsilon cosine is numerical objective, not strict defined Pearson',selection=old['selection'],formal_NN=dict(train_updates=44751600,val_epochs=5410620,offline_TRAIN_nonzero=8950320,selected_VAL=270531,selected_H_older_VAL=90177,total=59473248),new_encoder_head=0,engineering_NN_cap=4096,engineering_optimizer_updates=0,budget=dict(wall_seconds=21600,max_idle_GPU=2,max_jobs=9,RSS_each=8*1024**3,GPU_allocated_each=2*1024**3,root_bytes=160*1024**2,each_disk_free=20*1024**3,launch_min_MemAvailable=16*1024**3,launch_min_GPU_free_MiB=4096,new_arrays=0),scheduler=mock,forbidden=['old test','DTU','ds004408','new data/weights','mixed loss','LR search','new backbones','score-driven budget extension','globalGPU/MPS setting changes'],acceptance='Objective/optimization diagnostic; all9 fixed fits retained; not innovation or independent confirmation')
    dump(ROOT/'PROTOCOL.json',protocol)
    device=None;admission=[]
    for d in ['cuda:0','cuda:1']:
        free,observed,available=resources(d);admission.append(dict(device=d,free_GPU_MiB=free,observed_compute_pids=observed,MemAvailable_bytes=available))
        if device is None and not observed and free>=4096 and available>=16*1024**3:device=d
    assert device is not None,'No actually idle GPU for engineering; preserve freeze, no formal launch'
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    data=Data(1024,'train');ids=np.concatenate([np.array([r['offset']+q for q in (0,8,9,10,11,12,13,14)]) for r in data.rows[:8]])
    NN=0;checks=[]
    for arm in ARMS:
        model=Conditional(1024,arm).to(device);model.train();x,h,a,y,ok=data.batch_indices(ids,arm,device)
        for stage in ['fresh_zero','nonzero_D14_last']:
            if stage=='nonzero_D14_last':
                cp=torch.load(D14/f'fits/{arm}_seed1024/last.pt',map_location=device,weights_only=True);model.load_state_dict(cp['model']);del cp
            model.zero_grad(set_to_none=True);res=model(x,h);NN+=len(res);pred=a.clone();pred[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*res.double();pred.retain_grad()
            if stage=='fresh_zero':assert torch.count_nonzero(res)==0 and torch.equal(pred,a)
            loss=objective(pred,y);loss.backward();assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
            pn=pred.detach().cpu().numpy();yn=y.cpu().numpy();npvalue=float(numpy_loss(pn,yn).mean());assert abs(npvalue-float(loss.detach()))<1e-12
            direction=rng.normal(size=pn.shape);direction/=np.linalg.norm(direction);eps=1e-5
            fd=(numpy_loss(pn+eps*direction,yn).mean()-numpy_loss(pn-eps*direction,yn).mean())/(2*eps)
            grad=pred.grad.detach().cpu().numpy();exact=float(np.sum(grad*direction));assert abs(fd-exact)<1e-8
            gm=2*(pn-yn)/(len(pn)*192);gp=grad.copy();gm[~ok]=0;gp[~ok]=0
            checks.append(dict(arm=arm,stage=stage,NN_windows=len(res),loss=float(loss.detach()),independent_numpy_loss_abs=abs(npvalue-float(loss.detach())),output_finite_difference_abs=abs(fd-exact),learnable_gradient_norm_ratio_PCC_over_MSE=float(np.linalg.norm(gp)/np.linalg.norm(gm)),learnable_gradient_cosine=float(np.sum(gp*gm)/(np.linalg.norm(gp)*np.linalg.norm(gm))),all_parameter_gradients_finite=True,GRU_recurrent_gradient_nonzero_scalars=int(torch.count_nonzero(model.gru.weight_hh_l0.grad)),optimizer_updates=0,batch_windows=len(ids),fallback_windows=int(np.sum(~ok)),target_counts=target_counts(yn)))
        del model,x,h,a,y,res,pred,loss;gc.collect();torch.cuda.empty_cache();guard()
    assert NN==288 and NN<=4096
    dump(ROOT/'ENGINEERING_CHECK.json',dict(passed=True,checks=checks,scheduler=mock,guard_mock=size_mock,target_counts=counts,synthetic_constants=synthetic_constants,exact_centered_zero_target_has_zero_objective_gradient=True,synthetic_constant_prediction_finite_eps_gradient=True,independent_numpy_separate_norm_floor=True,NN_windows=NN,new_encoder_head=0,optimizer_updates=0,learning_model_fits=0,wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),device=device,prelaunch_resources=admission,formal_models_reconstructed_clean=True))
    print(json.dumps(dict(passed=True,NN=NN,protocol_sha256=sha(ROOT/'PROTOCOL.json'))),flush=True)
if __name__=='__main__':prepare()
