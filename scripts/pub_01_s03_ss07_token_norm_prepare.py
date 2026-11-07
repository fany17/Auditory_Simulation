"""D16 one-operator formula/axis/RNG/real-path checks, no fitting or updates."""
import gc
from pub_01_s03_ss07_token_norm_core import *
from pub_01_s03_ss07_token_norm_queue import resources,scheduler_checks
def numpy_norm(x):
    z=x.astype('f8');return ((z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)).astype('f4')
def grad_stats(g):
    a=g.detach().cpu().double().numpy();return dict(nonzero_scalars=int(np.count_nonzero(a)),L2_norm=float(np.linalg.norm(a)),max_abs=float(np.max(abs(a))))
def prepare():
    start=time.time();cpu=time.process_time();torch.set_num_threads(2)
    assert not (ROOT/'PROTOCOL.json').exists()
    old=read(D15/'PROTOCOL.json');assert sha(D15/'PROTOCOL.json')=='b80f9aeffada112c012ce23f944279027d4219250fdf2999a28c4878d336506f'
    assert read(D15/'coordinator_review/REVIEW.json')['passed']
    for path,digest in old['source_pins'].items():assert sha(path)==digest,path
    inputs={k:v for k,v in old['inputs'].items() if not k.endswith('/last.pt')}
    for arm in ['H_full','C_match']:
        path=D15/f'fits/{arm}_seed1024/last.pt';st=path.stat();inputs[str(path)]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(path))
    for path,item in inputs.items():
        st=Path(path).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'] and sha(path)==item['sha256'],path
    pins=dict(old['source_pins']);paths=[D15/'PROTOCOL.json',D15/'coordinator_review/REVIEW.json',ROOT/'PLAN.md',ROOT/'COORDINATOR_DECISION.md',ROOT/'D15_SCIENTIFIC_REVIEW.json']+sorted((ROOT/'input').glob('pub_01_s03_ss07_token_norm_*.py'))
    for seed in SEEDS:
        for arm in ARMS:
            folder=D15/f'fits/{arm}_seed{seed}';paths += [folder/'COMPLETED.json',folder/'HISTORY.jsonl',folder/'SELECTED_METRICS.json',folder/'FIRST_BATCH_GRADIENTS.jsonl']+[folder/f'OFFLINE_TRAIN_epoch{e}.json' for e in [-1,0,4,9,19]]
    for path in paths:pins[str(path)]=sha(path)
    # No normalization RNG/parameters/running statistics are introduced.
    for seed in SEEDS:
        for arm in ARMS:
            baseline=previous.Conditional(seed,arm);model=Conditional(seed,arm)
            aa,bb=baseline.state_dict(),model.state_dict();assert aa.keys()==bb.keys() and all(torch.equal(aa[k],bb[k]) for k in aa)
            assert sum(p.numel() for p in model.parameters())==(164784 if arm=='P_prior' else 165792)
    rng=np.random.default_rng(20261102);raw=rng.normal(size=(3,2,32)).astype('f4');t=torch.tensor(raw,requires_grad=True)
    state=torch.get_rng_state().clone();norm=normalize_tokens(t);assert torch.equal(state,torch.get_rng_state())
    formula_error=float(np.max(abs(norm.detach().numpy()-numpy_norm(raw))));assert formula_error<2e-6
    changed=t.detach().clone();changed[0,1]+=torch.arange(32,dtype=torch.float32)*10
    after=normalize_tokens(changed);assert torch.equal(norm.detach()[1:],after[1:]) and torch.equal(norm.detach()[0,0],after[0,0])
    norm.square().sum().backward();assert torch.isfinite(t.grad).all()
    constant=torch.full((3,2,32),.1,requires_grad=True);yy=normalize_tokens(constant);(yy*torch.arange(32)).sum().backward();assert torch.isfinite(yy).all() and torch.isfinite(constant.grad).all()
    mock=scheduler_checks()
    protocol=dict(frozen=True,time=time.time(),source_pins=pins,inputs=inputs,seeds=SEEDS,arms=ARMS,epochs=20,train_windows=252454,train_effective=248620,val_effective=30059,batch_size=64,last_batch_size=38,updates_per_fit=78900,updates_total=710100,permutations=old['permutations'],max_per_GPU=old['max_per_GPU'],max_jobs=9,optimizer=old['optimizer'],objective=old['objective'],objective_dtype=old['objective_dtype'],only_scientific_change='After final token GELU before GRU: F.layer_norm(tokens,(32,),weight=None,bias=None,eps=1e-5), all H/C/P',normalization=dict(axis='Last32 features of each independent Bx2 token; no batch/time/record/split sharing',variance='population correction0',denominator='sqrt(var+1e-5)',affine_parameters=0,running_statistics=False,train_eval_same=True,additional_arithmetic_not_equal_FLOPs=True,representation_scale_shift_information_changes=True),offline_TRAIN_epochs=old['offline_TRAIN_epochs'],offline_outputs=old['offline_outputs'],target_window_counts=old['target_window_counts'],constant_policy=old['constant_policy'],selection=old['selection'],formal_NN=old['formal_NN'],new_encoder_head=0,engineering_NN_cap=4096,engineering_optimizer_updates=0,budget=old['budget'],first_batch_diagnostics='Existing forward/backward only, detached scalar pre/post token RMS/max/mean/variance and GRU saturation plus5module gradients; no extra forward/update/graph retention; shuffled input changes',scheduler=mock,forbidden=old['forbidden'],acceptance='Separately judge path conditioning, H vs same-normalized C/P and Ccap, legal history replacement, costs and confirmation/novelty limits; keep all negatives')
    dump(ROOT/'PROTOCOL.json',protocol)
    device=None
    for d in ['cuda:0','cuda:1']:
        free,pids,mem=resources(d)
        if not pids and free>=4096 and mem>=16*1024**3:device=d;break
    assert device is not None
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    gpu_rng=torch.cuda.get_rng_state(device).clone();gpu_t=torch.from_numpy(raw).to(device);gpu_n=normalize_tokens(gpu_t);assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    gpu_error=float(np.max(abs(gpu_n.cpu().numpy()-numpy_norm(raw))));assert gpu_error<2e-6
    data=Data(1024,'train');ids=np.concatenate([np.array([r['offset']+q for q in (0,8,9,10,11,12,13,14)]) for r in data.rows[:8]])
    NN=0;checks=[]
    for arm in ARMS:
        model=Conditional(1024,arm).to(device);model.train();x,h,a,y,ok=data.batch_indices(ids,arm,device);h.requires_grad_(True)
        for stage in ['fresh_zero','fresh_fixed_nonzero_output_probe']:
            if stage!='fresh_zero':
                with torch.no_grad():model.output.weight.copy_(.01*torch.sin(torch.arange(model.output.weight.numel(),device=device,dtype=torch.float32).reshape_as(model.output.weight)*.017));model.output.bias.zero_()
            model.zero_grad(set_to_none=True);h.grad=None;model.capture_diagnostics=True;res=model(x,h);NN+=len(res)
            pred=a.clone();pred[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*res.double()
            if stage=='fresh_zero':assert torch.count_nonzero(res)==0 and torch.equal(pred,a)
            loss=objective(pred,y);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            input_gradient=grad_stats(h.grad);hh=grad_stats(model.gru.weight_hh_l0.grad)
            if stage!='fresh_zero':assert input_gradient['nonzero_scalars']>0 and hh['nonzero_scalars']>0
            checks.append(dict(arm=arm,stage=stage,NN_windows=len(res),loss=float(loss.detach()),token_input_gradient=input_gradient,GRU_hh_gradient=hh,modules=module_gradient_summary(model),token_state=model.take_token_diagnostics(),optimizer_updates=0))
            if arm=='H_full' and stage!='fresh_zero':
                model.capture_diagnostics=False
                with torch.no_grad():again=model(x,h);NN+=len(again)
                assert torch.equal(res.detach(),again) and model.last_scalar_diagnostics is None
        del model,x,h,a,y,res,pred,loss;gc.collect();torch.cuda.empty_cache();guard()
    # Fixed late-weight distribution-change diagnostic, never a trained candidate.
    for arm in ['H_full','C_match']:
        cp=torch.load(D15/f'fits/{arm}_seed1024/last.pt',map_location=device,weights_only=True)
        for normalized in [False,True]:
            model=(Conditional if normalized else previous.Conditional)(1024,arm).to(device);model.load_state_dict(cp['model']);model.train();captured={}
            def token_hook(module,args,out):captured['raw']=out
            def gru_hook(module,args,out):captured['in']=args[0];captured['states']=out[0]
            hooks=[model.token.register_forward_hook(token_hook),model.gru.register_forward_hook(gru_hook)]
            x,h,a,y,ok=data.batch_indices(ids,arm,device);h.requires_grad_(True);res=model(x,h);NN+=len(res);pred=a.clone();pred[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*res.double();loss=objective(pred,y);loss.backward()
            checks.append(dict(arm=arm,stage='D15_late_same_weights_distribution_diagnostic',normalization_applied=normalized,NN_windows=len(res),loss=float(loss.detach()),token_input_gradient=grad_stats(h.grad),GRU_hh_gradient=grad_stats(model.gru.weight_hh_l0.grad),token_state=token_state_summary(captured['raw'].reshape(len(x),2,32),captured['in'],captured['states']),optimizer_updates=0,not_a_trained_normalized_candidate=True))
            for hook in hooks:hook.remove()
            del model,x,h,a,y,res,pred,loss,captured,hooks;gc.collect();torch.cuda.empty_cache();guard()
        del cp
    assert NN==528 and NN<=4096
    dump(ROOT/'ENGINEERING_CHECK.json',dict(passed=True,checks=checks,formula_CPU_max_abs=formula_error,formula_GPU_max_abs=gpu_error,per_token_axis_independence=True,constant_input_output_and_gradient_finite=True,shared_state_dict_and_parameter_counts_all9_exact=True,CPU_CUDA_normalization_RNG_unchanged=True,diagnostic_flag_output_exact=True,NN_windows=NN,optimizer_updates=0,learning_model_fits=0,new_encoder_head=0,wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),formal_models_reconstructed_clean=True))
    print(json.dumps(dict(passed=True,NN_windows=NN,protocol_sha256=sha(ROOT/'PROTOCOL.json'))),flush=True)
if __name__=='__main__':prepare()
