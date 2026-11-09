"""D18 one bounded TRAIN-only engineering pass. All temporary updates discarded."""
import gc,io,traceback
from torch import nn
from pub_01_s03_ss07_preprojection_runtime import *
from pub_01_s03_ss07_preprojection_reference import ccap_value_and_vjp
EXTRA=dict(reference_encoder_prefix_windows=0,reference_projection_windows=0,reference_head_windows=0,readout_only_windows=0,readout_VJP_calls=0,attention_query_VJP_calls=0,temporary_optimizer_updates=0)
TOTAL={};ATTEMPT=[]
LIVE_MODEL=None
CLOCK={}
def add_counts(count):
    for k,v in count.items():TOTAL[k]=TOTAL.get(k,0)+v
def error_check(actual,reference,absolute,relative):
    a=np.asarray(actual,dtype='f8');b=np.asarray(reference,dtype='f8');err=float(np.max(abs(a-b)));bound=absolute+relative*float(np.max(abs(b)))
    assert np.isfinite(a).all() and np.isfinite(b).all() and err<=bound,(err,bound)
    return dict(max_abs=err,bound=bound)
def rng_state(device):return torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone()
def restore_rng(state,device):torch.set_rng_state(state[0]);torch.cuda.set_rng_state(state[1],device)
def assert_rng_equal(a,b):assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
def parameter_bytes(state):
    buffer=io.BytesIO();torch.save(dict(trainable_state=state,seed=31415,arm='P_prediction_history',epoch=19,validation_r=-.12345678901234567,protocol_sha256='0'*64,frozen_assets='Source pins only; no repeated E0/Ccap/position/optimizer'),buffer)
    size=buffer.tell();buffer.seek(0);loaded=torch.load(buffer,map_location='cpu',weights_only=True)['trainable_state'];assert state.keys()==loaded.keys() and all(torch.equal(state[k],loaded[k]) for k in state)
    return size
def readout_bridge(data,device,ids):
    ids=np.asarray(ids);ids=ids[data.local[ids]>=9];folder=D8/f'seed{data.seed}'
    z=np.load(folder/'train_latent.npy',mmap_mode='r');p=np.load(folder/'train_P.npy',mmap_mode='r')
    zz=np.array(z[ids],dtype='f8');pp=np.array(p[ids],dtype='f8');del z,p
    cotangent=np.broadcast_to(np.sin(np.arange(192)*.017)[None],pp.shape).copy()/len(ids)
    expected,vjp=ccap_value_and_vjp(zz,pp,data.statistics,data.coefficient,data.bias,data.calibration,cotangent)
    module=data.torch_readout().to(device);tz=torch.tensor(zz,device=device,requires_grad=True);tp=torch.tensor(pp[:,None],device=device,requires_grad=True)
    value=module(tz,tp);EXTRA['readout_only_windows']+=len(ids)
    grads=torch.autograd.grad(value,(tz,tp),torch.tensor(cotangent,device=device));EXTRA['readout_VJP_calls']+=1
    result=dict(windows=len(ids),value=error_check(value.detach().cpu().numpy(),expected,1e-10,1e-9),Z_VJP=error_check(grads[0].cpu().numpy(),vjp['latent'],1e-9,1e-8),head_VJP=error_check(grads[1][:,0].cpu().numpy(),vjp['head'],1e-9,1e-8),not_encoder_or_head_forward=True)
    del module,tz,tp,value,grads
    return result
def native_prediction(model,batch,grad=True):
    # Original AuditoryTransformer.forward, not the newly extracted prefix.
    z=model.encoder(batch['x']).transpose(1,2);native=model.head(z);prediction=model.readout(z,native)
    EXTRA['reference_encoder_prefix_windows']+=len(z);EXTRA['reference_projection_windows']+=len(z);EXTRA['reference_head_windows']+=len(z)
    return z,native,torch.where(batch['valid'][:,None],prediction,batch['anchor'])
def engineering(device):
    global LIVE_MODEL
    start=time.time();cpu=time.process_time();CLOCK.update(wall=start,cpu=cpu);p=read(ROOT/'ENGINEERING_PROTOCOL.json');verify(p);guard()
    assert not (ROOT/'ENGINEERING_CHECK.json').exists()
    resource_before=admission(device);assert resource_before['admitted'];seed_all(1024,device);torch.cuda.reset_peak_memory_stats()
    results=[];reference_states={};checkpoint_sizes=[]
    for seed in SEEDS:
        data=RawPhase(seed,'train',ram_raw=False,lazy_anchor=True)
        ids0=np.array(p['TRAIN_cases'][str(seed)]['shuffle0']);ids1=np.array(p['TRAIN_cases'][str(seed)]['shuffle1']);cold=np.array(p['TRAIN_cases'][str(seed)]['cold'])
        bridge=readout_bridge(data,device,ids0);dump(ROOT/f'ENGINEERING_READOUT_seed{seed}.json',bridge);reference_train=None;reference_rng_after=None;common_initial=None
        for arm in ARMS:
            seed_all(seed,device);cuda_before_constructor=torch.cuda.get_rng_state(device).clone();model=make_model(seed,arm,data,device);model.train()
            assert torch.equal(cuda_before_constructor,torch.cuda.get_rng_state(device)),'CPU-only initialization must preserve CUDA RNG'
            LIVE_MODEL=model
            count_start=dict(model.count);initial={k:v.clone() for k,v in model.trainable_state().items()}
            frozen={k:v.detach().cpu().clone() for k,v in model.reference_encoder.state_dict().items()};buffers={k:v.detach().cpu().clone() for k,v in model.readout.named_buffers()}
            expected={'T_continue':3372617,'H_history':3389513,'C_current':3389513,'P_prediction_history':3390153}[arm]
            parameters=sum(v.numel() for v in model.parameters() if v.requires_grad);assert parameters==expected
            if arm=='H_history':common_initial={k:v.clone() for k,v in initial.items()}
            if arm=='C_current':assert common_initial.keys()==initial.keys() and all(torch.equal(common_initial[k],initial[k]) for k in initial)
            batch,target=data.batch(ids0,arm,device);state_before=rng_state(device)
            if arm=='H_history':
                model.eval()
                with torch.no_grad():z,native,raw=native_prediction(model,batch)
                folder=D8/f'seed{seed}';cz=np.load(folder/'train_latent.npy',mmap_mode='r');cp=np.load(folder/'train_P.npy',mmap_mode='r')
                raw_bridge=dict(Z=error_check(z.cpu().numpy(),np.array(cz[ids0]),1e-4,1e-6),P=error_check(native[:,0].cpu().numpy(),np.array(cp[ids0]),1e-4,1e-6),Ccap=error_check(raw.cpu().numpy(),data.anchor[ids0],1e-4,1e-6),scope='Fresh shuffled raw64 eval vs original global64 cached data; not same-latent algebra')
                del z,native,raw,cz,cp
                model.train();restore_rng(state_before,device)
                z,native,raw=native_prediction(model,batch)
                reference_train=raw.detach().cpu().numpy().copy();reference_rng_after=rng_state(device)
                del z,native,raw;restore_rng(state_before,device)
            else:raw_bridge=None
            assert_rng_equal(state_before,rng_state(device))
            opt=torch.optim.Adam(model.trainable_groups(),weight_decay=0.)
            step_rows=[];cached_capture={}
            def memory_before(module,args):cached_capture['query']=args[0];cached_capture['memory']=args[1]
            def memory_after(module,args,out):cached_capture['delta']=out
            hooks=[] if arm=='T_continue' else [model.memory.register_forward_pre_hook(memory_before),model.memory.register_forward_hook(memory_after)]
            for step,ids in enumerate((ids0,ids1)):
                if step:batch,target=data.batch(ids,arm,device)
                opt.zero_grad(set_to_none=True);torch.cuda.synchronize();step_start=time.time();step_cpu=time.process_time()
                model.capture=True;prediction=model(**batch)
                if step==0:
                    initial_bridge=error_check(prediction.detach().cpu().numpy(),reference_train,1e-6,0.)
                    assert_rng_equal(rng_state(device),reference_rng_after)
                    assert all(not x.training for x in model.reference_encoder.modules())
                    dump(ROOT/f'ENGINEERING_INITIAL_{arm}_seed{seed}.json',dict(raw_cache_bridge=raw_bridge,initial_same_raw_bridge=initial_bridge,CPU_and_CUDA_RNG_equal=True,E0_eval=True))
                if step==1 and arm!='T_continue':
                    scalar=(cached_capture['delta']*torch.sin(torch.arange(64,device=device)*.017)).sum()/cached_capture['delta'].numel()
                    dq=torch.autograd.grad(scalar,cached_capture['query'],retain_graph=True)[0];EXTRA['attention_query_VJP_calls']+=1
                    assert torch.isfinite(dq).all() and torch.count_nonzero(dq)>0
                    query_gradient=dict(nonzero_scalars=int(torch.count_nonzero(dq)),L2_norm=float(dq.double().square().sum().sqrt()))
                else:query_gradient=None
                loss=objective(prediction,target);loss.backward();grads=gradient_summary(model)
                assert grads['encoder']['nonzero_scalars']>0 and grads['head']['nonzero_scalars']>0 and grads['original_attention0']['nonzero_scalars']>0
                if arm!='T_continue':
                    for name in ('Q','K','V'):
                        if step==0:assert grads[f'memory_{name}_weight']['nonzero_scalars']==0
                        else:assert grads[f'memory_{name}_weight']['nonzero_scalars']>0
                opt.step();EXTRA['temporary_optimizer_updates']+=1;torch.cuda.synchronize()
                step_rows.append(dict(step=step,CPU_seconds=time.process_time()-step_cpu,wall_seconds=time.time()-step_start,includes='forward/loss/backward/Adam/synchronization; excludes batch loading',loss=float(loss.detach()),gradients=grads,query_VJP=query_gradient))
            for hook in hooks:hook.remove()
            p_probe=None
            if arm=='P_prediction_history':
                # Reuse existing query and original prior from step1; only new
                # Conv/KV algebra is counted, no encoder/head or full attention.
                with torch.no_grad():
                    query=cached_capture['query'].detach();valid=batch['valid']
                    # Both probe conditions use the SAME post-step2 weights.
                    before=model.memory_values(batch['x'],valid,prior=batch['prior'])
                    changed=batch['prior']+.01*torch.sin(torch.arange(192,device=device,dtype=torch.float64)*.017)
                    after=model.memory_values(batch['x'],valid,prior=changed)
                    q,k0,v0=model.memory.inputs(query,before);_,k1,v1=model.memory.inputs(query,after)
                    w=model.memory.attention.in_proj_weight;b=model.memory.attention.in_proj_bias
                    k0=nn.functional.linear(k0,w[64:128],b[64:128]);k1=nn.functional.linear(k1,w[64:128],b[64:128]);v0=nn.functional.linear(v0,w[128:],b[128:]);v1=nn.functional.linear(v1,w[128:],b[128:])
                    kd=float((k1-k0).abs().max());vd=float((v1-v0).abs().max());assert kd>0 and vd>0
                    p_probe=dict(projected_K_max_abs_difference=kd,projected_V_max_abs_difference=vd,extra_P_projection_windows=4*int(valid.sum()),extra_K_V_linear_operations=4,encoder_head_forwards=0)
            model.eval();cold_batch,_=data.batch(cold,arm,device)
            with torch.no_grad():
                pred=model(**cold_batch);assert torch.equal(pred[~cold_batch['valid']],cold_batch['anchor'][~cold_batch['valid']])
                snapshot=dict(model.count);bypass=model(**cold_batch,enabled=False);assert torch.equal(bypass,cold_batch['anchor']) and model.count==snapshot
            assert all(torch.equal(frozen[k],v.detach().cpu()) for k,v in model.reference_encoder.state_dict().items())
            assert all(torch.equal(buffers[k],v.detach().cpu()) for k,v in model.readout.named_buffers())
            assert any(not torch.equal(initial[k],v) for k,v in model.trainable_state().items())
            checkpoint_size=parameter_bytes(model.trainable_state());checkpoint_sizes.append(checkpoint_size)
            counts=counters_delta(model.count,count_start);add_counts(counts)
            record=dict(seed=seed,arm=arm,trainable_parameters=parameters,initial_same_raw_bridge=initial_bridge,raw_cache_bridge=raw_bridge,readout_value_VJP_bridge=bridge if arm=='H_history' else 'shared same-seed proof',steps=step_rows,P_content_probe=p_probe,counts=counts,checkpoint_bytes_in_memory=checkpoint_size,E0_and_Ccap_unchanged=True,cold_fallback_and_entire_bypass_exact=True,formal_initialization_clean_rebuild_required=True)
            results.append(record);ATTEMPT.append(record);dump(ROOT/f'ENGINEERING_{arm}_seed{seed}.json',record)
            LIVE_MODEL=None
            del model,opt,batch,target,cold_batch,pred,bypass,cached_capture,initial,frozen,buffers,hooks;gc.collect();torch.cuda.empty_cache();guard()
        del data;gc.collect()
    previous=p['prior_engineering_costs'];prior=previous['production'];prior_extra=previous['extra']
    assert EXTRA['temporary_optimizer_updates']==24
    assert EXTRA['temporary_optimizer_updates']+prior_extra.get('temporary_optimizer_updates',0)==p['engineering_caps']['temporary_optimizer_updates']==28
    assert TOTAL['current_encoder_prefix_windows']+EXTRA['reference_encoder_prefix_windows']+prior.get('current_encoder_prefix_windows',0)+prior_extra.get('reference_encoder_prefix_windows',0)<=8192
    assert TOTAL['current_head_windows']+EXTRA['reference_head_windows']+prior.get('current_head_windows',0)+prior_extra.get('reference_head_windows',0)<=8192
    assert TOTAL['frozen_encoder_prefix_windows']+prior.get('frozen_encoder_prefix_windows',0)<=16384
    result=dict(passed=True,engineering_protocol_sha256=sha(ROOT/'ENGINEERING_PROTOCOL.json'),results=results,production_component_counts=TOTAL,extra_component_counts=EXTRA,new_fits=0,formal_optimizer_updates=0,new_disk_arrays=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),largest_parameter_checkpoint_bytes=max(checkpoint_sizes),checkpoint_24_copies_upper_bytes=24*max(checkpoint_sizes),resource_before=resource_before,reference_synthetic_receipt='COORDINATOR_REFERENCE_SELFCHECK.json, already executed independently, not repeated',goal_complete=False)
    result['preserved_prior_attempt_costs']=previous
    dump(ROOT/'ENGINEERING_CHECK.json',result);print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    try:engineering(args.device)
    except BaseException:
        dump(ROOT/'ENGINEERING_FAILURE.json',dict(traceback=traceback.format_exc(),completed_rows=len(ATTEMPT),counts_completed_rows=TOTAL,unfinished_model_invoked_component_counts=LIVE_MODEL.count if LIVE_MODEL is not None else {},extra_counts=EXTRA,CPU_seconds=time.process_time()-CLOCK.get('cpu',time.process_time()),wall_seconds=time.time()-CLOCK.get('wall',time.time()),peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,warning='Invoked component windows include an incomplete operation if a kernel raised; exact completed internal operations are not fabricated',automatic_retry=False));raise
