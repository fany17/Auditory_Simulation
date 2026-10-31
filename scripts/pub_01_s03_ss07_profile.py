"""Discarded-weight train-cache engineering profile; no active source edits."""
import copy
import json
import time
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT,OLD,dump,sha
from pub_01_s03_ss07_incremental import NativeResidual,detach
from pub_01_s03_ss07_train import objective


def serial(model):return all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
def aggregated(model):
    values=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    return bool(torch.stack(values).all()) if values else True


def update(model,opt,x,y,i,checker,state=None):
    local=(i%2)*64
    if local==0:state=None
    opt.zero_grad(set_to_none=True);p,state,_=model(x[local:local+64],state,local,'engineering')
    loss=objective(p,y[local:local+64]);assert torch.isfinite(loss);loss.backward()
    assert checker(model);opt.step()
    # Same scalar conversion and detach as active trainer.
    loss_value=float(loss.detach())
    return detach(state)


def main():
    torch.set_num_threads(2);device='cuda:0';torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    x=torch.from_numpy(np.array(np.load(ROOT/'baseline/head_seed1024/train_latent.npy',mmap_mode='r')[:128],copy=True)).to(device)
    y=torch.from_numpy(np.array(np.load(OLD/'envelope_readout_seed1024/train_envelope.npy',mmap_mode='r')[:128],copy=True)).to(device)
    torch.manual_seed(789006);first=NativeResidual('m2',16).to(device);second=copy.deepcopy(first)
    first.core.gru.flatten_parameters();second.core.gru.flatten_parameters()
    opts=[torch.optim.Adam(m.parameters(),lr=.001) for m in [first,second]];states=[None,None]
    # Isolate the finite predicates from independent cuDNN backward noise:
    # EXACT same initial weights/Adam state and EXACT same real gradients.
    for i in range(8):
        local=(i%2)*64
        if local==0:states[0]=None
        opts[0].zero_grad(set_to_none=True);p,states[0],_=first(x[local:local+64],states[0],local,'engineering')
        loss=objective(p,y[local:local+64]);loss.backward()
        for a,b in zip(first.parameters(),second.parameters()):b.grad=a.grad.clone() if a.grad is not None else None
        snapshot=[v.grad.clone() if v.grad is not None else None for v in first.parameters()]
        assert serial(first) and aggregated(second)
        for a,b,previous in zip(first.parameters(),second.parameters(),snapshot):
            if previous is not None:assert torch.equal(a.grad,previous) and torch.equal(b.grad,previous)
        opts[0].step();opts[1].step();states[0]=detach(states[0])
    torch.cuda.synchronize()
    weight_diff=max(float((a.detach()-b.detach()).abs().max()) for a,b in zip(first.parameters(),second.parameters()));assert weight_diff==0
    grad_diff=max(float((a.grad-b.grad).abs().max()) for a,b in zip(first.parameters(),second.parameters()) if a.grad is not None);assert grad_diff==0
    adam_diff=max(float((opts[0].state[a][key]-opts[1].state[b][key]).abs().max())
        for a,b in zip(first.parameters(),second.parameters()) for key in ['exp_avg','exp_avg_sq']);assert adam_diff==0
    for parameter in first.parameters():
        if parameter.grad is None:continue
        for bad in [float('nan'),float('inf'),-float('inf')]:
            original=parameter.grad.flatten()[0].clone();parameter.grad.flatten()[0]=bad
            assert not serial(first) and not aggregated(first)
            parameter.grad.flatten()[0]=original
    assert serial(first) and aggregated(first)
    # Alternating short blocks under unchanged shared GPU contention. This is
    # NOT an isolated benchmark and cannot imply scientific-training speed.
    times={'serial':[],'aggregated':[]};models=[first,second];states=[None,None]
    for block in range(8):
        order=[0,1] if block%2==0 else [1,0]
        for j in order:
            checker=[serial,aggregated][j];name=['serial','aggregated'][j]
            torch.cuda.synchronize();started=time.perf_counter()
            for i in range(20):states[j]=update(models[j],opts[j],x,y,i,checker,states[j])
            torch.cuda.synchronize();times[name].append((time.perf_counter()-started)/20)
    # Check cost alone, same existing gradients, all same finite predicates.
    checks={}
    for name,checker in [('serial',serial),('aggregated',aggregated)]:
        torch.cuda.synchronize();started=time.perf_counter()
        for i in range(100):assert checker(first)
        torch.cuda.synchronize();checks[name]=(time.perf_counter()-started)/100
    dump(ROOT/'FINITE_PROFILE.json',{'engineering_only':True,'test_read':False,'fresh_discarded_weights':True,
        'active_training_sources_changed':False,'actual_first128_ownseed1024_train_cache_windows':True,
        'equivalence_method':'exact SAME real backward gradients and initial weights/Adam states; check predicates verified nonmutating; independent cuDNN backwards not assumed bitwise deterministic',
        'parameter_max_diff':weight_diff,'gradient_max_diff':grad_diff,'Adam_state_max_diff':adam_diff,
        'NaN_Inf_both_signs_in_each_gradient_tensor_detected':True,
        'whole_step_seconds_per_alternating_block':times,'whole_step_medians_seconds':{k:float(np.median(v)) for k,v in times.items()},
        'finite_check_alone_seconds':checks,'GPU_contention':'other authorized jobs remain active; timing is indicative only',
        'source':str(ROOT/'input/pub_01_s03_ss07_profile.py'),'source_sha256':sha(ROOT/'input/pub_01_s03_ss07_profile.py')})
    print(json.dumps({'status':'PASS','step_medians':{k:float(np.median(v)) for k,v in times.items()},'check_seconds':checks}))


if __name__=='__main__':main()
