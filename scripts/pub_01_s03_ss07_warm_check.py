"""2203 engineering only: synthetic gradients/state and actual VAL epoch-zero.

No scientific fit or optimizer step on real neural data. Artifacts in new ROOT.
"""
import argparse
import copy
import json
import platform
import time
import numpy as np
import torch
from pub_01_s03_ss07_warm import (ROOT,ROUND01,OLD,SEEDS,CONDITIONS,Data,WarmDecoder,
    original_p,make,rng_snapshot,rng_restore,rng_equal,optimizer,objective,detach,
    validate,read,dump,update,sha,disk_guard,precision)

def tensors(state):
    if torch.is_tensor(state):return [state]
    if isinstance(state,dict):return sum((tensors(v) for v in state.values()),[])
    if isinstance(state,(list,tuple)):return sum((tensors(v) for v in state),[])
    return []

def synthetic_checks(device):
    results=[]
    for condition in CONDITIONS:
        model=make(1024,condition,device);model.train();assert model.encoder.training==model.joint
        # Fixed engineering input, unrelated to neural arrays or target outcomes.
        gen=torch.Generator(device=device).manual_seed(317)
        x=torch.randn(64,192,64,device=device,generator=gen);y=torch.randn(64,1,192,device=device,generator=gen)
        spans=[(0,31,{'record':'synthetic_A'},0),(31,64,{'record':'synthetic_B'},0)]
        calls=[];hook=model.encoder.register_forward_pre_hook(lambda _,args:calls.append(tuple(args[0].shape)))
        opt=optimizer(model);reachable={name:False for name,p in model.named_parameters() if p.requires_grad}
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            if model.joint:z=model.encoder(x).transpose(1,2)
            else:
                with torch.no_grad():z=model.encoder(x).transpose(1,2)
            pred,state,active=model.latent_forward(z,spans)
            loss=objective(pred,y);loss.backward()
            for name,p in model.named_parameters():
                if name in reachable and p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0:reachable[name]=True
            opt.step();state=detach(state)
            assert all(v.grad_fn is None and not v.requires_grad for v in tensors(state))
        hook.remove();assert calls==[(64,192,64)]*3,calls
        missing=[k for k,v in reachable.items() if not v];assert not missing,(condition,missing)
        assert model.encoder.training==model.joint
        if not model.joint:assert all(p.grad is None for p in model.encoder.parameters())
        results.append({'condition':condition,'all_trainable_parameter_tensors_nonzero_gradient_by_step3':True,
            'trainable_tensor_count':len(reachable),'encoder_call_shapes':calls,'frozen_eval_preserved':not model.joint,
            'synthetic_steps_only':3,'real_data_optimizer_steps':0})
    # Nonzero projection is deliberate only here, so state/latent dependencies can be measured.
    # cuDNN GRU requires train mode for backward. This path consumes synthetic
    # z directly, so the stochastic encoder is not invoked by the probe.
    model=make(1024,'joint_past',device);model.train()
    with torch.no_grad():model.readout.residual.weight.fill_(.02)
    gen=torch.Generator(device=device).manual_seed(911)
    z=torch.randn(64,8,192,device=device,generator=gen,requires_grad=True)
    a={'record':'A'};spans=[(0,64,a,0)]
    p,state,_=model.latent_forward(z,spans)
    g=torch.autograd.grad(p[10].square().sum(),z,retain_graph=True)[0]
    assert g[0].abs().sum()>0 and g[8].abs().sum()>0
    assert torch.count_nonzero(g[9])==0 and torch.count_nonzero(g[11:])==0
    within={'earlier_i0_gradient_norm':float(g[0].norm()),'i_minus2_gradient_norm':float(g[8].norm()),
        'overlap_i_minus1_gradient_zero':True,'future_gradient_zero':True}
    native_pred,_,_=model.readout(z,None,0,'A')
    native_equation_diff=float((native_pred-p).abs().max())
    assert native_equation_diff==0
    # FP32 cuDNN TF32 uses different kernels at different recurrent lengths.
    # Structural partition identity is verified independently in CPU float64;
    # actual GPU global64/VAL tolerances remain unchanged below.
    model.readout.to(device='cpu',dtype=torch.float64)
    z=z.detach().cpu().double().requires_grad_()
    p,state,_=model.latent_forward(z,spans)
    # Chunk boundaries preserve numerical state; no detach inside a global batch.
    first,state_first,_=model.latent_forward(z[:32],[(0,32,a,0)])
    second,state_second,_=model.latent_forward(z[32:],[(0,32,a,32)],state_first,'A')
    partition_diff=float((p-torch.cat((first,second))).abs().max())
    assert partition_diff<1e-10
    cross=torch.autograd.grad(second[0].square().sum(),z,retain_graph=True)[0]
    assert cross[:32].abs().sum()>0
    detached=detach(state_first)
    continued,state_cont,_=model.latent_forward(z[32:],[(0,32,a,32)],detached,'A')
    assert (continued-second).abs().max()<1e-10
    after_detach=torch.autograd.grad(continued[0].square().sum(),z,retain_graph=True)[0]
    assert torch.count_nonzero(after_detach[:32])==0
    reset,_,_=model.latent_forward(z[32:],[(0,32,a,0)],None,None)
    assert (continued-reset).abs().max()>1e-8
    # Record B never depends on record A, including a mixed-record global64.
    mixed,_,_=model.latent_forward(z,[(0,31,a,0),(31,64,{'record':'B'},0)])
    clean,_,_=model.latent_forward(z[31:],[(0,33,{'record':'B'},0)])
    assert (mixed[31:]-clean).abs().max()<1e-10
    across=torch.autograd.grad(mixed[40].square().sum(),z)[0]
    assert torch.count_nonzero(across[:31])==0
    try:model.readout(z[:2],state_first,32,'B')
    except AssertionError:wrong_record_rejected=True
    else:raise AssertionError('wrong-record state accepted')
    epoch1,_,_=model.latent_forward(z,spans,None,None)
    assert (epoch1-p).abs().max()==0
    return {'parameters':results,'within_global64_history_gradient':within,'chunk_carry_not_reset':True,
        'partition_reference_dtype_device':'float64 CPU on2203','partition_reference_max_diff':partition_diff,
        'native_M4_same64_equation_max_diff':native_equation_diff,
        'GPU_different_recurrent_partition_bitwise_equivalence_claimed':False,
        'detach_cuts_cross_update_graph_not_numeric_state':True,'mixed_record_independence':True,
        'wrong_record_state_rejected':wrong_record_rejected,'epoch_reset_reproducible':True}

@torch.no_grad()
def actual_checks(device,folder):
    rows=[];rng_rows=[]
    for seed in SEEDS:
        reference=original_p(seed,device);ref_rng=rng_snapshot()
        joint_data=Data(seed,'val',device,True)
        boundary=joint_data.rows[0]['windows'];first=boundary//64*64
        if first==boundary:first=max(0,first-32)
        assert first<boundary<first+64
        x,y=joint_data.raw.batch(first,first+64,device)
        reference.train();rng_restore(ref_rng);ref_pred=reference(x);ref_after=rng_snapshot()
        for kind in ('p','current','past'):
            ref=original_p(seed,device);before=rng_snapshot();model=WarmDecoder(ref,'joint_'+kind)
            assert rng_equal(before,rng_snapshot()) and rng_equal(before,ref_rng)
            model.train();rng_restore(ref_rng);calls=[]
            h=model.encoder.register_forward_pre_hook(lambda _,a:calls.append(tuple(a[0].shape)))
            z=model.encoder(x).transpose(1,2)
            # Start the representative record tail with an engineering reset;
            # zero projection makes predictions independent of that history.
            spans=list(joint_data.spans(first,first+64))
            pred,_,_=model.latent_forward(z,spans);h.remove()
            diff=float((pred-ref_pred).abs().max());assert diff<=1e-6 and calls==[(64,192,64)]
            assert rng_equal(rng_snapshot(),ref_after)
            rng_rows.append({'seed':seed,'condition':'joint_'+kind,'cross_record_global_start':first,
                'constructor_cpu_all_cuda_rng_unchanged':True,'encoder_train_dropout_stream_matches_reconstructed_legacy':True,
                'encoder_call_shapes':calls,'zero_projection_train_waveform_max_diff':diff})
        del reference
        for condition in CONDITIONS:
            model=make(seed,condition,device);data=Data(seed,'val',device,model.joint)
            result,pred=validate(model,data,continuous=False,keep_predictions=True)
            mode=condition.split('_')[0]
            if condition.endswith('_p'):
                base=pred.copy();base_loss=result['negative_Pearson'];base_mode=mode
            assert mode==base_mode
            waveform_diff=float(np.max(np.abs(pred-base)));lossdiff=abs(result['negative_Pearson']-base_loss)
            assert waveform_diff<=1e-6 and lossdiff<=1e-7
            legacy=read(ROUND01/f'finetune/native_P_seed{seed}/completed.json')
            assert legacy['epochs']==5 and legacy['updates']==19725 and legacy['selected_epoch']==4
            legacy_lossdiff=abs(result['negative_Pearson']-legacy['initial_validation_negative_Pearson'])
            assert legacy_lossdiff<=1e-7
            rows.append({'seed':seed,'condition':condition,**result,'same_row_P_waveform_max_diff':waveform_diff,
                'same_row_P_loss_abs_diff':lossdiff,'original_FT_initial_VAL_loss_abs_diff':legacy_lossdiff,
                'real_data_optimizer_steps':0})
            update(folder/'status.json',{'state':'ACTUAL_ZERO_VAL','completed_cells':len(rows),'time':time.time()})
            del model,data
    return {'epoch_minus1_actual_all49_VAL_rows':rows,'reconstructed_legacy_RNG_and_global_batch_rows':rng_rows}

def main(device,attempt):
    assert torch.cuda.is_available();disk_guard()
    assert attempt.isalnum();folder=ROOT/f'engineering_{attempt}';folder.mkdir(exist_ok=False)
    started=time.time();synthetic=synthetic_checks(device);dump(folder/'SYNTHETIC_GRADIENT_STATE.json',synthetic)
    actual=actual_checks(device,folder);dump(folder/'ACTUAL_ZERO_VAL_RNG.json',actual)
    # Direct historical runtime RNG/backend snapshots were not persisted by FT.
    # Current-source reconstruction passes, but choose common implementation
    # for all 18 cells BEFORE any new scientific fit, irrespective of outcomes.
    reuse={'legacy_FT_reused':False,'new_fits':18,'decision_basis':'historical per-device runtime RNG/backend snapshots absent; reconstructed legacy equivalence is engineering evidence, not direct provenance of historical stochastic streams; use common round02 implementation for all six cells',
        'decision_before_any_new_fit':not (ROOT/'fits').exists(),'old_FT_preserved_as_historical_reference':True,
        'legacy_checkpoint_recorded_keys':['model','seed','epoch','validation_loss'],'score_triggered':False}
    assert reuse['decision_before_any_new_fit']
    dump(ROOT/'LEGACY_FT_REUSE_DECISION.json',reuse)
    sources={str(p):sha(p) for p in sorted((ROOT/'input').glob('pub_01_s03_ss07_warm*.py'))}
    dump(ROOT/'ENGINEERING_COMPLETED.json',{'passed':True,'time':time.time(),'seconds':time.time()-started,
        'actual_VAL_cells':len(actual['epoch_minus1_actual_all49_VAL_rows']),'actual_VAL_windows_per_cell':30500,
        'synthetic_optimizer_steps_disclosed':18,'real_data_optimizer_steps':0,'test_read':False,
        'reuse_decision_sha256':sha(ROOT/'LEGACY_FT_REUSE_DECISION.json'),'sources':sources,
        'proofs':{str(folder/n):sha(folder/n) for n in ['SYNTHETIC_GRADIENT_STATE.json','ACTUAL_ZERO_VAL_RNG.json']},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'precision':precision(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}})
    print('WARM_ENGINEERING_PASS',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--attempt',required=True);a=p.parse_args();main(a.device,a.attempt)
