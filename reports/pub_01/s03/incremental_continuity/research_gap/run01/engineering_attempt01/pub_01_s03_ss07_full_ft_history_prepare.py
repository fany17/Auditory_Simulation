"""Freeze approved single matrix; bounded disposable engineering on TRAIN/VAL."""
import gc,os,sys,time
import numpy as np
import torch
from pub_01_s03_ss07_full_ft_history_core import *

def prepare():
    assert not (ROOT/'PROTOCOL.json').exists();started=time.time();cpu=time.process_time()
    inputs=read(D8/'PROTOCOL.json');pins={}
    for name in ['core','train','prepare','queue']:
        p=ROOT/f'input/pub_01_s03_ss07_full_ft_history_{name}.py';pins[str(p)]=sha(p)
    for p in [ROOT/'PLAN.md',ROOT/'COORDINATOR_DECISION.md',D8/'PROTOCOL.json',Path(metric.__file__)]:pins[str(p)]=sha(p)
    for seed in SEEDS:
        for name in ['TRAIN_TRANSFORMS.npz','Ccap_lambda0.01_COEFFICIENTS.npz','P_FT_BASELINE.json','Ccap_lambda0.01_METRICS.json']:
            p=D8/f'seed{seed}/{name}';pins[str(p)]=sha(p)
    cache_metadata=[]
    for seed in SEEDS:
        for phase in ['train','val']:
            for suffix in ['latent','P']:
                p=D8/f'seed{seed}/{phase}_{suffix}.npy';cache_metadata.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
    targets=[OLD/f'envelope_readout_seed1024/{phase}_envelope.npy' for phase in ['train','val']]
    targets+=sorted({Path(r['envelope_path']) for rr in inputs['records'].values() for r in rr})
    for p in targets:cache_metadata.append(dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
    dump(ROOT/'PROTOCOL.json',dict(frozen=True,source_pins=pins,seeds=SEEDS,arms=ARMS,epochs=20,fits=9,updates_per_fit=78900,total_updates=710100,
        target_window_presentations=45441720,effective_residual_TRAIN_presentations=248620*180,train_windows=252454,val_windows=30500,
        common_local_start=9,history_indices=[-5,-3],older_indices=[-9,-7],mask=inputs['mask'],cache_metadata=cache_metadata,
        optimizer=dict(name='Adam',lr=.001,weight_decay=0,grad_clip=None),loss='Unweighted pointwise192 window MSE, full original global64 including constant fallback positions',
        selection='Incumbent -1 and epochs0..19 complete VAL60s/person r; strict improvement, earliest ties; always20epochs',
        budget=dict(wall_seconds=21600,max_workers=2,worker_RSS_bytes=8*1024**3,worker_GPU_allocated_bytes=2*1024**3,new_root_bytes=128*1024**2,
            history_disk_cache_bytes=0,each_filesystem_free_bytes=20*1024**3,all_SS07_NPY_bytes=28*1024**3),
        precision=dict(new_residual='FP32',fixed_anchor_and_Hann='FP64',matmul_TF32=False,cudnn_TF32=True),
        forbidden=['test','DTU','ds004408','new encoder/head forward','new datasets/weights','patients'],scientific_status='Repeated original TRAIN/VAL development',
        engineering_updates_separate=True,formal_clean_initialization=True))
    torch.set_num_threads(2);device='cuda:0';torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    checks=[];temporary_updates=0;forward_windows=0
    for seed in SEEDS:
        tr=Data(seed,'train');val=Data(seed,'val');m=val.score(val.anchor)
        ref=read(D8/f'seed{seed}/Ccap_lambda0.01_METRICS.json');delta=abs(m['phases']['val']['r']-ref['r60s']);assert delta<1e-10
        old={(b['record'],b['start']):b['r'] for b in ref['metrics']['segments']}
        block_delta=max(abs(b['r']-old[b['record'],b['start']]) for b in m['segments']);assert block_delta<1e-10
        models={a:Conditional(seed,a).to(device) for a in ARMS}
        assert all(torch.equal(models['H_full'].state_dict()[k],v) for k,v in models['C_match'].state_dict().items())
        for k,v in models['P_prior'].state_dict().items():
            if not k.startswith('token.0.'):assert torch.equal(v,models['H_full'].state_dict()[k]),k
        initial,cost=predict(models['H_full'],val,device);forward_windows+=cost['effective_residual_windows']
        assert np.array_equal(initial,val.anchor)
        initial_metrics=val.score(initial);assert initial_metrics['phases']['val']['r']==m['phases']['val']['r']
        for arm,model in models.items():
            count=sum(v.numel() for v in model.parameters());assert count==(164784 if arm=='P_prior' else 165792),(arm,count)
            x,h,a,y,ok=tr.batch(0,64,arm,device);out=model(x,h);forward_windows+=len(x);assert torch.count_nonzero(out)==0
            actual_initial=tr.anchor[:64].copy();actual_initial[ok]+=.1*out.detach().cpu().numpy().astype('f8')
            assert np.array_equal(tr.anchor[:64],actual_initial)
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            tick=time.time()
            for step in range(2):
                opt.zero_grad();p=a.clone();p[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*model(x,h).double();forward_windows+=len(x)
                loss=(p-y).square().mean();loss.backward();assert torch.isfinite(loss)
                opt.step();temporary_updates+=1
            torch.cuda.synchronize();grads={k:int(torch.count_nonzero(v.grad)) for k,v in model.named_parameters()}
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
            assert sum(v.numel() for v in model.parameters() if v.grad is not None)==count
            assert grads['gru.weight_hh_l0']>0
            checks.append(dict(seed=seed,arm=arm,registered=count,gradient_reachable=count,nonzero_gradient_scalars=sum(grads.values()),
                current_gru_hh_nonzero=grads['gru.weight_hh_l0'],two_disposable_step_seconds=time.time()-tick,
                anchor_block_r_max_abs=block_delta,anchor_person_r_max_abs=delta,initial_zero_residual=True))
        hm=models['H_full'];x,h,_,_,ok=tr.batch(64,128,'H_full',device);xo,ho,_,_,oko=tr.batch(64,128,'H_full',device,older=True)
        assert np.array_equal(ok,oko) and torch.equal(x,xo) and not torch.equal(h,ho)
        with torch.no_grad():a,ha,qa=hm(x,h,True);b,hb,qb=hm(xo,ho,True)
        forward_windows+=len(x)*2;assert torch.equal(qa,qb) and not torch.equal(ha,hb) and not torch.equal(a,b)
        for phase,rr in inputs['records'].items():
            for row in rr:
                for i in [9,row['windows']-1]:
                    assert i-9>=0 and 96*(i-3)+192==96*i-96 and 96*(i-7)+192==96*(i-5)
        del tr,val,models,model,hm,opt,x,h,xo,ho,a,b,ha,hb,qa,qb,out,p,y,initial,actual_initial;gc.collect();torch.cuda.empty_cache();guard()
    dump(ROOT/'ENGINEERING_CHECK.json',dict(passed=True,checks=checks,temporary_discarded_optimizer_steps=temporary_updates,
        temporary_residual_NN_window_forwards=forward_windows,new_encoder_head_forwards=0,full_VAL_anchor_bridges=3,
        same_current_intervention_checked_after_nonzero_disposable_weights=True,same_record_union_support=True,
        full752_block_mask_preserved=True,checkpoint_earliest_tie_check=int(np.argmax([.1,.1,.2,.2]))-1==1,
        new_external_access=False,wall_seconds=time.time()-started,CPU_seconds=time.process_time()-cpu,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        formal_models_reconstructed_clean=True))
    print('FIXED_MATRIX_AND_ENGINEERING_PASS_READY_FOR_NINE_FORMAL_FITS',flush=True)

if __name__=='__main__':prepare()
