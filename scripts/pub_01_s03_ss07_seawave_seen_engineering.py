"""D27 bounded CPU-only engineering and final source freeze."""
import argparse
import importlib.util
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

import pub_01_s03_ss07_seawave_seen_data as data
import pub_01_s03_ss07_seawave_seen_model as model
import pub_01_s03_ss07_seawave_seen_train as train

ROOT=data.ROOT
ENGINEERING=ROOT/"ENGINEERING.json"


def prepare():
    inv=data.inventory()
    data.write(train.INVENTORY,inv)
    val=inv["phase_records"]["val"]
    sizes=[row["common_samples"] for row in val]
    calls=sum(math.ceil(n/data.CORE) for n in sizes)
    actual=sum(min(n,min(n,s+data.CORE)+data.HALO)-max(0,s-data.HALO)
               for n in sizes for s in range(0,n,data.CORE))
    personal=inv["personal"]
    budget=dict(
        initial=dict(inv["initial"],fits=3,epochs_per_fit=201,updates=3*201*inv["initial"]["updates"]),
        personal=dict(fits=147,subjects=49,cap_epochs_per_fit=21,
                      windows_per_all_people_pass=sum(v["windows"] for v in personal.values()),
                      unique_times_per_all_people_pass=sum(v["unique_record_sample_times"] for v in personal.values()),
                      target_sample_presentations_per_all_people_pass=sum(v["target_sample_presentations"] for v in personal.values()),
                      updates_per_all_people_pass=sum(v["updates"] for v in personal.values()),
                      updates_cap=3*21*sum(v["updates"] for v in personal.values()),per_subject=personal),
        validation=dict(unique_record_times_per_full_PASS=sum(sizes),calls_per_full_PASS=calls,
                        halo_input_times_per_full_PASS=actual,initial_full_passes=603,
                        personal_equivalent_full_passes_cap=66,
                        initial_calls_cap=603*calls,personal_calls_cap=66*calls),
        disk=dict(writer_GiB=16,reserve_GiB=64,physical_windows_on_disk=0),
        engineering_cap=dict(model_forward_batches=8,initial_updates=2,personal_updates=1,CUDA=0))
    data.write(ROOT/"BUDGET.json",budget)


def optim(net,stage,epoch_size):
    specs=dict(data.SETTINGS[stage],batch=2)
    opt=torch.optim.AdamW(net.parameters(),lr=specs["lr"],weight_decay=specs["wd"])
    return opt,model.scheduler(opt,specs,epoch_size)


def scheduler_resume():
    rows={}
    for stage in ("initial","personal"):
        before=torch.nn.Linear(2,1)
        opt,sched=optim(before,stage,5)
        after=torch.nn.Linear(2,1)
        newopt,newsched=optim(after,stage,5)
        path=ROOT/f"RESUME_CANARY_{stage}.pt"
        recorded=[]
        for epoch in range(14):
            sched.step()
            if epoch>=10:
                newsched.step()
                assert opt.param_groups[0]["lr"]==newopt.param_groups[0]["lr"]
                assert opt.param_groups[0]["weight_decay"]==newopt.param_groups[0]["weight_decay"]
            for batch in range(3):
                sched.batch_step()
                if (epoch,batch)==(9,0):
                    first_parameter=next(before.parameters())
                    opt.state[first_parameter]=dict(step=torch.tensor(28.),
                        exp_avg=torch.ones_like(first_parameter),exp_avg_sq=2*torch.ones_like(first_parameter))
                    train.save_boundary(path,dict(epoch=9,next_step=1),before,opt,sched,"cpu")
                    expected=(random.random(),float(np.random.random()),torch.rand(7))
                    payload=torch.load(path,map_location="cpu",weights_only=True)
                    train.recover(payload,after,newopt,newsched,"cpu")
                    got=(random.random(),float(np.random.random()),torch.rand(7))
                    assert got[:2]==expected[:2] and torch.equal(got[2],expected[2])
                    assert (payload["epoch"],payload["next_step"])==(9,1)
                    restored=newopt.state[next(after.parameters())]
                    assert restored["step"].item()==28 and torch.equal(restored["exp_avg_sq"],2*torch.ones_like(next(after.parameters())))
                elif epoch>=9:
                    newsched.batch_step()
                if epoch>=9:
                    assert opt.param_groups[0]["lr"]==newopt.param_groups[0]["lr"]
                    assert opt.param_groups[0]["weight_decay"]==newopt.param_groups[0]["weight_decay"]
                    recorded.append((epoch,batch,opt.param_groups[0]["lr"],opt.param_groups[0]["weight_decay"]))
        assert sched.restarts>=1 and model.scheduler_state(sched)==model.scheduler_state(newsched)
        rows[stage]=dict(restarts=sched.restarts,exact_lr_wd_continuation=True,
                         actual_Adam_moments_and_step_restored=True,checked_batch_positions=len(recorded),first_and_last=(recorded[0],recorded[-1]))
    return rows


def engineer():
    data.check(not torch.cuda.is_initialized(),"CPU only")
    train.configure(1024,2,"cpu")
    wall,cpu=time.monotonic(),time.process_time()
    inv=data.read(train.INVENTORY)
    rows=inv["phase_records"]["train"][:2]
    assert all(r["phase"]=="train" for r in rows)
    signals=data.Signals("cpu")
    initial_grid=data.Windows(rows,signals,dict(data.SETTINGS["initial"],batch=2))
    initial=model.model(.2)
    initial.train()
    parameters=sum(p.numel() for p in initial.parameters())
    assert initial.calRp(19,initial.dilationPerLayer[19][1],3)==249
    opt,sched=optim(initial,"initial",initial_grid.windows)
    sched.step()
    raw,truth=signals.raw(rows[0])
    x,y=initial_grid.batch(0)
    expected=((np.asarray(raw[:320],dtype=np.float32)-signals.mean)/signals.std).astype(np.float32)
    assert np.array_equal(x[0].numpy(),expected.T) and np.array_equal(y[0,0].numpy(),truth[:320].astype(np.float32))
    i=initial_grid.counts[0]//2
    xb,yb=initial_grid.batch(i)
    if initial_grid.counts[0]%2:
        assert np.array_equal(yb[1,0].numpy(),signals.raw(rows[1])[1][:320].astype(np.float32))
    train.step(initial,opt,sched,"initial",x,y)
    x,y=initial_grid.batch(1)
    train.step(initial,opt,sched,"initial",x,y)
    gradients={k:float(dict(initial.named_parameters())[k].grad.norm()) for k in
               ("init_conv.0.conv.weight_v","final_conv.2.conv.weight")}
    assert min(gradients.values())>0 and all(math.isfinite(v) for v in gradients.values())
    state=train.clone(initial)
    same=model.wave.WaveNet_regressor(64,32,32,1,20,5,.2)
    same.load_state_dict(state,strict=True)
    initial.eval();same.eval()
    full,_=signals.on_device(rows[0])
    with torch.no_grad():
        wrapped=initial(full[None,:,:320])
        direct=same(full[None,:,:320])
        exact=float((wrapped-direct).abs().max())
        assert exact==0 and float(wrapped.abs().max())>0
        engineering_n=4608
        wave=initial(full[None,:,:engineering_n])[0,0].numpy().astype(np.float64)
    halo_calls=[]
    chunked=model.continuous(initial,full[:,:engineering_n],call=halo_calls.append)
    halo_error=float(np.max(np.abs(wave-chunked)))
    assert len(halo_calls)==2 and halo_error<=1e-6
    row=dict(rows[0],common_samples=engineering_n)
    eligible=[dict(start=0,samples=3839,eligible=True)]
    direct_score=model.record_scores(row,wave,truth[:engineering_n],eligible)[0]
    path=data.HOME/"s03_current_readout_internal_audit_20261101/input/pub_01_s03_ss07_generalization_core.py"
    spec=importlib.util.spec_from_file_location("d27_old_score_canary",path)
    metric=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    windows=np.lib.stride_tricks.sliding_window_view(wave,192)[::96].copy()
    old=metric.measure(windows,truth[:engineering_n],row,eligible)["segments"][0]
    score_error={k:abs(old[k]-direct_score[k]) for k in ("r","MSE")}
    assert max(score_error.values())<=1e-12
    branches=[]
    for _ in range(2):
        train.configure(1024,2,"cpu")
        ft=model.model(.65)
        ft.load_state_dict(state,strict=True)
        fo,fs=optim(ft,"personal",5)
        branches.append((ft,fo,fs,torch.get_rng_state().clone()))
    a,ao,_,ar=branches[0];b,bo,_,br=branches[1]
    assert not ao.state and not bo.state and torch.equal(ar,br)
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in state)
    ft_grid=data.Windows(rows,signals,dict(data.SETTINGS["personal"],batch=2))
    fs=model.scheduler(ao,dict(data.SETTINGS["personal"],batch=2),ft_grid.windows)
    a.train()
    fs.step()
    fx,fy=ft_grid.batch(0)
    ft_loss=train.step(a,ao,fs,"personal",fx,fy)
    assert math.isfinite(ft_loss) and not bo.state
    assert all(torch.equal(b.state_dict()[k],v) and torch.equal(initial.state_dict()[k],v) for k,v in state.items())
    sched_result=scheduler_resume()
    train.guard("cpu")
    assert torch.get_num_threads()==2 and not torch.cuda.is_initialized()
    data.write(ENGINEERING,dict(passed=True,process=train.identity(),
        TRAIN_records=[r["record"] for r in rows],parameters=parameters,receptive_field=249,
        input_and_target_bridge=True,author_nonzero_state_max_abs=exact,gradient_norm_after_two_MSE_updates=gradients,
        halo_forward_calls=halo_calls,halo_max_abs=halo_error,shared_metric_abs=score_error,
        personal_independent_start=True,FT_objective=ft_loss,scheduler_resume=sched_result,
        updates=dict(initial=2,personal=1),model_forward_batches=dict(optimization=3,author_equivalence=2,halo_full=1,halo_parts=2),
        wall_seconds=time.monotonic()-wall,CPU_seconds=time.process_time()-cpu,CUDA=0,VAL_read=0,old_test_reads=0,
        notes="All three actual optimization states stayed in memory and are not formal assets; resume checkpoint used synthetic linear tensors."))


def freeze():
    inv=data.read(train.INVENTORY)
    data.check(data.read(ENGINEERING)["passed"],"Actual CPU-only bounded evidence")
    data.check(all(data.stamp(p)==v for p,v in inv["inputs"].items()),"Actual frozen inputs")
    data.write(train.GATE,dict(frozen=True,inventory_sha256=data.digest(train.INVENTORY),
        sources=train.pinned_sources(),budget_sha256=data.digest(ROOT/"BUDGET.json"),
        settings=data.SETTINGS,seeds=list(data.SEEDS),initial_fits=3,personal_fits=147,
        model=dict(in_channels=64,channels=32,out=1,layers=20,cycle=5,kernel=3,receptive_field=249),
        use_native_author_code=True,train_only_norm=str(data.NORM),VAL_mask=str(data.MASK),
        halo=dict(core=4096,left=124,right=124),VAL_people=49,VAL_blocks=752,
        no_population_global_FT=True,old_test_reads=0,
        resource_caps=dict(writer_GiB=16,reserve_GiB=64,RSS_worker_GiB=32,CUDA_worker_GiB=40),
        checkpoint_frequency_updates=2048,formal_requires_completed_D26_and_exited_workers=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","engineer","freeze"))
    {"prepare":prepare,"engineer":engineer,"freeze":freeze}[parser.parse_args().stage]()
