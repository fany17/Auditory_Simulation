"""Bounded actual D26 engineering and append-only protocol preparation."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

import pub_01_s03_ss07_contrastive_history_data as data
import pub_01_s03_ss07_contrastive_history_model as model
import pub_01_s03_ss07_contrastive_history_readout as readout
import pub_01_s03_ss07_contrastive_history_train as train

CONFIG=data.read(data.NATIVE_CONFIG)


def seed_net(seed,route):
    train.configure(seed)
    return model.CL(CONFIG,seed,route).cuda()


def fresh_digest(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def objective_gradient(net):
    return {k:float(t.grad.norm()) for k,t in net.inspect.items()}


def apply_one(loss,opt):
    data.check(torch.isfinite(loss).item(),"Finite bounded objective")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def engineering():
    inv=data.read(train.INVENTORY)
    wall,cpu=time.monotonic(),time.process_time()
    seed=1024
    tr=data.Coordinates("train",inv,audio=True)
    selec=list(tr.selections(seed,True))
    h,c=(seed_net(seed,route) for route in ("H","C"))
    h_state,c_state=(fresh_digest(net) for net in (h,c))
    data.check(h_state.keys()==c_state.keys() and all(torch.equal(v,c_state[k]) for k,v in h_state.items()),
               "Paired exact initial state")
    data.check(len(list(h.parameters()))==len({id(v) for v in h.parameters()}),"No duplicated parameters")
    opt={role:torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01) for role,net in (("H",h),("C",c))}
    prepared_rows=tr.cl_rows
    check_batch=tr.contrastive(selec[0],"H")
    prepared,mask,past,audio=check_batch
    ids=selec[0][2]
    current_bridge=float(np.max(np.abs(prepared[:8]-tr.windows_at(tr.cl_records[ids[:8]],tr.cl_starts[ids[:8]]))))
    data.check(current_bridge==0,"Raw normalization equals original prepared windows")
    features=[row for row in inv["features"] if row["story"] in {r["story"] for r in tr.rows}]
    loader=model.native.StoryBatches(data.PREPARED,features,"train",128)
    original_x,original_audio=next(loader.batches(seed,True))
    data.check(np.array_equal(original_x.numpy(),prepared) and np.array_equal(original_audio.numpy(),audio)
               and len(set(selec[0][1]))==128,"Actual native negative/time/audio parity")
    valid_ids=ids[mask]
    for offset,slot in zip(data.NEAR,past):
        j=valid_ids[0];row=tr.rows[tr.cl_records[j]]
        data.check(prepared_rows[j]["record_id"]==row["record_id"]
                   and np.array_equal(slot[0],tr.eeg(tr.cl_records[j])[tr.cl_starts[j]+offset:tr.cl_starts[j]+offset+192]),
                   "Same-record real historical slot")
    base=model.native.TransformerCLIP(CONFIG).cuda().eval()
    base.eeg.load_state_dict(h.eeg.backbone.state_dict())
    base.audio.load_state_dict(h.audio.state_dict())
    base.log_scale.data.copy_(h.log_scale)
    h.eval();c.eval()
    with torch.no_grad():
        audio_tensor=torch.from_numpy(audio).cuda()
        z0,f0=base(torch.from_numpy(prepared).cuda(),audio_tensor)
        loss0=model.native.contrastive(base,torch.from_numpy(prepared).cuda(),audio_tensor)
        eq={}
        for role,net in (("H",h),("C",c)):
            x,m,p,a=tr.contrastive(selec[0],role)
            packed=train.pack(x,m,p)
            z,f=net(packed,audio_tensor)
            loss=model.native.contrastive(net,packed,audio_tensor)
            eq[role]=dict(EEG_max=float((z-z0).abs().max()),audio_max=float((f-f0).abs().max()),
                          CE_abs=float((loss-loss0).abs()))
            data.check(max(eq[role].values())<=1e-7,"Zero-memory native eval equivalence")
    del base
    cl={}
    for role,net in (("H",h),("C",c)):
        net.train()
        values=[]
        for n in range(2):
            net.inspect={}
            loss=train.cl_step(net,tr.contrastive(selec[n],role))
            apply_one(loss,opt[role])
            grad=objective_gradient(net)
            grad["out_proj"]=float(net.eeg.memory.attention.out_proj.weight.grad.norm())
            data.check(all(np.isfinite(list(grad.values()))) and grad["query"]>0 and grad["audio"]>0
                       and grad["out_proj"]>0,"Actual shared graph gradients")
            if n==0:
                data.check(grad["slot1"]==grad["slot2"]==0,"Zero gate blocks slot gradients on step one")
            else:
                data.check(min(grad["slot1"],grad["slot2"])>0,"Both history slot gradients activated")
            values.append(grad)
        net.inspect=None
        cl[role]=values
    supervised={}
    first=tr.rows[0]["windows"]-13
    last=first+64
    data.check(len(set(tr.record_ids[first:last]))==2,"Actual mixed-record original batch")
    check_y=np.load(tr.rows[1]["envelope_path"],mmap_mode="r",allow_pickle=False).reshape(-1)[:192]
    data.check(np.array_equal(np.asarray(check_y,np.float32),tr.y[tr.rows[0]["windows"],0]),"Raw target bridge")
    for arm in data.ARMS:
        source=h if arm[0]=="H" else c
        # Independent encoder copy so the CH fixture cannot mutate the CC starting state.
        import copy
        encoder=copy.deepcopy(source.eeg).cuda().eval()
        head=model.make_head(seed).cuda()
        decoder=model.Decoder(encoder,head,arm[1])
        packed,y=train.task_grid(tr,first,last,arm[1])
        with torch.no_grad():
            z=encoder(packed,arm[1]).transpose(1,2)
            x,valid,past=packed
            query=encoder.query(x)
            local=encoder.backbone.output(query)
            output=encoder.project(query,x,valid,past,arm[1])
            data.check(torch.equal(output[~valid],local[~valid]) and (~valid).any(),"Native local prefix")
            cache_error=float((head(z)-decoder(packed)).abs().max())
            data.check(cache_error<=1e-7,"Actual cached versus raw prediction")
            if arm=="HH":
                ids,starts=tr.record_ids[first:last],tr.starts[first:last]
                far=tr.past(ids,starts,data.FAR)
                xfar=tuple(torch.from_numpy(v).cuda() for v in far)
                far_z=encoder.project(query,x,valid,xfar,"H")
                copy_z=encoder.project(query,x,valid,None,"C")
                data.check(torch.equal(far_z[~valid],local[~valid]) and torch.equal(copy_z[~valid],local[~valid]),
                           "One actual cached current query and identical local prefix")
        apply_one(model.native_ft.objective(head(z),y),torch.optim.Adam(head.parameters(),lr=.001))
        decoder.train()
        opt_ft=torch.optim.Adam([dict(params=encoder.parameters(),lr=1e-5),
                                 dict(params=head.parameters(),lr=1e-3)],weight_decay=0)
        apply_one(model.native_ft.objective(decoder(packed),y),opt_ft)
        supervised[arm]=dict(cache_error=cache_error,head_updates=1,FT_updates=1,
                             real_target_windows=64,prefix_rows=int((~valid).sum()))
    data.save(data.ROOT/"NEURAL_ENGINEERING_V02.json",dict(passed=True,process=train.identity(),
        paired_initial_state=True,native_eval_equivalence=eq,native_window_bridge=current_bridge,
        CL_updates=dict(H=2,C=2),gradients=cl,supervised=supervised,
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),CPU_seconds=time.process_time()-cpu,
        elapsed_seconds=time.monotonic()-wall,old_test_reads=0))
    train.guard()


def ccap_fixture():
    # Real legal D8 TRAIN cache slices exercise the already-vetted ridge helper.
    # No old checkpoint enters a D26 model. This is one engineering fixture.
    started,cpu=time.monotonic(),time.process_time()
    tr=data.Coordinates("train",data.read(train.INVENTORY))
    d8=data.HOME/"s03_finetuned_readout_transfer_20261101/seed1024"
    z=np.load(d8/"train_latent.npy",mmap_mode="r",allow_pickle=False)
    p=np.load(d8/"train_P.npy",mmap_mode="r",allow_pickle=False)
    rows=tr.rows[:12]
    limit=sum(r["windows"] for r in rows)
    targets={r["record"]:np.load(r["envelope_path"],mmap_mode="r",allow_pickle=False).reshape(-1)[:r["common_samples"]] for r in rows}
    fit,stats=model.ridge.readout(rows,z[:limit],p[:limit],targets,model.ridge_helpers)
    a,corrected=readout.apply_readout(z[:rows[0]["windows"]],p[:rows[0]["windows"]],fit)
    wave,keep=model.metrics.fuse(corrected,rows[0]["common_samples"])
    data.check(np.array_equal(a[:4],corrected[:4]) and keep.sum()==(rows[0]["windows"]-1)*96+190
               and np.isfinite(wave).all(),"TRAIN-only Ccap fallback and positive Hann support")
    data.save(data.ROOT/"CCAP_ENGINEERING_V02.json",dict(passed=True,process=train.identity(),
              TRAIN_records=len(rows),fixture_windows=limit,fit=stats,affine=fit[-1],
              elapsed_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
              new_neural_updates=0,new_neural_forwards=0,engineering_ridge_fits=1,
              first4_fallback_equal=True,positive_Hann_support=int(keep.sum()),old_test_reads=0))


def record_v01():
    old=data.ROOT/"ENGINEERING_V01.log"
    source=(data.ROOT/"input_v01/pub_01_s03_ss07_contrastive_history_engineering.py")
    data.check("boolean index did not match indexed array" in old.read_text(),"Exact V01 exception")
    data.save(data.ROOT/"NEURAL_V01_REACHED.json",dict(passed_neural_assertions=True,
        wall_seconds=None,CPU_seconds=None,loss_or_grad_numeric_capture=None,
        exact_completed_updates=dict(CL_H=2,CL_C=2,HEAD_HH=1,HEAD_CH=1,HEAD_CC=1,FT_HH=1,FT_CH=1,FT_CC=1),
        reached="first TRAIN Ccap affine record; neural assertions were earlier",
        source_sha256=data.digest(source),log_sha256=data.digest(old),
        failed_Ccap_before_solution=True,engineering_ridge_fits=0))




def gradients_only():
    start,cpu=time.monotonic(),time.process_time()
    seed=1024
    tr=data.Coordinates("train",data.read(train.INVENTORY),audio=True)
    chosen=list(tr.selections(seed,True))[:2]
    results={}
    state=None
    for role in ("H","C"):
        net=seed_net(seed,role).train()
        current=fresh_digest(net)
        if state is None:
            state=current
        else:
            data.check(all(torch.equal(value,current[name]) for name,value in state.items()),"Shared seed initialization")
        opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
        measurements=[]
        for index,selection in enumerate(chosen):
            net.inspect={}
            batch=tr.contrastive(selection,role)
            loss=train.cl_step(net,batch)
            apply_one(loss,opt)
            grad=objective_gradient(net)
            grad["out_proj"]=float(net.eeg.memory.attention.out_proj.weight.grad.norm())
            data.check(all(np.isfinite(list(grad.values()))) and min(grad["query"],grad["audio"],grad["out_proj"])>0,
                       "Actual query/audio/output gradients")
            data.check((grad["slot1"]==grad["slot2"]==0) if index==0 else
                       min(grad["slot1"],grad["slot2"])>0,"Both actual slot gradients")
            row=dict(step=index+1,current_windows=128,audio_windows=128,
                     slot_encoder_windows=2*int(batch[1].sum()),gradients=grad)
            data.save(data.ROOT/f"GRADIENT_V02_{role}_STEP{index+1}.json",row)
            measurements.append(row)
        results[role]=measurements
        del net,opt
    data.save(data.ROOT/"GRADIENT_ENGINEERING_V02.json",dict(passed=True,seed=seed,
        extra_CL_updates=4,extra_head_or_FT_updates=0,process=train.identity(),history=results,
        elapsed_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
        CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def prepare():
    data.ROOT.mkdir(exist_ok=True)
    inv=data.inventory()
    data.save(train.INVENTORY,inv)
    compact={k:inv[k] for k in ("counts","allowed_phases","VAL_people","VAL_blocks","VAL_shared_family_records")}
    compact.update(pinned_payload_stats=len(inv["inputs"]),inventory_sha256=data.digest(train.INVENTORY))
    data.save(data.ROOT/"INPUT_RECEIPT.json",compact)


def freeze():
    old=data.read(data.ROOT/"NEURAL_V01_REACHED.json")
    ccap=data.read(data.ROOT/"CCAP_ENGINEERING_V02.json")
    data.check(old["passed_neural_assertions"] and ccap["passed"],"Actual bounded engineering")
    numerical=data.ROOT/"GRADIENT_ENGINEERING_V02.json"
    data.check(numerical.is_file() and data.read(numerical)["passed"],"Actual numeric history gradients")
    combined=dict(passed=True,neural_v01=old,Ccap_v02=ccap,gradients_v02=data.read(numerical))
    if train.ENGINEERING.is_file():
        data.check(data.read(train.ENGINEERING)==combined,"Frozen engineering receipt")
    else:
        data.save(train.ENGINEERING,combined)
    inv=data.read(train.INVENTORY)
    data.check(all(data.stamp(path)==value for path,value in inv["inputs"].items()),"Frozen cohort stats")
    data.save(train.PROTOCOL,dict(frozen=True,inventory_sha256=data.digest(train.INVENTORY),
              budget_sha256=data.digest(data.ROOT/"BUDGET.json"),plan_sha256=data.digest(data.ROOT/"PLAN.md"),
              sources=train.static_pins(),engineering_sha256=data.digest(train.ENGINEERING),
              axes=dict(current=192,hop=96,ready=864,near=list(data.NEAR),far=list(data.FAR)),
              arms=list(data.ARMS),seeds=list(data.SEEDS),CL_pairs=3,CL_unique_jobs=6,
              head_jobs=9,FT_jobs=9,Ccap_jobs=9,CL_batch=128,CL_joint_patience=15,CL_epoch_cap=100,
              head_train_batch=64,head_val_batch=512,head_joint_patience=10,head_epoch_cap=250,
              FT_epochs=5,FT_final_epoch=4,lambda_fixed=.01,readout_prefix_windows=4,
              loss="native bidirectional CE and supervised float32 centered cosine eps1e-6",
              selection="final Ccap mean3 fixed epoch4; tie CC CH HH",
              caps=dict(writer_GiB=64,reserve_GiB=64,RSS_per_worker_GiB=32,CUDA_reserved_per_worker_GiB=40),
              source_family_counts=dict(shared_VAL=29,unseen_VAL=20),
              old_test_reads=0,exact_resume="saved model optimizer scheduler Torch CPU/CUDA RNG and batch cursor"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","engineer","ccap","record-v01","gradients","freeze"))
    stage=parser.parse_args().stage
    {"prepare":prepare,"engineer":engineering,"ccap":ccap_fixture,"record-v01":record_v01,"gradients":gradients_only,"freeze":freeze}[stage]()
