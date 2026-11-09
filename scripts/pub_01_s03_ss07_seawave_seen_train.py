"""D27 original whole-record validation and exact boundary resume."""
import argparse
import json
import math
import os
import random
import resource
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import pub_01_s03_ss07_seawave_seen_data as data
import pub_01_s03_ss07_seawave_seen_model as source

GIB=1024**3
GATE=data.ROOT/"PROTOCOL_V04.json"
INVENTORY=data.ROOT/"INVENTORY.json"
FORMAL=data.ROOT/"formal"
WORKER_START=time.monotonic()
WORKER_CPU=time.process_time()


def identity():
    fields=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
    return dict(pid=os.getpid(),uid=os.getuid(),startticks=int(fields[19]),gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))


def configure(seed,threads,device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    if device=="cuda":
        data.check(torch.cuda.device_count()==1 and (os.environ.get("CUDA_VISIBLE_DEVICES") or "").startswith("GPU-"),
                   "Single named GPU")
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()


def guard(device):
    data.check(shutil.disk_usage(data.ROOT).free>=64*GIB,"SATA free reserve")
    data.check(sum(p.stat().st_size for p in data.ROOT.rglob("*") if p.is_file())<=16*GIB,"D27 root cap")
    cap=32 if device=="cuda" else 4
    data.check(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=cap*GIB,"Worker RSS")
    if device=="cuda":
        data.check(torch.cuda.max_memory_reserved()<=40*GIB,"Worker CUDA reserved")


def pinned_sources():
    paths=set(Path(__file__).parent.glob("pub_01_s03_ss07_seawave_seen_*.py"))
    paths.update(source.AUTHOR/name for name in source.AUTHOR_HASHES)
    paths.update([data.ROOT/"PLAN_V04.md",data.ROOT/"COORDINATOR_DECISION.md",data.MASK,
                  data.ROOT/"GEOMETRY_RECEIPT.json",data.ROOT/"BUDGET.json",data.ROOT/"ENGINEERING.json",
                  data.ROOT/"QUEUE_ENGINEERING_V04.json"])
    return {str(p.resolve()):data.digest(p) for p in sorted(paths)}


def admit():
    protocol=data.read(GATE)
    data.check(protocol["frozen"] and protocol["inventory_sha256"]==data.digest(INVENTORY)
               and protocol["sources"]==pinned_sources(),"Frozen exact source chain")
    inv=data.read(INVENTORY)
    for path,expected in inv["inputs"].items():
        data.check(data.stamp(path)==expected,f"Frozen public input: {path}")
    return inv,protocol


def snapshot_rng(device):
    name,keys,position,has_gauss,cached=np.random.get_state()
    return dict(python=random.getstate(),numpy=(name,torch.from_numpy(keys.astype(np.int64)),position,
                                               has_gauss,cached),torch=torch.get_rng_state(),
                CUDA=torch.cuda.get_rng_state() if device=="cuda" else None)


def restore_rng(saved,device):
    random.setstate(saved["python"])
    name,keys,position,has_gauss,cached=saved["numpy"]
    np.random.set_state((name,keys.numpy().astype(np.uint32),position,has_gauss,cached))
    torch.set_rng_state(saved["torch"])
    if device=="cuda":
        torch.cuda.set_rng_state(saved["CUDA"])


def atomic_torch(path,value):
    temporary=path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with temporary.open("xb") as stream:
        torch.save(value,stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def immutable(path,value):
    if path.exists():
        data.check(data.read(path)==value,"Append-only score receipt")
        return
    tmp=path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    data.write(tmp,value)
    tmp.rename(path)


def clone(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def recover(value,net,opt,sched,device):
    net.load_state_dict(value["model"],strict=True)
    opt.load_state_dict(value["optimizer"])
    source.restore_scheduler(sched,value["scheduler"])
    restore_rng(value["rng"],device)


def save_boundary(path,state,net,opt,sched,device):
    payload=dict(state,model=net.state_dict(),optimizer=opt.state_dict(),
                 scheduler=source.scheduler_state(sched),rng=snapshot_rng(device))
    atomic_torch(path,payload)


def validation(net,signals,rows,masks,shared_families):
    blocks=[]
    calls=frames=0
    def observe(n):
        nonlocal calls,frames
        calls+=1
        frames+=n
    for row in rows:
        x,_=signals.on_device(row)
        _,y=signals.raw(row)
        prediction=source.continuous(net,x,call=observe)
        blocks.extend(source.record_scores(row,prediction,y,masks[row["record"]]))
    result=source.aggregate(blocks)
    result.update(VAL_model_calls=calls,VAL_model_input_sample_times=frames,
                  VAL_unique_sample_times=sum(r["common_samples"] for r in rows))
    result["families"]={name:summary for name,selected in (
        ("shared",[b for b in blocks if b["source_family"] in shared_families]),
        ("unseen",[b for b in blocks if b["source_family"] not in shared_families]))
        if selected for summary in [{k:v for k,v in source.aggregate(selected).items() if k!="blocks"}]}
    return result


def step(net,opt,sched,stage,x,y):
    opt.zero_grad(set_to_none=True)
    loss=source.objective(stage,net(x),y)
    data.check(torch.isfinite(loss).item(),"Finite supervised objective")
    loss.backward()
    opt.step()
    sched.batch_step()
    return float(loss.detach())


def fit(stage,seed,subject,resume):
    inv,_=admit()
    device="cuda"
    configure(seed,4,device)
    guard(device)
    specs=data.SETTINGS[stage]
    check_seed=FORMAL/f"seed{seed}"
    folder=check_seed/"initial" if stage=="initial" else check_seed/"personal"/subject
    folder.mkdir(parents=True,exist_ok=resume)
    signals=data.Signals(device)
    all_train=inv["phase_records"]["train"]
    all_val=inv["phase_records"]["val"]
    train=all_train if stage=="initial" else [r for r in all_train if r["subject"]==subject]
    val=all_val if stage=="initial" else [r for r in all_val if r["subject"]==subject]
    grid=data.Windows(train,signals,specs)
    geo=inv["initial"] if stage=="initial" else inv["personal"][subject]
    data.check((grid.windows,grid.batches)==(geo["windows"],geo["updates"]),"Exact full epoch")
    expected_val=sum(b["eligible"] for b in data.read(data.MASK)["blocks"]
                     if b["record"] in {r["record"] for r in val})
    data.check((stage=="initial" and len(val)==49 and expected_val==752)
               or (stage=="personal" and len(val)==1 and expected_val>0),"Original score cohort")
    masks={r["record"]:[] for r in val}
    for block in data.read(data.MASK)["blocks"]:
        if block["record"] in masks:
            masks[block["record"]].append(block)
    net=source.model(specs["dropout"]).cuda()
    initial_sha=None
    if stage=="personal":
        base=check_seed/"initial"
        recorded=data.read(base/"COMPLETED.json")
        checkpoint=base/"SELECTED.pt"
        initial_sha=data.digest(checkpoint)
        data.check(recorded["passed"] and recorded["selected_sha256"]==initial_sha,"Shared initial source")
        net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
        data.check(all(m.p==.65 for m in net.modules() if isinstance(m,torch.nn.Dropout)),"Published FT dropout")
    opt=torch.optim.AdamW(net.parameters(),lr=specs["lr"],weight_decay=specs["wd"])
    sched=source.scheduler(opt,specs,grid.windows)
    state=dict(protocol_sha256=data.digest(GATE),seed=seed,stage=stage,subject=subject,
               initial_sha256=initial_sha,epoch=0,next_step=0,updates=0,partial_loss=0.,
               partial_windows=0,last_batch=0,history=[],best=None,best_score=-math.inf,best_epoch=None,
               best_metrics=None,stale=0,wall_prior=0.,CPU_prior=0.,VAL_calls=0,VAL_model_points=0,
               previous_peak_cuda=0)
    file=folder/"CURRENT.pt"
    if resume and file.exists():
        payload=torch.load(file,map_location="cpu",weights_only=True)
        data.check(all(payload[k]==state[k] for k in ("protocol_sha256","seed","stage","subject","initial_sha256")),
                   "Exact resume identity")
        recover(payload,net,opt,sched,device)
        state.update({k:payload[k] for k in state})
        for row in state["history"]:
            immutable(folder/f"EPOCH_{row['epoch']:03}.json",row)
    elif resume:
        data.check(not list(folder.glob("EPOCH_*.json")),"Existing epoch receipts without a saved boundary")
    entry_wall,entry_cpu=time.monotonic(),time.process_time()
    prior_wall,prior_cpu=state["wall_prior"],state["CPU_prior"]
    def persist():
        state["wall_prior"]=prior_wall+time.monotonic()-entry_wall
        state["CPU_prior"]=prior_cpu+time.process_time()-entry_cpu
        state["previous_peak_cuda"]=max(state["previous_peak_cuda"],torch.cuda.max_memory_reserved())
        save_boundary(file,state,net,opt,sched,device)
    def score():
        current=validation(net,signals,val,masks,set(inv["train_source_families"]))
        data.check(len(current["people"])==len(val) and len(current["blocks"])==expected_val,
                   "Fixed validation eligibility")
        state["VAL_calls"]+=current["VAL_model_calls"]
        state["VAL_model_points"]+=current["VAL_model_input_sample_times"]
        return current
    if stage=="personal" and state["best_epoch"] is None:
        result=score()
        state.update(best=clone(net),best_epoch=-1,best_score=result["r"],best_metrics=result)
        persist()
        immutable(folder/"EPOCH_MINUS1.json",dict(r=result["r"],MSE=result["MSE"],epoch=-1,
                                                  initial_sha256=initial_sha,updates=0))
    elif stage=="personal" and state["best_epoch"]==-1 and not state["history"]:
        result=state["best_metrics"]
        immutable(folder/"EPOCH_MINUS1.json",dict(r=result["r"],MSE=result["MSE"],epoch=-1,
                                                  initial_sha256=initial_sha,updates=0))
    cap=specs["epochs"]
    while state["epoch"]<cap and (stage=="initial" or state["stale"]<specs["patience"]):
        epoch=state["epoch"]
        net.train()
        if state["next_step"]==0:
            sched.step()
        for i in range(state["next_step"],grid.batches):
            x,y=grid.batch(i)
            loss=step(net,opt,sched,stage,x,y)
            state["partial_loss"]+=loss*len(x)
            state["partial_windows"]+=len(x)
            state["updates"]+=1
            state["next_step"]=i+1
            state["last_batch"]=len(x)
            if state["next_step"]%2048==0:
                guard(device)
                persist()
        data.check(state["partial_windows"]==grid.windows and state["next_step"]==grid.batches
                   and state["last_batch"]==geo["tail"],"Actual full sequential pass and tail")
        result=score()
        if result["r"]>state["best_score"]:
            state.update(best=clone(net),best_epoch=epoch,best_score=result["r"],best_metrics=result,stale=0)
        else:
            state["stale"]+=1
        row=dict(epoch=epoch,r=result["r"],MSE=result["MSE"],windows=grid.windows,updates=state["updates"],
                 train_objective=state["partial_loss"]/grid.windows,stale=state["stale"],
                 VAL_model_calls=result["VAL_model_calls"],VAL_model_input_sample_times=result["VAL_model_input_sample_times"])
        state["history"].append(row)
        state.update(epoch=epoch+1,next_step=0,partial_windows=0,partial_loss=0.,last_batch=0)
        persist()
        immutable(folder/f"EPOCH_{epoch:03}.json",row)
        print(json.dumps(dict(stage=stage,seed=seed,subject=subject,**row)),flush=True)
        guard(device)
    for name,value in (("SELECTED.pt",state["best"]),("FINAL.pt",clone(net))):
        path=folder/name
        if path.exists():
            prior=torch.load(path,map_location="cpu",weights_only=True)
            data.check(prior.keys()==value.keys() and all(torch.equal(prior[k],v) for k,v in value.items()),
                       "Existing final weight evidence must match committed state")
        else:
            atomic_torch(path,value)
    immutable(folder/"SELECTED_VAL.json",state["best_metrics"])
    immutable(folder/"COMPLETED.json",dict(
        passed=True,seed=seed,stage=stage,subject=subject,selected_epoch=state["best_epoch"],
        selected_r=state["best_score"],selected_sha256=data.digest(folder/"SELECTED.pt"),
        selected_prefix_updates=(state["best_epoch"]+1)*grid.batches,initial_sha256=initial_sha,
        complete_epochs=len(state["history"]),actual_updates=state["updates"],geometry=geo,
        windows=grid.windows*len(state["history"]),VAL_model_calls=state["VAL_calls"],
        VAL_model_input_sample_times=state["VAL_model_points"],CPU_seconds=state["CPU_prior"],
        elapsed_seconds=state["wall_prior"],peak_cuda_reserved=state["previous_peak_cuda"],
        process=identity(),parameters=sum(v.numel() for v in net.parameters()),old_test_reads=0,
        population_global_FT=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("initial","personal"))
    parser.add_argument("--seed",required=True,type=int,choices=data.SEEDS)
    parser.add_argument("--subject")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    try:
        data.check((args.stage=="personal")==bool(args.subject),"Explicit subject for personal stage")
        fit(args.stage,args.seed,args.subject,args.resume)
    except BaseException as error:
        data.write(data.ROOT/f"FAILURE_{args.stage}_{args.seed}_{os.getpid()}.json",
                   dict(message=str(error),process=identity(),traceback=traceback.format_exc(),
                        uncommitted_training_updates=None,checkpoint_interval_updates=2048,
                        note="Actual uncommitted attempts are unknown. At most 2048 completed updates can be between two committed boundaries."))
        raise
    finally:
        print(json.dumps(dict(event="worker_timing",process=identity(),
                   wall_seconds=time.monotonic()-WORKER_START,CPU_seconds=time.process_time()-WORKER_CPU)),flush=True)
