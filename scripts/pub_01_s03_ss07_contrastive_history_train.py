"""D26 resumable native contrastive, matched head, and fixed joint FT stages."""
import argparse
import json
import math
import os
import resource
import shutil
import sys
import time

WORKER_START = time.monotonic()
WORKER_CPU = time.process_time()
import traceback
from pathlib import Path

import numpy as np
import torch

import pub_01_s03_ss07_contrastive_history_data as data
import pub_01_s03_ss07_contrastive_history_model as model

GIB = 1024**3
PROTOCOL = data.ROOT/"PROTOCOL_V05.json"
ENGINEERING = data.ROOT/"ENGINEERING.json"
INVENTORY = data.ROOT/"INPUTS.json"
FORMAL = data.ROOT/"formal"


def identity():
    fields = Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
    return dict(pid=os.getpid(),uid=os.getuid(),startticks=int(fields[19]),
                gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))


def status(event, **fields):
    print(json.dumps(dict(event=event,**fields),allow_nan=False),flush=True)


def guard():
    data.check(shutil.disk_usage(data.ROOT).free >= 64*GIB,"SATA reserve")
    data.check(sum(p.stat().st_size for p in data.ROOT.rglob("*") if p.is_file()) <=64*GIB,"Writer cap")
    data.check(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <=32*GIB,"Worker RSS cap")
    if torch.cuda.is_initialized():
        data.check(torch.cuda.max_memory_reserved()<=40*GIB,"Per-worker CUDA cap")


def configure(seed):
    data.check(torch.cuda.device_count()==1 and (os.environ.get("CUDA_VISIBLE_DEVICES") or "").startswith("GPU-"),
               "One named GPU per worker")
    model.native_ft.seed_all(seed,"cuda:0")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True


def static_pins():
    paths = {Path(module.__file__).resolve() for name,module in list(sys.modules.items())
             if name.startswith("pub_01_") and getattr(module,"__file__",None)}
    paths.update(p.resolve() for p in Path(__file__).parent.glob("pub_01_s03_ss07_contrastive_history*.py"))
    paths.update((data.NATIVE_CONFIG,data.FEATURES,data.FAMILY,data.MASK,
                  data.PREPARED/"manifest_frozen.json",data.ENVELOPE/"manifest_frozen.json",
                  data.D19/"PREPARATION_METADATA.json",data.ROOT/"PLAN.md",data.ROOT/"BUDGET.json",
                  data.ROOT/"COORDINATOR_DECISION.md",ENGINEERING,data.ROOT/"INPUT_RECEIPT.json"))
    paths.update(data.PREPARED/f"{p}_windows.json" for p in data.PHASES)
    paths.update(data.D19/f"fit/seed{s}_ANCHOR.json" for s in data.SEEDS)
    return {str(p):data.digest(p) for p in sorted(paths)}


def admit():
    p=data.read(PROTOCOL)
    data.check(p["frozen"] and p["inventory_sha256"]==data.digest(INVENTORY),"Frozen D26 protocol")
    data.check(all(data.digest(path)==h for path,h in p["sources"].items()) and all(p["sources"].get(path)==h for path,h in static_pins().items()),"Exact current source and transitive imports")
    inv=data.read(INVENTORY)
    for path,value in inv["inputs"].items():
        data.check(data.stamp(path)==value,f"Input stat changed: {path}")
    return inv,p


def pack(current,valid,past):
    tensor=lambda x:torch.from_numpy(x).cuda()
    return tensor(current),tensor(valid),None if past is None else tuple(tensor(x) for x in past)


def cl_step(net,batch):
    x,valid,past,audio=batch
    return model.native.contrastive(net,pack(x,valid,past),torch.from_numpy(audio).cuda())


def task_grid(coords,first,last,route):
    x,valid,past,y=coords.grid(first,last,route)
    return pack(x,valid,past),torch.from_numpy(y).cuda()


def clone(net):
    return {name:value.detach().cpu().clone() for name,value in net.state_dict().items()}


def immutable_torch(path,value):
    if not path.exists():
        journal(path,value)


def journal(path,value):
    # This file is the mutable resume pointer. Epoch receipts and all selected
    # checkpoints are append-only. Preserve an interrupted temporary write.
    temp=path.parent/f".checkpoint_{os.getpid()}_{time.monotonic_ns()}.tmp"
    with temp.open("xb") as stream:
        torch.save(value,stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


class Job:
    def __init__(self,folder,net,optimizer,scheduler=None,resume=False):
        self.folder=folder
        if resume:
            folder.mkdir(exist_ok=True)
        else:
            folder.mkdir()
        self.net,self.optimizer,self.scheduler=net,optimizer,scheduler
        self.file=folder/"CURRENT.pt"
        self.history=[]
        self.best=None
        self.best_score=math.inf
        self.best_epoch=None
        self.stale=0
        self.step=0
        self.loss=0.
        self.windows=0
        self.updates=0
        self.coverage={}
        self.run_wall=0.
        self.run_cpu=0.
        self.peak=0
        self.active_wall=time.monotonic()
        self.active_cpu=time.process_time()
        if resume and self.file.exists():
            value=torch.load(self.file,map_location="cpu",weights_only=True)
            data.check(value["protocol_sha256"]==data.digest(PROTOCOL),"Resume on the frozen protocol")
            net.load_state_dict(value["model"])
            optimizer.load_state_dict(value["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(value["scheduler"])
            torch.set_rng_state(value["cpu_rng"])
            torch.cuda.set_rng_state(value["gpu_rng"])
            for name in ("history","best","best_score","best_epoch","stale","step","loss","windows",
                         "updates","coverage","run_wall","run_cpu","peak"):
                setattr(self,name,value[name])
            for row in self.history:
                self.publish(row)
        elif resume:
            data.check(not list(folder.glob("EPOCH_*.json")),"Epoch evidence without recovery state")

    def persist(self):
        peak=max(self.peak,torch.cuda.max_memory_reserved())
        value={name:getattr(self,name) for name in ("history","best","best_score","best_epoch",
               "stale","step","loss","windows","updates","coverage")}
        value.update(model=self.net.state_dict(),optimizer=self.optimizer.state_dict(),
                     scheduler=self.scheduler.state_dict() if self.scheduler is not None else None,
                     cpu_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(),
                     protocol_sha256=data.digest(PROTOCOL),
                     run_wall=self.run_wall+time.monotonic()-self.active_wall,
                     run_cpu=self.run_cpu+time.process_time()-self.active_cpu,peak=peak)
        journal(self.file,value)

    def update(self,loss,n,period,verify_gradients=False):
        data.check(torch.isfinite(loss).item(),"Finite objective")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if verify_gradients:
            data.check(all(torch.isfinite(v.grad).all().item() for v in self.net.parameters() if v.grad is not None),
                       "Finite joint FT gradients")
        self.optimizer.step()
        self.step+=1
        self.loss+=float(loss.detach())*n
        self.windows+=n
        self.updates+=1
        if period and self.step%period==0:
            guard()
            self.persist()

    def publish(self,row):
        path=self.folder/f"EPOCH_{row['epoch']:03}.json"
        if path.is_file():
            data.check(data.read(path)==row,"Immutable epoch receipt")
        else:
            temp=path.with_name(f".{path.name}_{os.getpid()}_{time.monotonic_ns()}")
            data.save(temp,row)
            temp.rename(path)

    def finish_epoch(self,epoch,validation,expected_steps,expected_windows,**more):
        data.check(epoch==len(self.history) and self.step==expected_steps
                   and self.windows==expected_windows,"Exact epoch coverage")
        data.check(math.isfinite(validation),"Finite validation objective")
        if validation<self.best_score:
            self.best_score,self.best_epoch,self.best=validation,epoch,clone(self.net)
            self.stale=0
        else:
            self.stale+=1
        row=dict(epoch=epoch,train=self.loss/self.windows,val=validation,stale=self.stale,
                 epoch_updates=self.step,total_updates=self.updates,epoch_windows=self.windows,**more)
        self.history.append(row)
        self.loss=self.windows=self.step=0
        if self.scheduler is not None:
            self.scheduler.step()
        self.persist()
        self.publish(row)
        status("epoch",path=str(self.folder),**{k:row[k] for k in ("epoch","train","val","stale","total_updates")})
        guard()

    def complete(self,**more):
        path=self.folder/"COMPLETED.json"
        if path.exists():
            return
        immutable_torch(self.folder/"SELECTED.pt",self.best)
        immutable_torch(self.folder/"FINAL.pt",clone(self.net))
        data.save(path,dict(passed=True,history=self.history,actual_epochs=len(self.history),
                 selected_epoch=self.best_epoch,selected_val=self.best_score,
                 selected_prefix_updates=self.history[self.best_epoch]["total_updates"],
                 selected_prefix_windows=sum(row["epoch_windows"] for row in self.history[:self.best_epoch+1]),
                 updates=self.updates,CPU_seconds=self.run_cpu+time.process_time()-self.active_cpu,
                 elapsed_seconds=self.run_wall+time.monotonic()-self.active_wall,
                 cuda_peak_reserved=max(self.peak,torch.cuda.max_memory_reserved()),process=identity(),**more))


def barrier(seed_root,folders,epoch,patience):
    receipts=[f/f"EPOCH_{epoch:03}.json" for f in folders]
    abort=seed_root/"ABORT.json"
    while not all(p.is_file() for p in receipts):
        data.check(not abort.is_file(),"Peer ended; keep exact checkpoints for review")
        time.sleep(2)
    return all(data.read(p)["stale"]>=patience for p in receipts)


@torch.no_grad()
def cl_validate(net,coords,selections,role):
    net.eval()
    return float(np.mean([float(cl_step(net,coords.contrastive(item,role))) for item in selections]))


def cl_train(seed,role,coords,seed_root,resume):
    tr,va=coords
    configure(seed)
    net=model.CL(data.read(data.NATIVE_CONFIG),seed,role).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.1)
    folders=[seed_root/f"CL_{name}" for name in ("H","C")]
    folder=seed_root/f"CL_{role}"
    job=Job(folder,net,opt,sched,resume=resume)
    validation=list(va.selections(20261015,False))
    val_budget=va.cl_coverage(validation,role)
    for epoch in range(100):
        if epoch>=len(job.history):
            net.train()
            selections=list(tr.selections(seed+epoch,True))
            for selection in selections[job.step:]:
                loss=cl_step(net,tr.contrastive(selection,role))
                job.update(loss,data.CL_BATCH,period=256)
            train_budget=tr.cl_coverage(selections,role,job.coverage)
            job.finish_epoch(epoch,cl_validate(net,va,validation,role),820,104960,
                             train_coverage=train_budget,val_coverage=val_budget)
        if barrier(seed_root,folders,epoch,15):
            break
    selected=job.history[:job.best_epoch+1]
    selected_cost=dict(EEG_current=sum(x["train_coverage"]["current_windows"] for x in selected),
                       EEG_memory_slots=sum(x["train_coverage"]["history_slot_presentations"] for x in selected),
                       unique=selected[-1]["train_coverage"]["cumulative"])
    job.complete(phase="contrastive",role=role,seed=seed,epochs_cap=100,selected_prefix_cost=selected_cost,
                 parameters=sum(p.numel() for p in net.parameters()),
                 stopping="paired_patience" if epoch<99 else "maximum100")


def encoder_from(seed_root,seed,arm):
    role=arm[0]
    net=model.CL(data.read(data.NATIVE_CONFIG),seed,role)
    net.load_state_dict(torch.load(seed_root/f"CL_{role}"/"SELECTED.pt",map_location="cpu",weights_only=True))
    return net.eeg


@torch.no_grad()
def head_cache(encoder,coords,route):
    encoder.eval()
    result=torch.empty((coords.windows,8,192),device="cuda")
    for first in range(0,coords.windows,64):
        stop=min(coords.windows,first+64)
        packed,_=task_grid(coords,first,stop,route)
        result[first:stop]=encoder(packed,route).transpose(1,2)
    return result,torch.from_numpy(np.array(coords.y,copy=True)).cuda()


@torch.no_grad()
def cached_val(net,cache):
    x,y=cache
    net.eval()
    total=0.
    for first in range(0,len(x),512):
        xx,yy=x[first:first+512],y[first:first+512]
        total+=float(model.native_ft.objective(net(xx),yy))*len(xx)
    return total/len(x)


def measure_cache(encoder,coords,route):
    wall,cpu=time.monotonic(),time.process_time()
    cache=head_cache(encoder,coords,route)
    torch.cuda.synchronize()
    active=int((coords.starts>=data.READY).sum())
    return cache,dict(phase=coords.phase,global_B64_batches=math.ceil(coords.windows/64),
                      current_encoder_windows=coords.windows,history_encoder_windows=2*active,
                      CUDA_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)


def heads(seed,side,coords,seed_root,resume):
    configure(seed)
    names=("HH",) if side=="H" else ("CH","CC")
    jobs={}
    for arm in names:
        route=arm[1]
        encoder=encoder_from(seed_root,seed,arm).requires_grad_(False).cuda()
        (train,train_cache),(valid,val_cache)=(measure_cache(encoder,phase,route) for phase in coords)
        del encoder
        net=model.make_head(seed).cuda()
        opt=torch.optim.Adam(net.parameters(),lr=.001)
        job=Job(seed_root/f"HEAD_{arm}",net,opt,resume=resume)
        data.save(job.folder/f"CACHE_{os.getpid()}.json",
                  dict(seed=seed,arm=arm,route=route,process=identity(),train=train_cache,val=val_cache,
                       scope="Actual encoder and target staging, includes on-demand record loading."))
        jobs[arm]=(job,train,valid)
    folders=[seed_root/f"HEAD_{arm}" for arm in data.ARMS]
    for epoch in range(250):
        for job,(x,y),valid in jobs.values():
            if epoch<len(job.history):
                continue
            job.net.train()
            started,cpu=time.monotonic(),time.process_time()
            for first in range(job.step*64,len(x),64):
                xx,yy=x[first:first+64],y[first:first+64]
                job.update(model.native_ft.objective(job.net(xx),yy),len(xx),period=1024)
            optimize=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu)
            started,cpu=time.monotonic(),time.process_time()
            validation=cached_val(job.net,valid)
            evaluate=dict(wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu)
            job.finish_epoch(epoch,validation,3945,252454,head_optimize=optimize,head_cached_VAL=evaluate,
                             VAL_windows=30500,VAL_batches=60)
        if barrier(seed_root,folders,epoch,10):
            break
    for arm,(job,_,__) in jobs.items():
        job.complete(phase="head",arm=arm,seed=seed,epochs_cap=250,
                     head_parameters=sum(p.numel() for p in job.net.parameters()),
                     cached_encoder_current_windows_per_attempt=282954,cached_encoder_slot_windows_per_attempt=557358,
                     cache_receipts=sorted(p.name for p in job.folder.glob("CACHE_*.json")),
                     head_optimize={k:sum(r["head_optimize"][k] for r in job.history) for k in ("wall_seconds","CPU_seconds")},
                     head_cached_VAL={k:sum(r["head_cached_VAL"][k] for r in job.history) for k in ("wall_seconds","CPU_seconds")},
                     timing_note="Job lifetime includes pair barrier and may overlap other jobs. Optimization, cache and validation are separately measured.",
                     stopping="paired_patience" if epoch<249 else "maximum250")


@torch.no_grad()
def decoder_val(net,coords,route):
    total=0.
    net.eval()
    for first in range(0,coords.windows,64):
        stop=min(coords.windows,first+64)
        packed,y=task_grid(coords,first,stop,route)
        total+=float(model.native_ft.objective(net(packed),y))*(stop-first)
    return total/coords.windows


def ft_train(seed,arm,coords,seed_root,resume):
    configure(seed)
    route=arm[1]
    encoder=encoder_from(seed_root,seed,arm)
    head=model.make_head(seed)
    head.load_state_dict(torch.load(seed_root/f"HEAD_{arm}"/"SELECTED.pt",map_location="cpu",weights_only=True))
    net=model.Decoder(encoder,head,route).cuda()
    opt=torch.optim.Adam([{"params":net.encoder.parameters(),"lr":1e-5},
                          {"params":net.head.parameters(),"lr":1e-3}],weight_decay=0)
    tr,va=coords
    job=Job(seed_root/f"FT_{arm}",net,opt,resume=resume)
    initial=job.folder/"EPOCH_MINUS1.json"
    if not initial.exists():
        data.check(not job.history and job.step==0,"Initial stage evidence before updates")
        actual=decoder_val(net,va,route)
        old=data.read(seed_root/f"HEAD_{arm}"/"COMPLETED.json")["selected_val"]
        data.check(abs(actual-old)<=1e-4,"Cached head to raw FT bridge")
        immutable_torch(job.folder/"INCUMBENT.pt",clone(net))
        data.save(initial,dict(epoch=-1,val=actual,head_cache_val=old,abs_difference=abs(actual-old)))
    for epoch in range(len(job.history),5):
        net.train()
        for first in range(job.step*64,tr.windows,64):
            stop=min(tr.windows,first+64)
            packed,y=task_grid(tr,first,stop,route)
            job.update(model.native_ft.objective(net(packed),y),stop-first,period=512,verify_gradients=True)
        job.finish_epoch(epoch,decoder_val(net,va,route),3945,252454,
                         train_active_windows=248620,VAL_windows=30500,VAL_active_windows=30059)
    job.complete(phase="FT",arm=arm,seed=seed,final_epoch=4,final_model="FINAL.pt",
                 parameters=sum(p.numel() for p in net.parameters()),stopping="fixed5")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("cl","head","ft"))
    parser.add_argument("--seed",type=int,choices=data.SEEDS,required=True)
    parser.add_argument("--role",required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    inv,p=admit()
    seed_root=FORMAL/f"seed{args.seed}"
    data.check(seed_root.is_dir(),"Queue must create seed scope")
    coords=tuple(data.Coordinates(ph,inv,audio=args.stage=="cl") for ph in data.PHASES)
    funcs={"cl":cl_train,"head":heads,"ft":ft_train}
    status("process",stage=args.stage,role=args.role,seed=args.seed,**identity())
    try:
        funcs[args.stage](args.seed,args.role,coords,seed_root,args.resume)
    except BaseException as error:
        path=seed_root/f"FAILURE_{args.stage}_{args.role}_{os.getpid()}.json"
        data.save(path,dict(process=identity(),message=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        status("worker_timing",stage=args.stage,role=args.role,seed=args.seed,
               worker_wall_seconds=time.monotonic()-WORKER_START,worker_CPU_seconds=time.process_time()-WORKER_CPU)


if __name__=="__main__":
    main()
