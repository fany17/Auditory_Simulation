"""Run the immutable audited author model and scheduler through the D27 adapter."""
import importlib.util
import math
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F

import pub_01_s03_ss07_seawave_seen_data as data

AUTHOR=data.ROOT/"author"
AUTHOR_HASHES={
    "EEGWaveRegressor.py":"bac621abfd5c4553d9b4be790da7e5af81d26932ee3e0dede86564503950f0f0",
    "cyclic_scheduler.py":"4ec5aefb6e92637984b2e1d82e70f7f62367d0dfda38cc6443ebc25e4c7a5ebf",
    "util.py":"8520aa37c0d0e1551dd5d7bdaa5f25b692f252fbdc62dea47486cec665e6b50d",
    "LICENSE":"8b53e6bd069c180598bb4dd0c121d5a86b89d2a856c7f35002a5426e2340b69c",
}


def import_exact(name,filename):
    path=AUTHOR/filename
    data.check(data.digest(path)==AUTHOR_HASHES[filename],"Audited author bytes")
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wave=import_exact("d27_author_wave","EEGWaveRegressor.py")
cyclic=import_exact("d27_author_schedule","cyclic_scheduler.py")
util=import_exact("d27_author_util","util.py")


def model(dropout):
    # Direct author class, keeping weight_norm and initialization order.
    return wave.WaveNet_regressor(64,32,32,1,20,5,dropout)


def scheduler(optimizer, settings, windows):
    return cyclic.CyclicLRWithRestarts(optimizer,batch_size=settings["batch"],
           epoch_size=windows,restart_period=10,t_mult=1.2,policy="cosine",min_lr=1e-7)


def scheduler_state(obj):
    excluded={"optimizer","policy_fn","eta_on_restart_cb","eta_on_iteration_cb"}
    return {key:value for key,value in vars(obj).items() if key not in excluded}


def restore_scheduler(obj,saved):
    data.check(saved.keys()==scheduler_state(obj).keys(),"Exact scheduler state layout")
    for key in ("base_lrs","base_weight_decays","min_lrs","batch_size","epoch_size","policy","t_mult"):
        data.check(getattr(obj,key)==saved[key],f"Frozen scheduler {key}")
    for key,value in saved.items():
        setattr(obj,key,value)


def objective(stage,output,target):
    mse=F.mse_loss(output,target)
    return mse if stage=="initial" else .8*mse+.2*util.pearson_loss(output,target,axis=-1)


@torch.no_grad()
def continuous(net,x,call=None):
    net.eval()
    n=x.shape[-1]
    parts=[]
    for start in range(0,n,data.CORE):
        end=min(n,start+data.CORE)
        first=max(0,start-data.HALO)
        last=min(n,end+data.HALO)
        prediction=net(x[None,:,first:last])[0,0]
        if call:
            call(last-first)
        parts.append(prediction[start-first:end-first])
    return torch.cat(parts).cpu().numpy().astype(np.float64)


def support_stop(n):
    return ((n-192)//96)*96+191


def correlate(x,y):
    a=x-x.mean()
    b=y-y.mean()
    norm=np.sqrt((a@a)*(b@b))
    data.check(np.isfinite(norm) and norm>0,"Every frozen block has defined correlation")
    return float(a@b/norm)


def record_scores(row,pred,target,mask):
    n=row["common_samples"]
    data.check(pred.shape==(n,) and len(target)>=n and np.isfinite(pred).all(),"Continuous legal support")
    stop=support_stop(n)
    points=[]
    for b in mask:
        if not b["eligible"]:
            continue
        a=max(1,b["start"])
        z=min(stop,b["start"]+3840)
        data.check(z-a==b["samples"],"Old positive Hann common support")
        x=pred[a:z]
        y=np.asarray(target[a:z],dtype=np.float64)
        points.append(dict(record=row["record"],subject=row["subject"],source_family=row["source_family"],
                           start=b["start"],samples=z-a,r=correlate(x,y),MSE=float(np.mean((x-y)**2))))
    return points


def aggregate(blocks):
    groups=defaultdict(list)
    for b in blocks:
        groups[b["subject"]].append(b)
    people=[dict(subject=name,blocks=len(rows),r=float(np.mean([r["r"] for r in rows])),
                 MSE=float(np.mean([r["MSE"] for r in rows]))) for name,rows in sorted(groups.items())]
    data.check(len(people)>0,"Eligible participants")
    return dict(r=float(np.mean([r["r"] for r in people])),MSE=float(np.mean([r["MSE"] for r in people])),
                people=people,blocks=blocks)
