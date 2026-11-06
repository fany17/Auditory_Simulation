"""D26 fixed final-epoch TRAIN-only Ccap and old prediction-independent VAL mask."""
import argparse
import math
import time

WORKER_START=time.monotonic()
WORKER_CPU=time.process_time()

import numpy as np
import torch

import pub_01_s03_ss07_contrastive_history_data as data
import pub_01_s03_ss07_contrastive_history_model as model
import pub_01_s03_ss07_contrastive_history_train as train


@torch.no_grad()
def decode(net,coords,route,intervene=False):
    net.eval()
    names=("near","far","copy") if intervene else ("native",)
    latents={name:np.empty((coords.windows,8,192),np.float32) for name in names}
    predictions={name:np.empty((coords.windows,192),np.float32) for name in names}
    for first in range(0,coords.windows,64):
        stop=min(coords.windows,first+64)
        packed,_=train.task_grid(coords,first,stop,route)
        x,valid,past=packed
        query=net.encoder.query(x)
        paths=[(names[0],route,past)]
        if intervene:
            far=coords.past(coords.record_ids[first:stop],coords.starts[first:stop],data.FAR)
            paths.extend((("far","H",None if far is None else tuple(torch.from_numpy(v).cuda() for v in far)),
                          ("copy","C",None)))
        for name,mode,memory in paths:
            z=net.encoder.project(query,x,valid,memory,mode).transpose(1,2)
            p=net.head(z)[:,0]
            latents[name][first:stop]=z.cpu().numpy()
            predictions[name][first:stop]=p.cpu().numpy()
        if first%32768==0:
            train.guard()
    return {key:(latents[key],predictions[key]) for key in names}


def target_files(coords):
    return {row["record"]:np.load(row["envelope_path"],mmap_mode="r",allow_pickle=False).reshape(-1)[:row["common_samples"]]
            for row in coords.rows}


def apply_readout(z,p,fit):
    stats,coef,bias,affine=fit
    base=affine["a"]*p.astype(np.float64)+affine["b"]
    result=base.copy()
    norms=model.ridge_helpers.normalize
    first=4
    u=norms(z[first:].astype(np.float64).reshape(-1,1536),stats[0])
    v=norms(base[first:],stats[1])
    w=norms(u*u,stats[2])
    x=np.concatenate((u/math.sqrt(1536),v/math.sqrt(192),w/math.sqrt(1536)),axis=1)
    result[first:]+=x@coef+bias
    data.check(np.array_equal(base[:4],result[:4]),"Current model affine fallback")
    return base,result


def score(rows,targets,masks,waves,active=False):
    records=[]
    for row,pred in zip(rows,waves):
        key=row["record"]
        records.append(model.metrics.measure(pred,targets[key],row,masks[key]))
    result=model.metrics.aggregate(records)
    data.check(len(result["people"])==49 and len(result["segments"])==752,"Complete native primary score")
    if active:
        points=[]
        for row,pred in zip(rows,waves):
            key=row["record"]
            wave,keep=model.metrics.fuse(pred,row["common_samples"])
            for b in masks[key]:
                if not b["eligible"]:
                    continue
                start=max(b["start"],960)
                stop=b["start"]+3840
                valid=keep[start:stop]
                p=wave[start:stop][valid];y=targets[key][start:stop][valid]
                points.append(dict(subject=row["subject"],source_family=row["source_family"],
                                   record=key,start=b["start"],retained_samples=len(p),
                                   r=model.metrics.corr(p,y),MSE=float(np.mean((p-y)**2))))
        people=sorted({v["subject"] for v in points})
        result["pure_memory"]=dict(r=float(np.mean([np.mean([v["r"] for v in points if v["subject"]==s])
                                                   for s in people])),
                                   MSE=float(np.mean([np.mean([v["MSE"] for v in points if v["subject"]==s])
                                                     for s in people])),
                                   people=len(people),blocks=len(points),records=points,first_physical_sample=960)
    return result


def run(seed,arm):
    inv,_=train.admit()
    train.configure(seed)
    tr,va=(data.Coordinates(phase,inv) for phase in data.PHASES)
    seed_root=train.FORMAL/f"seed{seed}"
    out=seed_root/f"READOUT_{arm}"
    out.mkdir()
    start,cpu=time.monotonic(),time.process_time()
    fitfolder=seed_root/f"FT_{arm}"
    info=data.read(fitfolder/"COMPLETED.json")
    data.check(info["final_epoch"]==4 and info["updates"]==19725,"Fixed actual FT terminal checkpoint")
    checkpoint=fitfolder/"FINAL.pt"
    net=model.Decoder(train.encoder_from(seed_root,seed,arm),model.make_head(seed),arm[1]).cuda()
    net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    z,p=decode(net,tr,arm[1])["native"]
    fit,ledger=model.ridge.readout(tr.rows,z,p,target_files(tr),model.ridge_helpers)
    del z,p
    stats,coef,bias,affine=fit
    with (out/"READOUT.npz").open("xb") as handle:
        np.savez(handle,coef=coef,bias=bias,a=affine["a"],b=affine["b"],
                 **{f"{k}_{i}":v for i,item in enumerate(stats) for k,v in item.items()})
    mask=data.read(data.MASK)["blocks"]
    masks={r["record"]:[b for b in mask if b["record"]==r["record"]] for r in va.rows}
    targets=target_files(va)
    pred=decode(net,va,arm[1],intervene=arm=="HH")
    metrics={}
    for path,(z,p) in pred.items():
        native,calibrated,ccap=[],[],[]
        for _,zz,pp in model.ridge.blocks(va.rows,z,p):
            base,corrected=apply_readout(zz,pp,fit)
            native.append(pp.astype(np.float64))
            calibrated.append(base)
            ccap.append(corrected)
        metrics[path]=dict(Ccap=score(va.rows,targets,masks,ccap,active=arm=="HH"))
        if path in ("native","near"):
            metrics[path].update(raw=score(va.rows,targets,masks,native),
                                 affine=score(va.rows,targets,masks,calibrated))
            raw=metrics[path]["raw"]["segments"]
            aff=metrics[path]["affine"]["segments"]
            data.check(max(abs(a["r"]-b["r"]) for a,b in zip(raw,aff))<=1e-10,"Positive affine invariance")
    for path,value in metrics.items():
        data.save(out/f"SCORE_{path}.json",value)
    primary=metrics["near" if arm=="HH" else "native"]
    r={k:v["phases"]["val"]["r"] for k,v in primary.items()}
    baseline=data.read(data.D19/f"fit/seed{seed}_ANCHOR.json")["anchor"]["val"]["r"]
    data.save(out/"COMPLETED.json",dict(passed=True,seed=seed,arm=arm,epoch=4,
              FT_checkpoint_sha256=data.digest(checkpoint),TRAIN_fit=ledger,lambda_fixed=.01,
              affine=affine,VAL_r=r,anchor_r=baseline,delta_anchor=r["Ccap"]-baseline,
              VAL_pure_memory={k:v["Ccap"]["pure_memory"]["r"] for k,v in metrics.items()} if arm=="HH" else {},
              encoder_current_windows=282954,
              encoder_slots=557358+(4*30059 if arm=="HH" else 0),
              history_interventions=2 if arm=="HH" else 0,
              CPU_seconds=time.process_time()-cpu,elapsed_seconds=time.monotonic()-start,
              process=train.identity(),new_neural_updates=0,old_test_reads=0))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",required=True,type=int,choices=data.SEEDS)
    p.add_argument("--arm",required=True,choices=data.ARMS)
    a=p.parse_args()
    try:
        run(a.seed,a.arm)
    finally:
        train.status("worker_timing",stage="readout",role=a.arm,seed=a.seed,
                     worker_wall_seconds=time.monotonic()-WORKER_START,worker_CPU_seconds=time.process_time()-WORKER_CPU)
