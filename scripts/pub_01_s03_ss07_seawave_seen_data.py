"""D27 frozen public data, record-local window indexing and old common support."""
import json
import math
from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

SATA=Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
HOME=Path("/home/fanyu/auditory_simulation_m6a/pub_01")
ROOT=SATA/"s03_seawave_seen_reference_20261106"
D26=SATA/"s03_contrastive_history_20261106"
OLD=HOME/"s03_strong_baselines_20261015/transformer_available01"
BASE=HOME/"s03_incremental_continuity_20261101"
NORM=OLD/"prepared/train_only_normalization.npz"
MASK=BASE/"block_eligibility/val.json"
D26_INPUT_SHA="6385bde08161beb1dd507568aeb3c6d32d2f2be9087f423aec04107c1e943697"
SEEDS=(1024,2026,31415)
SETTINGS={"initial":dict(frame=320,hop=32,batch=32,epochs=201,lr=.0001,wd=.00001,dropout=.2),
          "personal":dict(frame=512,hop=6,batch=8,epochs=21,patience=5,lr=.00001,wd=.01,dropout=.65)}
CORE,HALO=4096,124


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def stamp(path):
    s=Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def check(ok,message):
    if not ok:
        raise RuntimeError(message)


def write(path,data):
    with Path(path).open("x") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n")


def size(row,frame,hop):
    return max(0,(row["common_samples"]-frame)//hop+1)


def geometry(rows,s):
    frame,hop,batch=(s[k] for k in ("frame","hop","batch"))
    n=sum(size(r,frame,hop) for r in rows)
    check(n>0 and all(size(r,frame,hop)>0 for r in rows),"Usable record-local windows")
    samples=sum((size(r,frame,hop)-1)*hop+frame for r in rows)
    return dict(records=len(rows),windows=n,updates=math.ceil(n/batch),tail=n%batch or batch,
                unique_record_sample_times=samples,target_sample_presentations=n*frame,
                repeated_target_sample_presentations=n*frame-samples)


def inventory():
    inv=D26/"INPUTS.json"
    check(digest(inv)==D26_INPUT_SHA,"Frozen D26 cohort")
    src=read(inv)
    train=sorted(src["phases"]["train"],key=lambda r:r["record"])
    val=src["phases"]["val"]
    check(len(train)==426 and len(val)==49 and len({r["subject"] for r in val})==49,"Original TRAIN/VAL")
    check(all(r["phase"]=="train" for r in train) and all(r["phase"]=="val" for r in val),"TRAIN/VAL only")
    groups=defaultdict(list)
    for row in train:
        groups[row["subject"]].append(row)
    names=sorted({r["subject"] for r in val})
    check(len(groups)==70 and all(subject in groups for subject in names),"VAL seen-person cohort")
    initial=geometry(train,SETTINGS["initial"])
    personal={name:geometry(groups[name],SETTINGS["personal"]) for name in names}
    check((initial["windows"],initial["updates"],initial["tail"])==(755187,23600,19),"Initial geometry")
    check(tuple(sum(v[k] for v in personal.values()) for k in ("records","windows","updates"))
          ==(305,2846186,355800),"Exact 49-person geometry")
    mask=read(MASK)
    check(mask["prediction_independent"] and mask["eligible_blocks"]==752 and mask["records"]==[r["record"] for r in val],
          "Original ordered score mask")
    train_families={r["source_family"] for r in train}
    check(sum(r["source_family"] in train_families for r in val)==29,"Original audio family groups")
    files={str(NORM)}|{r[k] for r in train+val for k in ("eeg_path","envelope_path")}
    pins={path:stamp(path) for path in sorted(files)}
    check(all(src["inputs"][p]==v for p,v in pins.items()),"Reuse actual D26 frozen input identities")
    return dict(phase_records=dict(train=train,val=val),subjects=names,train_source_families=sorted(train_families),
                initial=initial,personal=personal,inputs=pins,VAL_blocks=752,VAL_shared_source_records=29,
                source_inventory=str(inv),source_inventory_sha256=D26_INPUT_SHA)


class Signals:
    def __init__(self,device):
        import torch
        self.torch=torch
        self.device=device
        with np.load(NORM,allow_pickle=False) as scaling:
            self.mean,self.std=scaling["mean"].copy(),scaling["std"].copy()
        check(self.mean.shape==self.std.shape==(64,) and np.all(self.std>0),"Original TRAIN-only scale")
        self.records={}

    def raw(self,row):
        array=np.load(row["eeg_path"],mmap_mode="r",allow_pickle=False)
        eeg=array.T if array.shape[0]==64 else array
        y=np.load(row["envelope_path"],mmap_mode="r",allow_pickle=False).reshape(-1)
        n=row["common_samples"]
        check(eeg.ndim==2 and eeg.shape[1]==64 and len(eeg)>=n and len(y)>=n,"Common EEG/envelope support")
        return eeg[:n],y[:n]

    def on_device(self,row):
        key=row["record"]
        if key not in self.records:
            x,y=self.raw(row)
            x=((np.asarray(x,dtype=np.float32)-self.mean)/self.std).astype(np.float32)
            y=np.array(y,dtype=np.float32,copy=True)
            check(np.isfinite(x).all() and np.isfinite(y).all(),"Finite public common support")
            self.records[key]=(self.torch.from_numpy(np.ascontiguousarray(x.T)).to(self.device),
                               self.torch.from_numpy(y[None]).to(self.device))
        return self.records[key]


class Windows:
    def __init__(self,rows,signals,settings):
        self.rows,self.signals,self.settings=rows,signals,settings
        self.counts=[size(r,settings["frame"],settings["hop"]) for r in rows]
        self.ends=np.cumsum(self.counts).tolist()
        self.windows=self.ends[-1]
        self.batches=math.ceil(self.windows/settings["batch"])
        self.views={}

    def get(self,i):
        if i not in self.views:
            x,y=self.signals.on_device(self.rows[i])
            length,hop=self.settings["frame"],self.settings["hop"]
            self.views[i]=tuple(a.unfold(-1,length,hop).transpose(0,1) for a in (x,y))
        return self.views[i]

    def batch(self,step):
        batch=self.settings["batch"]
        first,last=step*batch,min((step+1)*batch,self.windows)
        check(first<last,"Original tail batch")
        pairs=[]
        while first<last:
            i=bisect_right(self.ends,first)
            start=0 if i==0 else self.ends[i-1]
            end=min(last,self.ends[i])
            pairs.append(tuple(a[first-start:end-start] for a in self.get(i)))
            first=end
        return tuple(self.signals.torch.cat(a,dim=0) for a in zip(*pairs))
