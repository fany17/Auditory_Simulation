"""D28 frozen public raw-EEG ridge design and exact streaming sufficient statistics."""
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.linalg.blas import dsyrk

SATA=Path("/sata5/fanyu/auditory_simulation_m6a/pub_01")
HOME=Path("/home/fanyu/auditory_simulation_m6a/pub_01")
ROOT=SATA/"s03_matched_raw_ridge_20261107"
D27=SATA/"s03_seawave_seen_reference_20261106"
SOURCE_INVENTORY=D27/"INVENTORY.json"
SOURCE_INVENTORY_SHA="9f9ccde5d9645213ce88adba6833ccba8ea3a4c685871ab18c7d55dc612f8732"
NORM=HOME/"s03_strong_baselines_20261015/transformer_available01/prepared/train_only_normalization.npz"
MASK=HOME/"s03_incremental_continuity_20261101/block_eligibility/val.json"
LAGS=tuple(range(33))
ALPHAS=(1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.,10.,100.)
BLOCK=4096
DIM=64*len(LAGS)


def check(ok,message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path,data):
    with Path(path).open("x") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def stamp(path):
    stat=Path(path).stat()
    return dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def inventory():
    check(digest(SOURCE_INVENTORY)==SOURCE_INVENTORY_SHA,"Exact D27 input scope")
    original=read(SOURCE_INVENTORY)
    train=sorted(original["phase_records"]["train"],key=lambda r:(r["subject"],r["record"]))
    val=original["phase_records"]["val"]
    check(len(train)==426 and len(val)==49 and all(r["phase"]=="train" for r in train)
          and all(r["phase"]=="val" for r in val),"Frozen public TRAIN/VAL")
    check(sum(r["common_samples"] for r in train)==24297126
          and sum(r["common_samples"] for r in val)==2935045,"Frozen full-sample geometry")
    subjects=original["subjects"]
    individual=[r for r in train if r["subject"] in subjects]
    check(len(individual)==305 and len(subjects)==49 and len({r["subject"] for r in train})==70,"Known people")
    mask=read(MASK)
    check(mask["prediction_independent"] and mask["records"]==[r["record"] for r in val]
          and mask["eligible_blocks"]==752,"Original positive-support mask")
    for path,expected in original["inputs"].items():
        check(stamp(path)==expected,f"Input identity {path}")
    return dict(train=train,val=val,inputs=original["inputs"],personal_subjects=subjects,
                groups={s:[r["record"] for r in train if r["subject"]==s] for s in sorted({r["subject"] for r in train})},
                personal_samples=sum(r["common_samples"] for r in individual),
                train_source_families=original["train_source_families"])


def normalization():
    with np.load(NORM,allow_pickle=False) as f:
        mean,std=f["mean"].copy(),f["std"].copy()
    check(mean.shape==std.shape==(64,) and (std>0).all(),"Existing full TRAIN EEG scale")
    return mean,std


def load_record(row,norm,limit=None):
    eeg=np.load(row["eeg_path"],mmap_mode="r",allow_pickle=False)
    eeg=eeg.T if eeg.shape[0]==64 else eeg
    raw_y=np.load(row["envelope_path"],mmap_mode="r",allow_pickle=False).reshape(-1)
    common=row["common_samples"]
    check(eeg.ndim==2 and eeg.shape[1]==64 and len(eeg)>=common and len(raw_y)>=common,"Common support")
    n=common if limit is None else min(limit,common)
    mean,std=norm
    normalized=((np.asarray(eeg[:n],dtype=np.float32)-mean)/std).astype(np.float32).astype(np.float64)
    target=np.asarray(raw_y[:n],dtype=np.float32).astype(np.float64)
    check(np.isfinite(normalized).all() and np.isfinite(target).all(),"Finite input and target")
    return normalized,target


def design(normalized,first,stop,lags=LAGS):
    # The returned Fortran matrix is feature x target time. In each column,
    # chronological lag is the major index and EEG channel is the minor index.
    n,channels=normalized.shape
    check(0<=first<stop<=n and stop-first<=BLOCK,"Record-local design block")
    size=stop-first
    x=np.zeros((channels*len(lags),size),dtype=np.float64,order="F")
    ordered=x.T.reshape(size,len(lags),channels)
    for j,delay in enumerate(lags):
        available=max(0,min(stop,n-delay)-first)
        if available:
            ordered[:available,j,:]=normalized[first+delay:first+delay+available]
    return x


class Sufficient:
    def __init__(self,dimension):
        self.n=0
        self.gram=np.zeros((dimension,dimension),dtype=np.float64,order="F")
        self.x=np.zeros(dimension)
        self.xy=np.zeros(dimension)
        self.minimum=np.full(dimension,np.inf)
        self.maximum=np.full(dimension,-np.inf)
        self.y=self.yy=0.

    def add(self,x,y):
        check(x.shape==(len(self.x),len(y)) and x.dtype==y.dtype==np.float64,"Float64 statistics")
        self.gram=dsyrk(1.,x,beta=1.,c=self.gram,lower=1,overwrite_c=1)
        self.x+=x.sum(axis=1)
        self.minimum=np.minimum(self.minimum,x.min(axis=1))
        self.maximum=np.maximum(self.maximum,x.max(axis=1))
        self.xy+=x@y
        self.y+=float(y.sum())
        self.yy+=float(y@y)
        self.n+=len(y)

    def merge(self,other):
        self.gram+=other.gram
        self.x+=other.x
        self.xy+=other.xy
        self.minimum=np.minimum(self.minimum,other.minimum)
        self.maximum=np.maximum(self.maximum,other.maximum)
        self.y+=other.y
        self.yy+=other.yy
        self.n+=other.n

    def arrays(self):
        return dict(n=np.int64(self.n),gram=self.gram,x=self.x,xy=self.xy,y=self.y,yy=self.yy,
                    minimum=self.minimum,maximum=self.maximum)

    @classmethod
    def load(cls,archive):
        state=cls(len(archive["x"]))
        state.n=int(archive["n"])
        state.gram=np.array(archive["gram"],order="F",copy=True)
        state.x=archive["x"].copy()
        state.xy=archive["xy"].copy()
        state.minimum=archive["minimum"].copy()
        state.maximum=archive["maximum"].copy()
        state.y=float(archive["y"])
        state.yy=float(archive["yy"])
        return state


def solutions(stats,alphas=ALPHAS):
    check(stats.n>0,"Nonempty sufficient statistics")
    gram=stats.gram+np.tril(stats.gram,-1).T
    n=stats.n
    xm=stats.x/n
    ym=stats.y/n
    cov=gram/n-np.outer(xm,xm)
    var=cov.diagonal()
    check(var.min()>=-1e-10*max(1.,float(abs(var).max())),"Stable population feature variances")
    const=stats.minimum==stats.maximum
    check(np.all(var[~const]>0),"Positive nonconstant design variances")
    cov[const,:]=0
    cov[:,const]=0
    xs=np.sqrt(np.where(const,1.,var))
    yv=stats.yy/n-ym**2
    check(yv>0,"Finite nonconstant training target")
    ys=np.sqrt(yv)
    scaled=cov/np.outer(xs,xs)
    scaled=(scaled+scaled.T)*.5
    rhs=(stats.xy/n-xm*ym)/(xs*ys)
    rhs[const]=0
    eigen,basis=eigh(scaled,driver="evr",check_finite=True)
    ridge=np.array(alphas,dtype=np.float64)
    check(eigen.min()>-min(ridge)*.1,"Numerical eigensystem supports fixed alpha")
    w=basis@((basis.T@rhs)[:,None]/(eigen[:,None]+ridge))
    residual=np.max(np.abs(scaled@w+w*ridge-rhs[:,None]),axis=0)
    check(np.isfinite(w).all() and residual.max()<=1e-7*max(1.,float(abs(rhs).max())),
          "Exact normalized ridge residual")
    beta=ys*w/xs[:,None]
    intercept=ym-xm@beta
    return dict(weights=w,raw_weights=beta,raw_intercepts=intercept,
                design_mean=xm,design_scale=xs,constant=const,target_mean=ym,target_scale=ys,
                alphas=ridge,normal_equation_residual=residual,smallest_eigenvalue=eigen.min())
