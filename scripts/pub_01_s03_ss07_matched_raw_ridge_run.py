"""D28 complete CPU ridge matrix, safe record-boundary resume and original VAL scoring."""
import argparse
import csv
import json
import math
import os
import resource
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits

import pub_01_s03_ss07_matched_raw_ridge_data as data

D27_INPUT=data.D27/"input_v04"
sys.path.insert(0,str(D27_INPUT))
import pub_01_s03_ss07_seawave_seen_model as scorer

INVENTORY=data.ROOT/"INVENTORY.json"
GATE=data.ROOT/"PROTOCOL.json"
FORMAL=data.ROOT/"formal"
GIB=1024**3


def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    path=Path(f"/proc/{pid}")
    f=(path/"stat").read_text().rsplit(")",1)[1].split()
    return dict(pid=pid,uid=path.stat().st_uid,startticks=int(f[19]),state=f[0])


def live(record):
    try:
        state=identity(record["pid"])
        return all(state[k]==record[k] for k in ("pid","uid","startticks")) and state["state"] not in ("Z","X")
    except FileNotFoundError:
        return False


def guard(threads):
    data.check(os.environ.get("CUDA_VISIBLE_DEVICES")=="" and not scorer.torch.cuda.is_initialized(),"D28 CPU only")
    info=threadpool_info()
    data.check(all(v.get("num_threads",0)<=threads for v in info),"Actual BLAS/OMP thread cap")
    data.check(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=8*GIB,"RSS cap")
    data.check(shutil.disk_usage(data.ROOT).free>=64*GIB,"SATA reserve")
    data.check(sum(p.stat().st_size for p in data.ROOT.rglob("*") if p.is_file())<=8*GIB,"SATA writer cap")


def pins():
    paths=set(Path(__file__).parent.glob("pub_01_s03_ss07_matched_raw_ridge_*.py"))
    paths.update((data.SOURCE_INVENTORY,data.MASK,data.ROOT/"PLAN.md",data.ROOT/"COORDINATOR_DECISION.md",
                  data.ROOT/"COORDINATOR_GEOMETRY.json",data.ROOT/"BUDGET.json",
                  data.ROOT/"ENGINEERING.json",Path(scorer.__file__).resolve(),
                  data.HOME/"s03_current_readout_internal_audit_20261101/input/pub_01_s03_ss07_generalization_core.py",
                  D27_INPUT/"pub_01_s03_ss07_seawave_seen_data.py"))
    paths.update(scorer.AUTHOR/name for name in scorer.AUTHOR_HASHES)
    return {str(p.resolve()):data.digest(p) for p in sorted(paths)}


def atomic_npz(path,arrays):
    scratch=path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with scratch.open("xb") as stream:
        np.savez(stream,**arrays)
        stream.flush()
        os.fsync(stream.fileno())
    scratch.replace(path)


def publish(path,value):
    if path.exists():
        data.check(data.read(path)==value,"Existing evidence must match exactly")
    else:
        scratch=path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
        data.write(scratch,value)
        scratch.rename(path)


def save_stats(path,stats,meta):
    atomic_npz(path,dict(**stats.arrays(),metadata=np.array(json.dumps(meta,sort_keys=True))))


def read_stats(path):
    with np.load(path,allow_pickle=False) as arc:
        return data.Sufficient.load(arc),json.loads(str(arc["metadata"]))


def fingerprint():
    return dict(protocol_sha256=data.digest(GATE),inventory_sha256=data.digest(INVENTORY))


def record_moments(row,stats,norm):
    clock,cpu=time.monotonic(),time.process_time()
    x,y=data.load_record(row,norm)
    for start in range(0,len(y),data.BLOCK):
        end=min(len(y),start+data.BLOCK)
        stats.add(data.design(x,start,end),y[start:end])
    return dict(record=row["record"],target_times=len(y),BLAS_blocks=math.ceil(len(y)/data.BLOCK),
                CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-clock)


def checked_prefix(stats,meta,rows,header,complete=False):
    data.check(meta["fingerprint"]==header,"Resume from original source and input scope")
    done=meta["record_ledger"]
    data.check([r["record"] for r in done]==[r["record"] for r in rows[:len(done)]]
               and stats.n==sum(r["common_samples"] for r in rows[:len(done)]),
               "Exact committed record prefix")
    data.check(not complete or len(done)==len(rows),"Finished subject statistics")
    return len(done)


def fit_stats(inv,folder,norm,threads):
    sdir=folder/"stats"
    sdir.mkdir(exist_ok=True)
    by_id={r["record"]:r for r in inv["train"]}
    header=fingerprint()
    subject_paths={}
    for subject,records in inv["groups"].items():
        rows=[by_id[key] for key in records]
        final=sdir/f"{subject}.npz"
        work=sdir/f"{subject}_CURRENT.npz"
        if final.exists():
            data.check(not work.exists(),"One committed subject archive")
            stats,meta=read_stats(final)
            checked_prefix(stats,meta,rows,header,complete=True)
        else:
            if work.exists():
                stats,meta=read_stats(work)
                offset=checked_prefix(stats,meta,rows,header)
            else:
                stats=data.Sufficient(data.DIM)
                meta=dict(fingerprint=header,subject=subject,record_ledger=[])
                offset=0
            for row in rows[offset:]:
                meta["record_ledger"].append(record_moments(row,stats,norm))
                save_stats(work,stats,meta)
                guard(threads)
            checked_prefix(stats,meta,rows,header,complete=True)
            work.rename(final)
        subject_paths[subject]=final
        del stats
    g=sdir/"G.npz"
    if g.exists():
        stats,meta=read_stats(g)
        checked_prefix(stats,meta,inv["train"],header,complete=True)
    else:
        clock,cpu=time.monotonic(),time.process_time()
        stats=data.Sufficient(data.DIM)
        meta=dict(fingerprint=header,subject="G",record_ledger=[],merged_subjects=list(subject_paths))
        for subject,path in subject_paths.items():
            block,rec=read_stats(path)
            stats.merge(block)
            meta["record_ledger"].extend(rec["record_ledger"])
        checked_prefix(stats,meta,inv["train"],header,complete=True)
        meta["merge_wall_seconds"]=time.monotonic()-clock
        meta["merge_CPU_seconds"]=time.process_time()-cpu
        save_stats(g,stats,meta)
    ledger=meta
    return dict(G=g,**{s:subject_paths[s] for s in inv["personal_subjects"]}),ledger


def fit_coefficients(paths,folder,threads):
    destination=folder/"coefficients"
    destination.mkdir(exist_ok=True)
    result={}
    for name,source in paths.items():
        target=destination/f"{name}.npz"
        source_stamp=data.stamp(source)
        if target.exists():
            with np.load(target,allow_pickle=False) as f:
                meta=json.loads(str(f["metadata"]))
                data.check(meta["stats_stamp"]==source_stamp and meta["fingerprint"]==fingerprint(),
                           "Existing coefficient source")
        else:
            begin,cpu=time.monotonic(),time.process_time()
            stats,_=read_stats(source)
            fit=data.solutions(stats)
            meta=dict(stats_stamp=source_stamp,fingerprint=fingerprint(),
                      fit=name,TRAIN_samples=stats.n,decompositions=1,ridge_candidates=9,
                      elapsed_seconds=time.monotonic()-begin,CPU_seconds=time.process_time()-cpu,
                      constant_features=int(fit["constant"].sum()))
            atomic_npz(target,dict(**fit,metadata=np.array(json.dumps(meta))))
        result[name]=target,meta
        guard(threads)
    return result


def score_record(row,norm,paths,masks):
    x,y32=data.load_record(row,norm)
    # The historical common-support metric uses the original continuous target
    # precision. The optimization target remains the old float32 value.
    target=np.load(row["envelope_path"],mmap_mode="r",allow_pickle=False).reshape(-1)[:len(y32)]
    weights,intercepts=[],[]
    for path in (paths["G"][0],paths[row["subject"]][0]):
        with np.load(path,allow_pickle=False) as f:
            weights.append(f["raw_weights"].copy())
            intercepts.append(f["raw_intercepts"].copy())
    beta=np.concatenate(weights,axis=1)
    intercept=np.concatenate(intercepts)
    prediction=np.empty((len(x),len(intercept)),dtype=np.float64)
    for first in range(0,len(x),data.BLOCK):
        end=min(len(x),first+data.BLOCK)
        prediction[first:end]=data.design(x,first,end).T@beta+intercept
    records=[]
    for i,kind in enumerate(("G","I")):
        for j,alpha in enumerate(data.ALPHAS):
            blocks=scorer.record_scores(row,prediction[:,i*9+j],target,masks)
            records.extend(dict(**b,kind=kind,alpha=alpha,story=row["story"]) for b in blocks)
    return records


def val_scores(inv,folder,norm,paths,threads):
    destination=folder/"VAL"
    destination.mkdir(exist_ok=True)
    mask=data.read(data.MASK)["blocks"]
    collected=[]
    for index,row in enumerate(inv["val"]):
        file=destination/f"RECORD_{index:02}.json"
        expected=[b for b in mask if b["record"]==row["record"]]
        if file.exists():
            saved=data.read(file)
            data.check(saved["fingerprint"]==fingerprint() and saved["record"]==row["record"],
                       "Existing original VAL record")
        else:
            elapsed,cpu=time.monotonic(),time.process_time()
            rows=score_record(row,norm,paths,expected)
            saved=dict(fingerprint=fingerprint(),record=row["record"],blocks=rows,
                       target_times=row["common_samples"],CPU_seconds=time.process_time()-cpu,
                       wall_seconds=time.monotonic()-elapsed)
            publish(file,saved)
        data.check(len(saved["blocks"])==18*sum(b["eligible"] for b in expected),"All per-record candidate blocks")
        collected.extend(saved["blocks"])
        guard(threads)
    data.check(len(collected)==752*18,"Exact VAL candidate block count")
    return collected


def csv_publish(path,rows):
    if path.exists():
        with path.open(newline="") as stream:
            existing=list(csv.DictReader(stream))
        expected=[{k:str(v) for k,v in row.items()} for row in rows]
        data.check(existing==expected,"Existing CSV is identical")
        return
    scratch=path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    with scratch.open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scratch.rename(path)


def selected(inv,folder,paths,all_blocks):
    group={}
    for b in all_blocks:
        key=(b["kind"],"G" if b["kind"]=="G" else b["subject"],b["alpha"])
        group.setdefault(key,[]).append(b)
    grid={}
    output=[]
    for (kind,fit,alpha),rows in group.items():
        agg=scorer.aggregate(rows)
        grid[(fit,alpha)]=agg
        output.append(dict(kind=kind,fit=fit,alpha=alpha,r=agg["r"],MSE=agg["MSE"],
                           participants=len(agg["people"]),blocks=len(agg["blocks"])))
    fits=["G",*inv["personal_subjects"]]
    data.check(len(grid)==450,"Exactly 50?9 scored candidates")
    chosen={fit:max(data.ALPHAS,key=lambda a:grid[(fit,a)]["r"]) for fit in fits}
    selected_blocks=[b for b in all_blocks if b["alpha"]==chosen["G" if b["kind"]=="G" else b["subject"]]]
    report={}
    families=set(inv["train_source_families"])
    for kind in ("G","I"):
        rows=[b for b in selected_blocks if b["kind"]==kind]
        result=scorer.aggregate(rows)
        data.check(len(result["blocks"])==752 and len(result["people"])==49,"Original complete VAL support")
        result["by_family"]={key:scorer.aggregate([b for b in rows if (b["source_family"] in families)==(key=="shared")])
                             for key in ("shared","unseen")}
        result["by_story"]={key:scorer.aggregate([b for b in rows if b["story"]==key])
                            for key in sorted({b["story"] for b in rows})}
        report[kind]=result
    csv_publish(folder/"CANDIDATE_SCORES.csv",sorted(output,key=lambda r:(r["fit"],r["alpha"])))
    csv_publish(folder/"VAL_BLOCKS.csv",all_blocks)
    csv_publish(folder/"SELECTED_VAL_BLOCKS.csv",selected_blocks)
    publish(folder/"SELECTED_VAL.json",dict(alpha_by_fit=chosen,metrics=report))
    values={}
    for fit,alpha in chosen.items():
        with np.load(paths[fit][0],allow_pickle=False) as f:
            j=data.ALPHAS.index(alpha)
            for key in ("design_mean","design_scale","target_mean","target_scale","constant"):
                values[f"{fit}_{key}"]=f[key]
            for key in ("weights","raw_weights"):
                values[f"{fit}_{key}"]=f[key][:,j]
            values[f"{fit}_raw_intercept"]=f["raw_intercepts"][j]
            values[f"{fit}_alpha"]=np.float64(alpha)
    file=folder/"SELECTED_COEFFICIENTS.npz"
    if file.exists():
        with np.load(file,allow_pickle=False) as f:
            data.check(set(f.files)==set(values) and all(np.array_equal(f[k],v) for k,v in values.items()),
                       "Same previously published coefficients")
    else:
        atomic_npz(file,values)
    return {k:{r:v[r] for r in ("r","MSE")} for k,v in report.items()}


def run(resume,threads):
    gate=data.read(GATE)
    review=data.read(data.ROOT/"COORDINATOR_EXECUTION_REVIEW.json")
    data.check(gate["frozen"] and review["passed"] and review["protocol_sha256"]==data.digest(GATE)
               and gate["sources"]==pins() and gate["inventory_sha256"]==data.digest(INVENTORY),
               "Independent actual frozen source and inputs")
    inv=data.read(INVENTORY)
    data.check(all(data.stamp(p)==expected for p,expected in inv["inputs"].items()),"Original frozen payload stat")
    if resume:
        attempts=sorted(FORMAL.glob("ATTEMPT_*.json"))
        data.check(attempts and not (FORMAL/"COMPLETED.json").exists()
                   and not any(live(data.read(p)["process"]) for p in attempts),"Only an incomplete, exited attempt")
        index=len(attempts)
    else:
        FORMAL.mkdir()
        index=0
    clock,cpu=time.monotonic(),time.process_time()
    data.write(FORMAL/f"ATTEMPT_{index:03}.json",dict(process=identity(),threads=threads,**fingerprint()))
    try:
        with threadpool_limits(limits=threads):
            guard(threads)
            norm=data.normalization()
            paths,ledger=fit_stats(inv,FORMAL,norm,threads)
            publish(FORMAL/"MOMENT_LEDGER.json",ledger)
            coef=fit_coefficients(paths,FORMAL,threads)
            blocks=val_scores(inv,FORMAL,norm,coef,threads)
            result=selected(inv,FORMAL,coef,blocks)
            guard(threads)
            publish(FORMAL/"COMPLETED.json",dict(passed=True,process=identity(),**fingerprint(),scores=result,
                TRAIN_records=len(ledger["record_ledger"]),TRAIN_points=sum(r["target_times"] for r in ledger["record_ledger"]),
                per_record_G_moment_calls=sum(r["BLAS_blocks"] for r in ledger["record_ledger"]),
                TRAIN_subject_accumulators=70,distinct_fit_statistics=50,eigensystems=50,ridge_candidates=450,
                VAL_records=49,VAL_people=49,VAL_eligible_blocks=752,VAL_candidate_blocks=len(blocks),
                CPU_seconds_this_attempt=time.process_time()-cpu,elapsed_seconds_this_attempt=time.monotonic()-clock,
                peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                gpu=0,new_neural_forwards=0,new_neural_optimizations=0,old_test_reads=0))
    except BaseException as error:
        data.write(FORMAL/f"FAILURE_{index:03}.json",dict(process=identity(),message=str(error),
                  traceback=traceback.format_exc(),uncommitted_target_times=None,
                  CPU_seconds_this_attempt=time.process_time()-cpu,elapsed_seconds_this_attempt=time.monotonic()-clock))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--threads",type=int,default=16,choices=range(1,17))
    args=parser.parse_args()
    run(args.resume,args.threads)
