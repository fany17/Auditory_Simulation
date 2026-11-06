"""D28 bounded CPU engineering and final source freeze."""
import argparse
import importlib.util
import math
import time

import numpy as np
from threadpoolctl import threadpool_info,threadpool_limits

import pub_01_s03_ss07_matched_raw_ridge_data as data
import pub_01_s03_ss07_matched_raw_ridge_run as run


def prepare():
    inv=data.inventory()
    data.write(run.INVENTORY,inv)
    independent=data.read(data.ROOT/"COORDINATOR_GEOMETRY.json")
    n=inv["personal_samples"]
    data.check(n==17232052 and len(inv["groups"])==70,"Independent personal scope")
    g=sum(r["common_samples"] for r in inv["train"])
    val=sum(r["common_samples"] for r in inv["val"])
    personal={person:dict(records=len(rows),targets=sum(r["common_samples"] for r in rows))
              for person in inv["personal_subjects"]
              for rows in [[r for r in inv["train"] if r["subject"]==person]]}
    data.check(independent["personal"]==personal
               and independent["TRAIN_unique_record_target_times"]==g
               and independent["VAL_unique_record_target_times"]==val
               and independent["dimension"]==data.DIM,"Match independent preflight geometry")
    budget=dict(TRAIN_shared_points=g,TRAIN_personal_logical_points=n,
                shared_record_statistics_physical_points=g,logical_sum_G_and_I_points=g+n,
                TRAIN_records=426,TRAIN_personal_records=305,raw_subject_accumulators=70,
                TRAIN_rank_k_calls=sum(math.ceil(r["common_samples"]/data.BLOCK) for r in inv["train"]),
                fitted_statistics=50,eigensystems=50,candidates=450,
                VAL_points=val,VAL_candidate_prediction_points=val*18,VAL_candidate_blocks=752*18,
                features=data.DIM,per_record_block_max=4096,alphas=list(data.ALPHAS),
                engineering_max=dict(synthetic_small_solutions=2,actual_2112_solutions=1,
                                     actual_TRAIN_target_rows=2048,max_samples_read_per_record=4640,
                                     TRAIN_scoring_segments_of_4608=2,VAL_performance=0),
                author_independent_geometry_sha256=data.digest(data.ROOT/"COORDINATOR_GEOMETRY.json"),
                resource=dict(engineering_threads=4,formal_threads=16,RSS_GiB=8,writer_GiB=8,SATA_reserve_GiB=64),
                no_GPU_or_NN=True)
    data.write(data.ROOT/"BUDGET.json",budget)


def synthetic():
    rows=[np.c_[np.linspace(1.,2.,7),np.zeros(7)],np.c_[np.linspace(5.,3.,5),np.zeros(5)]]
    targets=[np.linspace(.2,.7,7),np.linspace(-1.,3.,5)]
    lags=(0,1,2)
    arrays=[data.design(v,0,len(v),lags) for v in rows]
    # Last row of the first record reads that record only, with two absent lags.
    data.check(np.array_equal(arrays[0][:,-1],[rows[0][-1,0],0,0,0,0,0]),"Explicit record tail")
    data.check(np.array_equal(arrays[1][:,0],rows[1][:3].reshape(-1)),"Lag major layout")
    block=data.design(rows[0],0,2,lags)
    data.check(np.array_equal(block,arrays[0][:,:2]) and block[4,1]==rows[0][3,0],
               "Within-record next-32 dependency is not a block boundary")
    q=data.Sufficient(6)
    q.add(arrays[0],targets[0])
    raw_rows=[dict(record="a",common_samples=7),dict(record="b",common_samples=5)]
    fixture=data.ROOT/"RESUME_CANARY_V01"
    fixture.mkdir()
    prefix=fixture/"CURRENT.npz"
    fake={"synthetic":"boundary"}
    metadata=dict(fingerprint=fake,record_ledger=[dict(record="a")])
    run.save_stats(prefix,q,metadata)
    resumed,stored=run.read_stats(prefix)
    offset=run.checked_prefix(resumed,stored,raw_rows,fake)
    data.check(offset==1,"Exact subject prefix restored")
    resumed.add(arrays[offset],targets[offset])
    stored["record_ledger"].append(dict(record="b"))
    data.check(run.checked_prefix(resumed,stored,raw_rows,fake,complete=True)==2 and resumed.n==12,
               "Each physical record added once")
    immutable=fixture/"SUMMARY.json"
    run.publish(immutable,{"records":2})
    mtime=immutable.stat().st_mtime_ns
    run.publish(immutable,{"records":2})
    rejected=False
    try:
        run.publish(immutable,{"records":3})
    except RuntimeError:
        rejected=True
    data.check(rejected and immutable.stat().st_mtime_ns==mtime,"Same result reused, conflicting evidence refused")
    alpha=.01
    streaming=data.solutions(resumed,(alpha,))
    x=np.concatenate(arrays,axis=1).T
    y=np.concatenate(targets)
    m=x.mean(0);s=x.std(0);s[s==0]=1
    yz=(y-y.mean())/y.std();xz=(x-m)/s
    direct=np.linalg.solve(xz.T@xz/len(y)+alpha*np.eye(x.shape[1]),xz.T@yz/len(y))
    error=float(abs(streaming["weights"][:,0]-direct).max())
    data.check(error<=1e-11 and int(streaming["constant"].sum())==3,"Direct normalized ridge bridge")
    return dict(passed=True,two_small_solutions=True,coefficient_max_abs=error,
                zero_variance_columns=3,tail_and_block_border_verified=True,
                resumed_record_prefix=offset,exactly_two_records_after_resume=True,
                same_receipt_preserved=True,conflict_rejected=True)


def engineer():
    clock,cpu=time.monotonic(),time.process_time()
    inv=data.read(run.INVENTORY)
    with threadpool_limits(limits=4):
        run.guard(4)
        synthetic_result=synthetic()
        norm=data.normalization()
        train=inv["train"][:2]
        examples=[]
        fit=data.Sufficient(data.DIM)
        for row in train:
            data.check(row["phase"]=="train","Engineering TRAIN only")
            x,y=data.load_record(row,norm,limit=4640)
            data.check(len(x)==len(y)==4640,"Bounded actual record slice")
            fit.add(data.design(x,0,1024),y[:1024])
            examples.append((row,x,y))
        data.check(fit.n==2048,"Exact real engineering support")
        solved=data.solutions(fit,(.01,))
        path=data.HOME/"s03_current_readout_internal_audit_20261101/input/pub_01_s03_ss07_generalization_core.py"
        spec=importlib.util.spec_from_file_location("d28_old_score_bridge",path)
        legacy=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        errors=[]
        for row,x,y in examples:
            pred=np.concatenate([(data.design(x,first,min(first+4096,4608)).T@solved["raw_weights"]
                                 +solved["raw_intercepts"])[:,0] for first in (0,4096)])
            clipped=dict(row,common_samples=4608)
            mask=[dict(start=0,samples=3839,eligible=True)]
            actual=run.scorer.record_scores(clipped,pred,y[:4608],mask)[0]
            waves=np.lib.stride_tricks.sliding_window_view(pred,192)[::96].copy()
            native=legacy.measure(waves,y[:4608],clipped,mask)["segments"][0]
            errors.append({k:abs(actual[k]-native[k]) for k in ("r","MSE")})
        data.check(max(max(x.values()) for x in errors)<1e-12,"Original common-support numeric bridge")
        run.guard(4)
        actual_threads=threadpool_info()
    data.write(data.ROOT/"ENGINEERING.json",dict(passed=True,process=run.identity(),
          synthetic=synthetic_result,actual_records=[r["record"] for r,_,_ in examples],actual_rows=2048,
          actual_features=data.DIM,samples_read_per_record=4640,TRAIN_predictions_of_4608=2,
          shared_metric_absolute_errors=errors,normal_equation_residual=float(solved["normal_equation_residual"][0]),
          three_total_engineering_solutions=3,engineering_coefficients_saved=0,VAL_performance_calls=0,
          neural_forwards=0,gpu=0,actual_threadpools=actual_threads,
          elapsed_seconds=time.monotonic()-clock,CPU_seconds=time.process_time()-cpu,old_test_reads=0))


def freeze():
    data.check(data.read(data.ROOT/"ENGINEERING.json")["passed"],"Actual bounded engine")
    inv=data.read(run.INVENTORY)
    data.check(all(data.stamp(path)==stat for path,stat in inv["inputs"].items()),"Original input identities")
    data.write(run.GATE,dict(frozen=True,inventory_sha256=data.digest(run.INVENTORY),sources=run.pins(),
          features=data.DIM,lag_samples=list(data.LAGS),positive_lag="EEG[t+lag] predicts target[t]",
          tail="normalized EEG zero padding only at each record tail before feature standardization",
          regression="G/n + alpha*I in normalized design and target, no jitter; intercept via training centering",
          model_fits=50,candidates=450,alphas=list(data.ALPHAS),TRAIN_only_statistics=True,
          evaluation="original continuous envelope precision on old positive Hann support, 752 blocks then 49 people",
          training_target="old float32 envelope promoted to float64 for all fitting math",
          selection="G original full 49-person macro r; each I original person's block mean r; ties smaller alpha",
          max_threads=16,formal_gpu=0,no_neural=True,old_test_reads=0))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","engineer","freeze"))
    {"prepare":prepare,"engineer":engineer,"freeze":freeze}[parser.parse_args().stage]()
