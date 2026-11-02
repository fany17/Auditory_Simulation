"""Freeze source/inputs/target-only support before the fixed diagnostic NN pass."""
import sys,time,os
from pathlib import Path
import numpy as np
from pub_01_s03_ss07_error_decomposition_core import *

def prepare():
    started=time.time();assert not (ROOT/'PROTOCOL.json').exists()
    old=read(D12/'PROTOCOL.json');assert sha(D12/'PROTOCOL.json')=='b8e3552f33557be86609aa244c6d837ef63d9d765f518b5e3df58b46d89289f3'
    d8=read(BASE/'s03_finetuned_readout_transfer_20261101/PROTOCOL.json');records=d8['records'];masks={};counts={};inputs={}
    for item in old['cache_metadata']:
        p=Path(item['path']);st=p.stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'];assert sha(p)==item['sha256'];inputs[str(p)]=dict(item)
    for seed in SEEDS:
        for arm in ARMS:
            p=D12/f'fits/{arm}_seed{seed}/last.pt';st=p.stat();inputs[str(p)]=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(p))
    for phase in ['train','val']:
        blocks=[];windows=0
        for row in records[phase]:
            assert row['offset']==windows;windows+=row['windows'];n=row['common_samples'];weight,keep=weights(row['windows'],n)
            original=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
            truth=np.asarray(original,dtype='f4').astype('f8') if phase=='train' else np.asarray(original,dtype='f8')
            assert len(truth)>=n
            for first in range(0,n-3840+1,3840):
                k=keep[first:first+3840];y=truth[first:first+3840][k]
                reason='insufficient_overlap_support' if len(y)<3800 else 'nonfinite_target' if not np.isfinite(y).all() else 'constant_target' if np.std(y)==0 else None
                blocks.append(dict(phase=phase,record=row['record'],subject=row['subject'],start=first,stop=first+3840,covered=int(k.sum()),eligible=reason is None,reason=reason))
        tally={}
        for b in blocks:
            if b['eligible']:tally[b['subject']]=tally.get(b['subject'],0)+1
        assert set(tally)=={r['subject'] for r in records[phase]},'Explicitly preserve all persons; no target-only empty-person omission'
        masks[phase]=dict(blocks=blocks,person_blocks=tally,people=len(tally),records=len(records[phase]),windows=windows,
            eligible_blocks=sum(b['eligible'] for b in blocks),target_only=True)
        counts[phase]=dict(records=len(records[phase]),people=len(tally),blocks=masks[phase]['eligible_blocks'],windows=windows,
            effective_windows=sum(r['windows']-9 for r in records[phase]))
    actual={(b['record'],b['start'],b['covered'],b['eligible']) for b in masks['val']['blocks']}
    reference={(b['record'],b['start'],b['samples'],b['eligible']) for b in old['mask']['blocks']}
    assert actual==reference and masks['val']['eligible_blocks']==752 and masks['val']['people']==49
    oldtrain=read(BASE/'s03_incremental_continuity_20261101/block_eligibility/train.json')
    assert {(b['record'],b['start'],b['covered'],b['eligible']) for b in masks['train']['blocks']}=={(b['record'],b['start'],b['samples'],b['eligible']) for b in oldtrain['blocks']}
    dump(ROOT/'SUPPORT_MASK.json',masks)
    pins={}
    for name in ['core','prepare','run','queue','report']:
        p=ROOT/f'input/pub_01_s03_ss07_error_decomposition_{name}.py';pins[str(p)]=sha(p)
    for p in [ROOT/'PLAN.md',ROOT/'COORDINATOR_MATH_AND_SCOPE.md',ROOT/'SUPPORT_MASK.json',D12/'PROTOCOL.json',
        D12/'input/pub_01_s03_ss07_full_ft_history_core.py',BASE/'s03_current_readout_internal_audit_20261101/input/pub_01_s03_ss07_generalization_core.py']:
        pins[str(p)]=sha(p)
    for path,digest in old['source_pins'].items():
        assert sha(path)==digest,path;pins[path]=digest
    protocol=dict(frozen=True,time=time.time(),deadline_unix=time.time()+1800,source_pins=pins,inputs=inputs,records=records,support=counts,
        seeds=SEEDS,arms=ARMS,checkpoint='Every last epoch19, no selection',effective_residual_NN_windows=2508111,
        expected_per_seed_residual_windows=836037,new_encoder_head_windows=0,optimizer_updates=0,learning_model_fits=0,
        descriptive_projection_statistics_not_learning_model_fits=True,primary_projection_rows=9*(counts['train']['blocks']+counts['val']['blocks']),
        target_units=dict(train='Original record envelope cast float32 then float64, same inherited window-cache values',val='Original continuous record target float64'),
        projection=dict(centering='Within each fixed covered physical60s block',variance='Population ddof0',alpha='dot(u,v)/dot(u,u)',
            e='v-alpha*u; standard centered population variance/covariance; mean(e) reported',
            r='((1+alpha)*r_anchor+cov(e,y)/(sd(a)*sd(y)))/sqrt((1+alpha)^2+var(e)/var(a))',
            MSE_bias='2*mean(d)*(mean(a)-mean(y))+mean(d)^2',MSE_parallel='(alpha^2+2*alpha)*var(a)-2*alpha*cov(a,y)',
            MSE_orthogonal='var(e)-2*cov(e,y)',r_aggregation='Evaluate identity blockwise, then block->person->phase; never plug average alpha into nonlinear identity'),
        undefined_policy='Nonfinite/zero variance explicit None, person preserved; any missing metric makes that full-person/phase metric undefined; no epsilon/ridge or row dropping',
        low_variance_policy='var<=64*float64_eps*mean(x^2) flags only; positive finite denominators still computed, no exclusions',
        gradients=dict(phase='TRAIN only, once per seed at frozen anchor',coordinates='Predicted window p before raw residual factor0.1; same i>=9 learnability mask',
            gM='2*(anchor_window-y_float32)/(252454*192); all original windows including fallback constants in denominator; not online SGD/Adam gradient',
            gF='2*(fused_anchor-y_float32)/covered_points/(70*person_blocks)',
            gR='(r*u/dot(u,u)-yc/sqrt(dot(u,u)*dot(yc,yc)))/(70*person_blocks)',
            adjoint='For every original record, scatter physical gradients of all fixed blocks, then h[t]/weight[96i+t] for each window coordinate, finally i>=9 mask',
            pairs=['M/F','F/R','M/R'],aggregation='Sum squared norms/dots, then cosine; record/person distributions retained, not average cosines',
            no_gradient_cache=True,angular_identity='P_perp gF=(2*norm(u)*norm(yc)/n)*gR before Hann; positive scale, no intrinsic single-block angular conflict',
            fallback='i<9 constant derivative zero; partial tails/outside eligible blocks have gF/gR0 but gM retains192-point supervision including zero-Hann endpoints'),
        budget=dict(wall_seconds=1800,max_idle_GPUs=2,worker_RSS_bytes=8*1024**3,worker_GPU_allocated_bytes=2*1024**3,root_bytes=128*1024**2,history_gradient_disk_cache_bytes=0,each_disk_free_bytes=20*1024**3),
        forbidden=['old test','DTU','ds004408','older replay','new training','deployable postprocessing','VAL alpha selection'],scientific_status='Fixed diagnostic, not a learned model or causal optimization inference')
    dump(ROOT/'PROTOCOL.json',protocol)
    check=synthetic_checks();check.update(wall_seconds=time.time()-started,CPU_seconds=time.process_time(),new_NN=0)
    dump(ROOT/'ENGINEERING_CHECK.json',check)
    print(dict(passed=True,support=counts,projection_rows=protocol['primary_projection_rows'],NN_windows=2508111,protocol_sha256=sha(ROOT/'PROTOCOL.json')),flush=True)

if __name__=='__main__':prepare()
