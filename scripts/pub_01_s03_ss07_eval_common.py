"""Prediction/metric algebra only. No dataset/checkpoint loader or launcher.

Numerical work belongs on server2203. Dataset readers must satisfy the final
checkpoint + prediction-independent-mask manifest BEFORE calling inference.
"""
import numpy as np
import torch
from pub_01_s03_ss07_history import causal_ema,predict_with_context
from pub_01_s03_ss07_incremental import summary


def aligned_chunks(length,global_offset=0):
    first=0
    while first<length:
        last=min(length,first+64-(global_offset+first)%64)
        yield first,last;first=last


@torch.no_grad()
def predict_raw_phase(decoder,records,device):
    """FT original global64 batches, including mixed-record batches.

    Unlike a stateful residual head, FT's raw decoder never splits a batch
    at a record boundary. Stream exact original phase order, then scatter.
    records yields (record_id, float32[N,192,64]) for ONE phase.
    """
    pending=[];destinations=[];output={};decoder.eval()
    def flush():
        if not pending:return
        x=torch.from_numpy(np.stack(pending)).to(device);p=decoder(x)
        assert p.shape==(len(pending),1,192)
        if not torch.isfinite(p).all():raise RuntimeError('nonfinite fine-tuned raw prediction')
        values=p[:,0].cpu().numpy()
        for (record,local),value in zip(destinations,values):output[record][local]=value
        pending.clear();destinations.clear()
    for record,windows in records:
        assert record not in output and windows.ndim==3 and windows.shape[1:]==(192,64)
        output[record]=np.empty((len(windows),192),dtype=np.float32)
        for local,value in enumerate(windows):
            pending.append(value);destinations.append((record,local))
            if len(pending)==64:flush()
    flush();return output


@torch.no_grad()
def predict_windows(model,z,record_key,mode='full',global_offset=0):
    assert z.ndim==3 and z.shape[1:]==(8,192)
    model.eval();state=None;pieces=[]
    for first,last in aligned_chunks(len(z),global_offset):
        p,state,_=model(z[first:last],state,first,record_key,mode=mode)
        assert p.shape==(last-first,1,192)
        if not torch.isfinite(p).all():raise RuntimeError('nonfinite window prediction; condition fails')
        pieces.append(p[:,0].cpu().numpy())
    return np.concatenate(pieces)


@torch.no_grad()
def intervene_aligned(model,z,record_key,kind='true',global_offset=0,donor_observations=None,neutral=None,permutation=None,mode='full'):
    """Same reviewed intervention algebra, original global64 record splits.

    Preserving batch partition avoids cuDNN TF32 convolution rounding changes
    from repacking record tails. Current z and all score coordinates fixed.
    """
    assert model.conditioning=='past'
    assert kind in ('true','wrong_history','shuffled','reset','content_blind','older_outside_fused_support')
    observations=torch.cat([summary(model.head(z[a:b]) if model.family=='output_prior' else z[a:b])
        for a,b in aligned_chunks(len(z),global_offset)])
    values=observations
    if kind=='wrong_history':
        assert donor_observations is not None and donor_observations.shape==observations.shape;values=donor_observations
    elif kind=='shuffled':
        assert permutation is not None and sorted(permutation.tolist())==list(range(len(z)))
        values=observations.index_select(0,permutation)
    elif kind=='content_blind':
        assert neutral is not None and neutral.shape==(observations.shape[1],);values=neutral[None].expand_as(observations)
    state=None;contexts=[]
    for first,last in aligned_chunks(len(z),global_offset):
        before,_,state=model.core.safe_history(values[first:last],state,first,record_key);contexts.append(before)
    context=torch.cat(contexts)
    if kind=='reset':
        before,_,_=model.core.evolve(observations[:,None,:],None);context=before[:,0]
    elif kind=='older_outside_fused_support':
        before,_,_=model.core.evolve(observations[:1,None,:],None);context=torch.cat([before[:,0],context[:-1]])
    pieces=[]
    for first,last in aligned_chunks(len(z),global_offset):
        prediction=predict_with_context(model,z[first:last],context[first:last],mode)
        if not torch.isfinite(prediction).all():raise RuntimeError('nonfinite intervention prediction')
        pieces.append(prediction[:,0].cpu().numpy())
    return np.concatenate(pieces)


def fuse(windows,common_samples,tau=0.):
    assert windows.shape==((common_samples-192)//96+1,192)
    if not np.isfinite(windows).all():raise RuntimeError('nonfinite window predictions')
    total=np.zeros(common_samples,dtype=np.float64);weight=np.zeros_like(total);h=np.hanning(192)
    for i,value in enumerate(windows):
        s=i*96;total[s:s+192]+=value*h;weight[s:s+192]+=h
    covered=weight>1e-8;total[covered]/=weight[covered]
    return causal_ema(total,covered,tau),covered


def correlation(prediction,truth):
    assert prediction.shape==truth.shape and prediction.ndim==1
    if not np.isfinite(prediction).all() or np.std(prediction)==0:
        raise RuntimeError('undefined prediction; no selective block/trial removal')
    assert np.isfinite(truth).all() and np.std(truth)>0,'fixed eligible target changed'
    r=float(np.corrcoef(prediction,truth)[0,1]);assert np.isfinite(r)
    return r


def score_record(windows,target,row,mask,tau=0.):
    fused,covered=fuse(windows,row['common_samples'],tau);target=np.asarray(target).reshape(-1)
    assert len(target)>=row['common_samples'] and mask['record']==row['record']
    segments=[]
    for block in mask['blocks']:
        s=block['start'];keep=covered[s:s+3840]
        assert len(keep)==3840 and int(keep.sum())==block['samples']
        if not block['eligible']:continue
        a=fused[s:s+3840][keep];b=np.asarray(target[s:s+3840],dtype=np.float64)[keep]
        r=correlation(a,b);error=float(np.mean((a-b)**2))
        segments.append({k:row[k] for k in ['record','phase','subject','story','source_family']}|
            {'start':s,'samples':len(a),'r':r,'MSE_descriptive':error,
             'R2_descriptive':float(1-np.sum((a-b)**2)/np.sum((b-b.mean())**2))})
    assert len(segments)==sum(b['eligible'] for b in mask['blocks'])
    return segments,fused


def summarize_absolute(segments):
    people=[]
    for phase in ('test_seen','test_heldout'):
        for subject in sorted({r['subject'] for r in segments if r['phase']==phase}):
            rows=[r for r in segments if r['phase']==phase and r['subject']==subject]
            people.append({'phase':phase,'subject':subject,'r':float(np.mean([r['r'] for r in rows])),
                'blocks':len(rows),'records':len({r['record'] for r in rows})})
    means={phase:float(np.mean([r['r'] for r in people if r['phase']==phase]))
        for phase in ('test_seen','test_heldout') if any(r['phase']==phase for r in people)}
    weighted=2/3*means['test_seen']+1/3*means['test_heldout'] if len(means)==2 else None
    return {'phase_macro_r':means,'weighted_r':weighted,'people':people,'blocks':len(segments)}


def paired_arrays(results,conditions,seeds,eligible_keys=None):
    """Require exact SAME keys across all requested seed/condition cells.

    eligible_keys is frozen metadata/mask membership, NEVER prediction quality.
    Main contrasts use ALL keys; donor interventions pass their OWN subset and
    compare the corresponding true-history subset, never the whole main score.
    """
    canonical=None;metadata=None;arrays=[]
    allowed=None if eligible_keys is None else set(eligible_keys)
    for condition in conditions:
        seed_values=[]
        for seed in seeds:
            source=results[(condition,seed)];mapping={(r['record'],r['start']):r for r in source}
            assert len(mapping)==len(source),'duplicate score keys'
            if allowed is not None:
                assert allowed.issubset(mapping),'missing predeclared paired key'
                mapping={key:mapping[key] for key in allowed}
            keys=sorted(mapping)
            if canonical is None:
                canonical=keys;metadata=[mapping[key] for key in keys];assert canonical
            assert keys==canonical,'different coverage cannot be averaged or subtracted'
            for key,original in zip(canonical,metadata):
                assert all(mapping[key][field]==original[field] for field in ['phase','subject','story','source_family','samples'])
            values=np.array([mapping[key]['r'] for key in canonical]);assert np.isfinite(values).all()
            seed_values.append(values)
        arrays.append(np.stack(seed_values))
    return np.stack(arrays),metadata


def descriptive_family_slices(diff,blocks):
    """Coverage-explicit per-family / leave-one-family-out descriptions."""
    assert diff.ndim==3 and diff.shape[2]==len(blocks)
    output=[]
    for family in sorted({b['source_family'] for b in blocks}):
        for mode in ['family_only','leave_family_out']:
            ids=[i for i,b in enumerate(blocks) if (b['source_family']==family)==(mode=='family_only')]
            coverage={};means={}
            for phase in ['test_seen','test_heldout']:
                people=sorted({blocks[i]['subject'] for i in ids if blocks[i]['phase']==phase})
                values=[]
                for subject in people:
                    selected=[i for i in ids if blocks[i]['phase']==phase and blocks[i]['subject']==subject]
                    values.append(diff[:,:,selected].mean(2))
                coverage[phase]={'people':len(people),'blocks':sum(blocks[i]['phase']==phase for i in ids)}
                means[phase]=np.stack(values,2).mean((1,2)).tolist() if values else None
            weighted=(2/3*np.array(means['test_seen'])+1/3*np.array(means['test_heldout'])).tolist() if all(v is not None for v in means.values()) else None
            output.append({'family':family,'slice':mode,'coverage':coverage,'phase_mean_differences':means,
                'weighted_differences':weighted,'descriptive_changed_population_not_a_new_primary_endpoint':True})
    return output
