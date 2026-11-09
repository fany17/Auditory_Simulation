"""Read-only history interventions; current native z NEVER changes.

This module has no dataset/checkpoint loader. Call only after the evaluation
plan is frozen. Donor selection/permutations are metadata-defined upstream.
"""
import numpy as np
import torch
from torch.nn import functional as F
from pub_01_s03_ss07_incremental import summary


@torch.no_grad()
def contexts(model,observations,chunk=64):
    """Each row is context BEFORE that parity's current observation update."""
    assert model.conditioning=='past' and observations.ndim==2
    state=None;pieces=[]
    for first in range(0,len(observations),chunk):
        before,_,state=model.core.safe_history(observations[first:first+chunk],state,first,'intervention_record')
        pieces.append(before)
    return torch.cat(pieces)


def predict_with_context(model,z,context,mode='full'):
    """Replicate only prediction algebra; verify against frozen forward first."""
    delta=model.residual(context)
    if model.family=='output_prior':
        return model.head(z)+.1*F.interpolate(delta[:,None,:],size=192,mode='linear',align_corners=False)
    if model.family=='m4':
        coefficients=delta.reshape(-1,8,3);transformed=z
        for j,lag in enumerate((4,8,16)):
            transformed=transformed+.1*coefficients[:,:,j,None]*(F.pad(z[:,:,:-lag],(lag,0))-z)
        return model.head(transformed)
    innovation=z
    if model.family=='m5':
        prior=F.interpolate(model.prior(context).reshape(-1,8,6),size=192,mode='linear',align_corners=False)
        innovation=z-prior
        if mode=='prior_only':return model.head(prior)
    return model.head(z+.1*torch.bmm(delta.reshape(-1,8,8),innovation))


@torch.no_grad()
def intervene(model,z,kind='true',donor_observations=None,neutral=None,permutation=None,mode='full'):
    """Entire-record intervention: every window receives rebuilt context.

    wrong_history takes a fixed donor observation sequence of equal length.
    shuffled takes an upstream-frozen permutation of observations. It is an
    OFFLINE distribution intervention and may contain donor later times; no
    deployable/causal performance claim follows from it.
    older_outside_fused_support uses context_before(i-1), not i's own parity.
    This changes both recency and parity phase; it is NOT a pure delay effect.
    """
    assert kind in ('true','wrong_history','shuffled','reset','content_blind','older_outside_fused_support')
    assert z.ndim==3 and z.shape[1:]==(8,192)
    obs=summary(model.head(z)) if model.family=='output_prior' else summary(z)
    if kind=='wrong_history':
        assert donor_observations is not None and donor_observations.shape==obs.shape
        values=donor_observations
    elif kind=='shuffled':
        assert permutation is not None and sorted(permutation.tolist())==list(range(len(obs)))
        values=obs.index_select(0,permutation)
    elif kind=='content_blind':
        assert neutral is not None and neutral.shape==(obs.shape[1],)
        # Same record-local time and update count, train-only neutral content.
        values=neutral[None].expand_as(obs)
    else:values=obs
    context=contexts(model,values)
    if kind=='reset':
        # core.context bias is retained: reset means empty history, not h=0
        # AFTER context projection, which would be a different intervention.
        before,_,_=model.core.evolve(obs[:,None,:],None)
        context=before[:,0]
    elif kind=='older_outside_fused_support':
        before,_,_=model.core.evolve(obs[:1,None,:],None)
        context=torch.cat((before[:,0],context[:-1]),0)
    pieces=[predict_with_context(model,z[i:i+64],context[i:i+64],mode) for i in range(0,len(z),64)]
    return torch.cat(pieces),context


def hann_support_audit(windows,common_samples):
    """Actual nonzero Hann support vs UNION of native input supports.

    The conservative union includes every baseline window contributing to
    ANY nonzero output coordinate of p_i. History's last source i-3 ends at
    96i-96, no later than that union's earliest start (half-open intervals).
    First/last/cold-start/coverage gaps are explicitly included.
    """
    assert windows==(common_samples-192)//96+1
    h=np.hanning(192);positive=np.flatnonzero(h>1e-8)
    weight=np.zeros(common_samples)
    for i in range(windows):weight[96*i:96*i+192]+=h
    covered=weight>1e-8;rows=[]
    for i in range(windows):
        coordinates=96*i+positive
        contributors=[]
        for j in range(max(0,i-2),min(windows,i+3)):
            local=coordinates-96*j;valid=(local>=0)&(local<192)
            if np.any(h[local[valid]]>1e-8):contributors.append(j)
        earliest=min(contributors)*96
        last=i-3 if i>=3 else None
        history_end=96*last+192 if last is not None else None
        assert history_end is None or history_end<=earliest
        assert np.all(covered[coordinates])
        rows.append({'window':i,'nonzero_prediction_start':int(coordinates[0]),
            'nonzero_prediction_end_inclusive':int(coordinates[-1]),
            'baseline_contributors':contributors,'fused_baseline_input_union_start':earliest,
            'older_history_last_source':last,'older_history_support_end_exclusive':history_end,
            'cold_start':last is None})
    return {'windows':windows,'common_samples':common_samples,'covered_samples':int(covered.sum()),
        'uncovered_samples':np.flatnonzero(~covered).tolist(),'rows':rows,
        'interpretation':'older history outside fused baseline input union; recency AND parity phase changed'}


def causal_ema(prediction,covered,tau,sampling_rate=64):
    """Post-Hann EMA, resets at every coverage gap; tau0 exact identity."""
    assert prediction.ndim==covered.ndim==1 and prediction.shape==covered.shape and tau>=0
    result=prediction.copy()
    if tau==0:return result
    alpha=np.exp(-1/(sampling_rate*tau));state=None
    for i in range(len(prediction)):
        if not covered[i]:state=None;continue
        assert np.isfinite(prediction[i])
        state=prediction[i] if state is None else alpha*state+(1-alpha)*prediction[i]
        result[i]=state
    return result
