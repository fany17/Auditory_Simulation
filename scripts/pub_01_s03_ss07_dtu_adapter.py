"""Proposed no-label/current3s DTU adapter. Not an evaluation launcher.

Proposal only until external protocol review; no fitted target-domain stats.
"""
import numpy as np


def native_windows(eeg,labels,scalp_names):
    assert eeg.ndim==2 and len(labels)==eeg.shape[1] and len(scalp_names)==64
    assert len(set(labels))==len(labels) and len(set(scalp_names))==64
    x=np.asarray(eeg[:,[labels.index(n) for n in scalp_names]],dtype=np.float64)
    assert np.isfinite(x).all()
    # Pointwise physical reference: no temporal support added.
    x=x-x.mean(1,keepdims=True)
    starts=np.arange(0,len(x)-192+1,96);windows=[];zero_variance=[]
    for s in starts:
        current=x[s:s+192];mean=current.mean(0);std=current.std(0,ddof=0)
        constant=std==0
        normalized=(current-mean)/np.where(constant,1.,std)
        normalized[:,constant]=0.
        assert np.isfinite(normalized).all()
        windows.append(normalized.astype(np.float32));zero_variance.append(int(constant.sum()))
    return np.stack(windows),starts,zero_variance
