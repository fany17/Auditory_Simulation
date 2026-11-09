"""Fixed four-readout algebra and record-local scoring; no fits, I/O or NN."""
import numpy as np
from pub_01_s03_ss07_openneuro_metrics import coverage,pearson


def normalized(x,stats,dtype):
    value=(np.asarray(x,dtype)-np.asarray(stats['mean'],dtype))/np.asarray(stats['std'],dtype)
    value[:,stats['constant']]=0
    return value


def readout(z,p,calibration,stats,coefficient=None,dtype=np.float64):
    dtype=np.dtype(dtype).type
    assert z.shape==(len(p),8,192) and p.shape==(len(z),192)
    assert dtype in (np.float32,np.float64) and calibration['a']>0
    base=dtype(calibration['a'])*np.asarray(p,dtype)+dtype(calibration['b'])
    if coefficient is None:return base
    c=normalized(np.asarray(z[4:],dtype).reshape(-1,1536),stats[0],dtype)
    pp=normalized(base[4:],stats[1],dtype)
    q=normalized(c*c,stats[2],dtype)
    features=np.concatenate((c/np.sqrt(dtype(1536)),pp/np.sqrt(dtype(192)),q/np.sqrt(dtype(1536))),axis=1)
    result=base.copy()
    result[4:]+=features@np.asarray(coefficient['coef'],dtype)+np.asarray(coefficient['bias'],dtype)
    assert result.dtype==dtype and np.array_equal(result[:4],base[:4])
    return result


def fuse(pred,length,dtype=np.float64):
    dtype=np.dtype(dtype).type;pred=np.asarray(pred,dtype)
    assert pred.shape==((length-192)//96+1,192) and np.isfinite(pred).all()
    total=np.zeros(length,dtype);weight=np.zeros(length,dtype);h=np.hanning(192).astype(dtype)
    for i,value in enumerate(pred):
        start=i*96;total[start:start+192]+=value*h;weight[start:start+192]+=h
    covered=weight>0;total[covered]/=weight[covered]
    return total,covered


def maybe_corr(x,y):
    x=np.asarray(x,'f8');y=np.asarray(y,'f8')
    if len(x)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():return None
    if x.max()==x.min() or y.max()==y.min():return None
    return pearson(x,y)


def measure(pred,target,row,seed,role):
    pred=np.asarray(pred,'f8');target=np.asarray(target,'f8')[:row['length']]
    assert pred.shape==(row['windows'],192) and np.isfinite(pred).all() and np.isfinite(target).all()
    n=len(target);w=len(pred);h=np.hanning(192);fused,keep=fuse(pred,n)
    starts,weights=coverage(n);assert int(keep.sum())==row['covered'] and np.array_equal(keep,weights>0)
    yw=np.stack([target[s:s+192] for s in starts])
    xc=pred-pred.mean(1,keepdims=True);yc=yw-yw.mean(1,keepdims=True)
    defined=(np.ptp(pred,axis=1)>0)&(np.ptp(yw,axis=1)>0)
    r3=(xc[defined]*yc[defined]).sum(1)/np.sqrt((xc[defined]**2).sum(1)*(yc[defined]**2).sum(1))
    weighted=np.zeros(n);disagreement=np.zeros(n)
    contributor_min=np.full(n,np.inf);contributor_max=np.full(n,-np.inf)
    for i,value in enumerate(pred):
        s=i*96;weighted[s:s+192]+=h*(value-target[s:s+192])**2
        contributor_min[s:s+192]=np.minimum(contributor_min[s:s+192],np.where(h>0,value,np.inf))
        contributor_max[s:s+192]=np.maximum(contributor_max[s:s+192],np.where(h>0,value,-np.inf))
    weighted[keep]/=weights[keep]
    left=pred[:-1,96:];right=pred[1:,:96];alpha=h[96:]/(h[96:]+h[:96])
    for i in range(w-1):
        s=(i+1)*96;disagreement[s:s+96]=alpha*(1-alpha)*(left[i]-right[i])**2
    error=(fused-target)**2
    identity=float(abs(weighted[keep]-error[keep]-disagreement[keep]).max())
    assert identity<=1e-10*max(1.,float(weighted[keep].max()))
    def span(start,stop):
        k=keep[start:stop];p=fused[start:stop][k];y=target[start:stop][k]
        constant=bool(k.any() and contributor_min[start:stop][k].min()==contributor_max[start:stop][k].max())
        return dict(covered=int(k.sum()),r=None if constant else maybe_corr(p,y),
            MSE=float(error[start:stop][k].mean()) if k.any() else None,
            weighted_single_MSE=float(weighted[start:stop][k].mean()) if k.any() else None,
            disagreement=float(disagreement[start:stop][k].mean()) if k.any() else None)
    blocks=[]
    for block in row['blocks']:
        if not block['eligible']:continue
        value=span(block['start'],block['stop']);assert value['covered']==block['covered']
        # Null r stays in the fixed support and invalidates its comparison; never drop it.
        blocks.append(dict(subject=row['subject'],record=row['record'],seed=seed,role=role,
            start=block['start'],stop=block['stop'],**value))
    tail=span(row['tail']['start'],n)
    if not row['tail']['eligible_r']:tail['r']=None
    last=(w-1)*96
    result=dict(subject=row['subject'],record=row['record'],seed=seed,role=role,windows=w,
        blocks=blocks,all_primary_blocks_defined=all(r['r'] is not None for r in blocks),
        r3s=float(r3.mean()) if defined.all() else None,r3s_sum_defined=float(r3.sum()),
        defined_window_r=int(defined.sum()),undefined_window_r=int((~defined).sum()),
        window_MSE=float(((pred-yw)**2).mean()),whole_record=span(0,n),tail=tail,
        tail_start=row['tail']['start'],tail_length=n-row['tail']['start'],
        uniform_overlap_MSE=float(((left-right)**2).mean()),overlap_pairs=w-1,
        Hann_identity_max_abs=identity,max_abs_window_prediction=float(abs(pred).max()),
        max_abs_fused_prediction=float(abs(fused[keep]).max()),
        first4_window_MSE=float(((pred[:4]-yw[:4])**2).mean()),
        prefix_before_learning=span(0,min(385,n)),first_learning_overlap=span(385,min(480,n)),
        final_single_window_tail=span(last+96,min(last+191,n)))
    return result,fused,keep


def precision_diagnostic(reference64,pred32,target,row):
    assert pred32.dtype==np.float32
    fused32,keep=fuse(pred32,row['length'],np.float32)
    fused64,k=fuse(reference64,row['length'],np.float64);assert np.array_equal(keep,k)
    target=np.asarray(target,'f8')[:row['length']]
    p32=fused32.astype('f8');error=p32-fused64
    return dict(record=row['record'],whole_record_waveform_max_abs=float(abs(error[k]).max()),
        fused_MSE32=float(((p32[k]-target[k])**2).mean()),fused_MSE64=float(((fused64[k]-target[k])**2).mean()),
        whole_record_r32=maybe_corr(p32[k],target[k]),whole_record_r64=maybe_corr(fused64[k],target[k]),
        readout32_Hann32=True,blocks=[dict(start=b['start'],stop=b['stop'],
            r32=maybe_corr(p32[b['start']:b['stop']][k[b['start']:b['stop']]],target[b['start']:b['stop']][k[b['start']:b['stop']]]),
            r64=maybe_corr(fused64[b['start']:b['stop']][k[b['start']:b['stop']]],target[b['start']:b['stop']][k[b['start']:b['stop']]]))
            for b in row['blocks'] if b['eligible']])


def synthetic_check():
    rng=np.random.default_rng(20261102);w=87;n=(w-1)*96+192+29
    target=rng.normal(size=n);p=np.stack([target[i*96:i*96+192] for i in range(w)])+rng.normal(scale=.1,size=(w,192))
    from pub_01_s03_ss07_openneuro_metrics import mechanical_mask
    row=dict(subject='synthetic',record='synthetic',**mechanical_mask(target))
    result,_,_=measure(p,target,row,1024,'synthetic')
    assert result['all_primary_blocks_defined'] and len(result['blocks'])==2 and result['Hann_identity_max_abs']<1e-12
    constant,_,_=measure(np.full_like(p,.1),target,row,1024,'constant')
    assert len(constant['blocks'])==2 and not constant['all_primary_blocks_defined']
    # Check the actual constants, feature order, first4 fallback and full FP32 arithmetic.
    z=rng.normal(size=(10,8,192)).astype('f4');p=rng.normal(size=(10,192)).astype('f4')
    stats=[dict(mean=np.zeros(d),std=np.ones(d),constant=np.zeros(d,bool)) for d in [1536,192,1536]]
    co=dict(coef=rng.normal(scale=.001,size=(3264,192)),bias=rng.normal(size=192))
    cal=dict(a=.2,b=.3);q=readout(z,p,cal,stats,co)
    c=z[4:].astype('f8').reshape(-1,1536);base=.2*p.astype('f8')+.3
    explicit=base.copy();explicit[4:]+=c/np.sqrt(1536)@co['coef'][:1536]+base[4:]/np.sqrt(192)@co['coef'][1536:1728]+co['bias']
    explicit[4:]+=(c*c)/np.sqrt(1536)@co['coef'][1728:]
    difference=float(abs(q-explicit).max());assert difference<1e-12
    q32=readout(z,p,cal,stats,co,np.float32);assert q32.dtype==np.float32
    return dict(passed=True,Hann_identity=result['Hann_identity_max_abs'],dense_vs_split_readout_max_abs=difference,
        constant_predictions_preserved_as_undefined=True,full_FP32_readout_dtype=True,no_real_data_or_NN=True)
