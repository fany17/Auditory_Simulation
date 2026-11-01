"""D7 frozen algebraic readouts and record-level fixed-Hann metrics only."""
import numpy as np

ROLES=('P','Ponly','S32','U32','C','Ccap')
LAMBDAS={'Ponly':.0001,'S32':.0001,'U32':.0001,'C':.0001,'Ccap':.01}

def normalize(x,stats):
    out=(x-stats['mean'])/stats['std'];out[:,stats['constant']]=0
    return out

def predict(z,p,calibration,stats,coefficient=None):
    """Every row is one current window. No history input or fitted statistic."""
    assert z.ndim==3 and z.shape[1:]==(8,192) and p.shape==(len(z),192)
    assert np.isfinite(z).all() and np.isfinite(p).all() and calibration['a']>0
    base=calibration['a']*p.astype(np.float64)+calibration['b']
    if coefficient is None:return base
    c=normalize(z[4:].astype(np.float64).reshape(-1,1536),stats[0])
    xp=normalize(base[4:],stats[1])/np.sqrt(192);xz=c/np.sqrt(1536)
    result=base.copy();result[4:]+=xz@coefficient['BZ']+xp@coefficient['BP']+coefficient['bias']
    if coefficient.get('BQ') is not None:
        q=normalize(c*c,stats[5])/np.sqrt(1536);result[4:]+=q@coefficient['BQ']
    assert np.array_equal(result[:4],base[:4]) and np.isfinite(result).all()
    return result

def corr(a,b):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert a.max()!=a.min() and b.max()!=b.min(),'constant prediction or target; no selective exclusion'
    x=a-a.mean();y=b-b.mean();value=float(x@y/np.sqrt((x@x)*(y@y)))
    assert np.isfinite(value);return value

def fuse(pred,n):
    assert pred.shape==((n-192)//96+1,192) and np.isfinite(pred).all()
    out=np.zeros(n);weight=np.zeros(n);w=np.hanning(192)
    for i,p in enumerate(pred):
        s=i*96;out[s:s+192]+=p*w;weight[s:s+192]+=w
    keep=weight>1e-8;out[keep]/=weight[keep]
    return out,keep

def measure(pred,target,row,blocks):
    n=row['common_samples'];w=(n-192)//96+1;assert pred.shape==(w,192)
    target=np.asarray(target,np.float64).reshape(-1);assert len(target)>=n and np.isfinite(target[:n]).all()
    # Same float32 window targets as the original supervised envelope cache;
    # continuous60s truth below retains the original file's float64 view.
    yw=np.stack([target[i*96:i*96+192] for i in range(w)]).astype(np.float32).astype(np.float64)
    x=pred-pred.mean(1,keepdims=True);y=yw-yw.mean(1,keepdims=True)
    target_constant=np.ptp(yw,axis=1)==0;pred_constant=np.ptp(pred,axis=1)==0
    defined=~(target_constant|pred_constant);r3=np.full(w,np.nan)
    r3[defined]=(x[defined]*y[defined]).sum(1)/np.sqrt((x[defined]*x[defined]).sum(1)*(y[defined]*y[defined]).sum(1))
    assert np.isfinite(r3[defined]).all()
    fused,keep=fuse(pred,n);h=np.hanning(192);weight=np.zeros(n);weighted=np.zeros(n);disagreement=np.zeros(n)
    contributor_min=np.full(n,np.inf);contributor_max=np.full(n,-np.inf)
    for i,p in enumerate(pred):
        s=i*96;weight[s:s+192]+=h;weighted[s:s+192]+=(p-target[s:s+192])**2*h
        contributor_min[s:s+192]=np.minimum(contributor_min[s:s+192],np.where(h>0,p,np.inf))
        contributor_max[s:s+192]=np.maximum(contributor_max[s:s+192],np.where(h>0,p,-np.inf))
    weighted[keep]/=weight[keep];fused_error=(fused-target[:n])**2
    a=pred[:-1,96:];b=pred[1:,:96];pair_mse=((a-b)**2).mean(1);den=(a.var(1)+b.var(1))/2;valid=den>0
    alpha=h[96:]/(h[96:]+h[:96])
    for i in range(w-1):
        s=(i+1)*96;disagreement[s:s+96]=alpha*(1-alpha)*(a[i]-b[i])**2
    identity=float(abs(weighted[keep]-fused_error[keep]-disagreement[keep]).max())
    assert identity<=1e-10*max(1.,float(weighted[keep].max()))
    meta={k:row[k] for k in ('record','phase','subject','story')};meta['source_family']=row.get('source_family',row['story'])
    segments=[]
    for block in blocks:
        s=block['start'];k=keep[s:s+3840];assert len(k)==3840 and int(k.sum())==block['samples']
        if not block['eligible']:continue
        p=fused[s:s+3840][k];t=target[s:s+3840][k]
        assert contributor_min[s:s+3840][k].min()!=contributor_max[s:s+3840][k].max(),'constant contributing predictions in primary block'
        segments.append(dict(**meta,start=s,samples=int(k.sum()),r=corr(p,t),MSE=float(fused_error[s:s+3840][k].mean()),
            weighted_single_MSE=float(weighted[s:s+3840][k].mean()),disagreement=float(disagreement[s:s+3840][k].mean())))
    assert len(segments)==sum(b['eligible'] for b in blocks)
    return dict(**meta,windows=w,r3s=float(r3.mean()) if defined.all() else None,window_r_defined=int(defined.sum()),window_r_undefined=int((~defined).sum()),
        target_constant_windows=int(target_constant.sum()),prediction_constant_windows=int(pred_constant.sum()),window_MSE=float(((pred-yw)**2).mean()),covered_samples=int(keep.sum()),
        overlap_pairs=w-1,valid_normalized_pairs=int(valid.sum()),overlap_SSE_mean_sum=float(pair_mse.sum()),
        normalized_overlap_num=float(pair_mse[valid].sum()),normalized_overlap_den=float(den[valid].sum()),
        overlap_MSE=float(pair_mse.mean()),overlap_NRMSE=float(np.sqrt(pair_mse[valid].sum()/den[valid].sum())) if valid.any() else None,
        Hann_identity_max_abs=identity,segments=segments)

def aggregate(records):
    segments=[b for r in records for b in r['segments']];people=[];phases={}
    for phase in sorted({r['phase'] for r in records}):
        pp=[]
        for subject in sorted({b['subject'] for b in segments if b['phase']==phase}):
            bb=[b for b in segments if b['phase']==phase and b['subject']==subject]
            person=dict(phase=phase,subject=subject,blocks=len(bb),records=len({b['record'] for b in bb}),
                **{key:float(np.mean([b[key] for b in bb])) for key in ('r','MSE','weighted_single_MSE','disagreement')})
            people.append(person);pp.append(person)
        rr=[r for r in records if r['phase']==phase];windows=sum(r['windows'] for r in rr);pairs=sum(r['overlap_pairs'] for r in rr)
        denominator=sum(r['normalized_overlap_den'] for r in rr)
        undefined=sum(r['window_r_undefined'] for r in rr)
        phases[phase]=dict(people=len(pp),blocks=sum(p['blocks'] for p in pp),records=len(rr),windows=windows,window_r_undefined=undefined,window_r_defined=windows-undefined,
            **{key:float(np.mean([p[key] for p in pp])) for key in ('r','MSE','weighted_single_MSE','disagreement')},
            r3s=sum(r['r3s']*r['windows'] for r in rr)/windows if undefined==0 else None,window_MSE=sum(r['window_MSE']*r['windows'] for r in rr)/windows,
            overlap_MSE=sum(r['overlap_SSE_mean_sum'] for r in rr)/pairs,overlap_pairs=pairs,
            valid_normalized_pairs=sum(r['valid_normalized_pairs'] for r in rr),overlap_NRMSE=float(np.sqrt(sum(r['normalized_overlap_num'] for r in rr)/denominator)) if denominator>0 else None)
    weighted=None
    if set(phases)=={'test_seen','test_heldout'}:
        weighted={k:2/3*phases['test_seen'][k]+1/3*phases['test_heldout'][k] for k in ('r','MSE','weighted_single_MSE','disagreement')}
    return dict(phases=phases,weighted=weighted,people=people,segments=segments,records=records,
        Hann_identity_max_abs=max(r['Hann_identity_max_abs'] for r in records))

def math_check():
    rng=np.random.default_rng(77101);count=50;n=(count-1)*96+192+7
    target=rng.normal(size=n);pred=np.stack([target[i*96:i*96+192] for i in range(count)])+rng.normal(scale=.2,size=(count,192))
    row=dict(record='synthetic',phase='test_seen',subject='synthetic',story='synthetic',source_family='synthetic',common_samples=n)
    _,keep=fuse(pred,n);blocks=[dict(start=0,samples=int(keep[:3840].sum()),eligible=True)] if n>=3840 else []
    m=measure(pred,target,row,blocks);assert m['overlap_pairs']==49 and len(m['segments'])==1 and m['Hann_identity_max_abs']<1e-12
    others=[]
    for record,phase,subject,noise in [('second','test_seen','synthetic',.5),('third','test_seen','B',1.5),('fourth','test_heldout','C',.8)]:
        r=dict(row,record=record,phase=phase,subject=subject);q=np.stack([target[i*96:i*96+192] for i in range(count)])+rng.normal(scale=noise,size=(count,192))
        others.append(measure(q,target,r,blocks))
    summary=aggregate([m]+others);seen_expected=((m['segments'][0]['r']+others[0]['segments'][0]['r'])/2+others[1]['segments'][0]['r'])/2
    assert abs(summary['phases']['test_seen']['r']-seen_expected)<1e-14
    assert abs(summary['weighted']['r']-(2/3*seen_expected+1/3*others[2]['segments'][0]['r']))<1e-14
    short_constant=pred.copy();short_constant[0]=.1;undefined=measure(short_constant,target,row,blocks)
    assert undefined['r3s'] is None and undefined['window_r_undefined']==1 and len(undefined['segments'])==1
    assert aggregate([undefined])['phases']['test_seen']['r3s'] is None
    try:measure(np.full_like(pred,.1),target,row,blocks)
    except AssertionError:pass
    else:raise AssertionError('constant primary block must fail despite floating Hann rounding')
    truth=np.stack([target[i*96:i*96+192] for i in range(count)])
    st=[dict(mean=np.zeros(d),std=np.ones(d),constant=np.zeros(d,dtype=bool)) for d in (1536,192,48,1536,1536,1536)]
    z=rng.normal(size=(count,8,192));p=rng.normal(size=(count,192));cal=dict(a=.2,b=.3)
    co=dict(BZ=rng.normal(size=(1536,192))*.001,BP=rng.normal(size=(192,192))*.001,bias=np.zeros(192),BQ=None)
    before=predict(z,p,cal,st,co);changed=z.copy();changed[9:]+=10;after=predict(changed,p,cal,st,co)
    assert np.array_equal(before[:9],after[:9]) and np.array_equal(before[:4],(.2*p+.3)[:4])
    ponly=dict(**co);ponly['BZ']=np.zeros_like(co['BZ']);assert np.array_equal(predict(z,p,cal,st,ponly),predict(changed,p,cal,st,ponly))
    try:corr(np.full(10,.1),np.arange(10.))
    except AssertionError:pass
    else:raise AssertionError('exact nonbinary constant must fail')
    return dict(passed=True,current_only_and_first4_canaries=True,Ponly_zeroZ=True,exact_constant_rejection=True,Hann_identity_max_abs=m['Hann_identity_max_abs'],
        complete_60s_block_and_person_macro_and_weighted_split=True,undefined_auxiliary_3s_null_without_dropping_primary=True)
