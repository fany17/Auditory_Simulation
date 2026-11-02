"""Prespecified person aggregation and two-comparison inference; no model/data I/O."""
import numpy as np


def pearson(x,y):
    x=np.asarray(x,'f8');y=np.asarray(y,'f8')
    assert x.ndim==y.ndim==1 and len(x)==len(y) and len(x)>=2
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert x.max()>x.min() and y.max()>y.min(),'Constant predictions/targets cannot be silently dropped'
    a=x-x.mean();b=y-y.mean()
    return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))


def coverage(length):
    starts=np.arange(0,length-192+1,96,dtype=np.int64)
    out=np.zeros(length,'f8');h=np.hanning(192)
    for start in starts:out[start:start+192]+=h
    return starts,out


def mechanical_mask(target):
    target=np.asarray(target);length=len(target)
    starts,weights=coverage(length);covered=weights>0
    blocks=[]
    for start in range(0,length-3840+1,3840):
        mask=covered[start:start+3840];y=target[start:start+3840][mask]
        finite=bool(np.isfinite(y).all());nonconstant=bool(len(y)>0 and finite and y.max()>y.min())
        blocks.append(dict(start=start,stop=start+3840,covered=int(mask.sum()),finite_target=finite,
            nonconstant_target=nonconstant,eligible=bool(mask.sum()>=3800 and finite and nonconstant)))
    tail_start=length//3840*3840
    y=target[tail_start:][covered[tail_start:]]
    finite=bool(np.isfinite(y).all());nonconstant=bool(len(y)>0 and finite and y.max()>y.min())
    tail=dict(start=tail_start,stop=length,covered=len(y),finite_target=finite,
        nonconstant_target=nonconstant,eligible_r=bool(len(y)>=192 and finite and nonconstant))
    return dict(length=length,windows=len(starts),covered=int(covered.sum()),blocks=blocks,tail=tail)


def aggregate_person(rows):
    """Rows are per-seed/subject/record/block correlations. No across-record concatenation."""
    grouped={}
    for row in rows:
        key=(row['subject'],row['seed'])
        grouped.setdefault(key,[]).append(row)
    subjects={}
    for (subject,seed),values in sorted(grouped.items()):
        assert np.isfinite([r['r'] for r in values]).all()
        identities=[(r['record'],r['start'],r['stop']) for r in values]
        assert len(set(identities))==len(identities),'Duplicate record/block identity'
        clips={}
        for row in values:clips.setdefault(row['record'],[]).append(row['r'])
        subjects.setdefault(subject,{})[seed]=dict(equal_block=float(np.mean([r['r'] for r in values])),
            equal_clip=float(np.mean([np.mean(v) for v in clips.values()])),blocks=len(values),records=len(clips),
            block_identities=sorted(identities))
    result={}
    for subject,seeds in subjects.items():
        assert set(seeds)=={1024,2026,31415}
        assert len({s['blocks'] for s in seeds.values()})==1
        reference=next(iter(seeds.values()))['block_identities']
        assert all(s['block_identities']==reference for s in seeds.values()),'Paired seeds need identical record/block support'
        result[subject]=dict(equal_block=float(np.mean([s['equal_block'] for s in seeds.values()])),
            equal_clip=float(np.mean([s['equal_clip'] for s in seeds.values()])),seeds=seeds)
    return result


def exact_signflip(difference):
    d=np.asarray(difference,'f8')
    assert d.ndim==1 and 1<=len(d)<=19 and np.isfinite(d).all()
    observed=abs(float(d.mean()));total=1<<len(d);count=0
    for start in range(0,total,65536):
        patterns=np.arange(start,min(start+65536,total),dtype=np.uint32)
        signs=1.-2.*((patterns[:,None]>>np.arange(len(d),dtype=np.uint32))&1)
        statistic=np.abs((signs@d)/len(d))
        count+=int(np.count_nonzero(statistic>=observed-1e-15))
    return dict(statistic=observed,p=count/total,extreme=count,enumerated=total,
        tie_absolute_tolerance=1e-15,null_assumption='Independent people; symmetric/exchangeable signs under null')


def two_primary_inference(person_differences):
    d=np.asarray(person_differences,'f8')
    assert d.ndim==2 and d.shape[1]==2 and 1<=len(d)<=19 and np.isfinite(d).all()
    tests=[exact_signflip(d[:,i]) for i in range(2)]
    p=np.array([t['p'] for t in tests]);order=np.argsort(p,kind='stable')
    adjusted=np.empty(2);adjusted[order]=np.minimum(1.,np.maximum.accumulate(p[order]*[2,1]))
    rng=np.random.default_rng(20261102)
    indices=rng.integers(0,len(d),size=(20000,len(d)))
    boot=d[indices].mean(axis=1)
    intervals=np.quantile(boot,[.0125,.9875],axis=0)
    for i,test in enumerate(tests):
        test.update(mean_difference=float(d[:,i].mean()),p_Holm2=float(adjusted[i]),
            Bonferroni_family95_approximate_CI=intervals[:,i].tolist(),bootstrap_people=20000,
            bootstrap_seed=20261102,bootstrap_quantiles=[.0125,.9875])
    return tests


def synthetic_check():
    assert exact_signflip([1,2,3])['p']==.25
    assert exact_signflip([0,0,0])['p']==1.
    assert exact_signflip([1,-1])['p']==1.
    x=np.arange(8000,dtype='f4');mask=mechanical_mask(x)
    assert len(mask['blocks'])==2 and all(r['eligible'] for r in mask['blocks'])
    assert mask['tail']['eligible_r']
    assert not any(r['eligible'] for r in mechanical_mask(np.ones(8000))['blocks'])
    assert abs(pearson(x,2*x+1)-1)<1e-15
    rows=[dict(subject='sub-001',seed=s,record=r,start=b*3840,stop=(b+1)*3840,r=v)
          for s in [1024,2026,31415] for r,b,v in [('a',0,.1),('a',1,.3),('b',0,.8)]]
    result=aggregate_person(rows)['sub-001']
    assert abs(result['equal_block']-.4)<1e-15 and abs(result['equal_clip']-.5)<1e-15
    tests=two_primary_inference(np.array([[1.,2.],[2.,3.],[3.,4.]]))
    assert all(t['p_Holm2']==.5 for t in tests)
    mismatch=[dict(r) for r in rows];mismatch[3]['stop']+=1;failed=False
    try:aggregate_person(mismatch)
    except AssertionError:failed=True
    assert failed,'Equal block counts with mismatched identities must fail'
    failed=False
    try:aggregate_person(rows+[dict(rows[0])])
    except AssertionError:failed=True
    assert failed,'Duplicate identities must fail'
    return dict(passed=True,signflip_enumeration=True,constant_target_rule=True,
        equal_block_vs_equal_clip=True,person_shared_bootstrap=True,
        paired_block_identity_mismatch_rejected=True,duplicate_blocks_rejected=True,no_actual_data=True)
