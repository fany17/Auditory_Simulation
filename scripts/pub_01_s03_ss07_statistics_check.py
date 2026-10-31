"""Synthetic paired-statistics checks; no neural data or performance input."""
import json
import numpy as np
from pub_01_s03_ss07_baseline import ROOT,dump,sha
from pub_01_s03_ss07_statistics import layout,point_estimates,crossed_seed_person,crossed_family_sensitivity,summarize


def main():
    blocks=[{'phase':phase,'subject':f'{phase}_p{p}','record':f'{phase}_p{p}_r{r}',
        'start':j*3840,'source_family':f'family{r}'}
        for phase in ['test_seen','test_heldout'] for p in range(3) for r in range(2) for j in range(2)]
    constant=np.ones((6,5,len(blocks)))*.125
    point,samples=crossed_seed_person(constant,blocks,draws=200)
    assert np.allclose(point,.125,atol=1e-15) and np.allclose(samples,.125,atol=1e-15)
    sensitivity=crossed_family_sensitivity(constant,blocks,draws=200)
    assert np.allclose(sensitivity,.125,atol=1e-15)
    zero=summarize(np.zeros_like(constant),blocks,[f'c{i}' for i in range(6)],draws=200)
    assert all(row['weighted_Bonferroni_M_approx_bootstrap_interval']==[0.,0.] for row in zero['contrasts'])
    # Manual replicate verifies SAME person identities across ALL selected seeds.
    rng=np.random.default_rng(42);variable=rng.normal(size=(6,5,len(blocks)))
    _,_,groups,person=layout(variable,blocks);rng=np.random.default_rng(20261101)
    seeds=rng.integers(5,size=(100,5));seen=rng.integers(3,size=(100,3));held=rng.integers(3,size=(100,3))
    expected=[]
    for c in range(6):
        expected.append(2/3*np.mean(person[c,seeds[0]][:,groups[0][seen[0]]])+
            1/3*np.mean(person[c,seeds[0]][:,groups[1][held[0]]]))
    _,actual=crossed_seed_person(variable,blocks,draws=100)
    assert np.max(np.abs(np.array(expected)-actual[:,0,0]))<1e-15
    # Identical paired contrasts MUST share every bootstrap sample.
    duplicated=np.repeat(variable[:1],6,axis=0);_,values=crossed_seed_person(duplicated,blocks,draws=200)
    assert np.array_equal(values[0],values[5])
    bad=variable.copy();bad[0,0,0]=np.nan
    try:layout(bad,blocks)
    except AssertionError:pass
    else:raise AssertionError('undefined model prediction was silently accepted')
    dump(ROOT/'STATISTICS_ENGINEERING_CHECK.json',{'status':'PASS','synthetic_only':True,'test_read':False,
        'constant_effect_and_zero_effect_preserved':True,'manual_crossed_seed_person_replicate_verified':True,
        'all_contrasts_share_draws':True,'nonfinite_rejected':True,
        'sources':{str(ROOT/'input'/n):sha(ROOT/'input'/n) for n in
            ['pub_01_s03_ss07_statistics.py','pub_01_s03_ss07_statistics_check.py']}})
    print(json.dumps({'status':'PASS','manual_replicate_max_error':float(np.max(np.abs(np.array(expected)-actual[:,0,0])))}))


if __name__=='__main__':main()
