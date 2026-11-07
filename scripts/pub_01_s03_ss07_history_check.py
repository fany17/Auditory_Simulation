"""Synthetic-only scientific support/intervention checks on server2203."""
import json
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT,dump,sha
from pub_01_s03_ss07_incremental import NativeResidual,summary
from pub_01_s03_ss07_history import intervene,contexts,hann_support_audit,causal_ema


def main():
    torch.set_num_threads(2);torch.manual_seed(7321);rows=[]
    for family in ['m1','m2','m3','m4','m5','output_prior']:
        model=NativeResidual(family,16).eval()
        # Nonzero learned residual, so identity cannot pass trivially.
        with torch.no_grad():
            model.residual.weight.normal_(0,.1);model.residual.bias.normal_(0,.1)
        z=torch.randn(137,8,192);fixed=z.clone()
        with torch.no_grad():
            native,state,_=model(z[:64],None,0,'one')
            second,state,_=model(z[64:128],state,64,'one')
            third,_,_=model(z[128:],state,128,'one');native=torch.cat([native,second,third])
            prediction,ctx=intervene(model,z)
            error=float((prediction-native).abs().max());assert error<1e-6
            obs=summary(model.head(z)) if family=='output_prior' else summary(z)
            # Future perturbation cannot change any true-history context so far.
            changed=z.clone();changed[70:]+=100
            _,changed_ctx=intervene(model,changed)
            assert torch.equal(ctx[:72],changed_ctx[:72])
            blind,blind_ctx=intervene(model,z,'content_blind',neutral=torch.ones(obs.shape[1])*.25)
            reset,reset_ctx=intervene(model,z,'reset')
            assert not torch.equal(blind_ctx[4:],reset_ctx[4:])
            wrong,wrong_ctx=intervene(model,z,'wrong_history',donor_observations=obs.flip(0))
            assert not torch.equal(wrong_ctx[4:],ctx[4:])
            shuffled,_=intervene(model,z,'shuffled',permutation=torch.arange(len(z)-1,-1,-1))
            assert torch.equal(shuffled,wrong)
            older,older_ctx=intervene(model,z,'older_outside_fused_support')
            assert torch.equal(older_ctx[1:],ctx[:-1])
            assert torch.equal(z,fixed)
            # Interventions applied EVERY window, not just first60s state.
            assert not torch.equal(wrong[-20:],prediction[-20:])
            if family=='m5':
                prior,pc=intervene(model,z,mode='prior_only')
                reference,_,_=model(z[:64],None,0,'prior',mode='prior_only')
                assert float((prior[:64]-reference).abs().max())<1e-6
            rows.append({'family':family,'native_forward_max_abs_diff':error,'current_z_unchanged':True,
                'every_window_wrong_history':True,'time_preserving_neutral_distinct_from_reset':True})
    supports=[hann_support_audit(n,(n-1)*96+192+tail) for n in [1,2,3,4,17] for tail in [0,1,95]]
    assert all(x['rows'][0]['cold_start'] for x in supports)
    values=np.array([99.,1.,3.,999.,10.,12.]);covered=np.array([False,True,True,False,True,True])
    assert np.array_equal(causal_ema(values,covered,0),values)
    smoothed=causal_ema(values,covered,.25)
    assert smoothed[1]==1 and smoothed[4]==10 and smoothed[0]==99 and smoothed[3]==999
    assert 1<smoothed[2]<3 and 10<smoothed[5]<12
    dump(ROOT/'HISTORY_ENGINEERING_CHECK.json',{'status':'PASS','synthetic_only':True,'test_read':False,
        'sources':{str(ROOT/'input'/name):sha(ROOT/'input'/name) for name in
            ['pub_01_s03_ss07_history.py','pub_01_s03_ss07_history_check.py']},
        'families':rows,'support_cases':supports,'post_Hann_EMA_gap_reset_identity_verified':True,
        'no_performance_or_neural_causality_conclusion':True})
    print(json.dumps({'status':'PASS','families':rows,'support_cases':len(supports)}))


if __name__=='__main__':main()
