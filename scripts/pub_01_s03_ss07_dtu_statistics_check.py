"""Synthetic external statistics check, never external predictions."""
import json
from pathlib import Path
import numpy as np
from pub_01_s03_ss07_dtu_statistics import paired


def main():
    names=[f'fixture{i}' for i in range(6)]
    constant=paired(np.full((6,5,18,60),.125),names,draws=200)
    assert all(np.allclose(r['Bonferroni_M_approx_interval'],.125) for r in constant['contrasts'])
    assert all(np.allclose(r['leave_one_person_out_descriptive'],.125) for r in constant['contrasts'])
    source=np.random.default_rng(188).normal(size=(1,5,18,60));duplicated=np.repeat(source,6,axis=0)
    result=paired(duplicated,names,draws=200)
    assert result['contrasts'][0]['Bonferroni_M_approx_interval']==result['contrasts'][5]['Bonferroni_M_approx_interval']
    p=source.mean(3)[0];expected=[float(np.delete(p,i,axis=1).mean()) for i in range(18)]
    assert expected==result['contrasts'][0]['leave_one_person_out_descriptive']
    bad=duplicated.copy();bad[0,0,0,0]=np.nan
    try:paired(bad,names,draws=100)
    except AssertionError:pass
    else:raise AssertionError('nonfinite accepted')
    path=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101/external_dtu/STATISTICS_ENGINEERING_CHECK.json')
    with path.open('x') as f:json.dump({'PASS':True,'synthetic_only':True,'no_external_model_scores':True,
        'constant_effect_preserved':True,'shared_contrast_draws':True,'LOPO_person_axis_verified':True,'nonfinite_rejected':True},f,indent=2)
    print('PASS synthetic external statistics')


if __name__=='__main__':main()
