"""External18-person analysis. No data/model loader, server2203 only."""
import numpy as np


def paired(differences,contrast_names,draws=20000):
    # K contrasts, fixed FIVE trained seeds, EIGHTEEN people, SIXTY trials.
    assert differences.shape==(len(contrast_names),5,18,60)
    assert len(contrast_names) in [6,7] and len(set(contrast_names))==len(contrast_names)
    assert np.isfinite(differences).all()
    person=differences.mean(3);k=len(contrast_names);rng=np.random.default_rng(20261105)
    samples=np.empty((k,draws))
    for first in range(0,draws,100):
        n=min(100,draws-first);seeds=rng.integers(5,size=(n,5));people=rng.integers(18,size=(n,18))
        values=person[:,seeds,:].mean(2)
        samples[:,first:first+n]=np.take_along_axis(values,people[None],2).mean(2)
    rows=[]
    for i,name in enumerate(contrast_names):
        rows.append({'contrast':name,'mean_r_difference':float(person[i].mean()),
            'paired_seed_r_differences':person[i].mean(1).tolist(),
            'paired_person_r_differences':person[i].mean(0).tolist(),
            'approx_95_interval':np.quantile(samples[i],[.025,.975]).tolist(),
            'Bonferroni_M_approx_interval':np.quantile(samples[i],[.05/(2*k),1-.05/(2*k)]).tolist(),
            'leave_one_person_out_descriptive':[float(np.delete(person[i],p,axis=1).mean()) for p in range(18)]})
    return {'endpoint':'trial attended r -> person -> five seeds; fixed3166/3200 coverage',
        'contrasts':rows,'M':k,'draws':draws,'rng':20261105,'bootstrap':'CROSSED seed/person, same draws for all conditions',
        'interpretation':'approximate intervals conditional on these external stimuli; no finite-sample coverage guarantee',
        'joint_internal_external_FWER_not_claimed':True,'1080trials_not_independent_n':True,
        'auxiliary_attended_vs_unattended_accuracy_descriptive_only':True}
