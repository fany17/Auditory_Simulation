"""Paired SS07 statistics, no data/model/test loader or automatic execution.

diff[K,S,B] is the SAME fixed block set for K declared paired contrasts,
S training seeds. People and seeds are CROSSED, never nested replicates.
Execute numerical work on server2203 only.
"""
import numpy as np


def layout(diff,blocks):
    assert diff.ndim==3 and diff.shape[2]==len(blocks) and np.isfinite(diff).all()
    assert len({(b['record'],b['start']) for b in blocks})==len(blocks)
    phases=['test_seen','test_heldout'];people=[];indices=[];groups=[]
    for g,phase in enumerate(phases):
        subjects=sorted({b['subject'] for b in blocks if b['phase']==phase});assert subjects
        group=[]
        for subject in subjects:
            ids=np.array([i for i,b in enumerate(blocks) if b['phase']==phase and b['subject']==subject])
            group.append(len(people));people.append(subject);indices.append(ids)
        groups.append(np.array(group))
    assert len(set(people))==len(people),'seen/heldout person groups must be disjoint'
    assert sum(len(i) for i in indices)==len(blocks)
    per_person=np.stack([diff[:,:,i].mean(2) for i in indices],2)
    return people,indices,groups,per_person


def point_estimates(per_person,groups):
    seen=per_person[:,:,groups[0]].mean(2);held=per_person[:,:,groups[1]].mean(2)
    return np.stack([2/3*seen+1/3*held,seen,held],2)


def crossed_seed_person(diff,blocks,draws=20000,seed=20261101):
    people,indices,groups,per_person=layout(diff,blocks)
    k,s,_=per_person.shape;rng=np.random.default_rng(seed);output=np.empty((k,draws,3))
    for first in range(0,draws,100):
        n=min(100,draws-first)
        selected_seeds=rng.integers(s,size=(n,s))
        # One person draw per stratum/replicate, shared by EVERY seed/model.
        seen=rng.integers(len(groups[0]),size=(n,len(groups[0])))
        held=rng.integers(len(groups[1]),size=(n,len(groups[1])))
        mean_seeds=per_person[:,selected_seeds,:].mean(2)
        a=np.take_along_axis(mean_seeds[:,:,groups[0]],seen[None],2).mean(2)
        b=np.take_along_axis(mean_seeds[:,:,groups[1]],held[None],2).mean(2)
        output[:,first:first+n]=np.stack([2/3*a+1/3*b,a,b],2)
    return point_estimates(per_person,groups),output


def crossed_family_sensitivity(diff,blocks,draws=20000,seed=20261103):
    people,indices,groups,per_person=layout(diff,blocks)
    families=sorted({b['source_family'] for b in blocks});assert all(families)
    family_ids=np.array([families.index(b['source_family']) for b in blocks])
    rng=np.random.default_rng(seed);k,s,_=diff.shape;output=np.empty((k,draws,3))
    for first in range(0,draws,100):
        n=min(100,draws-first);selected_seeds=rng.integers(s,size=(n,s))
        person_weights=rng.exponential(size=(n,len(people)))
        family_weights=rng.exponential(size=(n,len(families)))
        # Positive weights keep every person eligible with sparse shared story.
        person_weights=np.maximum(person_weights,np.finfo(float).tiny)
        family_weights=np.maximum(family_weights,np.finfo(float).tiny)
        mean_seeds=diff[:,selected_seeds,:].mean(2);values=[]
        for ids in indices:
            weight=family_weights[:,family_ids[ids]]
            values.append((mean_seeds[:,:,ids]*weight[None]).sum(2)/weight.sum(1)[None])
        values=np.stack(values,2);est=[]
        for group in groups:
            weight=person_weights[:,group]
            est.append((values[:,:,group]*weight[None]).sum(2)/weight.sum(1)[None])
        output[:,first:first+n]=np.stack([2/3*est[0]+1/3*est[1],est[0],est[1]],2)
    return output


def summarize(diff,blocks,contrast_names,draws=20000):
    assert len(contrast_names)==len(diff) and len(set(contrast_names))==len(contrast_names)
    m=len(contrast_names);assert m in (5,6,7),'fixed family size must be declared before scores'
    per_seed,bootstrap=crossed_seed_person(diff,blocks,draws)
    sensitivity=crossed_family_sensitivity(diff,blocks,draws)
    result=[]
    for i,name in enumerate(contrast_names):
        item={'contrast':name,'seed_paired_weighted_differences':per_seed[i,:,0].tolist(),
            'weighted_mean_difference':float(per_seed[i,:,0].mean()),
            'weighted_95_approx_bootstrap_interval':np.quantile(bootstrap[i,:,0],[.025,.975]).tolist(),
            'weighted_Bonferroni_M_approx_bootstrap_interval':np.quantile(bootstrap[i,:,0],[.05/(2*m),1-.05/(2*m)]).tolist(),
            'weighted_crossed_Exp_sensitivity_interval_not_calibrated_CI':np.quantile(sensitivity[i,:,0],[.025,.975]).tolist(),
            'supporting_endpoints':{}}
        for j,name_ in [(1,'seen'),(2,'heldout')]:
            item['supporting_endpoints'][name_]={'mean_difference':float(per_seed[i,:,j].mean()),
                'exploratory_95_interval':np.quantile(bootstrap[i,:,j],[.025,.975]).tolist(),
                'not_in_weighted_M_multiplicity_guarantee':True}
        result.append(item)
    return {'contrasts':result,'M':m,'primary_endpoint':'original 2/3 seen + 1/3 heldout macro r',
        'draws':draws,'seed_person_rng':20261101,'family_sensitivity_rng':20261103,
        'shared_person_draw_across_all_seeds_and_contrasts':True,
        'interpretation':'Bonferroni-adjusted approximate bootstrap intervals; no finite-sample/confirmatory guarantee',
        'family_sensitivity_interpretation':'positive crossed weights; NOT calibrated frequentist confidence intervals',
        'other_contrast_family_joint_FWER_not_claimed':True}
