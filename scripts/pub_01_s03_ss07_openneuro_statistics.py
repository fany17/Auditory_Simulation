"""Freeze and apply person-level two-primary statistics, preserving undefined outcomes."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from pub_01_s03_ss07_openneuro_transport import ROOT
import pub_01_s03_ss07_openneuro_metrics as metric

SEEDS=(1024,2026,31415);ROLES=('P_FT_star','Ccap','W','F');PAIRS=(('Ccap','P_FT_star'),('F','W'))


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def family_inference(columns):
    """None endpoint remains unestimable; reserve its Holm slot at p=1."""
    available=[np.asarray(c,'f8') for c in columns if c is not None]
    assert available and len({len(c) for c in available})==1
    n=len(available[0]);assert 1<=n<=19
    rng=np.random.default_rng(20261102);indices=rng.integers(0,n,size=(20000,n))
    tests=[];raw=[]
    for column in columns:
        if column is None:tests.append(None);raw.append(1.);continue
        d=np.asarray(column,'f8');test=metric.exact_signflip(d)
        test.update(mean_difference=float(d.mean()),Bonferroni_family95_approximate_CI=np.quantile(d[indices].mean(1),[.0125,.9875]).tolist())
        tests.append(test);raw.append(test['p'])
    order=np.argsort(raw,kind='stable');adjusted=np.empty(2)
    adjusted[order]=np.minimum(1.,np.maximum.accumulate(np.asarray(raw)[order]*[2,1]))
    for i,test in enumerate(tests):
        if test is not None:test.update(p_Holm2=float(adjusted[i]),bootstrap_replicates=20000,bootstrap_seed=20261102)
    return tests


def freeze():
    assert not any((ROOT/f'evaluation_seed{s}/STARTED.json').exists() for s in SEEDS)
    d=np.array([[1.,2.],[2.,3.],[3.,4.]])
    own=family_inference([d[:,0],d[:,1]]);reference=metric.two_primary_inference(d)
    assert all(own[i]['p']==reference[i]['p'] and own[i]['p_Holm2']==reference[i]['p_Holm2'] and own[i]['Bonferroni_family95_approximate_CI']==reference[i]['Bonferroni_family95_approximate_CI'] for i in range(2))
    missing=family_inference([d[:,0],None]);assert missing[1] is None and missing[0]['p_Holm2']==.5
    dump('STATISTICS_PROTOCOL.json',dict(frozen_before_any_real_NN=True,source_sha256=sha(__file__),
        source_metric_sha256=sha(metric.__file__),primary_pairs=PAIRS,independent_unit='person, maximum19',
        aggregation='Separate Pearson per fixed record/block; equal blocks within person, then3seeds within person; no cross-record concatenation',
        exact_signflip=dict(statistic='abs(mean(person differences))',all_signs_including_observed=True,tie_absolute=1e-15,
            assumption='Independent people and symmetric/exchangeable signs under null'),
        multiplicity='Holm over2 fixed hypotheses; unestimable endpoint reserves its slot at p=1 internally but is reported undefined',
        bootstrap=dict(people_paired=True,shared_draws_across_primary=True,replicates=20000,seed=20261102,
            quantiles=[.0125,.9875],meaning='Bonferroni family95 approximate percentile intervals'),
        failures='No prediction/role-dependent input exclusion; constant/nonfinite or missing fixed block makes affected model/primary unestimable',
        precision='W/F fullFP32 readout and Hann is separate diagnostic; a diagnostic failure does not delete valid FP64 primary data',
        synthetic_checks_passed=True,real_NN_forwards=0))
    print('STATISTICS_PROTOCOL_FROZEN_NO_REAL_NN')


def run():
    protocol=read(ROOT/'EVALUATION_PROTOCOL.json');stats=read(ROOT/'STATISTICS_PROTOCOL.json')
    assert stats['source_sha256']==sha(__file__) and stats['source_metric_sha256']==sha(metric.__file__)
    expected={(r['subject'],r['record'],b['start'],b['stop']) for r in protocol['rows'] for b in r['blocks'] if b['eligible']}
    assert len(expected)==988
    models={};blocks=[];people=[];summaries=[];source_pins={}
    for role in ROLES:
        allblocks=[];records=[];complete=True;perseed=[];issues=[]
        for seed in SEEDS:
            p=ROOT/f'evaluation_seed{seed}/{role}_METRICS.json';j=read(p);source_pins[str(p)]=sha(p)
            bb=[dict(b) for r in j['records'] for b in r['blocks']]
            keys=[(b['subject'],b['record'],b['start'],b['stop']) for b in bb]
            valid=j['passed'] and len(keys)==len(set(keys)) and set(keys)==expected and all(b['r'] is not None for b in bb)
            complete=complete and valid;allblocks+=bb;records+=j['records'];issues+=j['failures']
            perseed.append(dict(seed=seed,passed=valid,records=len(j['records']),blocks=len(bb),
                undefined_primary_blocks=j['undefined_primary_blocks'],precision_failures=j.get('precision_failures',[])))
        blocks+=allblocks
        model=metric.aggregate_person(allblocks) if complete else None;models[role]=model
        if model:
            for subject,v in model.items():people.append(dict(subject=subject,role=role,r=v['equal_block'],equal_clip_r=v['equal_clip']))
        windows=sum(r['windows'] for r in records);undefined=sum(r['undefined_window_r'] for r in records)
        summary=dict(role=role,complete=complete,seeds=perseed,issues=issues,
            r=float(np.mean([v['equal_block'] for v in model.values()])) if model else None,
            equal_clip_r=float(np.mean([v['equal_clip'] for v in model.values()])) if model else None,
            r3s=sum(r['r3s_sum_defined'] for r in records)/windows if windows and not undefined else None,
            undefined_3s_windows=undefined,windows=windows,
            window_MSE=sum(r['window_MSE']*r['windows'] for r in records)/windows if windows else None,
            maximum_window_abs=max((r['max_abs_window_prediction'] for r in records),default=None),
            maximum_fused_abs=max((r['max_abs_fused_prediction'] for r in records),default=None),
            Hann_identity_max_abs=max((r['Hann_identity_max_abs'] for r in records),default=None))
        # The physical MSE endpoints use the same fixed block/person/seed hierarchy.
        for field in ['MSE','weighted_single_MSE','disagreement']:
            values=[]
            if complete:
                for subject in sorted(model):
                    ss=[np.mean([b[field] for b in allblocks if b['subject']==subject and b['seed']==s]) for s in SEEDS]
                    values.append(np.mean(ss))
            summary[field]=float(np.mean(values)) if values else None
        summaries.append(summary)
    columns=[];contrasts=[];subjects=sorted({r['subject'] for r in protocol['rows']})
    for a,b in PAIRS:
        valid=models[a] is not None and models[b] is not None
        d=np.array([models[a][s]['equal_block']-models[b][s]['equal_block'] for s in subjects]) if valid else None
        columns.append(d);contrasts.append(dict(comparison=f'{a}-{b}',estimable=valid,
            person_differences=[dict(subject=s,difference=float(x)) for s,x in zip(subjects,d)] if valid else None))
    tests=family_inference(columns) if any(c is not None for c in columns) else [None,None]
    for contrast,test in zip(contrasts,tests):contrast['inference']=test
    for name,rows in [('PERSON_RESULTS.csv',people),('BLOCK_RESULTS.csv',blocks)]:
        if not rows:continue
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (ROOT/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    dump('EXTERNAL_SUMMARY.json',dict(models=summaries,primary=contrasts,people=19,shared_narratives=1,
        inference_scope='Conditional on shared narrative/speaker and frozen offline/transductive unlabeled scalar adaptation',
        precision_comparisons='Record/block detail retained in per-seed W/F_METRICS; no precision or role selection',
        source_pins=source_pins,statistics_protocol_sha256=sha(ROOT/'STATISTICS_PROTOCOL.json'),goal_complete=False))
    print('EXTERNAL_SUMMARY_WRITTEN_GOAL_NOT_PREJUDGED')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','run']);args=p.parse_args()
    freeze() if args.mode=='freeze' else run()
