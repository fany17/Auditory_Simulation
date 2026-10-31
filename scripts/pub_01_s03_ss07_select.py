"""Strict three-paired-seed selection; immutable child plans, no test access."""
import argparse
import csv
import json
import math
from pathlib import Path
import time
from pub_01_s03_ss07_baseline import ROOT,read,dump,sha


def collect(stage,conditions,seeds):
    rows=[];complete={};missing=[];failed=[]
    failures=[]
    f=ROOT/('SCREEN_FAILURES.jsonl' if stage=='screen' else 'FULL_FAILURES.jsonl')
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():failures.append(json.loads(line))
    failure_keys={(r['condition'],r['seed']) for r in failures}
    for config in conditions:
        valid=[];coverage=None
        for seed in seeds:
            folder=ROOT/f'{stage}/{config["id"]}_seed{seed}';path=folder/'completed.json'
            if not path.exists():
                key=(config['id'],seed)
                (failed if key in failure_keys else missing).append({'condition':config['id'],'seed':seed})
                continue
            d=read(path);assert d['seed']==seed and d['condition']==config and d['test_read'] is False
            assert d['stage']==stage and d['protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
            assert d['updates']==d['epochs']*3945
            macro=d['validation_macro'];value=macro['macro_r'];assert math.isfinite(value)
            here=(macro['subjects'],macro['blocks'])
            if coverage is None:coverage=here
            assert coverage==here==(49,752),'different evaluation coverage cannot be averaged'
            val=read(folder/'validation_macro.json');assert val['checkpoint_selection'].startswith('original fullval3s')
            assert len(val['people'])==49 and len(val['segments'])==752
            valid.append(value);rows.append({'condition':config['id'],'seed':seed,'stage':stage,
                'validation_macro_r':value,'validation_3s_loss':d['best_validation_negative_Pearson'],
                'selected_epoch':d['selected_epoch'],'epochs':d['epochs'],'updates':d['updates'],
                'seconds':d['seconds'],'target_presentations':d['supervised_target_samples']})
        if len(valid)==len(seeds):complete[config['id']]=math.fsum(valid)/len(seeds)
    return rows,complete,missing,failed


def screen_selection():
    if (ROOT/'FULL_SELECTION.json').exists():
        selection=read(ROOT/'FULL_SELECTION.json')
        assert selection['parent_protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
        return selection
    protocol=read(ROOT/'INCREMENTAL_PROTOCOL.json');seeds=protocol['screening_seeds'];conditions=protocol['conditions']
    rows,means,missing,failed=collect('screen',conditions,seeds)
    if missing:raise RuntimeError('screen incomplete; never average missing seeds: '+json.dumps(missing))
    if failed:
        path=ROOT/'SCREEN_FAILURE_REVIEW_REQUIRED.json'
        if not path.exists():dump(path,{'failed':failed,'parent_protocol_sha256':sha(ROOT/'INCREMENTAL_PROTOCOL.json'),
            'disposition':'explicit review required; no silent discard or success-only seed mean'})
        raise RuntimeError('scientific/engineering failure review required before selection')
    assert len(rows)==54 and len(means)==18
    selected=[];approved=set();catalog={c['id']:c for c in protocol['full_conditions']}
    for family in ('m1','m2','m3','m4','m5'):
        alternatives=[c for c in conditions if c['id'] in (f'{family}_h16',f'{family}_h32')]
        best=sorted(alternatives,key=lambda c:(-means[c['id']],c['width'],c['id']))[0]
        selected.append(best['id']);approved.add(best['id'])
        for id_ in [f'{family}_current_h{best["width"]}',f'static_{family}_h{best["width"]}']:
            assert id_ in catalog;approved.add(id_)
        if family=='m3':approved.add(f'm3_reconstruct_h{best["width"]}')
        if family=='m4':approved.add('static_alignment')
        if family=='m5':approved.add(f'm5_prior_h{best["width"]}')
    # Width rule for final output prior is primary's H; not chosen from scores.
    approved.update(['output_prior_h16','output_prior_h32'])
    with (ROOT/'SCREEN_LEDGER.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    dump(ROOT/'FULL_SELECTION.json',{'frozen':True,'epoch':time.time(),'parent_protocol_sha256':sha(ROOT/'INCREMENTAL_PROTOCOL.json'),
        'selected_family_condition_ids':selected,'approved_condition_ids':sorted(approved),
        'full_initial_seeds':seeds,'full_budget':protocol['full_budget'],
        'ranking_means_all18':means,'required_successful_paired_seeds':seeds,
        'all54_success_and_common49person752block_coverage_verified':True,
        'failure_handling':'no missing/failed-seed averaging; failure would have blocked this selection',
        'controls_not_yet_started':'approved deterministic family/width catalog, scheduled only by immutable later final-primary child',
        'test_read':False,'screen_weights_reused':False})
    print(json.dumps(read(ROOT/'FULL_SELECTION.json')),flush=True)


def primary_selection():
    if (ROOT/'FULL_PRIMARY_SELECTION.json').exists():
        selection=read(ROOT/'FULL_PRIMARY_SELECTION.json')
        assert selection['parent_full_selection_sha256']==sha(ROOT/'FULL_SELECTION.json')
        return selection
    p=read(ROOT/'INCREMENTAL_PROTOCOL.json');selection=read(ROOT/'FULL_SELECTION.json')
    assert selection['parent_protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
    catalog={c['id']:c for c in p['full_conditions']};conditions=[catalog[i] for i in selection['selected_family_condition_ids']]
    rows,means,missing,failed=collect('full',conditions,p['screening_seeds'])
    assert not missing and not failed and len(rows)==15
    family_order={f:i for i,f in enumerate(['m1','m2','m3','m4','m5'])}
    best=sorted(conditions,key=lambda c:(-means[c['id']],c['width'],family_order[c['family']],c['id']))[0]
    family=best['family'];h=best['width']
    controls=[f'{family}_current_h{h}',f'static_{family}_h{h}',f'output_prior_h{h}']
    if family=='m3':controls.append(f'm3_reconstruct_h{h}')
    if family=='m4':controls.append('static_alignment')
    if family=='m5':controls.append(f'm5_prior_h{h}')
    assert all(c in selection['approved_condition_ids'] for c in controls)
    dump(ROOT/'FULL_PRIMARY_SELECTION.json',{'frozen':True,'epoch':time.time(),
        'parent_full_selection_sha256':sha(ROOT/'FULL_SELECTION.json'),
        'primary':best['id'],'full_family_validation_means':means,
        'same_full_budget_control_ids':controls,'final_seeds':p['final_seeds'],
        'native_P_five_seeds_already_authorized':True,'EMA_control':p['EMA_control'],
        'strong_finetune_control':p['strong_control'],'extra_seeds_fixed_before_any_test':True,
        'five_seed_primary_not_reselected_after_extra_seeds':True,'test_read':False})
    print(json.dumps(read(ROOT/'FULL_PRIMARY_SELECTION.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['screen','primary']);a=p.parse_args()
    (screen_selection if a.action=='screen' else primary_selection)()
