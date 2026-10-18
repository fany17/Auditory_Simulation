"""Separate post-execution review and paired subject analysis on server2203."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from pub_01_s03_temporal_scales import TemporalScaleDecoder,model_parameters
from pub_01_s03_ss03_data import dump,digest

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_architecture_20261019')


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def corr(x,y):
    a=x.astype(np.float64);b=y.astype(np.float64);a-=a.mean();b-=b.mean()
    return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))


def cpu_check(model,record,scaling,expected):
    raw=np.load(record['eeg_path'],mmap_mode='r');eeg=raw.T if raw.shape[0]==64 else raw
    starts=np.arange(40)*96
    xx=np.stack([(np.asarray(eeg[s:s+192],dtype=np.float32)-scaling['mean'])/scaling['std'] for s in starts]).astype(np.float32)
    with torch.no_grad():pred=model(torch.tensor(xx)).numpy()[:,:,0]
    positions=starts[:,None]+np.arange(192)[None,:];window=np.hanning(192)
    support=np.zeros(3936);prediction=np.zeros(3936)
    np.add.at(support,positions.ravel(),np.broadcast_to(window,positions.shape).ravel())
    np.add.at(prediction,positions.ravel(),(pred*window).ravel())
    keep=support[:3840]>1e-8;prediction=prediction[:3840][keep]/support[:3840][keep]
    target=np.load(record['envelope_path'],mmap_mode='r').reshape(-1)[:3840][keep]
    return abs(corr(target,prediction)-float(expected['pearson_r']))


def main():
    protocol=json.loads((ROOT/'formal_protocol.json').read_text());manifest=json.loads((ROOT/'cache01/manifest.json').read_text())
    constant=json.loads((ROOT/'constant_train_mean_control.json').read_text());opened=json.loads((ROOT/'test_opened.json').read_text())
    assert constant['utc_epoch']<opened['utc_epoch'] and constant['Pearson_r'] is None and constant['prediction_peak_to_peak']==0
    out=ROOT/'review01';out.mkdir(exist_ok=False);torch.set_num_threads(4)
    upstream=Path(protocol['config']['server_root'])/protocol['config']['upstream_public_available']
    scaling=np.load(upstream/'prepared/train_only_normalization.npz');aggregates=[];all_persons=[];checks=[];cpu_errors=[];aggregation_errors=[]
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            run=ROOT/'formal01'/f'{variant}_seed{seed}';result=json.loads((run/'test01/result.json').read_text())
            train=json.loads((run/'train_completed.json').read_text());history=json.loads((run/'history.json').read_text())
            selected=max(history,key=lambda r:r['validation_subject_macro_60s_r']);checkpoint=torch.load(run/'best_validation.pt',map_location='cpu',weights_only=True)
            model=TemporalScaleDecoder(variant=variant);model.load_state_dict(checkpoint['model']);model.eval()
            assert train['best_step']==selected['step']==checkpoint['step']==result['checkpoint_step']
            assert train['optimizer_steps']==25000 and len(history)==10 and train['parameters']==6001 and model_parameters(model)==6001
            assert train['protocol_sha256']==result['formal_protocol_sha256']==digest(ROOT/'formal_protocol.json')
            assert result['test_evaluations']==1 and all(result['identity_canary'].values()) and result['raw_EEG_only']
            segments=read_csv(run/'test01/per_segment.csv');persons=read_csv(run/'test01/per_subject.csv')
            excluded=json.loads((run/'test01/excluded_segments.json').read_text())
            assert not excluded,'reported invalid test segments require explicit review before acceptance'
            assert len({(r['phase'],r['record'],r['start_sample']) for r in segments})==len(segments)
            for phase in ('test_seen','test_heldout'):
                members=[r for r in persons if r['phase']==phase]
                assert {r['subject'] for r in members}==set(manifest['phase_subjects'][phase])
                for member in members:
                    s=[r for r in segments if r['phase']==phase and r['subject']==member['subject']]
                    for metric in ('pearson_r','wrong_EEG_r'):
                        aggregation_errors.append(abs(np.mean([float(r[metric]) for r in s])-float(member[metric])))
                    all_persons.append(dict(variant=variant,seed=seed,**member))
                for metric in ('pearson_r','wrong_EEG_r'):
                    aggregation_errors.append(abs(np.mean([float(r[metric]) for r in members])-result['groups'][phase][metric]))
                first=next(r for r in manifest['records'] if r['phase']==phase)
                expected=next(r for r in segments if r['phase']==phase and r['record']==first['record'] and int(r['start_sample'])==0)
                cpu_errors.append(cpu_check(model,first,scaling,expected))
                aggregates.append({'variant':variant,'seed':seed,'group':phase,'subjects':len(members),
                    'pearson_r':result['groups'][phase]['pearson_r'],'wrong_EEG_r':result['groups'][phase]['wrong_EEG_r'],
                    'parameters':6001,'optimizer_steps':25000,'best_validation_step':train['best_step'],'best_validation_r':train['best_validation_r']})
            weighted=2/3*result['groups']['test_seen']['pearson_r']+1/3*result['groups']['test_heldout']['pearson_r']
            aggregation_errors.append(abs(weighted-result['weighted_2to1_r']))
            aggregates.append({'variant':variant,'seed':seed,'group':'weighted_2to1','subjects':82,'pearson_r':weighted,
                'wrong_EEG_r':2/3*result['groups']['test_seen']['wrong_EEG_r']+1/3*result['groups']['test_heldout']['wrong_EEG_r'],
                'parameters':6001,'optimizer_steps':25000,'best_validation_step':train['best_step'],'best_validation_r':train['best_validation_r']})
            checks.append({'variant':variant,'seed':seed,'budget_selection_identity_and_test_once':'PASS'})
    comparisons=[];group_support={}
    for phase in ('test_seen','test_heldout'):
        people=manifest['phase_subjects'][phase];group_support[phase]=True
        for control in ('long','middle'):
            delta=[]
            for person in people:
                def mean(v):return np.mean([float(r['pearson_r']) for r in all_persons if r['phase']==phase and r['subject']==person and r['variant']==v])
                delta.append(mean('multiscale')-mean(control))
            values=np.array(delta);rng=np.random.default_rng(20261019)
            boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
            ci=np.percentile(boot,[2.5,97.5]);family=np.percentile(boot,[.625,99.375]);supported=bool(family[0]>0)
            group_support[phase]&=supported
            comparisons.append({'group':phase,'contrast':'multiscale_minus_'+control,'subjects':len(values),'mean_delta_r':float(values.mean()),
                'CI95_low':float(ci[0]),'CI95_high':float(ci[1]),'Bonferroni_family95_low':float(family[0]),'Bonferroni_family95_high':float(family[1]),
                'persons_positive':int((values>0).sum()),'support_predeclared':supported})
    assert max(aggregation_errors)<1e-10
    if max(cpu_errors)>=1e-4:raise RuntimeError('independent raw EEG CPU correlation does not agree')
    source_ok=all(digest(ROOT/'input'/name)==value for name,value in protocol['source_sha256'].items());assert source_ok
    save(out/'RESULTS.csv',aggregates);save(out/'per_subject_all_seeds.csv',all_persons);save(out/'paired_comparisons.csv',comparisons)
    summary=[]
    for variant in protocol['variants']:
        for group in ('test_seen','test_heldout','weighted_2to1'):
            rows=[r for r in aggregates if r['variant']==variant and r['group']==group];values=[r['pearson_r'] for r in rows]
            summary.append({'variant':variant,'group':group,'mean_r':float(np.mean(values)),'SD_across3seeds':float(np.std(values,ddof=1)),
                'wrong_EEG_r_mean':float(np.mean([r['wrong_EEG_r'] for r in rows]))})
    save(out/'SUMMARY.csv',summary)
    decision='SUPPORTED_IN_BOTH_GROUPS' if all(group_support.values()) else 'GROUP_SPECIFIC_MIXED' if any(group_support.values()) else 'H1_NOT_SUPPORTED'
    review={'technical':'PASS','acceptance':'PASS_WITH_LIMITATION','parent_consistency':'PASS_SAME_SS03_PUBLIC_ONLY_NO_SS04',
        'runs':checks,'same_source_config_hashes':source_ok,'maximum_aggregate_error':max(aggregation_errors),
        'independent_CPU_first60s_segments_checked':len(cpu_errors),'maximum_raw_EEG_CPU_r_error':max(cpu_errors),
        'constant_train_mean_control':constant,'H1_decision':decision,'group_support':group_support,'comparisons':comparisons,
        'review_type':'separate same-agent post-execution pass; not external scientific signoff',
        'limitations':['single public available fold4','three model seeds; subject bootstrap does not cover all shared-story or training-set uncertainty',
            'physical_sync UNKNOWN; computational scales are not neural latencies','small one-layer temporal prototype; no general architecture/SOTA conclusion',
            'dilation matches parameters and nominal MACCs, not hardware latency','fixed25000 steps; validation-only checkpoint; no test-driven extension']}
    dump(out/'review.json',review);print(json.dumps({'H1_decision':decision,'summary':summary,'comparisons':comparisons,'max_CPU_error':max(cpu_errors)}),flush=True)


def save(path,rows):
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
