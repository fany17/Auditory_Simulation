#!/usr/bin/env python3
"""Separate output review: frozen scope/split, metrics, first-record direct check, rerun.
Must run on server2203 when reading the model/derivative arrays.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def csvread(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def avg(rows, key):
    values = [float(r[key]) for r in rows if r[key] != '']
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--rerun', type=Path, required=True)
    args = parser.parse_args()
    c = json.loads((args.run/'config_snapshot.json').read_text())
    summary = json.loads((args.run/'run_summary.json').read_text())
    scope = json.loads((args.run/'scope_frozen.json').read_text())
    subject = csvread(args.run/'metrics_per_subject.csv')
    records = csvread(args.run/'metrics_per_record.csv')
    alpha = csvread(args.run/'alpha_selection.csv')
    samples = csvread(args.run/'sample_manifest.csv')
    split = csvread(args.run/'split_manifest.csv')
    upstream = csvread(c['split_path'])
    ready = [r for r in upstream if r['record_status']=='READY' and r['selected_in_mode']=='True']
    expected_subjects = sorted({r['subject'] for r in ready})[:10]
    assert scope['subjects'] == expected_subjects and len(subject)==40
    expect = {r['source_recording']:r for r in ready if r['subject'] in expected_subjects}
    assert len(split)==len(expect)==len(samples)==scope['records']
    assert not csvread(args.run/'failed_subjects.csv') and not csvread(args.run/'skipped_records.csv')
    for row in split:
        old = expect[row['source_recording']]
        assert all(row[k]==old[k] for k in ('split','subject','source_family','raw_story'))
    for s in expected_subjects:
        families = {k:{r['source_family'] for r in split if r['subject']==s and r['split']==k} for k in ('train','val','test')}
        assert all(families.values())
        assert not(families['train'] & families['val'] or families['train'] & families['test'] or families['val'] & families['test'])
        matches = [r for r in subject if r['subject']==s]
        assert len({r['test_samples'] for r in matches})==len({r['train_samples'] for r in matches})==1
    for sample in samples:
        assert int(sample['common_finite_samples'])+int(sample['nonfinite_dropped'])==int(sample['common_samples'])-864
    for row in subject:
        keys = ('subject','model','condition')
        rr = [r for r in records if all(r[k]==row[k] for k in keys)]
        aa = [r for r in alpha if all(r[k]==row[k] for k in keys)]
        assert len(rr)==int(row['test_records']) and sum(int(r['samples']) for r in rr)==int(row['test_samples'])
        assert np.isclose(avg(rr,'r'),float(row['test_record_macro_r']),atol=1e-12)
        assert np.isclose(avg(rr,'r2'),float(row['test_record_macro_r2']),atol=1e-12)
        selected = [r for r in aa if r['selected']=='True']
        assert len(selected)==1 and float(selected[0]['alpha'])==float(row['alpha'])
        assert float(selected[0]['validation_record_macro_r'])==max(float(r['validation_record_macro_r']) for r in aa)
        assert int(row['undefined_r_records'])==0
    # Direct, independent first-test-record metric calculation with saved train parameters.
    source = sorted(r['source_recording'] for r in split if r['subject']==expected_subjects[0] and r['split']=='test')[0]
    index = {r['source_recording']:r for r in csvread(c['index_path'])}[source]
    eeg = np.load(index['eeg_path'],allow_pickle=False)
    if eeg.shape[0]==64:
        eeg = eeg.T
    env = np.load(index['envelope_path'],allow_pickle=False)
    if env.ndim==2:
        env = env.mean(axis=0 if env.shape[0]<=256 and env.shape[1]>env.shape[0] else 1)
    n = min(int(index['samples']),len(eeg),len(env))
    t = np.arange(n-864)
    mask = np.isfinite(env[t])
    for off in (0,832):
        for lag in range(0,33,4):
            mask &= np.isfinite(eeg[t+off+lag]).all(axis=1)
    t = t[mask]
    params = np.load(args.run/'server_only'/f'{expected_subjects[0]}_parameters.npz')
    direct = []
    for model, lags in c['models'].items():
        for cond, shift in c['conditions'].items():
            prefix = f'{model}_{cond}'
            x = np.column_stack([eeg[t+shift+lag] for lag in lags])
            y = (env[t]-params[prefix+'_mean_y'])/params[prefix+'_std_y']
            p = ((x-params[prefix+'_mean_x'])/params[prefix+'_std_x'])@params[prefix+'_weights']
            r = float(np.corrcoef(y,p)[0,1]); r2 = float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))
            old = [a for a in records if a['source_recording']==source and a['model']==model and a['condition']==cond][0]
            assert np.isclose(r,float(old['r']),atol=1e-10) and np.isclose(r2,float(old['r2']),atol=1e-10)
            direct.append({'model':model,'condition':cond,'r_error':abs(r-float(old['r'])),'r2_error':abs(r2-float(old['r2']))})
    rerun = csvread(args.rerun/'metrics_per_subject.csv')
    first = [r for r in subject if r['subject']==expected_subjects[0]]
    max_error = 0.0
    for row in first:
        new = [r for r in rerun if r['model']==row['model'] and r['condition']==row['condition']][0]
        for k in ('alpha','train_samples','test_samples','validation_record_macro_r','test_record_macro_r','test_record_macro_r2','test_pooled_r','test_pooled_r2'):
            error = abs(float(new[k])-float(row[k])); max_error=max(max_error,error)
            assert error < 1e-10
    for s in summary['summary']:
        rows = [r for r in subject if r['model']==s['model'] and r['condition']==s['condition']]
        for k in ('test_record_macro_r','test_record_macro_r2','test_pooled_r','test_pooled_r2'):
            assert np.isclose(avg(rows,k),s[k],atol=1e-12)
    result = {'decision':'PASS_WITH_LIMITATION','technical':'PASS','acceptance':'PASS_WITH_LIMITATION','parent_consistency':'PASS',
              'checks':['10 frozen subjects/unchanged scope','exact S02 family split','no skips/failures hidden','same four-condition time support',
                        'alpha selected only by maximum validation metric','record-to-subject-to-summary arithmetic',
                        'independent first-record direct Pearson/SSE formulas','first-subject reproduction'],
              'direct_metric_check':direct,'rerun_max_absolute_error':max_error,'scope_subjects':expected_subjects,'scope_records':len(split),
              'test_records_per_condition':len(records)//4,'review_type':'separate scripted and manual review pass by same agent; not external human signoff',
              'limits':['no held-out-subject/deep-model/semantic result','single mismatch is not a p-value','publisher coordinate is not hardware synchronization','S03 and PUB-01 remain ACTIVE']}
    (args.run/'technical_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
