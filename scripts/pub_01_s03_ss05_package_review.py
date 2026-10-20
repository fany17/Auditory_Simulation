"""Final lightweight package vs actual run evidence; negative outcomes remain valid."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main(report):
    cfg=json.loads((report/'FORMAL_PROTOCOL.json').read_text());results=read(report/'RESULTS.csv');summary=read(report/'SUMMARY.csv');contrasts=read(report/'primary_contrasts.csv')
    review=json.loads((report/'review.json').read_text());split=json.loads((report/'evidence/intake/split_frozen.json').read_text())
    upstream=json.loads((report/'evidence/intake/upstream_phase_audit.json').read_text())
    fit=json.loads((report/'evidence/intake/normalization_contract.json').read_text());errors=[]
    runs={(r['variant'],int(r['seed'])) for r in results}
    for variant,seed in runs:
        run=report/'evidence/formal_runs'/f'{variant}_seed{seed}';actual=json.loads((run/'result.json').read_text())
        train=json.loads((run/'train_completed.json').read_text())
        assert actual['variant']==train['variant']==variant and actual['seed']==train['seed']==seed and actual['selected_step']==train['best_step']
        for row in (r for r in results if r['variant']==variant and int(r['seed'])==seed):
            errors.append(abs(float(row['pearson_r'])-actual['conditions'][row['condition']]['subject_macro_r']))
    for row in summary:
        selected=[float(r['pearson_r']) for r in results if r['variant']==row['variant'] and r['condition']==row['condition']]
        errors.append(abs(np.mean(selected)-float(row['mean_r'])))
    mandatory=['EXECUTION_LOG.md','CODE_AUDIT.md','DATA_CONTRACT.md','FORMAL_PROTOCOL.json','RESULTS.csv','REPORT.md','REVIEW.md',
        'SERVER_MANIFEST.csv','training_and_history_scores.png','primary_paired_differences.png','history_interventions.png']
    checks={'required_outputs':all((report/p).is_file() for p in mandatory),'15actual_runs_105condition_rows':len(runs)==15 and len(results)==105,
        'all_five_variants_three_seeds':{r['variant'] for r in results}==set(cfg['variants']) and {int(r['seed']) for r in results}==set(cfg['seeds']),
        'same_frozen_budget_lag':all(int(r['optimizer_steps'])==2000 and int(r['lag_samples'])==cfg['lag_samples'] for r in results),
        'prospective_group_counts':all(int(r['subjects'])==21 and int(r['records'])==47 and int(r['source_families'])==7 for r in results),
        'six_frozen_primary_contrasts':{r['contrast'] for r in contrasts}==set(cfg['primary_contrasts']),
        'numbers_equal_actual_evidence':max(errors)<1e-10,
        'new_normalization_fit_only_newtrain':set(fit['fit_record_ids'])=={r['record'] for r in split['records'] if r['phase']=='train'} and not fit['old_normalization_used'],
        'old_exposed_test_excluded':all(r['old_phase']=='SS03_train' for r in split['records']),
        'upstream_exclusion_label_corrected':upstream['upstream_actual_phase_counts']=={'train':426,'val':49,'test_seen':69,'test_heldout':75} and upstream['frozen_split_bytes_unchanged'],
        'same_agent_disclosed':('同一agent' in (report/'REVIEW.md').read_text() and '第三方' in (report/'REVIEW.md').read_text()),
        'preceding_separate_review_pass':review['technical']=='PASS' and review['acceptance']=='PASS_WITH_LIMITATION',
        'no_signal_or_weight_payload_in_Git_package':not any(p.suffix.lower() in {'.npy','.npz','.pt','.bin','.safetensors','.wav','.bdf','.gz'} for p in report.rglob('*'))}
    result={'technical':'PASS' if all(checks.values()) else 'REWORK','acceptance':'PASS_WITH_LIMITATION' if all(checks.values()) else 'REWORK',
        'parent_consistency':'PASS_CURRENT_SS05_ONLY','checks':checks,'maximum_metric_difference':max(errors),
        'review_scope':'same-agent separate report/source-evidence pass; scientific positive/negative outcome is not a technical gate'}
    with (report/'package_review.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    if not all(checks.values()):raise RuntimeError('final package review failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);main(p.parse_args().report)
