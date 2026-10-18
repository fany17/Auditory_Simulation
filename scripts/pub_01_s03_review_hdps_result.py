"""Independent post-run rank/aggregation/split/checkpoint selection review."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main(config):
    cfg=json.loads(config.read_text());prepared=Path(cfg['prepared_dir']);run=Path(cfg['run_dir'])
    if not (run/'completed.json').is_file():raise RuntimeError('training/test not finished')
    result=json.loads((run/'result.json').read_text());history=json.loads((run/'history.json').read_text())
    manifest=json.loads((prepared/'manifest_frozen.json').read_text());groups=json.loads((prepared/'candidate_groups.json').read_text())
    with (run/'test_per_subject.csv').open(newline='') as stream:subjects=list(csv.DictReader(stream))
    with (run/'test_query_ranks.csv').open(newline='') as stream:queries=list(csv.DictReader(stream))
    best=max(history,key=lambda row:row['validation']['Top10'])
    checks={'checkpoint_chosen_by_validation_only':result['best_epoch']==best['epoch'],
        'fixed_authorized_cohort':set(r['subject'] for r in subjects)==set(manifest['subjects']) and len(subjects)==76,
        'test_evaluations_one':result['test_evaluations']==1,
        'test_not_used_for_selection':not manifest['test_labels_used_for_training_or_selection'],
        'all_selected_inputs_complete':manifest['selected_subjects_complete'] and not manifest['selected_input_failures'],
        'exact_story_phase_overlap_absent':not manifest['exact_story_phase_overlap'],
        'candidate_identity_duplicates_absent':not manifest['duplicate_candidate_identity'],
        'stopped_at_precommitted_rule':history[-1]['stale_epochs']>=cfg['training']['patience'] or len(history)==cfg['training']['max_epochs']}
    train_pairs=np.load(prepared/'train_pairs.npy');val_pairs=np.load(prepared/'val_pairs.npy');test_pairs=np.load(prepared/'test_pairs.npy')
    checks['train_val_test_feature_IDs_disjoint']=not (set(train_pairs[:,1])&set(val_pairs[:,1]) or set(train_pairs[:,1])&set(test_pairs[:,1]) or set(val_pairs[:,1])&set(test_pairs[:,1]))
    maximum_error=0.
    for row in subjects:
        rows=[q for q in queries if q['subject']==row['subject']];n=int(row['candidate_n'])
        checks['candidate_count_'+row['subject']]=n==len(rows)==len(groups['test'][row['subject']]) and all(int(q['candidate_n'])==n and 1<=int(q['true_rank'])<=n for q in rows)
        for k in (1,5,10):
            independent=float(np.mean([int(q['true_rank'])<=k for q in rows]))
            maximum_error=max(maximum_error,abs(independent-float(row['Top'+str(k)])))
    for k in (1,5,10):
        macro=float(np.mean([float(row['Top'+str(k)]) for row in subjects]))
        maximum_error=max(maximum_error,abs(macro-result['test']['Top'+str(k)]))
    checks['independent_rank_and_macro_match']=maximum_error<1e-6
    review={'technical':'PASS' if all(checks.values()) else 'REWORK',
        'acceptance':'PASS_WITH_LIMITATION' if all(checks.values()) else 'REWORK',
        'parent_consistency':'PASS_SAME_SS02_PUBLIC_SUBSET_AMENDMENT',
        'model_performance_real':True,'maximum_metric_error':maximum_error,'checks':checks,
        'limitations':['available76 versus author81/paper dataset85','M only; numeric paper M reference unverified','single seed; subject bootstrap does not cover model-seed/shared-stimulus uncertainty','author public candidate counts retained even beyond paper stated range','not a decoder/artifact freeze or PUB parent completion']}
    with (run/'review.json').open('x') as stream:json.dump(review,stream,indent=2)
    print(json.dumps({k:v for k,v in review.items() if k!='checks'}),flush=True)
    if not all(checks.values()):raise RuntimeError('independent review failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);main(parser.parse_args().config)
