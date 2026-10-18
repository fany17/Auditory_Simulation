"""Independent frozen2AFC identity, isolation, score and macro review."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))


def main(config,run_name):
    cfg=json.loads(config.read_text());work=Path(cfg['work']);run=work/run_name;prepared=work/'prepared'
    if not (run/'completed.json').exists():raise RuntimeError('test not complete')
    result=json.loads((run/'result.json').read_text());history=json.loads((run/'history.json').read_text());manifest=json.loads((prepared/'manifest_frozen.json').read_text())
    frozen=read(prepared/'test_queries_frozen.csv');scored=read(run/'test_queries_scored.csv');subjects=read(run/'test_per_subject.csv')
    records=manifest['records'];record_sets={p:[r for r in records if r['phase']==p] for p in ('train','val','test_seen','test_heldout')}
    train_people={r['subject'] for r in record_sets['train']};unseen_people={r['subject'] for r in record_sets['test_heldout']}
    train_stories={r['story'] for r in record_sets['train']};val_stories={r['story'] for r in record_sets['val']};test_stories={r['story'] for r in record_sets['test_seen']}
    checks={'checkpoint_minimizes_validation_only':result['best_epoch']==min(history,key=lambda r:r['validation_loss'])['epoch'],
        'seen_train_val_test_exact_story_disjoint':not (train_stories&val_stories or train_stories&test_stories or val_stories&test_stories),
        'heldout_people_never_in_train':not(train_people&unseen_people),
        'all_seen_people_in_train':{r['subject'] for r in record_sets['test_seen']}<=train_people,
        'train_only_normalization':manifest['normalization_fit_phase']=='train_only',
        'test_query_count_preserved':len(frozen)==len(scored)==manifest['test_query_count'],
        'unique_query_identities':len({(q['phase'],q['record_id'],q['eeg_window_index']) for q in scored})==len(scored),
        'one_frozen_test_pass':result['test_evaluations']==1,
        'original_hidden_packet_not_claimed':not result['official_test_equivalence'],
        'stopped_at_fixed_rule':history[-1]['stale']>=cfg['training']['patience'] or len(history)==cfg['training']['epochs']}
    frozen_keys={tuple(q[k] for k in ('phase','subject','record_id','eeg_window_index','match_start','mismatch_start','matched_candidate')) for q in frozen}
    scored_keys={tuple(q[k] for k in ('phase','subject','record_id','eeg_window_index','match_start','mismatch_start','matched_candidate')) for q in scored}
    checks['candidate_manifest_not_changed']=frozen_keys==scored_keys
    correct=defaultdict(list);null=defaultdict(list);bounds=[];scores_ok=[]
    for q in scored:
        r=records[int(q['record_id'])];match=int(q['match_start']);mismatch=int(q['mismatch_start'])
        bounds.append(abs(match-mismatch)==256 and 0<=min(match,mismatch) and max(match,mismatch)+192<=r['common_samples'] and q['subject']==r['subject'] and q['phase']==r['phase'])
        values=[float(q['score0']),float(q['score1'])];prediction=int(np.argmax(values));truth=int(q['matched_candidate'])
        scores_ok.append(all(np.isfinite(values)) and int(q['prediction'])==prediction and (q['correct']=='True')==(prediction==truth))
        correct[(q['phase'],q['subject'])].append(prediction==truth);null[(q['phase'],q['subject'])].append(q['EEG_shuffle_correct']=='True')
    checks['all_candidates_same_record_plusminus4s_and_supported']=all(bounds)
    checks['independent_score_argmax_and_labels_match']=all(scores_ok)
    maximum_error=0.
    for row in subjects:
        key=(row['group'],row['subject']);maximum_error=max(maximum_error,abs(float(np.mean(correct[key]))-float(row['accuracy_2afc'])),abs(float(np.mean(null[key]))-float(row['EEG_shuffle_accuracy'])))
    for group in ('test_seen','test_heldout'):
        rows=[r for r in subjects if r['group']==group]
        macro=np.mean([float(r['accuracy_2afc']) for r in rows]);maximum_error=max(maximum_error,abs(macro-result['test'][group]['accuracy_2afc']))
    expected=2/3*result['test']['test_seen']['accuracy_2afc']+1/3*result['test']['test_heldout']['accuracy_2afc']
    maximum_error=max(maximum_error,abs(expected-result['test']['weighted_2afc']));checks['independent_subject_macro_weighted_match']=maximum_error<1e-10
    review={'technical':'PASS' if all(checks.values()) else 'REWORK','acceptance':'PASS_WITH_LIMITATION' if all(checks.values()) else 'REWORK',
        'parent_consistency':'PASS_SAME_SS02_AVAILABLE_PUBLIC_AMENDMENT','maximum_metric_error':maximum_error,'checks':checks,
        'limitations':cfg['precommitted_differences'],'model_performance_real':True}
    with (run/'review.json').open('x') as stream:json.dump(review,stream,indent=2)
    print(json.dumps({k:v for k,v in review.items() if k not in ('checks','limitations')}),flush=True)
    if not all(checks.values()):raise RuntimeError('independent review failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-name',default='training_seed1024_attempt04');args=parser.parse_args();main(args.config,args.run_name)
