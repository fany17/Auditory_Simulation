"""Final lightweight report/package consistency pass; no positive-score gate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def table(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main(report):
    protocol=read(report/'FORMAL_PROTOCOL.json');results=table(report/'RESULTS.csv');summary=table(report/'SUMMARY.csv')
    review=read(report/'review.json');contrasts=table(report/'primary_contrasts.csv');error=[]
    names={(r['variant'],int(r['seed'])) for r in results}
    for variant,seed in names:
        run=report/'evidence/formal_runs'/f'{variant}_seed{seed}';actual=read(run/'result.json');train=read(run/'train_completed.json')
        assert actual['selected_step']==train['selected_step'] and train['steps']==protocol['optimizer_steps']
        if 'evidence_compaction' in train:
            compact=train['evidence_compaction'];source=Path(compact['full_source_path'])
            assert source.resolve().is_relative_to(Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_continuity_20261023/eeg_run01'))
            assert hashlib.sha256(source.read_bytes()).hexdigest()==compact['full_metadata_sha256']
            assert compact['training_sequence_verified_by_all_training_gate'] and compact['actual_sequence_steps']==3262 and compact['actual_unique_records']==249
        for row in (r for r in results if r['variant']==variant and int(r['seed'])==seed):
            expected=actual['conditions'][row['condition']]['r'];assert (row['pearson_r']=='')==(expected is None)
            if expected is not None:error.append(abs(float(row['pearson_r'])-expected))
            error.append(abs(float(row['MSE_normalized'])-actual['conditions'][row['condition']]['mse']))
    for row in summary:
        selected=[r for r in results if r['variant']==row['variant'] and r['condition']==row['condition']]
        valid=[float(r['pearson_r']) for r in selected if r['pearson_r']!='']
        assert (row['mean_r']=='')==(not valid)
        if valid:error.append(abs(float(row['mean_r'])-np.mean(valid)))
    metadata=read(report/'metadata01/trainval_manifest.json');trainrows=[r for r in metadata['records'] if r['phase']=='train' and r['score_blocks']]
    expected_spans={(r['record'],start) for r in trainrows for start in range(0,(r['samples']//3840)*3840,3840)}
    for seed in protocol['seeds']:
        ssl=read(report/'evidence/formal_runs'/f'pretrain_seed{seed}'/'pretrain_completed.json')
        if 'actual_span_sequence' in ssl:
            assert {(r['record'],r['start_sample']) for r in ssl['actual_span_sequence']}==expected_spans
            assert ssl['actual_spans']==len(ssl['actual_span_sequence'])==protocol['pretraining_steps']
        else:
            compact=ssl['evidence_compaction']
            assert compact['actual_span_sequence_verified_against_expected'] and compact['actual_span_count']==len(expected_spans)==protocol['pretraining_steps']
            assert ssl['actual_spans']==protocol['pretraining_steps'] and compact['full_source_path'] and len(compact['full_metadata_sha256'])==64
            source=Path(compact['full_source_path'])
            assert source.resolve().is_relative_to(Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_continuity_20261023/eeg_run01'))
            assert hashlib.sha256(source.read_bytes()).hexdigest()==compact['full_metadata_sha256']
            assert {(r['record'],r['start_sample']) for r in read(source)['actual_span_sequence']}==expected_spans
        assert set(ssl['actual_records'])==set(ssl['eligible_records'])=={r['record'] for r in trainrows}
        for name in ('m3_frozen','m3_finetune'):
            train=read(report/'evidence/formal_runs'/f'{name}_seed{seed}'/'train_completed.json')
            assert train['pretrained_identity']['checkpoint_sha256']==ssl['checkpoint_sha256']
    required=['EXECUTION_LOG.md','SPLIT_PROVENANCE.md','CODE_AUDIT.md','FORMAL_PROTOCOL.json','RESULTS.csv','REPORT.md','REVIEW.md',
        'learning_curves.png','model_scores.png','primary_paired_differences.png','history_interventions.png']
    checks={'required_outputs':all((report/name).is_file() for name in required),
        '39_actual_runs_243_conditions':len(names)==39 and len(results)==243,
        '13variants_3seeds':{r['variant'] for r in results}==set(protocol['variants']) and {int(r['seed']) for r in results}==set(protocol['seeds']),
        'same_supervised_steps':all(int(r['optimizer_steps'])==3262 for r in results),
        'correct_endpoint_blocks':all(int(r['blocks'])==(41 if r['condition'].startswith('history_') else 538) for r in results),
        'correct_audit_group_counts':all(int(r['subjects'])==36 and int(r['records'])==41 and int(r['source_families'])==5 for r in results),
        'eight_frozen_contrasts':{r['contrast'] for r in contrasts}==set(protocol['primary_contrasts']),
        'metrics_match_actual_evidence':max(error)<1e-10,
        'SSL_all_complete_train_spans_and_shared_weights':True,
        'post_review_actual_metrics_and_raw_CPU':review['technical']=='PASS' and review['raw_input_original_CPU_r_checked']>=200 and review['max_raw_CPU_r_error']<1e-4,
        'internal_audit_and_same_agent_disclosed':('INTERNAL' in (report/'REPORT.md').read_text(encoding='utf-8-sig') and '同一agent' in (report/'REVIEW.md').read_text(encoding='utf-8-sig')),
        'no_signal_or_weight_payload':not any(p.suffix.lower() in {'.npy','.npz','.pt','.pth','.bin','.safetensors','.wav','.bdf','.edf','.gz'} for p in report.rglob('*'))}
    result={'technical':'PASS' if all(checks.values()) else 'REWORK','acceptance':'PASS_WITH_LIMITATION' if all(checks.values()) else 'REWORK',
        'parent_consistency':'PASS_SS06_ONLY','checks':checks,'maximum_metric_difference':max(error),
        'scope':'same-agent separate package pass; negative/positive outcomes do not determine technical acceptance'}
    with (report/'package_review.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    assert all(checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);main(p.parse_args().report)
