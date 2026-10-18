"""Independent final report schema and actual-evidence cross-check (2203)."""
import argparse
import csv
import json
from pathlib import Path


def main(report):
    mandatory=['PROTOCOL_MATRIX.md','REPRO_REPORT.md','RESULTS.csv','REVIEW.md','COMMANDS.md','SERVER_MANIFEST.csv']
    checks={'mandatory_files_present':all((report/p).is_file() for p in mandatory)}
    with (report/'RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    checks['seventeen_rows_with_ten_original_unrun']=len(rows)==17 and sum(r['status']=='NOT_RUN' for r in rows)==10
    actual=[r for r in rows if r['status']=='COMPLETED'];checks['seven_actual_rows']=len(actual)==7
    metrics=('top1','top5','top10','accuracy_2afc','accuracy_5afc','pearson_r','r2')
    checks['unrun_numbers_empty']=all(not any(r.get(k) for k in metrics) and int(r['subjects'])==0 for r in rows if r['status']=='NOT_RUN')
    checks['original_official_not_claimed_completed']=all(r['protocol']!='OFFICIAL_PROTOCOL' for r in actual)
    errors=[]
    for r in actual:
        path=report/r['provenance'];data=json.loads(path.read_text());test=data['test']
        if r['method']=='HDPS_ConvConcatNet_M':
            for key,field in [('Top1','top1'),('Top5','top5'),('Top10','top10')]:errors.append(abs(float(r[field])-test[key]))
        else:
            group=next((g for g in ('test_seen','test_heldout') if r['split'].endswith(g)),None)
            key='accuracy_2afc' if r['method']=='Transformer_CLIP_single' else 'pearson_r'
            value=test[group][key] if group else test['weighted_2afc' if key=='accuracy_2afc' else 'weighted_pearson_r']
            errors.append(abs(float(r[key])-value))
        review=json.loads((path.parent/'review.json').read_text())
        if review['technical']!='PASS' or review['acceptance']!='PASS_WITH_LIMITATION':raise RuntimeError('actual result lacks independent acceptance')
    checks['actual_metrics_match_reviewed_evidence']=len(errors)==9 and max(errors)<1e-7
    checks['all_actual_acceptance_limited']=all(r['acceptance']=='PASS_WITH_LIMITATION' for r in actual)
    checks['no_signal_or_weight_payload_in_returned_evidence']=not any(p.suffix.lower() in ('.npy','.npz','.pt','.bin','.safetensors','.wav','.bdf','.edf') for p in (report/'evidence/available_20261015').rglob('*'))
    review={'technical':'PASS' if all(checks.values()) else 'REWORK','acceptance':'PASS_WITH_LIMITATION' if all(checks.values()) else 'REWORK',
        'parent_consistency':'PASS_SAME_SS02_PUBLIC_ONLY_NO_NEXT_NODE','checks':checks,'maximum_metric_error':max(errors),
        'review_type':'separate post-execution pass by same agent; not external scientific sign-off',
        'scope':'authorized available-cohort method reproductions; official full-paper rows remain unrun'}
    with (report/'available_package_review.json').open('x') as f:json.dump(review,f,indent=2)
    print(json.dumps(review),flush=True)
    if not all(checks.values()):raise RuntimeError('final package review failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);main(p.parse_args().report)
