"""Review copied report package against actual frozen run evidence on2203."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main(report):
    results=read(report/'RESULTS.csv');summary=read(report/'SUMMARY.csv');pairs=read(report/'paired_comparisons.csv')
    review=json.loads((report/'review.json').read_text());manifest=json.loads((report/'evidence/bootstrap01/manifest.json').read_text())
    keys={(r['variant'],r['seed']) for r in results};errors=[]
    for variant,seed in keys:
        evidence=report/'evidence/formal_runs'/f'{variant}_seed{seed}';data=json.loads((evidence/'result.json').read_text())
        for row in (r for r in results if r['variant']==variant and r['seed']==seed):
            group=row['group'];value=data['weighted_2to1_r'] if group=='weighted_2to1' else data['groups'][group]['pearson_r']
            errors.append(abs(value-float(row['pearson_r'])))
    for row in summary:
        selected=[r for r in results if r['variant']==row['variant'] and r['group']==row['group']]
        errors.append(abs(np.mean([float(r['pearson_r']) for r in selected])-float(row['mean_r'])))
    mandatory=['REPORT.md','REVIEW.md','DATA_CONTRACT.md','EXPERIMENT_MATRIX.md','COMMANDS.md','RESULTS.csv','SERVER_MANIFEST.csv','learning_and_results.png','paired_differences.png']
    checks={'mandatory_outputs':all((report/name).is_file() for name in mandatory),'twelve_runs_thirtysix_result_rows':len(keys)==12 and len(results)==36,
        'all_seeds_and_variants':{r['variant'] for r in results}=={'short','middle','long','multiscale'} and {int(r['seed']) for r in results}=={1024,2026,31415},
        'all_runs_same6001params25000steps':all(int(r['parameters'])==6001 and int(r['optimizer_steps'])==25000 for r in results),
        'actual_test_counts_fixed':all(int(r['subjects'])=={'test_seen':69,'test_heldout':13,'weighted_2to1':82}[r['group']] for r in results),
        'source_run_metrics_and_summary_equal':max(errors)<1e-10,
        'prior_separate_review_pass':review['technical']=='PASS' and review['acceptance']=='PASS_WITH_LIMITATION',
        'four_preplanned_pairs_and_family_intervals':len(pairs)==4 and all((float(r['Bonferroni_family95_low'])>0)==(r['support_predeclared']=='True') for r in pairs),
        'same_agent_review_disclosed':'不是第三方' in (report/'REVIEW.md').read_text(),
        'no_array_or_weight_payload_returned':not any(p.suffix.lower() in {'.npy','.npz','.pt','.safetensors','.bin','.wav','.bdf','.gz'} for p in report.rglob('*')),
        'upstream_input_identity_checks':all(manifest['checks'].values())}
    result={'technical':'PASS' if all(checks.values()) else 'REWORK','acceptance':'PASS_WITH_LIMITATION' if all(checks.values()) else 'REWORK',
        'parent_consistency':'PASS_SS03_ONLY','checks':checks,'maximum_report_metric_error':max(errors),
        'scope':'same-agent separate post-execution report/evidence check; not third-party scientific signoff'}
    with (report/'package_review.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    if not all(checks.values()):raise RuntimeError('package review failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);main(p.parse_args().report)
