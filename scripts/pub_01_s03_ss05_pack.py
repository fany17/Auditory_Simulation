"""Explicit lightweight report-evidence allowlist; never export signal/weight files."""
import csv
import json
from pathlib import Path
import shutil
from pub_01_s03_ss05_data import ROOT


def main():
    out=ROOT/'review01';assert json.loads((out/'review.json').read_text())['technical']=='PASS'
    evidence=out/'evidence';intake=evidence/'intake';intake.mkdir(parents=True,exist_ok=False)
    for name in ('started.json','split_frozen.json','normalization_contract.json','FORMAL_PROTOCOL.json','freeze_review.json','scalar_parameter_audit.json',
        'review_method_validation_smoke.json','formal_started.json','all_training_review.json','holdout_opened.json','formal_status.json','unit_frozen15.log'):
        shutil.copy2(ROOT/name,intake/name)
    for name in ('cue160.json','key160.json','gru600.json','CODEX_SYNTHETIC_AUDIT.json','unit_tests.log'):
        shutil.copy2(ROOT/'synthetic01'/name,intake/name)
    for name in ('actual_CPU_GPU_smoke.json','pilot.json','completed.json'):shutil.copy2(ROOT/'pilot01'/name,intake/('pilot_'+name))
    shutil.copy2(ROOT/'cache_trainval/manifest.json',intake/'trainval_cache_manifest.json');shutil.copy2(ROOT/'cache_holdout/manifest.json',intake/'holdout_cache_manifest.json')
    protocol=json.loads((ROOT/'FORMAL_PROTOCOL.json').read_text())
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            run=ROOT/'formal01'/f'{variant}_seed{seed}';target=evidence/'formal_runs'/run.name;target.mkdir(parents=True,exist_ok=False)
            for name in ('history.json','train_completed.json'):shutil.copy2(run/name,target/name)
            shutil.copy2(run/'holdout01/result.json',target/'result.json');shutil.copy2(run/'holdout01/opened.json',target/'opened.json')
    paths=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file():paths.append({'host':'server2203','path':str(p),'bytes':p.stat().st_size,
            'kind':'SERVER_ONLY_ARRAY' if p.suffix in ('.npy','.npz') else 'SERVER_ONLY_WEIGHTS' if p.suffix=='.pt' else 'source_or_structured_evidence',
            'signal_payload_return':'NO'})
    with (out/'SERVER_MANIFEST.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paths[0]));writer.writeheader();writer.writerows(paths)
    print(json.dumps({'lightweight_run_evidence':15,'path_snapshot_files':len(paths),'signal_payload_returned':0}))


if __name__=='__main__':main()
