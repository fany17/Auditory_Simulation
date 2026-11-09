"""Allowlist only lightweight evidence; scientific arrays and weights stay remote."""
import json
from pathlib import Path
import shutil
from pub_01_s03_ss06_data import ROOT,read


def main():
    out=ROOT/'delivery01';out.mkdir(exist_ok=False);review=ROOT/'review01'
    assert read(review/'review.json')['technical']=='PASS'
    shutil.copy2(ROOT/'FORMAL_PROTOCOL.json',out/'FORMAL_PROTOCOL.json')
    shutil.copytree(ROOT/'metadata01',out/'metadata01')
    for name in ('RESULTS.csv','SUMMARY.csv','per_record_all_seeds.csv','per_subject_all_seeds.csv','training_summary.csv','primary_contrasts.csv','history_deltas.csv',
                 'review.json','learning_curves.png','model_scores.png','primary_paired_differences.png','history_interventions.png'):
        shutil.copy2(review/name,out/name)
    metadata=out/'evidence';metadata.mkdir()
    for name in ('all_training_complete.json','all_training_review.json','ready_for_audit.json','output_controls_frozen.json',
                 'audit_opened.json','audit_assessments_complete.json','review_method_validation.json','resource_profile.json',
                 'code_provenance.json','observer_sources.json','formal_started.json','audit_waiter_started.json','environment.json'):
        shutil.copy2(ROOT/name,metadata/name)
    for folder in (ROOT/'formal01').iterdir():
        dest=metadata/'formal_runs'/folder.name;dest.mkdir(parents=True)
        for name in ('history.json','train_completed.json','pretrain_completed.json'):
            if (folder/name).is_file():shutil.copy2(folder/name,dest/name)
        for name in ('opened.json','result.json'):
            if (folder/'audit01'/name).is_file():shutil.copy2(folder/'audit01'/name,dest/name)
    assert not any(p.suffix in ('.npy','.npz','.pt','.bin','.wav','.gz') for p in out.rglob('*'))
    print(json.dumps({'lightweight_files':sum(p.is_file() for p in out.rglob('*')),'signal_or_weight_payload':0}),flush=True)


if __name__=='__main__':main()
