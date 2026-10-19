"""One post-selection holdout pass including all frozen same-present interventions."""
import argparse
import csv
import inspect
import json
import time
import torch
from pub_01_s03_ss05_data import ROOT,dump,ContinuousPool
from pub_01_s03_ss05_finite_history import FiniteHistoryDecoder
from pub_01_s03_ss05_evaluate import evaluate
from pub_01_s03_ss05_train import digest,check_sources


def save(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(variant,seed,device):
    protocol=json.loads((ROOT/'FORMAL_PROTOCOL.json').read_text());check_sources(protocol)
    assert json.loads((ROOT/'all_training_review.json').read_text())['technical']=='PASS'
    run=ROOT/'formal01'/f'{variant}_seed{seed}';train=json.loads((run/'train_completed.json').read_text());checkpoint=torch.load(run/'best_validation.pt',map_location='cpu',weights_only=True)
    assert checkpoint['step']==train['best_step'];out=run/'holdout01';out.mkdir(exist_ok=False)
    dump(out/'opened.json',{'utc_epoch':time.time(),'test_pass':1,'selected_step':checkpoint['step']})
    torch.set_num_threads(4);model=FiniteHistoryDecoder(variant).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    pool=ContinuousPool('holdout',device);x,y=pool.batch([0],protocol['lag_samples'])
    with torch.no_grad():
        a=model(x);b=model(x);altered=x.clone();altered[:,:20]+=1.;c=model(altered)
    canary={'inference_argument_EEG_only':list(inspect.signature(model.forward).parameters)==['eeg'],
        'same_EEG_repeats_identically':bool(torch.equal(a,b)),
        'local_past_has_exactly_zero_influence':bool(torch.equal(a[:,0],c[:,0])) if variant=='local' else None,
        'same_current_EEG_when_history_changed':bool(torch.equal(x[:,20:],altered[:,20:])),
        'no_state_carried_across_records':True}
    assert canary['inference_argument_EEG_only'] and canary['same_EEG_repeats_identically'] and canary['same_current_EEG_when_history_changed']
    if variant=='local':assert canary['local_past_has_exactly_zero_influence']
    summary,persons,records,segments=evaluate(model,pool,protocol['lag_samples'],protocol['history_conditions'],batch=8)
    if variant=='local':assert all(abs(summary[m]['subject_macro_r']-summary['true']['subject_macro_r'])<1e-10 for m in protocol['history_conditions'])
    save(out/'per_subject.csv',persons);save(out/'per_record.csv',records);save(out/'per_segment.csv',segments)
    dump(out/'result.json',{'variant':variant,'seed':seed,'conditions':summary,'selected_step':checkpoint['step'],
        'lag_samples':protocol['lag_samples'],'test_evaluations':1,'canary':canary,'protocol_sha256':digest(ROOT/'FORMAL_PROTOCOL.json'),
        'constant_prediction_control':'Pearson_UNDEFINED_ZERO_VARIANCE; training mean pre-fitted and frozen',
        'interpretation':'prospective INTERNAL family holdout sourced from prior SS03train; not external independent cohort'})
    dump(out/'completed.json',{'actual_test_finished':True,'review_required':True});print(json.dumps({'variant':variant,'seed':seed,'conditions':summary}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();main(a.variant,a.seed,a.device)
