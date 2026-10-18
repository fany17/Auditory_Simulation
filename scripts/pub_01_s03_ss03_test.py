"""Exactly one locked-test inference pass for one preselected checkpoint."""
import argparse
import json
from pathlib import Path
import time
import torch
from pub_01_s03_temporal_scales import TemporalScaleDecoder
from pub_01_s03_ss03_data import load_contract,dump,digest,arrays
from pub_01_s03_ss03_eval import evaluate,save_csv,identity_canary


def main(config,variant,seed,device):
    _,_,_,root=load_contract(config);protocol=json.loads((root/'formal_protocol.json').read_text())
    if not (root/'all_training_review.json').exists():raise RuntimeError('all12 train/val must finish before test')
    assert json.loads((root/'all_training_review.json').read_text())['technical']=='PASS'
    for name,value in protocol['source_sha256'].items():assert digest(root/'input'/name)==value
    out=root/'formal01'/f'{variant}_seed{seed}';train=json.loads((out/'train_completed.json').read_text())
    checkpoint=torch.load(out/'best_validation.pt',map_location='cpu',weights_only=True)
    assert checkpoint['step']==train['best_step'] and checkpoint['validation_r']==train['best_validation_r']
    test=out/'test01';test.mkdir(exist_ok=False);dump(test/'opened.json',{'utc_epoch':time.time(),'checkpoint_step':checkpoint['step'],'test_evaluation':1})
    torch.set_num_threads(4);model=TemporalScaleDecoder(variant=variant).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    groups={};persons=[];segments=[];excluded=[]
    for phase in ('test_seen','test_heldout'):
        x=arrays(config,phase,device)[0]
        if phase=='test_seen':
            canary=identity_canary(model,x[:4]);assert all(canary.values())
        summary,p,s,e=evaluate(model,config,phase,device,x);groups[phase]=summary;persons.extend(p);segments.extend(s);excluded.extend(e)
        del x
    weighted=2/3*groups['test_seen']['pearson_r']+1/3*groups['test_heldout']['pearson_r']
    save_csv(test/'per_subject.csv',persons);save_csv(test/'per_segment.csv',segments);dump(test/'excluded_segments.json',excluded)
    dump(test/'result.json',{'variant':variant,'seed':seed,'groups':groups,'weighted_2to1_r':weighted,'test_evaluations':1,
        'checkpoint_step':checkpoint['step'],'identity_canary':canary,'constant_mean_control':'Pearson_UNDEFINED_ZERO_VARIANCE',
        'raw_EEG_only':True,'formal_protocol_sha256':digest(root/'formal_protocol.json')})
    dump(test/'completed.json',{'status':'ACTUAL_TEST_COMPLETE','review_required':True})
    print(json.dumps({'variant':variant,'seed':seed,'weighted_r':weighted,'groups':groups}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();main(a.config,a.variant,a.seed,a.device)
