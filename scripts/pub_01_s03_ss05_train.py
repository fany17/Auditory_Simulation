"""One fixed SS05 from-scratch train/validation run, no holdout access."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from pub_01_s03_ss05_data import ROOT,dump,ContinuousPool
from pub_01_s03_ss05_finite_history import FiniteHistoryDecoder,pcc_loss
from pub_01_s03_ss05_evaluate import evaluate


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_sources(protocol):
    for name,value in protocol['source_sha256'].items():assert digest(ROOT/'input'/name)==value,'source modified after freeze'


def main(variant,seed,device):
    protocol=json.loads((ROOT/'FORMAL_PROTOCOL.json').read_text());check_sources(protocol)
    assert variant in protocol['variants'] and seed in protocol['seeds']
    out=ROOT/'formal01'/f'{variant}_seed{seed}';out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
    model=FiniteHistoryDecoder(variant).to(device);train=ContinuousPool('train',device);valid=ContinuousPool('validation',device)
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    best=-float('inf');best_step=None;history=[];started=time.perf_counter();loss_sum=0.;lag=protocol['lag_samples']
    for step in range(1,protocol['optimizer_steps']+1):
        ids=rng.choice(len(train.segments),protocol['batch_60s_segments'],replace=False);x,y=train.batch(ids,lag)
        model.train();opt.zero_grad(set_to_none=True);loss=pcc_loss(model(x),y)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite formal loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=float(loss.detach())
        if step%protocol['validation_every_steps']==0:
            summary,_,_,_=evaluate(model,valid,lag,('true',),batch=8);score=summary['true']['subject_macro_r']
            if score>best:
                best=score;best_step=step;torch.save({'model':model.state_dict(),'variant':variant,'seed':seed,'step':step,'validation_r':score},out/'best_validation.pt')
            row={'step':step,'train_window_negative_Pearson':loss_sum/protocol['validation_every_steps'],
                'validation_record_then_subject_macro_60s_r':score,'seconds':time.perf_counter()-started}
            history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(dict(variant=variant,seed=seed,**row)),flush=True);loss_sum=0.
    dump(out/'train_completed.json',{'variant':variant,'seed':seed,'optimizer_steps':step,'best_step':best_step,'best_validation_r':best,
        'seconds':time.perf_counter()-started,'holdout_access':False,'protocol_sha256':digest(ROOT/'FORMAL_PROTOCOL.json'),
        'instantiated_parameters':sum(p.numel() for p in model.parameters()),'all_training_blocks_same_record':True,
        'batch_order':'numpy.default_rng(seed).choice(train_segments,2,replace=False) each step; same for all variants'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();main(a.variant,a.seed,a.device)
