"""One from-scratch run at the write-once common formal optimizer budget."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from pub_01_s03_temporal_scales import TemporalScaleDecoder,model_parameters
from pub_01_s03_ss03_data import arrays,load_contract,dump,digest
from pub_01_s03_ss03_bootstrap import loss_fn
from pub_01_s03_ss03_eval import evaluate,identity_canary


def main(config,variant,seed,device):
    cfg,_,_,root=load_contract(config);frozen=root/'formal_protocol.json';protocol=json.loads(frozen.read_text())
    assert variant in protocol['variants'] and seed in protocol['seeds'];assert protocol['test_access_before_freeze'] is False
    for name,value in protocol['source_sha256'].items():assert digest(root/'input'/name)==value,'formal source differs'
    out=root/'formal01'/f'{variant}_seed{seed}';out.mkdir(parents=True,exist_ok=False)
    dump(out/'config_frozen.json',protocol);torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    rng=np.random.default_rng(seed);model=TemporalScaleDecoder(variant=variant).to(device)
    x,y=arrays(config,'train',device);vx,vy=arrays(config,'val',device)
    canary=identity_canary(model,x[:4]);assert all(canary.values());torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    # Reset model after canary so all variants share the exact initial draw.
    model=TemporalScaleDecoder(variant=variant).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    best=-float('inf');best_step=None;history=[];started=time.perf_counter();train_sum=0.
    for step in range(1,protocol['optimizer_steps']+1):
        ids=torch.as_tensor(rng.choice(len(x),protocol['batch_size'],replace=False),device=device)
        model.train();opt.zero_grad(set_to_none=True);loss=loss_fn(model(x[ids]),y[ids])
        if not torch.isfinite(loss):raise RuntimeError('nonfinite formal loss')
        loss.backward();opt.step();train_sum+=float(loss.detach())
        if step%protocol['validation_every_steps']==0 or step==protocol['optimizer_steps']:
            val,_,_,_=evaluate(model,config,'val',device,vx);score=val['pearson_r']
            if score>best:
                best=score;best_step=step;torch.save({'model':model.state_dict(),'step':step,'validation_r':score,'variant':variant,'seed':seed},out/'best_validation.pt')
            row={'step':step,'train_negative_Pearson_interval_mean':train_sum/protocol['validation_every_steps'],
                'validation_subject_macro_60s_r':score,'validation_subjects':val['subjects'],
                'validation_excluded_segments':val['excluded_segments'],'seconds':time.perf_counter()-started}
            history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True);train_sum=0.
    dump(out/'train_completed.json',{'variant':variant,'seed':seed,'optimizer_steps':step,'best_step':best_step,'best_validation_r':best,
        'parameters':model_parameters(model),'seconds':time.perf_counter()-started,'test_scores_computed':False,'identity_canary':canary,
        'batch_schedule':'numpy.default_rng(seed).choice(train_windows,64,replace=False) independently each step; same every variant',
        'protocol_sha256':digest(frozen)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();main(a.config,a.variant,a.seed,a.device)
