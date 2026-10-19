"""Exploratory delayed two-key last-write recall with interfering events.

Synthetic ONLY. No EEG files. Directly compare predictive-error and delta
memory variants against a well-established GRU recurrent baseline.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss05_synthetic_probe import GRUReference, masked_mse
from pub_01_s03_ss05_predictive_memory import PredictiveEventMemory, parameter_count


def generate(n, seed, length=48):
    rng=np.random.default_rng(seed)
    x=np.zeros((n,length,6),np.float32)
    y=np.zeros((n,length,1),np.float32)
    mask=np.zeros((n,length,1),np.float32)
    samples=[]
    x[:,:,5]=rng.normal(0,.05,size=(n,length))
    for sample in range(n):
        key_values=rng.choice([-1.,1.],size=2)
        events=[(int(rng.integers(2,7)),0,key_values[0]),
                (int(rng.integers(9,15)),1,key_values[1])]
        # A later overwrite of one independent key is the distractor.
        overwrite_key=int(rng.integers(0,2))
        overwrite_value=float(rng.choice([-1.,1.]))
        events.append((int(rng.integers(17,24)),overwrite_key,overwrite_value))
        key_values[overwrite_key]=overwrite_value
        query_key=int(rng.integers(0,2));qt=int(rng.integers(39,47))
        for t,k,v in events:
            x[sample,t,0]=1.;x[sample,t,1]=v;x[sample,t,3+k]=1.
        x[sample,qt,2]=1.;x[sample,qt,3+query_key]=1.
        y[sample,qt,0]=key_values[query_key]
        mask[sample,qt,0]=1.
        samples.append({'query_t':qt,'key':query_key,'last_write_t':max(t for t,k,v in events if k==query_key)})
    return tuple(map(torch.from_numpy,(x,y,mask))),samples


def metrics(model,data):
    x,y,mask=data
    model.eval()
    with torch.no_grad():
        pred=model(x)
        return {'mse':float(masked_mse(pred,y,mask)),
                'accuracy':float((((pred*y)>0).float()*mask).sum()/mask.sum())}


def run(name, seed, steps):
    torch.manual_seed(seed)
    model=(GRUReference() if name=='gru' else PredictiveEventMemory(input_channels=6,hidden=12,key_dim=6,variant=name))
    opt=torch.optim.Adam(model.parameters(),lr=.007)
    train,_=generate(1024,seed+11000)
    val,_=generate(128,seed+22000)
    test,_=generate(128,seed+33000)
    g=torch.Generator().manual_seed(seed)
    best=float('inf');best_step=0;chosen=None
    for step in range(1,steps+1):
        ids=torch.randperm(len(train[0]),generator=g)[:48]
        x,y,mask=(a[ids] for a in train)
        model.train();opt.zero_grad(set_to_none=True)
        loss=masked_mse(model(x),y,mask)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step in (1,steps) or step%40==0:
            v=metrics(model,val)['mse']
            if v<best:
                best,best_step=v,step;chosen={k:x.detach().clone() for k,x in model.state_dict().items()}
    model.load_state_dict(chosen)
    return {'variant':name,'parameters':parameter_count(model),'seed':seed,
        'steps':steps,'best_validation_step':best_step,
        'validation':metrics(model,val),'test':metrics(model,test)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=160);ap.add_argument('--variants',nargs='+',default=['local','state','predictive_state','assoc_no_pred','full','gru'])
    args=ap.parse_args();torch.set_num_threads(1)
    runs=[]
    for name in args.variants:
        o=run(name,1024,args.steps);runs.append(o);print(json.dumps(o),flush=True)
    summary={'task':'two-key delayed last-write retrieval with competing event',
        'real_EEG_used':False,'patient_data_used':False,
        'task_definition':'cue writes key0 then key1, later overwrite one key; late query asks key-specific most recent value',
        'label_use':'train/val checkpoint, test one final assessment (synthetic exploratory only)',
        'limitations':['single seed','one synthesized distribution','not a novelty benchmark','no equalized active parameters/FLOPs'],
        'runs':runs}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
