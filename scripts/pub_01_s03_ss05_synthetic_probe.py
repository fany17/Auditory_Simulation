"""CPU synthetic long-lag cue recall probe, NOT EEG/BCI/clinical evidence.

Any functioning GRU could solve this toy. Purpose: expose whether the new
state can carry information unavailable to the causal 3-tap local control.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from pub_01_s03_ss05_predictive_memory import PredictiveEventMemory, parameter_count
from torch import nn

class GRUReference(nn.Module):
    """Independent strong off-the-shelf recurrent sanity reference."""
    def __init__(self, input_channels=6, hidden=40):
        super().__init__()
        self.core = nn.GRU(input_channels, hidden, batch_first=True)
        self.predict = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.predict(self.core(x)[0])


def make_records(n, seed, length=32):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, length, 6), np.float32)
    y = np.zeros((n, length, 1), np.float32)
    mask = np.zeros((n, length, 1), np.float32)
    v = rng.choice([-1., 1.], size=n)
    cues = rng.integers(2, 7, size=n)
    queries = rng.integers(25, 31, size=n)
    x[:, :, 3:] = rng.normal(0, .08, size=(n, length, 3))
    ix = np.arange(n)
    x[ix, cues, 0], x[ix, cues, 1] = 1., v
    x[ix, queries, 2], y[ix, queries, 0], mask[ix, queries, 0] = 1., v, 1.
    assert (queries - cues).min() >= 19
    return tuple(map(torch.from_numpy, (x, y, mask)))


def masked_mse(pred, y, mask):
    return (((pred-y)**2)*mask).sum()/mask.sum()


def evaluate(model, records, counterfactual=True):
    x, y, mask = records
    model.eval()
    with torch.no_grad():
        pred = model(x)
        score = {
            'query_mse':float(masked_mse(pred,y,mask)),
            'query_sign_accuracy':float((((pred*y)>0).float()*mask).sum()/mask.sum()),
        }
        if counterfactual:
            altered=x.clone(); b,t=torch.where(x[:,:,0]==1)
            altered[b,t,1]=-x[b,t,1]   # only remote past, not current sample
            altered_pred=model(altered)
            score['counterfactual_cue_flip_prediction_delta']=float(((altered_pred-pred).abs()*mask).sum()/mask.sum())
            score['counterfactual_accuracy_original_labels']=float((((altered_pred*y)>0).float()*mask).sum()/mask.sum())
    return score


def train_variant(name, seed, steps):
    torch.manual_seed(seed)
    model=(GRUReference() if name=="gru" else PredictiveEventMemory(input_channels=6,hidden=12,key_dim=6,variant=name))
    opt=torch.optim.Adam(model.parameters(),lr=.007)
    train=make_records(1024,seed+10000)
    valid=make_records(128,seed+20000)
    test=make_records(128,seed+30000)
    g=torch.Generator().manual_seed(seed)
    best=float('inf');weights=None;chosen=None;started=time.perf_counter()
    for step in range(1,steps+1):
        idx=torch.randperm(len(train[0]),generator=g)[:48]
        x,y,mask=(a[idx] for a in train)
        model.train();opt.zero_grad(set_to_none=True)
        loss=masked_mse(model(x),y,mask);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step in (1,steps) or step%40==0:
            v=evaluate(model,valid,counterfactual=False)['query_mse']
            if v<best:
                best=v;chosen=step;weights={k:v.clone() for k,v in model.state_dict().items()}
    model.load_state_dict(weights)
    return dict(variant=name,seed=seed,steps=steps,selected_step=chosen,
        instantiated_parameters=parameter_count(model),
        validation=evaluate(model,valid),test=evaluate(model,test),
        seconds=round(time.perf_counter()-started,2))


def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
    a.add_argument('--steps',type=int,default=120);args=a.parse_args()
    torch.set_num_threads(1)
    runs=[]
    for name in ('local','state','predictive_state','assoc_no_pred','full','gru'):
        outcome=train_variant(name,seed=1024,steps=args.steps)
        runs.append(outcome);print(json.dumps(outcome),flush=True)
    summary={
        'task':'synthetic_long_lag_cue_recall',
        'real_EEG_used':False,'patient_data_used':False,
        'task_definition':'cue sign at t2..6, query at t25..30, identical present-window distribution',
        'input_shape':'[N,32,6]',
        'data_sizes':{'train':1024,'validation':128,'test':128},
        'purpose':'capability smoke; NOT model novelty or EEG result',
        'limitations':['one seed','small synthetic data','GRU trained as standard comparator','ablations match instantiated, not active parameters','GRU has a slightly different parameter count'],
        'runs':runs,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
