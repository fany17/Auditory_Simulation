"""Independent old-number comparison and labeled-event counterfactual toy audit."""
import json
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss05_intake import OUT,dump
from pub_01_s03_ss05_keyed_probe import generate,metrics
from pub_01_s03_ss05_synthetic_probe import GRUReference,masked_mse
from pub_01_s03_ss05_predictive_memory import PredictiveEventMemory


def fit(name):
    torch.manual_seed(1024);model=GRUReference() if name=='gru' else PredictiveEventMemory(input_channels=6,hidden=12,key_dim=6,variant=name)
    opt=torch.optim.Adam(model.parameters(),lr=.007);train,_=generate(1024,12024);val,_=generate(128,23024)
    g=torch.Generator().manual_seed(1024);best=float('inf');weights=None
    for step in range(1,161):
        ids=torch.randperm(1024,generator=g)[:48];x,y,mask=(a[ids] for a in train)
        model.train();opt.zero_grad(set_to_none=True);loss=masked_mse(model(x),y,mask);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step in (1,160) or step%40==0:
            score=metrics(model,val)['mse']
            if score<best:best=score;weights={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(weights);model.eval();return model


def main():
    torch.set_num_threads(1)
    old=json.loads((OUT/'input/historical_SYNTHETIC_RESULTS.json').read_text());a=json.loads((OUT/'synthetic01/cue160.json').read_text());b=json.loads((OUT/'synthetic01/key160.json').read_text());comparisons=[]
    for probe,new in [('probe_a',a),('probe_b',b)]:
        for row in old[probe]['results']:
            actual=next(r for r in new['runs'] if r['variant']==row['variant']);data=actual['test']
            acc=data['query_sign_accuracy'] if probe=='probe_a' else data['accuracy'];mse=data['query_mse'] if probe=='probe_a' else data['mse']
            comparisons.append({'probe':probe,'variant':row['variant'],'old_accuracy':row['test_accuracy'],'Codex_accuracy':acc,
                'old_mse':row['test_mse'],'Codex_mse':mse,'accuracy_absolute_difference':abs(acc-row['test_accuracy']),'mse_absolute_difference':abs(mse-row['test_mse'])})
    results=[];test,events=generate(128,34024);x,y,mask=test
    for name in ('state','assoc_no_pred','full','gru'):
        model=fit(name)
        with torch.no_grad():
            original=model(x);base=(original*mask).sum(1).flatten()
            relevant=x.clone();irrelevant=x.clone()
            for i,event in enumerate(events):
                relevant[i,event['last_write_t'],1]*=-1
                others=[int(t) for t in torch.where(x[i,:,0]==1)[0] if x[i,int(t),3+event['key']]==0]
                irrelevant[i,max(others),1]*=-1
            pr=(model(relevant)*mask).sum(1).flatten();pi=(model(irrelevant)*mask).sum(1).flatten();truth=(y*mask).sum(1).flatten()
            entry={'variant':name,'training_steps':160,'ground_truth_relevant_event':'last write to queried key; exact synthetic labels only',
                'original_sign_accuracy':float(((base*truth)>0).float().mean()),
                'relevant_flip_accuracy_reversed_truth':float(((pr*(-truth))>0).float().mean()),
                'irrelevant_flip_accuracy_original_truth':float(((pi*truth)>0).float().mean()),
                'relevant_flip_mean_abs_prediction_delta':float((pr-base).abs().mean()),
                'irrelevant_flip_mean_abs_prediction_delta':float((pi-base).abs().mean()),'gap_queries':[]}
            for gap in (0,16,32):
                altered=torch.zeros(len(x),48+gap,6);altered[:,:48]=x
                for i,event in enumerate(events):
                    qt=event['query_t'];altered[i,qt+gap]=x[i,qt];
                    if gap:altered[i,qt]=0
                prediction=model(altered);value=torch.stack([prediction[i,event['query_t']+gap,0] for i,event in enumerate(events)])
                entry['gap_queries'].append({'additional_gap_tokens':gap,'sign_accuracy':float(((value*truth)>0).float().mean())})
            results.append(entry)
    dump(OUT/'synthetic01/CODEX_SYNTHETIC_AUDIT.json',{'old_number_comparison':comparisons,'labeled_event_counterfactuals':results,
        'environment':{'torch':torch.__version__,'compute':'server2203 CPU'},'real_EEG':False,
        'limits':['single synthetic seed','query-delay extrapolation exploratory','no EEG relevant-event ground truth','not novelty or clinical evidence']})
    print(json.dumps({'comparisons':comparisons,'event_interventions':results}),flush=True)


if __name__=='__main__':main()
