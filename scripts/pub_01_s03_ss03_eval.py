"""Fixed 60s Hann reconstruction; no candidate input or test model selection."""
from collections import defaultdict
import csv
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss03_data import load_contract,arrays


def pcc(x,y):
    a=x-np.mean(x);b=y-np.mean(y);den=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return None if den==0 else float(np.sum(a*b)/den)


def predict(model,x,device,batch=512):
    answer=[];model.eval()
    with torch.no_grad():
        for first in range(0,len(x),batch):
            xx=x[first:first+batch]
            tensor=xx.to(device) if torch.is_tensor(xx) else torch.tensor(np.asarray(xx).copy(),device=device)
            answer.append(model(tensor).cpu().numpy()[:,:,0])
    return np.concatenate(answer)


def overlap(prediction,length):
    window=np.hanning(192);total=np.zeros(length);weight=np.zeros(length)
    for i,p in enumerate(prediction):
        start=i*96;total[start:start+192]+=p*window;weight[start:start+192]+=window
    keep=weight>1e-8;total[keep]/=weight[keep]
    return total,keep


def evaluate(model,config,phase,device='cuda:0',cached_x=None):
    _,_,_,root=load_contract(config);manifest=json.loads((root/'cache01/manifest.json').read_text())
    x=arrays(config,phase)[0] if cached_x is None else cached_x;segments=[];excluded=[]
    for r in manifest['records']:
        if r['phase']!=phase:continue
        first=r['phase_offset'];n=r['readout_windows'];pred=predict(model,x[first:first+n],device)
        if not np.isfinite(pred).all():raise RuntimeError('nonfinite evaluation prediction')
        total,keep=overlap(pred,r['common_samples'])
        null,_=overlap(np.roll(pred,max(1,n//2),axis=0),r['common_samples'])
        target=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)
        for start in range(0,r['common_samples']-3840+1,3840):
            mask=keep[start:start+3840];truth=np.asarray(target[start:start+3840],dtype=np.float64)[mask]
            a=total[start:start+3840][mask];b=null[start:start+3840][mask]
            identity={'phase':phase,'subject':r['subject'],'record':r['record'],'start_sample':start,'samples':len(truth)}
            score=pcc(truth,a);wrong=pcc(truth,b)
            if len(truth)<3800 or score is None or wrong is None:
                excluded.append(dict(identity,reason='support_or_zero_variance',target_std=float(np.std(truth)),prediction_std=float(np.std(a))));continue
            if not np.isfinite([score,wrong]).all():raise RuntimeError('nonfinite correlation')
            segments.append(dict(identity,pearson_r=score,wrong_EEG_r=wrong))
    persons=[]
    for subject in sorted({r['subject'] for r in segments}):
        selected=[r for r in segments if r['subject']==subject]
        persons.append({'phase':phase,'subject':subject,'segments':len(selected),
            'pearson_r':float(np.mean([r['pearson_r'] for r in selected])),
            'wrong_EEG_r':float(np.mean([r['wrong_EEG_r'] for r in selected]))})
    if not persons:raise RuntimeError('no scored persons')
    summary={'phase':phase,'subjects':len(persons),'segments':len(segments),'excluded_segments':len(excluded),
        'pearson_r':float(np.mean([r['pearson_r'] for r in persons])),
        'wrong_EEG_r':float(np.mean([r['wrong_EEG_r'] for r in persons])),
        'constant_train_mean_envelope_Pearson':None,'constant_control_reason':'UNDEFINED_ZERO_VARIANCE; not fabricated as r=0',
        'candidate_input':False}
    return summary,persons,segments,excluded


def save_csv(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def identity_canary(model,x):
    assert list(inspect.signature(model.forward).parameters)==['eeg']
    model.eval()
    with torch.no_grad():
        a=model(x);changed_target=torch.randn_like(a);b=model(x)
        wrong=model(torch.roll(x,64,1))
    return {'only_EEG_in_signature':True,'prediction_identical_without_any_target_argument':bool(torch.equal(a,b)),
        'EEG_time_perturbation_changes_prediction':bool(not torch.allclose(a,wrong)),
        'no_pretrained_encoder':all('Transformer' not in type(m).__name__ for m in model.modules())}
