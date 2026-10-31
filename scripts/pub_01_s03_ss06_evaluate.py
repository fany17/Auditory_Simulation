"""One frozen SS06 audit assessment including prespecified history controls."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from scipy.signal import lfilter
from m6a_public.continuity.execution import build_variant
from pub_01_s03_ss06_data import ROOT,LENGTH,WARMUP,Records,dump,read
from pub_01_s03_ss06_train import tensor,evaluate,aggregate,metrics,write_csv,reset_post_encoder


def fit_AR():
    lags=(1,2,4,8,16,32);xtx=np.zeros((7,7));xty=np.zeros(7);count=0
    for r in Records('train').rows:
        _,y=Records('train').arrays(r);y=np.asarray(y[:,0],np.float64)
        for first in range(32,len(y),65536):
            last=min(len(y),first+65536);z=np.column_stack([np.ones(last-first)]+[y[first-lag:last-lag] for lag in lags])
            xtx+=z.T@z;xty+=z.T@y[first:last];count+=len(z)
    regularizer=np.eye(7)*(.001*count);regularizer[0,0]=0
    coefficients=np.linalg.solve(xtx+regularizer,xty)
    return {'lags':lags,'coefficients':coefficients.tolist(),'train_samples':count,
        'fit_record_ids':[r['record'] for r in Records('train').rows],
        'fit':'ridge prediction of next normalized train envelope from true TRAIN past only; inference uses predicted outputs exclusively',
        'scope':'specific output-autocorrelation control, not exclusion of every possible acoustic prior'}


def output_filter(pred,kind,value,ar):
    pred=np.asarray(pred,np.float64).reshape(-1)
    if kind=='ema':return lfilter([1-value],[1,-value],pred)
    coefficients=np.asarray(ar['coefficients']);den=np.zeros(33);den[0]=1
    for lag,coef in zip(ar['lags'],coefficients[1:]):den[lag]=-(1-value)*coef
    # q[t] = gain*p_local[t] + (1-gain)*(intercept + AR(q_past)).
    return lfilter([1.],den,value*pred+(1-value)*coefficients[0])


@torch.no_grad()
def local_predictions(model,data,device):
    predictions={};model.eval()
    for r in data.rows:
        x,_=data.arrays(r);pieces=[];state=None
        for first in range(0,WARMUP+r['score_blocks']*LENGTH,LENGTH):
            out=model(tensor(x[first:first+LENGTH],device),state);state=out.state;pieces.append(out.prediction.cpu().numpy().reshape(-1))
        predictions[r['record']]=np.concatenate(pieces)
    return predictions


def score_filters(data,predictions,kind,value,ar):
    rows=[]
    for r in data.rows:
        _,y=data.arrays(r);p=output_filter(predictions[r['record']],kind,value,ar)
        if not np.isfinite(p).all():raise RuntimeError('unstable output prior')
        for block in range(r['score_blocks']):
            first=WARMUP+block*LENGTH
            rows.append({'record':r['record'],'subject':r['subject'],'source_family':r['source_family'],'block':block,
                **metrics(p[first:first+LENGTH],y[first:first+LENGTH])})
    return aggregate(rows),rows


def select_output_controls():
    assert (ROOT/'all_training_complete.json').exists() and not (ROOT/'audit_opened.json').exists()
    ar=fit_AR();result={'AR':ar,'by_seed':{}}
    for seed in read(ROOT/'FORMAL_PROTOCOL.json')['seeds']:
        cp=torch.load(ROOT/'formal01'/f'local_seed{seed}'/'best_validation.pt',map_location='cpu',weights_only=True)
        model=build_variant('local').cuda().eval();model.load_state_dict(cp['model'])
        data=Records('validation');pred=local_predictions(model,data,'cuda:0');selected={}
        for kind,values in (('ema',(0.,.9,.98,.995)),('ar',(.05,.2,.5,1.))):
            scores=[]
            for value in values:
                (summary,_,_),_=score_filters(data,pred,kind,value,ar)
                scores.append({'value':value,'validation':summary})
            best=max(scores,key=lambda row:row['validation']['r']);selected[kind]={'selected':best['value'],'candidates':scores}
        result['by_seed'][str(seed)]=selected
    dump(ROOT/'output_controls_frozen.json',result);print(json.dumps(result),flush=True)


def replace_state(real,other):
    return {k:real[k] if k in ('encoder','mode') else other[k] for k in real}


@torch.no_grad()
def histories(model,data,device):
    model.eval();rows=[];canaries=[]
    ordered=sorted(data.rows,key=lambda row:row['record'])
    for index,r in enumerate(ordered):
        eligible=[s for s in ordered if s['record']!=r['record'] and s['source_family']!=r['source_family'] and s['subject']!=r['subject']]
        assert eligible,'insufficient donor groups'
        donor=eligible[index%len(eligible)];x,y=data.arrays(r);dx,_=data.arrays(donor)
        prefix=tensor(x[:WARMUP],device);wrong_prefix=tensor(dx[:WARMUP],device);current=tensor(x[WARMUP:WARMUP+LENGTH],device)
        real=model(prefix,None).state;wrong=model(wrong_prefix,None).state
        reset=reset_post_encoder(real);mismatch=replace_state(real,wrong)
        states={'true':real,'state_reset':reset,'state_wrong':mismatch,'prefix_reset':None,'prefix_wrong':wrong}
        features={};outputs={}
        for condition,state in states.items():
            captured=[]
            hook=model.encoder.register_forward_hook(lambda module,args,out:captured.append(out[0].detach().clone()))
            out=model(current,state);hook.remove();outputs[condition]=out.prediction
            if captured:features[condition]=captured[0]
            rows.append({'condition':condition,'record':r['record'],'subject':r['subject'],'source_family':r['source_family'],
                'block':0,'donor_record':donor['record'],'donor_subject':donor['subject'],'donor_family':donor['source_family'],
                **metrics(out.prediction.cpu().numpy(),y[WARMUP:WARMUP+LENGTH])})
        if features:
            assert torch.equal(features['true'],features['state_reset']) and torch.equal(features['true'],features['state_wrong'])
        if set(real)<={'encoder'}:assert all(torch.equal(outputs['true'],outputs[k]) for k in ('state_reset','state_wrong'))
        canaries.append({'record':r['record'],'current_EEG_fixed':True,'state_only_current_local_features_exactly_equal':True,
            'same_phase_different_record_person_family_donor':True,'equal_60s_prefix_duration':True})
    return rows,canaries


def run(name,seed,device):
    torch.set_num_threads(1);folder=ROOT/'formal01'/f'{name}_seed{seed}';outdir=folder/'audit01';outdir.mkdir(exist_ok=False)
    assert read(ROOT/'all_training_complete.json')['technical']=='PASS'
    dump(outdir/'opened.json',{'epoch':time.time(),'assessment':1,'fixed_conditions_in_single_assessment':True})
    cp=torch.load(folder/'best_validation.pt',map_location=device,weights_only=True)
    model=build_variant(name).to(device).eval();model.load_state_dict(cp['model']);data=Records('audit_holdout')
    results={};all_records=[];all_persons=[]
    def save(condition,endpoint,rows):
        summary,records,persons=aggregate(rows);results[condition]={'endpoint':endpoint,**summary}
        write_csv(outdir/f'{condition}_segments.csv',rows)
        all_records.extend(dict(variant=name,seed=seed,condition=condition,endpoint=endpoint,**r) for r in records)
        all_persons.extend(dict(variant=name,seed=seed,condition=condition,endpoint=endpoint,**r) for r in persons)
    (_,_,_),rows=evaluate(model,data,device);save('true','all_complete60s_after60s_warmup',rows)
    if name=='m2':
        (_,_,_),rows=evaluate(model,data,device,periodic_reset=True);save('periodic_reset3s','all_complete60s_after60s_warmup',rows)
    if name=='local':
        controls=read(ROOT/'output_controls_frozen.json');pred=local_predictions(model,data,device)
        for kind in ('ema','ar'):
            selected=controls['by_seed'][str(seed)][kind]['selected']
            (_,_,_),rows=score_filters(data,pred,kind,selected,controls['AR']);save('local_'+kind,'all_complete60s_after60s_warmup',rows)
    rows,canaries=histories(model,data,device)
    for condition in ('true','state_reset','state_wrong','prefix_reset','prefix_wrong'):
        save('history_'+condition,'first60s_score_after_equal60s_prefix_only',[r for r in rows if r['condition']==condition])
    write_csv(outdir/'per_record.csv',all_records);write_csv(outdir/'per_subject.csv',all_persons)
    dump(outdir/'result.json',{'variant':name,'seed':seed,'selected_step':cp['step'],'conditions':results,
        'history_canaries':canaries,'audit_assessments':1,'interpretation':'INTERNAL/EXPLORATORY, prior SS03/SS05 training exposure',
        'history_endpoint_is_separate_from_full_record_primary':True})
    print(json.dumps({'variant':name,'seed':seed,'primary':results['true']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--select-output-controls',action='store_true');p.add_argument('--variant');p.add_argument('--seed',type=int);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.select_output_controls:select_output_controls()
    else:run(a.variant,a.seed,a.device)
