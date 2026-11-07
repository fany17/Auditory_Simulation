"""Validate observer numerical reconstruction on validation BEFORE audit opening."""
import json
import time
import numpy as np
import torch
from m6a_public.continuity.execution import build_variant
from pub_01_s03_ss06_data import ROOT,WARMUP,LENGTH,Records,dump
from pub_01_s03_ss06_train import tensor,metrics,reset_post_encoder
from pub_01_s03_ss06_evaluate import output_filter,replace_state
from pub_01_s03_ss06_review import raw_cpu_conditions
from pub_01_s03_ss06_contract import verify_contract,verify_pretrained


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert not (ROOT/'audit_opened.json').exists();data=Records('validation');ordered=sorted(data.rows,key=lambda r:r['record'])
    selected=[data.rows[0],data.rows[-1]];errors=[];mse_errors=[]
    for name in ('local','m1_multi','m2','gru','m4_adaptive','m5_full'):
        checkpoint=torch.load(ROOT/'pilot01'/f'{name}_seed1024'/'best_validation.pt',map_location='cpu',weights_only=True)
        model=build_variant(name).cuda().eval();model.load_state_dict(checkpoint['model'])
        for r in selected:
            reference=raw_cpu_conditions(name,checkpoint,r,data.rows)
            index=next(i for i,s in enumerate(ordered) if s['record']==r['record'])
            eligible=[s for s in ordered if s['record']!=r['record'] and s['subject']!=r['subject'] and s['source_family']!=r['source_family']]
            donor=eligible[index%len(eligible)];x,y=data.arrays(r);dx,_=data.arrays(donor)
            with torch.no_grad():
                real=model(tensor(x[:WARMUP],'cuda:0')).state;wrong=model(tensor(dx[:WARMUP],'cuda:0')).state
                for condition,state in (('history_true',real),('history_state_reset',reset_post_encoder(real)),('history_state_wrong',replace_state(real,wrong))):
                    out=model(tensor(x[WARMUP:WARMUP+LENGTH],'cuda:0'),state)
                    actual=metrics(out.prediction.cpu().numpy(),y[WARMUP:WARMUP+LENGTH]);expected=reference[condition]
                    errors.append(abs(actual['r']-expected['r']));mse_errors.append(abs(actual['mse']-expected['mse']))
    assert max(errors)<1e-4 and max(mse_errors)<1e-4
    rejected=[]
    for label,func in (
        ('formal_pretrain_wrong_seed',lambda:verify_contract('pretrain',999,3511)),
        ('formal_pretrain_wrong_budget',lambda:verify_contract('pretrain',1024,100)),
        ('reject_pilot_pretrained',lambda:verify_pretrained(1024,ROOT/'pilot01/pretrain_seed1024/pretrained.pt'))):
        try:func()
        except AssertionError:rejected.append(label)
        else:raise RuntimeError('guard failed: '+label)
    pred=np.random.default_rng(1024).normal(size=400);ar={'lags':[1,2,4,8,16,32],'coefficients':[.01,.3,.1,.1,.05,.02,.01]}
    expected=[]
    for t,value in enumerate(pred):
        prior=ar['coefficients'][0]+sum(c*expected[t-lag] for lag,c in zip(ar['lags'],ar['coefficients'][1:]) if t>=lag)
        expected.append(.2*value+.8*prior)
    actual=output_filter(pred,'ar',.2,ar);changed=pred.copy();changed[200:]+=10
    assert np.max(np.abs(actual-expected))<1e-12 and np.array_equal(actual[:200],output_filter(changed,'ar',.2,ar)[:200])
    dump(ROOT/'review_method_validation.json',{'technical':'PASS','phase':'validation_only_before_audit',
        'raw_original_CPU_vs_cached_GPU_correlations':len(errors),'max_r_error':max(errors),'max_MSE_error':max(mse_errors),
        'formal_guards_rejected':rejected,'predicted_output_AR_recurrence_and_causality':'PASS',
        'no_parameter_updates_or_new_model_selection':True,'epoch':time.time()})
    print(json.dumps({'method_validation':'PASS','cases':len(errors),'max_r_error':max(errors)}),flush=True)


if __name__=='__main__':main()
