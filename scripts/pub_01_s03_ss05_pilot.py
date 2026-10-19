"""Actual newtrain/validation pilot; no holdout arrays or old checkpoints."""
import json
from pathlib import Path
import socket
import time
import numpy as np
import torch
from pub_01_s03_ss05_data import ROOT,dump,ContinuousPool
from pub_01_s03_ss05_finite_history import FiniteHistoryDecoder,VARIANTS,pcc_loss,parameter_profile
from pub_01_s03_ss05_evaluate import evaluate


def main():
    if not (ROOT/'intake_completed.json').exists() or not (ROOT/'cache_trainval/completed.json').exists():raise RuntimeError('audit/data not ready')
    out=ROOT/'pilot01';out.mkdir(exist_ok=False);torch.set_num_threads(4)
    dump(out/'started.json',{'stage':'ACTUAL_CONTINUOUS_PUBLIC_EEG_TRAIN_VAL_PILOT','seed':1024,'holdout_access':False,'hostname':socket.gethostname()})
    train=ContinuousPool('train');valid=ContinuousPool('validation');x,y=train.batch([0],0);profiles=[]
    for device in ('cpu','cuda:0'):
        xx=x.to(device);yy=y.to(device)
        for variant in VARIANTS:
            torch.manual_seed(1024);model=FiniteHistoryDecoder(variant).to(device)
            profile=parameter_profile(model,xx,yy);assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            model.eval()
            with torch.no_grad():
                for _ in range(3):model(xx)
                if device.startswith('cuda'):torch.cuda.synchronize()
                started=time.perf_counter()
                for _ in range(10):model(xx)
                if device.startswith('cuda'):torch.cuda.synchronize()
                latency=(time.perf_counter()-started)/10
                local,tokens=model.encode(xx);h=tokens.unfold(1,21,1).permute(0,1,3,2)
                a=model.queries(local[:,20:],h[:,:,-1],h[:,:,:-1]);b=model.queries(local[:,20:],h[:,:,-1],torch.flip(h[:,:,:-1],dims=[2]))
                delta=float((a-b).abs().max())
            if variant=='local':assert delta==0,'local must have no past dependence'
            profiles.append(dict(variant=variant,device=device,batch1_60s_inference_seconds=latency,
                same_current_history_reverse_max_prediction_delta=delta,**profile))
    dump(out/'actual_CPU_GPU_smoke.json',{'status':'PASS','profiles':profiles,'holdout_access':False})
    runs=[]
    for lag in (0,8,16,32):
        for variant in VARIANTS:
            torch.manual_seed(1024);torch.cuda.manual_seed_all(1024);rng=np.random.default_rng(1024)
            model=FiniteHistoryDecoder(variant).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            started=time.perf_counter();losses=[]
            for step in range(1,101):
                ids=rng.choice(len(train.segments),2,replace=False);xx,yy=train.batch(ids,lag)
                model.train();opt.zero_grad(set_to_none=True);loss=pcc_loss(model(xx),yy)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite pilot loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
            summary,_,_,_=evaluate(model,valid,lag,('true',),batch=8)
            row={'variant':variant,'lag_samples':lag,'seed':1024,'optimizer_steps':100,'last25_train_negative_Pearson':float(np.mean(losses[-25:])),
                'validation_60s_subject_macro_r':summary['true']['subject_macro_r'],'seconds':time.perf_counter()-started}
            runs.append(row);(out/'progress.json').write_text(json.dumps(runs,indent=2));print(json.dumps(row),flush=True)
    lag_scores={lag:float(np.mean([r['validation_60s_subject_macro_r'] for r in runs if r['lag_samples']==lag])) for lag in (0,8,16,32)}
    chosen=max(lag_scores,key=lag_scores.get)
    dump(out/'pilot.json',{'runs':runs,'lag_selection':'maximum mean validation subject macro across all five100step variants; firstlag wins tie',
        'lag_scores':lag_scores,'selected_lag_samples':chosen,'formal_weights_reused':False,'holdout_signal_access':False,
        'fixed_exploration':'5variants×4lags×100steps,batch2; no change based on holdout or exposed SS03 tests'})
    dump(out/'completed.json',{'status':'FIVEGROUP_TRAIN_VAL_PILOT_COMPLETE','selected_lag':chosen,'next':'FREEZE_FORMAL_BEFORE_NEW_HOLDOUT_PAYLOAD_ACCESS'})


if __name__=='__main__':main()
