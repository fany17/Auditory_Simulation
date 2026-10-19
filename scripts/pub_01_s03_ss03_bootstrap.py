"""2203 alignment→actual CPU/GPU smoke→100-step train/val pilot only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from pub_01_s03_temporal_scales import TemporalScaleDecoder,DILATION_PRESETS,model_parameters
from pub_01_s03_ss03_data import prepare,arrays,load_contract,dump


def loss_fn(pred,target):
    a=pred-pred.mean(1,keepdim=True);b=target-target.mean(1,keepdim=True)
    return -torch.nn.functional.cosine_similarity(a,b,dim=1,eps=1e-6).mean()


def main(config):
    cfg,_,_,root=load_contract(config);torch.set_num_threads(4)
    dump(root/'bootstrap_started.json',{'started_utc_epoch':time.time(),'stage':'ACTUAL_SERVER_INPUT_AUDIT','source_commit':'9ffef2656d06e648073d5a7fa59e78ba03d5a648'})
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_pub_01_s03_temporal_scales.py'],check=True,cwd=Path(__file__).parent)
    prepare(config);x,y=arrays(config,'train');vx,vy=arrays(config,'val');smoke=[]
    for device in ('cpu','cuda:0'):
        for variant in DILATION_PRESETS:
            torch.manual_seed(1024);model=TemporalScaleDecoder(variant=variant).to(device)
            xx=torch.tensor(x[:64],device=device);yy=torch.tensor(y[:64],device=device)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            started=time.perf_counter();pred=model(xx);loss=loss_fn(pred,yy);opt.zero_grad();loss.backward();opt.step()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            model.eval()
            with torch.no_grad():
                for _ in range(5):model(xx)
                if device.startswith('cuda'):torch.cuda.synchronize()
                before=time.perf_counter()
                for _ in range(20):model(xx)
                if device.startswith('cuda'):torch.cuda.synchronize()
                latency=(time.perf_counter()-before)/20
                vl=loss_fn(model(torch.tensor(vx[:64],device=device)),torch.tensor(vy[:64],device=device))
            smoke.append({'device':device,'variant':variant,'parameters':model_parameters(model),'train_loss_engineering':float(loss.detach()),
                'val_loss_engineering':float(vl),'batch64_inference_seconds':latency,'memory_allocated':torch.cuda.max_memory_allocated() if device.startswith('cuda') else None})
    dump(root/'smoke.json',{'status':'ACTUAL_CPU_GPU_SMOKE_PASS','checks':smoke,'scientific_test_scores_computed':False})
    torch.manual_seed(1024);rng=np.random.default_rng(1024);model=TemporalScaleDecoder(variant='multiscale').cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];started=time.perf_counter()
    for step in range(1,101):
        ids=rng.choice(len(x),64,replace=False);xx=torch.tensor(x[ids],device='cuda');yy=torch.tensor(y[ids],device='cuda')
        opt.zero_grad(set_to_none=True);loss=loss_fn(model(xx),yy)
        if not torch.isfinite(loss):raise RuntimeError('pilot nonfinite loss')
        loss.backward();opt.step()
        if step%10==0:
            model.eval()
            with torch.no_grad():vl=loss_fn(model(torch.tensor(vx[:512],device='cuda')),torch.tensor(vy[:512],device='cuda'))
            row={'step':step,'train_loss':float(loss.detach()),'validation_first512_window_loss_engineering':float(vl),'seconds':time.perf_counter()-started}
            history.append(row);print(json.dumps(row),flush=True);model.train()
    dump(root/'pilot.json',{'seed':1024,'variant':'multiscale','optimizer_steps':100,'history':history,'test_access':False,'formal_weights_reused':False})
    dump(root/'bootstrap_completed.json',{'status':'SMOKE_AND_100STEP_TRAIN_VAL_PASS','next':'FREEZE_COMMON_FORMAL_BUDGET_BEFORE_TEST','test_access':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args().config)
