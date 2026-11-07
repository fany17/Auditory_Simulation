"""One approved fixed fit. Never loads a test, external dataset or encoder."""
import argparse,json,os,resource,time,traceback
import numpy as np
import torch
from pub_01_s03_ss07_full_ft_history_core import *

def run(seed,arm,device):
    started=time.time();cpu=time.process_time();torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
    p=read(ROOT/'PROTOCOL.json');assert p['epochs']==20 and p['source_pins'][str(Path(__file__))]==sha(__file__)
    assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    for path,digest in p['source_pins'].items():assert sha(path)==digest,path
    for item in p['cache_metadata']:
        st=Path(item['path']).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'],item['path']
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    folder=ROOT/f'fits/{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'STARTED.json',dict(seed=seed,arm=arm,pid=os.getpid(),time=started,protocol_sha256=sha(ROOT/'PROTOCOL.json')))
    tr=Data(seed,'train');val=Data(seed,'val');model=Conditional(seed,arm).to(device)
    parameters=sum(p.numel() for p in model.parameters());assert parameters==(164784 if arm=='P_prior' else 165792)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    incumbent=val.score(val.anchor);best=incumbent['phases']['val']['r'];selected=-1;history=[];updates=0;new_forward=0
    torch.save(dict(model=model.state_dict(),epoch=-1,seed=seed,arm=arm,validation_r=best),folder/'best_validation.pt')
    dump(folder/'INCUMBENT.json',incumbent)
    for epoch in range(20):
        es=time.time();model.train();total=0.;effective=0
        for first in range(0,tr.n,64):
            last=min(first+64,tr.n);x,h,a,y,ok=tr.batch(first,last,arm,device)
            opt.zero_grad(set_to_none=True);res=model(x,h);new_forward+=len(x)
            pred=a.clone();pred[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*res.double()
            loss=(pred-y).square().mean();assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step();updates+=1;effective+=len(x);total+=float(loss.detach())*(last-first)
            if updates%500==0:guard();assert time.time()-started<6*3600
        assert effective==248620
        prediction,vcost=predict(model,val,device);new_forward+=vcost['effective_residual_windows'];metrics=val.score(prediction);r=metrics['phases']['val']['r']
        improved=r>best
        if improved:
            best=r;selected=epoch
            torch.save(dict(model=model.state_dict(),epoch=epoch,seed=seed,arm=arm,validation_r=r),folder/'best_validation.pt.writing')
            os.replace(folder/'best_validation.pt.writing',folder/'best_validation.pt')
        item=dict(epoch=epoch,train_window_MSE=total/tr.n,validation=metrics['phases']['val'],updates=updates,selected_improvement=improved,
            effective_TRAIN_windows=effective,full_target_windows=tr.n,epoch_wall_seconds=time.time()-es)
        history.append(item)
        with (folder/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(item,allow_nan=False)+'\n')
        print(json.dumps(dict(seed=seed,arm=arm,**item)),flush=True);guard()
    assert updates==78900
    torch.save(dict(model=model.state_dict(),epoch=19,seed=seed,arm=arm),folder/'last.pt')
    cp=torch.load(folder/'best_validation.pt',map_location=device,weights_only=True);assert cp['epoch']==selected;model.load_state_dict(cp['model'])
    scores=[incumbent['phases']['val']['r']]+[h['validation']['r'] for h in history];assert int(np.argmax(scores))-1==selected
    prediction,cost=predict(model,val,device);new_forward+=cost['effective_residual_windows'];m=val.score(prediction)
    assert abs(m['phases']['val']['r']-best)<1e-10
    dump(folder/'SELECTED_METRICS.json',m)
    if arm=='H_full':
        older,cost=predict(model,val,device,older=True);new_forward+=cost['effective_residual_windows']
        dump(folder/'OLDER_METRICS.json',val.score(older))
    dump(folder/'COMPLETED.json',dict(seed=seed,arm=arm,epochs=20,updates=updates,selected_epoch=selected,selected_validation_r=best,
        new_trainable_parameters=parameters,target_window_presentations=tr.n*20,effective_residual_TRAIN_presentations=248620*20,
        new_residual_NN_window_forwards=new_forward,new_encoder_head_forwards=0,wall_seconds=time.time()-started,CPU_seconds=time.process_time()-cpu,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),test_or_external_access=False))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--seed',type=int,required=True);q.add_argument('--arm',choices=ARMS,required=True);q.add_argument('--device',required=True);a=q.parse_args()
    try:run(a.seed,a.arm,a.device)
    except BaseException:
        p=ROOT/f'fits/{a.arm}_seed{a.seed}/FAILURE.json'
        if p.parent.exists() and not p.exists():dump(p,dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
