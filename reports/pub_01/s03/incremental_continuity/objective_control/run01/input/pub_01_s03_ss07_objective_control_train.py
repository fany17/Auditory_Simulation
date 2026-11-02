"""Fixed D15 objective-only nine-fit training, TRAIN/VAL only."""
import argparse,traceback
from pub_01_s03_ss07_objective_control_core import *
def run(seed,arm,device):
    start=time.time();cpu=time.process_time();torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
    p=read(ROOT/'PROTOCOL.json');assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    assert read(ROOT/'COORDINATOR_EXECUTION_REVIEW.json')['approved_for_execution']
    for path,digest in p['source_pins'].items():assert sha(path)==digest,path
    for path,item in p['inputs'].items():
        st=Path(path).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'],path
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    folder=ROOT/f'fits/{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'STARTED.json',dict(seed=seed,arm=arm,pid=os.getpid(),time=start,protocol_sha256=sha(ROOT/'PROTOCOL.json')))
    tr=Data(seed,'train');val=Data(seed,'val');model=Conditional(seed,arm).to(device)
    parameters=sum(q.numel() for q in model.parameters());assert parameters==(164784 if arm=='P_prior' else 165792)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    initial=train_score(tr,tr.anchor);dump(folder/'OFFLINE_TRAIN_epoch-1.json',initial)
    incumbent=val.score(val.anchor);best=incumbent['phases']['val']['r'];selected=-1;history=[];updates=0;new_forward=0
    torch.save(dict(model=model.state_dict(),epoch=-1,seed=seed,arm=arm,validation_r=best),folder/'best_validation.pt')
    dump(folder/'INCUMBENT.json',incumbent)
    for epoch in range(20):
        es=time.time();model.train();total=0.;online_mse=0.;effective=0;order=permutation(seed,epoch,tr.n)
        order_log=dict(epoch=epoch,n=len(order),sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),first16=order[:16].tolist(),last16=order[-16:].tolist(),fallback_presentations=int(np.sum(tr.local[order]<9)))
        assert order_log['sha256']==next(x['sha256'] for x in p['permutations'] if x['seed']==seed and x['epoch']==epoch)
        with (folder/'PERMUTATIONS.jsonl').open('a') as f:f.write(json.dumps(order_log)+'\n')
        for first in range(0,tr.n,64):
            batch=order[first:first+64];x,h,a,y,ok=tr.batch_indices(batch,arm,device)
            opt.zero_grad(set_to_none=True);res=model(x,h);new_forward+=len(x)
            pred=a.clone();pred[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*res.double()
            loss=objective(pred,y);assert torch.isfinite(loss);loss.backward()
            assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
            if first==0:
                diagnostic=dict(epoch=epoch,updates_before_step=updates,batch_global_indices=batch.tolist(),effective_windows=len(x),fallback_windows=len(batch)-len(x),expected_upstream_zero_at_fresh_first_batch=epoch==0,modules=module_gradient_summary(model),new_forward_windows=0,extra_optimizer_updates=0,interpretation='Frozen shuffled first batch changes each epoch; activity diagnostic, not fixed-input gradient trend or selection criterion')
                with (folder/'FIRST_BATCH_GRADIENTS.jsonl').open('a') as f:f.write(json.dumps(diagnostic,allow_nan=False)+'\n')
            opt.step();updates+=1;effective+=len(x);total+=float(loss.detach())*len(batch);online_mse+=float((pred.detach()-y).square().mean())*len(batch)
            if updates%500==0:guard();assert time.time()-start<6*3600
        assert effective==248620 and order_log['fallback_presentations']==3834
        prediction,vcost=predict(model,val,device);new_forward+=vcost;metrics=val.score(prediction);del prediction;r=metrics['phases']['val']['r']
        improved=r>best
        if improved:
            best=r;selected=epoch;torch.save(dict(model=model.state_dict(),epoch=epoch,seed=seed,arm=arm,validation_r=r),folder/'best_validation.pt.writing');os.replace(folder/'best_validation.pt.writing',folder/'best_validation.pt')
        if epoch in (0,4,9,19):
            prediction,cost=predict(model,tr,device);new_forward+=cost;offline=train_score(tr,prediction);del prediction
            dump(folder/f'OFFLINE_TRAIN_epoch{epoch}.json',offline)
        item=dict(epoch=epoch,online_trajectory_negative_cosine=total/tr.n,online_trajectory_window_MSE=online_mse/tr.n,validation=metrics['phases']['val'],updates=updates,selected_improvement=improved,effective_TRAIN_windows=effective,full_target_windows=tr.n,epoch_wall_seconds=time.time()-es)
        history.append(item)
        with (folder/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(item,allow_nan=False)+'\n')
        print(json.dumps(dict(seed=seed,arm=arm,**item)),flush=True);guard()
    assert updates==78900
    torch.save(dict(model=model.state_dict(),epoch=19,seed=seed,arm=arm),folder/'last.pt')
    cp=torch.load(folder/'best_validation.pt',map_location=device,weights_only=True);assert cp['epoch']==selected;model.load_state_dict(cp['model'])
    scores=[incumbent['phases']['val']['r']]+[h['validation']['r'] for h in history];assert int(np.argmax(scores))-1==selected
    prediction,cost=predict(model,val,device);new_forward+=cost;m=val.score(prediction);del prediction
    assert abs(m['phases']['val']['r']-best)<1e-10;dump(folder/'SELECTED_METRICS.json',m)
    if arm=='H_full':
        older,cost=predict(model,val,device,older=True);new_forward+=cost;dump(folder/'OLDER_METRICS.json',val.score(older));del older
    expected=20*248620+20*30059+4*248620+30059+(30059 if arm=='H_full' else 0)
    assert new_forward==expected
    dump(folder/'COMPLETED.json',dict(seed=seed,arm=arm,epochs=20,updates=updates,selected_epoch=selected,selected_validation_r=best,new_trainable_parameters=parameters,target_window_presentations=tr.n*20,effective_residual_TRAIN_presentations=248620*20,new_residual_NN_window_forwards=new_forward,old_last19_TRAIN_forward_windows=0,new_encoder_head_forwards=0,wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),test_or_external_access=False))
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--seed',type=int,required=True);q.add_argument('--arm',choices=ARMS,required=True);q.add_argument('--device',required=True);a=q.parse_args()
    try:run(a.seed,a.arm,a.device)
    except BaseException:
        path=ROOT/f'fits/{a.arm}_seed{a.seed}/FAILURE.json'
        if path.parent.exists() and not path.exists():dump(path,dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
