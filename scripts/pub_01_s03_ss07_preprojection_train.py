"""D18 one full bounded fit, original TRAIN/VAL only; never imports test entrypoints."""
import argparse,traceback
from pub_01_s03_ss07_preprojection_runtime import *
from pub_01_s03_ss07_preprojection_metrics import evaluate
LIVE=dict(model=None,optimizer_updates=0,update_positions=0)
def run(seed,arm,device):
    start=time.time();cpu=time.process_time();p=read(ROOT/'PROTOCOL.json');pin=sha(ROOT/'PROTOCOL.json');verify(p)
    approval=read(ROOT/'COORDINATOR_EXECUTION_REVIEW.json');assert approval['approved_for_execution'] and approval['protocol_sha256']==pin
    assert read(ROOT/'ENGINEERING_CHECK.json')['passed'] and seed in SEEDS and arm in ARMS;guard();seed_all(seed,device);torch.cuda.reset_peak_memory_stats()
    folder=ROOT/f'fits/{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'STARTED.json',dict(time=time.time(),pid=os.getpid(),uid=os.getuid(),seed=seed,arm=arm,device=device,protocol_sha256=pin))
    train=RawPhase(seed,'train',ram_raw=True);val=RawPhase(seed,'val',ram_raw=True)
    model=make_model(seed,arm,train,device);parameters=sum(v.numel() for v in model.parameters() if v.requires_grad);assert parameters==p['registered_parameters'][arm]
    LIVE['model']=model
    frozen={k:v.detach().cpu().clone() for k,v in model.reference_encoder.state_dict().items()};buffers={k:v.detach().cpu().clone() for k,v in model.readout.named_buffers()}
    dump(folder/'DATA_SETUP.json',dict(train=train.setup,val=val.setup,RSS_KiB_after_setup=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,registered_trainable_parameters=parameters,reference_encoder_frozen=True))
    legacy=read(D16/f'fits/H_full_seed{seed}/INCUMBENT.json');best=legacy['phases']['val']['r'];selected=-1
    incumbent,_=evaluate(model,val,device,anchor_only=True)
    assert abs(incumbent['phases']['val']['r']-best)<=1e-10,'OldCcap incumbent identity; do not reinterpret raw-path errors as gain'
    incumbent['selection_reference_r_exact_from_frozen_D16']=best;dump(folder/'INCUMBENT.json',incumbent)
    baseline_train,_=evaluate(model,train,device,anchor_only=True);dump(folder/'OFFLINE_TRAIN_epoch-1.json',baseline_train)
    assert all(v==0 for v in model.count.values())
    save_checkpoint(folder/'best_validation.pt',model,seed,-1,best,pin)
    opt=torch.optim.Adam(model.trainable_groups(),weight_decay=0.);updates=0;positions=0;empty_positions=0;history=[];phase_counts=[]
    for epoch in range(20):
        epoch_start=time.time();order=permutation(seed,epoch,train.n);digest=hashlib.sha256(order.tobytes()).hexdigest()
        assert digest==next(x['sha256'] for x in p['permutations'] if x['seed']==seed and x['epoch']==epoch)
        with (folder/'PERMUTATIONS.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,sha256=digest,windows=train.n,fallback_presentations=int(np.sum(train.local[order]<9))))+'\n')
        model.train();loss_sum=0.;mse_sum=0.;effective=0;before=dict(model.count)
        for batch_index,first in enumerate(range(0,train.n,64)):
            ids=order[first:first+64];batch,target=train.batch(ids,arm,device);positions+=1;LIVE['update_positions']=positions;effective+=int(batch['valid'].sum())
            opt.zero_grad(set_to_none=True);model.capture=batch_index==0
            prediction=model(**batch);loss=objective(prediction,target);assert torch.isfinite(loss)
            if bool(batch['valid'].any()):
                loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters() if v.requires_grad)
                if batch_index==0:
                    record=dict(epoch=epoch,batch_global_indices=ids.tolist(),effective_windows=int(batch['valid'].sum()),fallback_windows=int((~batch['valid']).sum()),fresh_zero_attention=epoch==0 and arm!='T_continue',modules=gradient_summary(model),activation_diagnostics=model.last_diagnostics,extra_NN=0)
                    with (folder/'FIRST_BATCH_GRADIENTS.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                opt.step();updates+=1;LIVE['optimizer_updates']=updates
            else:empty_positions+=1
            loss_sum+=float(loss.detach())*len(ids);mse_sum+=float((prediction.detach()-target).square().mean())*len(ids)
            if positions%100==0:update(folder/'STATUS.json',dict(epoch=epoch,positions=positions,optimizer_updates=updates,empty_positions=empty_positions,elapsed_seconds=time.time()-start))
            if positions%500==0:guard();assert time.time()-start<86400
        assert effective==248620 and positions==(epoch+1)*3945
        phase_counts.append(dict(stage='TRAIN_updates',epoch=epoch,**counters_delta(model.count,before)))
        before=dict(model.count);metrics,_=evaluate(model,val,device);phase_counts.append(dict(stage='epoch_VAL',epoch=epoch,**counters_delta(model.count,before)))
        current=metrics['phases']['val'];score=current['r']
        if score>best:
            best=score;selected=epoch;save_checkpoint(folder/'best_validation.pt',model,seed,epoch,best,pin)
        record=dict(epoch=epoch,optimizer_updates=updates,update_positions=positions,empty_positions=empty_positions,effective_TRAIN_windows=effective,target_TRAIN_windows=train.n,online_trajectory_negative_cosine=loss_sum/train.n,online_trajectory_window_MSE=mse_sum/train.n,validation=current,selected_epoch_so_far=selected,epoch_wall_seconds=time.time()-epoch_start)
        history.append(record)
        with (folder/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(dict(seed=seed,arm=arm,**record),allow_nan=False),flush=True)
        if epoch in (0,4,9,19):
            before=dict(model.count);offline,_=evaluate(model,train,device);phase_counts.append(dict(stage='offline_TRAIN',epoch=epoch,**counters_delta(model.count,before)));dump(folder/f'OFFLINE_TRAIN_epoch{epoch}.json',offline)
        guard()
    assert positions==78900 and updates+empty_positions==positions and empty_positions==0
    save_checkpoint(folder/'last.pt',model,seed,19,history[-1]['validation']['r'],pin)
    cp=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=True);assert cp['seed']==seed and cp['arm']==arm and cp['epoch']==selected and cp['protocol_sha256']==pin
    model.load_trainable(cp['trainable_state']);before=dict(model.count)
    selected_metrics,older=evaluate(model,val,device,anchor_only=selected==-1,paired_older=arm=='H_history')
    phase_counts.append(dict(stage='selected_VAL_and_H_older_shared_query',selected_epoch=selected,**counters_delta(model.count,before)))
    assert abs(selected_metrics['phases']['val']['r']-best)<=1e-10
    dump(folder/'SELECTED_METRICS.json',selected_metrics)
    if older is not None:dump(folder/'OLDER_METRICS.json',older)
    assert all(torch.equal(frozen[k],v.detach().cpu()) for k,v in model.reference_encoder.state_dict().items())
    assert all(torch.equal(buffers[k],v.detach().cpu()) for k,v in model.readout.named_buffers())
    verify(p);guard();dump(folder/'COMPONENT_COUNTS.json',dict(stages=phase_counts,total=model.count,all_counts_are_invoked_windows_not_unique_examples=True))
    dump(folder/'COMPLETED.json',dict(seed=seed,arm=arm,epochs=20,optimizer_updates=updates,update_positions=positions,empty_positions=empty_positions,target_window_presentations=train.n*20,effective_TRAIN_presentations=248620*20,selected_epoch=selected,selected_validation_r=best,registered_trainable_parameters=parameters,component_forward_counts=model.count,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),E0_and_Ccap_exactly_unchanged=True,checkpoint_files=2,new_prediction_disk_arrays=0,old_test_or_external_access=False,protocol_sha256=pin,timer_scope='After imports; includes data/anchor/RAM setup, source verification,all training/evaluation/checkpoint work up to final receipt',goal_complete=False))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--device',required=True);args=parser.parse_args()
    try:run(args.seed,args.arm,args.device)
    except BaseException:
        folder=ROOT/f'fits/{args.arm}_seed{args.seed}'
        if folder.exists():dump(folder/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),optimizer_updates=LIVE['optimizer_updates'],update_positions=LIVE['update_positions'],invoked_component_counts=LIVE['model'].count if LIVE['model'] is not None else {},automatic_retry=False))
        raise
