"""Freeze D14, prove index/history/gradient equivalence without model fitting."""
import gc,subprocess
from pub_01_s03_ss07_order_control_core import *
def prepare():
    start=time.time();cpu=time.process_time();torch.set_num_threads(2)
    assert not (ROOT/'PROTOCOL.json').exists()
    prior=read(D13/'PROTOCOL.json');assert sha(D13/'PROTOCOL.json')=='47404c336f76365c48fb847ac59007067420ea6701a82f806c82418c7b02221c'
    for path,digest in prior['source_pins'].items():assert sha(path)==digest,path
    # Fresh full identities only of previously frozen inputs, no new download/fit.
    for path,item in prior['inputs'].items():
        st=Path(path).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'] and sha(path)==item['sha256'],path
    pins=dict(prior['source_pins'])
    for path in [D13/'PROTOCOL.json',D13/'SUPPORT_MASK.json',D13/'coordinator_review/REVIEW.json',D13/'coordinator_review/ORDER_HYPOTHESIS_EVIDENCE.json',ROOT/'PLAN.md',ROOT/'ENGINEERING_AMENDMENT.md']+[D13/f'results/seed{s}/SUMMARY.json' for s in SEEDS]+sorted((ROOT/'input').glob('pub_01_s03_ss07_order_control_*.py')):
        pins[str(path)]=sha(path)
    perm=[]
    rng_before=torch.get_rng_state().clone()
    for seed in SEEDS:
        for epoch in range(20):
            order=permutation(seed,epoch,252454);assert len(order[-38:])==38
            perm.append(dict(seed=seed,epoch=epoch,sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()))
    assert torch.equal(rng_before,torch.get_rng_state());del order
    protocol=dict(frozen=True,time=time.time(),source_pins=pins,inputs=prior['inputs'],seeds=SEEDS,arms=ARMS,epochs=20,train_windows=252454,train_effective=248620,val_effective=30059,batch_size=64,last_batch_size=38,updates_per_fit=78900,updates_total=710100,permutations=perm,permutation_stream='NumPy default_rng(SeedSequence([seed,20261102,epoch])), shared H/C/P per seed/epoch; independent of torch initialization',intervention='Only global TRAIN-window order; history source remains original record-local (5,3), older(9,7)',offline_TRAIN_epochs=[-1,0,4,9,19],old_reference='All nine D12 last19 once TRAIN only, same batch64; compare physical to D13',selection='VAL0..19 vs incumbent-1; strict > and earliest tie; selected VAL replay and H older once',optimizer=dict(name='Adam',lr=.001,weight_decay=0),formal_NN=dict(train_updates=44751600,val_epochs=5410620,offline_TRAIN_nonzero=8950320,selected_VAL=270531,selected_H_older_VAL=90177,new_models_total=59473248,old_last19_TRAIN=2237580,total=61710828),new_encoder_head=0,engineering_NN_cap=16384,engineering_optimizer_updates=0,budget=dict(wall_seconds=21600,max_idle_GPU=2,RSS_each=8*1024**3,GPU_allocated_each=2*1024**3,root_bytes=128*1024**2,each_disk_free=20*1024**3,new_arrays=0),forbidden=['old test','DTU','ds004408','new datasets','new weights','new encoder/head forwards','new loss/LR/capacity','permutation chosen by scores'],acceptance='Check optimization/order effect first; no method innovation or confirmatory generalization claim; preserve negative outcomes')
    dump(ROOT/'PROTOCOL.json',protocol)
    # Select only an actually idle GPU and record admission. Never evict others.
    device=None
    for gpu in [0,1]:
        apps=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if not any(x.strip().isdigit() for x in apps.splitlines()) and free>=4096:device=f'cuda:{gpu}';break
    assert device is not None,'No idle engineering GPU; frozen protocol retained, do not train'
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    data=Data(1024,'train');checks=[];NN=0
    for seed in SEEDS:
        for epoch in range(20):
            order=permutation(seed,epoch,data.n)
            assert all(np.any(data.local[order[first:first+64]]>=9) for first in range(0,data.n,64))
    del order
    # Complete metadata: all future, record-boundary, duplicate/omission checks.
    ids=np.flatnonzero(data.local>=9)
    for lag in (3,5,7,9):
        assert np.array_equal(data.record_id[ids-lag],data.record_id[ids])
        assert np.array_equal(data.local[ids-lag],data.local[ids]-lag)
    assert len(ids)==248620 and int(np.sum(data.local<9))==3834
    for seed in SEEDS:
        for arm in ARMS:
            a=Conditional(seed,arm);b=Conditional(seed,arm)
            assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    for arm in ARMS:
        model=Conditional(1024,arm).to(device)
        cp=torch.load(D12/f'fits/{arm}_seed1024/last.pt',map_location=device,weights_only=True);model.load_state_dict(cp['model']);model.eval()
        for first in (0,64,252416):
            last=min(first+64,data.n);indices=np.arange(first,last)
            old=frozen.Data.batch(data,first,last,arm,device);new=data.batch_indices(indices,arm,device)
            for a,b in zip(old[:4],new[:4]):assert torch.equal(a,b)
            assert np.array_equal(old[4],new[4])
            reverse=indices[::-1].copy();z=data.batch_indices(reverse,arm,device)
            with torch.no_grad():
                out=model(new[0],new[1]);back=model(z[0],z[1]);NN+=len(out)+len(back)
            assert torch.allclose(out,back.flip(0),atol=2e-6,rtol=2e-6)
        # Same actual mixed-record set in a new batch order: scalar objective and
        # complete parameter derivatives must match, no optimizer steps.
        indices=np.concatenate([np.array([r['offset']+q for q in (0,8,9,10,11,12,13,14)]) for r in data.rows[:8]])
        reverse=indices[::-1].copy();outputs=[];losses=[];grads=[]
        model.train() # cuDNN GRU backward requires train mode; no dropout here.
        for ix in (indices,reverse):
            x,h,a,y,ok=data.batch_indices(ix,arm,device);model.zero_grad(set_to_none=True)
            residual=model(x,h);NN+=len(residual);pred=a.clone();pred[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*residual.double()
            loss=(pred-y).square().mean();loss.backward();outputs.append(pred.detach().cpu());losses.append(float(loss.detach()))
            grads.append({k:q.grad.detach().cpu().clone() for k,q in model.named_parameters()})
        assert torch.allclose(outputs[0],outputs[1].flip(0),atol=2e-6,rtol=2e-6)
        assert abs(losses[0]-losses[1])<2e-7
        maxgrad=max(float((grads[0][k]-grads[1][k]).abs().max()) for k in grads[0]);assert maxgrad<2e-6
        checks.append(dict(arm=arm,contiguous_first64_tail38_tensor_exact=True,nonzero_reorder_equivalent=True,mixed_record_loss_abs=abs(losses[0]-losses[1]),parameter_gradient_max_abs=maxgrad,optimizer_updates=0))
        del model,cp,grads,outputs,old,new,z,out,back,residual,pred,loss,x,h,a,y;gc.collect();torch.cuda.empty_cache();guard()
    assert NN<=16384
    dump(ROOT/'ENGINEERING_CHECK.json',dict(passed=True,checks=checks,all60_permutations_bijective_and_torch_RNG_unchanged=True,complete_record_local_past_metadata_checked=True,NN_windows=NN,optimizer_updates=0,learning_model_fits=0,new_encoder_head=0,wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),device=device,formal_models_reconstructed_clean=True))
    print(json.dumps(dict(passed=True,NN=NN,protocol_sha256=sha(ROOT/'PROTOCOL.json'))),flush=True)
if __name__=='__main__':prepare()
