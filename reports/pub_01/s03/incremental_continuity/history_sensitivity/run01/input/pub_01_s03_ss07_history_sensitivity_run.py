"""D17 four fixed checkpoints x near/older; exactly384NN and8 input-only backward."""
import signal,traceback,gc
from pub_01_s03_ss07_history_sensitivity_core import *
COUNT=dict(residual_NN_windows=0,forward_calls=0,history_input_backward_calls=0,new_encoder_head=0,optimizer_updates=0,fits=0)
def verify(p):
    for path,digest in p['source_pins'].items():assert sha(path)==digest,path
    for path,item in p['inputs'].items():
        st=Path(path).stat();assert st.st_size==item['bytes'] and st.st_mtime_ns==item['mtime_ns'],path
        if 'sha256' in item:assert sha(path)==item['sha256']
    for cp in p['checkpoints']:
        st=Path(cp['path']).stat();assert st.st_size==cp['bytes'] and st.st_mtime_ns==cp['mtime_ns'] and sha(cp['path'])==cp['sha256']
def probe_once(model,current,anchor,history,condition):
    captured={}
    def token_hook(module,args,out):captured['token_before_norm']=out.reshape(48,2,32)
    def gru_hook(module,args,out):captured['token_after_norm']=args[0];captured['GRU_states']=out[0];captured['GRU_last_h']=out[0][:,-1]
    def film_hook(module,args,out):
        gamma,beta=out.chunk(2,1);tanh=torch.tanh(gamma);captured.update(film_gamma=gamma,film_beta=beta,film_tanh_gamma=tanh,film_gain_delta=.1*tanh,film_bias_delta=.1*beta)
    def current_hook(module,args,out):captured['current_q']=out
    def output_hook(module,args,out):captured['film_modulated_current']=args[0]
    hooks=[model.token.register_forward_hook(token_hook),model.gru.register_forward_hook(gru_hook),model.film.register_forward_hook(film_hook),model.current.register_forward_hook(current_hook),model.output.register_forward_hook(output_hook)]
    h=history.detach().clone().requires_grad_(True)
    try:
        assert COUNT['forward_calls']<8 and COUNT['residual_NN_windows']+48<=384
        COUNT['forward_calls']+=1;COUNT['residual_NN_windows']+=48
        residual=model(current,h)
        captured['input_history']=h;captured['residual']=residual;captured['final_prediction_float64']=anchor+.1*residual.double()
        direction=torch.sin(torch.arange(192,device=h.device,dtype=torch.float32)*.017)
        scalar=(residual*direction).sum()/48
        assert COUNT['history_input_backward_calls']<8
        COUNT['history_input_backward_calls']+=1
        gradient=torch.autograd.grad(scalar,h,create_graph=False,retain_graph=False,allow_unused=False)[0]
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        arrays={k:v.detach().cpu().numpy().copy() for k,v in captured.items()}
        stats={k:node_stats(v,k in ('input_history','token_before_norm','token_after_norm','GRU_states')) for k,v in arrays.items()}
        grad=grad_stats(gradient);assert grad['finite']
        return arrays,dict(condition=condition,nodes=stats,history_input_gradient=grad,probe_value=float(scalar.detach()),probe_definition='residual directional sum /48, fixed sin(arange192*.017); not target loss')
    finally:
        for hook in hooks:hook.remove()
def run():
    start=time.time();cpu=time.process_time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('D17 900s cap')));signal.alarm(900)
    p=read(ROOT/'PROTOCOL.json');review=read(ROOT/'COORDINATOR_EXECUTION_REVIEW.json');assert review['approved_for_execution'] and review['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
    verify(p);assert metadata()==p['metadata'];torch.set_num_threads(2)
    while True:
        resources=[admission(d) for d in ('cuda:0','cuda:1')];device=next((x['device'] for x in resources if x['admitted']),None)
        if device is not None:break
        guard();time.sleep(5) # No neural work while waiting; same900s total cap.
    dump(ROOT/'RESOURCE_ADMISSION.json',dict(time=time.time(),resources=resources,chosen=device));guard()
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    results=[];cached={}
    for cp in p['checkpoints']:
        seed=cp['seed']
        if seed not in cached:cached[seed]=fixed_inputs(seed,device)
        current,anchor,histories,meta=cached[seed];snapshot=current.detach().clone();a_snapshot=anchor.detach().clone()
        weights=torch.load(cp['path'],map_location='cpu',weights_only=True);assert weights['seed']==seed and weights['arm']=='H_full' and weights['epoch']==cp['epoch']
        model=frozen.Conditional(seed,'H_full').to(device);model.load_state_dict(weights['model']);model.train()
        assert not any(isinstance(m,torch.nn.modules.dropout._DropoutNd) for m in model.modules())
        assert not any(getattr(m,'track_running_stats',False) for m in model.modules())
        for parameter in model.parameters():parameter.requires_grad_(False)
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};states={};records=[]
        for condition in ('near','older'):
            arrays,row=probe_once(model,current,anchor,histories[condition],condition);states[condition]=arrays;records.append(row)
            assert torch.equal(current,snapshot) and torch.equal(anchor,a_snapshot);guard()
        assert all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
        diff={k:difference_stats(states['near'][k],states['older'][k]) for k in states['near']}
        assert diff['current_q']['exactly_equal_element_fraction']==1
        item=dict(seed=seed,checkpoint_kind=cp['kind'],checkpoint_epoch=cp['epoch'],effective_windows=48,metadata_windows=64,metadata=meta,conditions=records,near_minus_older=diff,current_and_anchor_tensor_equal=True,weights_exactly_unchanged=True,mode='train with frozen params, input grad only')
        dump(ROOT/f"H_seed{seed}_{cp['kind']}.json",item);results.append(item)
        del model,weights,states,records,arrays,before,snapshot,a_snapshot;gc.collect();torch.cuda.empty_cache()
    assert COUNT['residual_NN_windows']==384 and COUNT['forward_calls']==8 and COUNT['history_input_backward_calls']==8
    guard();new_arrays=[str(x.relative_to(ROOT)) for x in ROOT.rglob('*') if x.suffix in ('.npy','.npz','.pt')];assert not new_arrays
    dump(ROOT/'COMPLETED.json',dict(passed=True,protocol_sha256=sha(ROOT/'PROTOCOL.json'),**COUNT,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_files=4,root_bytes_before_receipt=sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file()),new_array_files=new_arrays,timer_scope='run() after imports, includes source verification/GPU wait/load/subset statistics/forward/backward, excludes imports and final receipt serialization',goal_complete=False,limits=['48 fixed effective TRAIN windows only','near/older jointly alter content and recency; older may preserve information','one fixed local direction per forward, not full Jacobian or all directions','float32 exact equality is local arithmetic, not global information proof','No target loss, new VAL/test or training; next method requires separate coordinator decision']))
    signal.alarm(0);print(json.dumps(dict(passed=True,**COUNT)),flush=True)
if __name__=='__main__':
    try:run()
    except BaseException:
        dump(ROOT/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),**COUNT,automatic_neural_retry=False));raise
