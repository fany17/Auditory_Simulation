"""Authorized non-fitting fresh-path/saturation probe; never formal weights."""
import gc
from pub_01_s03_ss07_objective_control_core import *
def stat_summary(value):
    x=value.detach().cpu().double().numpy()
    return dict(max_abs=float(np.max(abs(x))),RMS=float(np.sqrt(np.mean(x*x))),fraction_abs_ge_10=float(np.mean(abs(x)>=10)),fraction_abs_ge_20=float(np.mean(abs(x)>=20)),fraction_abs_ge_one_minus_1e6=float(np.mean(abs(x)>=1-1e-6)))
def grad_summary(value):
    x=value.detach().cpu().double().numpy()
    return dict(nonzero_scalars=int(np.count_nonzero(x)),norm=float(np.linalg.norm(x)),max_abs=float(np.max(abs(x))))
def main():
    start=time.time();cpu=time.process_time();torch.set_num_threads(2);device='cuda:0'
    from pub_01_s03_ss07_objective_control_queue import resources
    free,pids,mem=resources(device);assert not pids and free>=4096 and mem>=16*1024**3
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    data=Data(1024,'train');ids=np.concatenate([np.array([r['offset']+q for q in (0,8,9,10,11,12,13,14)]) for r in data.rows[:8]])
    rows=[];NN=0
    for arm in ['H_full','C_match']:
        for stage in ['fresh_fixed_nonzero_output_probe','unchanged_D14_last']:
            model=Conditional(1024,arm).to(device);model.train()
            if stage=='fresh_fixed_nonzero_output_probe':
                with torch.no_grad():
                    ids_w=torch.arange(model.output.weight.numel(),device=device,dtype=torch.float32).reshape_as(model.output.weight)
                    model.output.weight.copy_(.01*torch.sin(ids_w*.017));model.output.bias.zero_()
            else:
                cp=torch.load(D14/f'fits/{arm}_seed1024/last.pt',map_location=device,weights_only=True);model.load_state_dict(cp['model']);del cp
            captured={}
            def token_hook(module,args,out):captured['token']=out;out.retain_grad()
            def gru_hook(module,args,out):captured['states']=out[0];out[0].retain_grad()
            hooks=[model.token.register_forward_hook(token_hook),model.gru.register_forward_hook(gru_hook)]
            x,h,a,y,ok=data.batch_indices(ids,arm,device);h.requires_grad_(True);res=model(x,h);NN+=len(res)
            pred=a.clone();pred[torch.from_numpy(np.flatnonzero(ok)).to(device)]+=.1*res.double()
            loss=objective(pred,y);loss.backward();assert torch.isfinite(h.grad).all()
            result=dict(arm=arm,stage=stage,NN_windows=len(res),loss=float(loss.detach()),token=stat_summary(captured['token']),GRU_states=stat_summary(captured['states']),token_input_gradient=grad_summary(h.grad),token_output_gradient=grad_summary(captured['token'].grad),GRU_state_output_gradient=grad_summary(captured['states'].grad),GRU_hh_gradient=grad_summary(model.gru.weight_hh_l0.grad),GRU_ih_gradient=grad_summary(model.gru.weight_ih_l0.grad),module_gradient_snapshot=module_gradient_summary(model),optimizer_updates=0,learning_model_fit=False)
            if stage=='fresh_fixed_nonzero_output_probe':assert result['token_input_gradient']['nonzero_scalars']>0 and result['GRU_hh_gradient']['nonzero_scalars']>0
            rows.append(result)
            for hook in hooks:hook.remove()
            del model,x,h,a,y,res,pred,loss,captured,hooks;gc.collect();torch.cuda.empty_cache();guard()
    assert NN==192
    dump(ROOT/'PATH_PROBE.json',dict(passed=True,source_sha256=sha(__file__),rows=rows,NN_windows=NN,optimizer_updates=0,learning_model_fits=0,new_encoder_head=0,probe_only_output_weight='.01*sin(arange(numel)*.017), bias0; never used for formal init',limits='Fixed64 metadata batch; C token input is duplicated current, not actual temporal history; late-state saturation/gradient evidence does not identify all optimization causes or justify LR/model change',wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(passed=True,NN_windows=NN,rows=rows)),flush=True)
if __name__=='__main__':main()
