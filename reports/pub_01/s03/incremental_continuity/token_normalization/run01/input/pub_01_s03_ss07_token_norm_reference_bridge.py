"""Minimal real nonzero GPU/manual-normalized reference bridge; no fitting."""
import sys,gc
from pub_01_s03_ss07_token_norm_core import *
sys.path.insert(0,str(ROOT/'coordinator_input'))
import pub_01_s03_ss07_token_norm_independent_review as independent
from pub_01_s03_ss07_token_norm_queue import resources
def main():
    start=time.time();cpu=time.process_time();torch.set_num_threads(2)
    assert sha(ROOT/'PROTOCOL.json')=='24204cbde539ca165e76b312b53817dc4dc07e4bbdc2a365b17f73d5557bb647'
    protocol=read(ROOT/'PROTOCOL.json')
    for name in ['core','prepare','train','queue']:
        path=ROOT/f'input/pub_01_s03_ss07_token_norm_{name}.py';assert sha(path)==protocol['source_pins'][str(path)]
    assert sha(independent.__file__)=='1bd999ec8d1e09bb3f7141f4ee9f8ab54ad8594f8bcf333c1d1c0ccf814e15f2'
    assert sha(independent.a.__file__)=='0113a30704b0fe2f13ae0e23aa53b2527afa29cfefc3c0c784789f384229cc64'
    device=None
    for candidate in ['cuda:0','cuda:1']:
        free,pids,mem=resources(candidate)
        if not pids and free>=4096 and mem>=16*1024**3:device=candidate;break
    assert device is not None
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    data=Data(1024,'train');ids=np.concatenate([np.array([r['offset']+q for q in (0,8,9,10,11,12,13,14)]) for r in data.rows[:8]])
    rows=[];gpu_NN=0;manual_NN=0
    for arm in ARMS:
        model=Conditional(1024,arm).to(device);model.eval()
        with torch.no_grad():model.output.weight.copy_(.01*torch.sin(torch.arange(model.output.weight.numel(),device=device,dtype=torch.float32).reshape_as(model.output.weight)*.017));model.output.bias.zero_()
        x,h,a,y,ok=data.batch_indices(ids,arm,device)
        with torch.no_grad():out=model(x,h).cpu().numpy();gpu_NN+=len(out)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        reference=independent.manual_normalized_residual(state,x.cpu().numpy(),h.cpu().numpy());manual_NN+=len(reference)
        delta=float(np.max(abs(out-reference)));assert np.allclose(out,reference,atol=2e-6,rtol=2e-6)
        rows.append(dict(seed=1024,arm=arm,metadata_batch_windows=len(ids),effective_windows=len(out),GPU_residual_NN=len(out),manual_residual_NN=len(reference),max_abs_residual_output_difference=delta,model='fresh fixed nonzero output probe; never formal initialization',mode='eval, no dropout or persistent state'))
        del model,x,h,a,y,out,reference,state;gc.collect();torch.cuda.empty_cache();guard()
    assert gpu_NN==manual_NN==144
    dump(ROOT/'REFERENCE_BRIDGE.json',dict(passed=True,source_sha256=sha(__file__),independent_source_sha256=sha(independent.__file__),independent_helper_sha256=sha(independent.a.__file__),rows=rows,additional_GPU_residual_NN=gpu_NN,additional_manual_CPU_residual_NN=manual_NN,NN_windows=gpu_NN+manual_NN,optimizer_updates=0,learning_model_fits=0,new_encoder_head=0,scope='Only fixed48-window per-arm output bridge; no repeated528-window engineering or full TRAIN traversal',wall_seconds=time.time()-start,CPU_seconds=time.process_time()-cpu,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()))
    print(json.dumps(dict(passed=True,NN_windows=gpu_NN+manual_NN,rows=rows)),flush=True)
if __name__=='__main__':main()
