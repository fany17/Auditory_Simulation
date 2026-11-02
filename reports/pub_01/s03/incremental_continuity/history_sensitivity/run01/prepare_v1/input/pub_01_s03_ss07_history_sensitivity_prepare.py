"""D17 stat/source/checkpoint freeze and synthetic scalar tests; 0 NN/backward."""
from pub_01_s03_ss07_history_sensitivity_core import *
def prepare():
    start=time.time();cpu=time.process_time();guard();assert not (ROOT/'PROTOCOL.json').exists()
    assert sha(D16/'PROTOCOL.json')=='56f4be8214ec7b5402b1c6ea3e4ff2154f082150b95c30538743bd1fa170b3a2'
    old=read(D16/'PROTOCOL.json');assert read(D16/'coordinator_review/REVIEW.json')['passed']
    # Source pins are lightweight prior evidence. Large cached arrays retain stat identity and inherited SHA, not a new full-file hash traversal.
    source_pins=dict(old['source_pins'])
    for path,digest in source_pins.items():assert sha(path)==digest,path
    extra=[D16/'PROTOCOL.json',D16/'coordinator_review/REVIEW.json',ROOT/'COORDINATOR_DECISION.md',ROOT/'PLAN.md',ROOT/'D16_SCIENTIFIC_REVIEW.json']+sorted((ROOT/'input').glob('*.py'))
    for path in extra:source_pins[str(path)]=sha(path)
    inputs={};checkpoints=[]
    for seed in (1024,2026,31415):
        folder=D8/f'seed{seed}'
        for name in ('train_latent.npy','train_P.npy'):
            path=folder/name;v=old['inputs'][str(path)];st=path.stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns']
            inputs[str(path)]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,inherited_D16_sha256_not_recomputed=v['sha256'])
        for name in ('TRAIN_TRANSFORMS.npz','Ccap_lambda0.01_COEFFICIENTS.npz','P_FT_BASELINE.json'):
            path=folder/name;digest=sha(path);assert digest==old['source_pins'][str(path)]
            st=path.stat();inputs[str(path)]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=digest)
    for seed,kind,epoch,name in CHECKPOINTS:
        path=D16/f'fits/H_full_seed{seed}'/name;st=path.stat()
        checkpoints.append(dict(seed=seed,kind=kind,epoch=epoch,path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(path)))
    tests=axis_tests();dump(ROOT/'AXIS_TESTS.json',dict(**tests,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,timer_scope='Whole preparation up to tests, not scalar-test-only'))
    protocol=dict(frozen=True,time=time.time(),source_pins=source_pins,inputs=inputs,checkpoints=checkpoints,metadata=metadata(),scope='Fixed original TRAIN48 effective only; no target loading, full TRAIN traversal, new VAL/test, model fitting or optimizer',centering='axis0 across48windows with every token-position/feature/timepoint coordinate fixed; aggregate centered squares over all coordinates',relative_denominator='sqrt((mean(near**2)+mean(older**2))/2); exact zero ->null, no epsilon',centered_relative_denominator='same pooled RMS after axis0 centering, zero ->null',probe='sum(residual*sin(arange(192,dtype=float32)*0.017))/48; once autograd.grad per existing forward, only history',NN_cap=384,forward_calls_cap=8,backward_cap=8,optimizer_updates=0,fits=0,new_encoder_head=0,model_mode='train(), no dropout or running statistics; all params requires_grad=False; history only requires_grad=True',numeric='D16 CUDA float32 architecture, matmul TF32=False / cuDNN TF32=True, anchor+0.1*res.double()',resources=dict(wall_seconds=900,RSS_bytes=8*1024**3,GPU_allocated_bytes=2*1024**3,root_bytes=8*1024**2,each_disk_free_bytes=20*1024**3,empty_GPU_before_launch=True),prepared_NN=0,prepared_backward=0,goal_complete=False)
    dump(ROOT/'PROTOCOL.json',protocol);dump(ROOT/'PREPARED.json',dict(passed=True,protocol_sha256=sha(ROOT/'PROTOCOL.json'),CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,NN=0,backward=0,fits=0,optimizer_updates=0,new_encoder_head=0,large_array_bytes_read=0));guard()
    print(json.dumps(dict(passed=True,protocol_sha256=sha(ROOT/'PROTOCOL.json'),NN=0)))
if __name__=='__main__':prepare()
