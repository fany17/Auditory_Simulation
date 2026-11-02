"""Three seed workers, at most two verified idle GPUs, shared30min deadline."""
import os,subprocess,time,traceback
from pathlib import Path
from pub_01_s03_ss07_error_decomposition_core import *
PY='/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python'
RUNNING={}
def stop_own():
    out=[]
    for d,j in RUNNING.items():
        p=j['process'];proc=Path(f'/proc/{p.pid}')
        if p.poll() is None:
            assert proc.stat().st_uid==os.getuid() and (proc/'stat').read_text().split()[21]==j['start_ticks']
            assert 'pub_01_s03_ss07_error_decomposition_run.py' in (proc/'cmdline').read_bytes().decode().replace('\0',' ')
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                assert proc.stat().st_uid==os.getuid() and (proc/'stat').read_text().split()[21]==j['start_ticks'];p.kill();p.wait(timeout=10)
            out.append(dict(pid=p.pid,seed=j['seed'],returncode=p.returncode,identity_verified=True))
        j['log'].close()
    return out
def run():
    started=time.time();protocol=read(ROOT/'PROTOCOL.json');assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    dump(ROOT/'QUEUE_STARTED.json',dict(pid=os.getpid(),time=started,protocol_sha256=sha(ROOT/'PROTOCOL.json'),source_sha256=sha(__file__)))
    pending=list(SEEDS);done=[]
    while pending or RUNNING:
        guard()
        for device,j in list(RUNNING.items()):
            code=j['process'].poll()
            if code is None:continue
            j['log'].close();done.append(dict(seed=j['seed'],pid=j['process'].pid,returncode=code));del RUNNING[device]
            assert code==0,f'Worker failure, no retry: {done[-1]}'
        for device in ['cuda:0','cuda:1']:
            if device in RUNNING or not pending:continue
            free=int(subprocess.check_output(['nvidia-smi',f'--id={device[-1]}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            pids=subprocess.check_output(['nvidia-smi',f'--id={device[-1]}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
            if free<4096 or any(x.strip().isdigit() for x in pids.splitlines()):continue
            seed=pending.pop(0);log=(ROOT/f'seed{seed}.log').open('xb');env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4');env.pop('CUDA_VISIBLE_DEVICES',None)
            proc=subprocess.Popen([PY,str(ROOT/'input/pub_01_s03_ss07_error_decomposition_run.py'),'--seed',str(seed),'--device',device],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            ticks=Path(f'/proc/{proc.pid}/stat').read_text().split()[21]
            RUNNING[device]=dict(seed=seed,process=proc,log=log,start_ticks=ticks)
            with (ROOT/'WORKER_LAUNCHES.jsonl').open('a') as f:f.write(json.dumps(dict(seed=seed,device=device,pid=proc.pid,uid=os.getuid(),start_ticks=ticks,time=time.time(),prelaunch_compute_pids=[]))+'\n')
        status=dict(time=time.time(),pending=pending,running={d:dict(seed=j['seed'],pid=j['process'].pid) for d,j in RUNNING.items()},completed=done)
        t=ROOT/'STATUS.writing';t.write_text(json.dumps(status,indent=2));os.replace(t,ROOT/'STATUS.json')
        if pending or RUNNING:time.sleep(5)
    costs=[read(ROOT/f'results/seed{s}/COMPLETED.json') for s in SEEDS]
    assert sum(c['effective_residual_NN_windows'] for c in costs)==2508111
    dump(ROOT/'COMPLETED.json',dict(workers=done,effective_residual_NN_windows=2508111,new_encoder_head_windows=0,learning_model_fits=0,optimizer_updates=0,
        block_descriptive_projection_statistics=sum(c['block_descriptive_projection_statistics'] for c in costs),auxiliary_region_projection_statistics=sum(c['auxiliary_region_projection_statistics'] for c in costs),
        queue_wall_seconds=time.time()-started,CPU_worker_seconds=sum(c['CPU_seconds'] for c in costs),goal_complete=False,independent_review_required=True))
if __name__=='__main__':
    try:run()
    except BaseException:
        error=traceback.format_exc();stopped=stop_own();dump(ROOT/'QUEUE_FAILURE.json',dict(traceback=error,stopped_own_workers=stopped,automatic_retry=False));raise
