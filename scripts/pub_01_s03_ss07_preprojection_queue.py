"""D18 two-worker/one-per-GPU fixed12fit queue; no automatic retry."""
import signal,traceback
from pub_01_s03_ss07_preprojection_runtime import *
TRAIN=ROOT/'input/pub_01_s03_ss07_preprojection_train.py';RUNNING={}
def identity(pid,job):
    p=Path(f'/proc/{pid}')
    try:
        args=(p/'cmdline').read_bytes().split(b'\0');ticks=(p/'stat').read_text().split()[21]
        return p.stat().st_uid==os.getuid() and ticks==job['start_ticks'] and str(TRAIN).encode() in args and str(job['seed']).encode() in args and job['arm'].encode() in args
    except FileNotFoundError:return False
def stop_own():
    stopped=[]
    for pid,job in list(RUNNING.items()):
        if job['process'].poll() is not None:continue
        if identity(pid,job):os.killpg(pid,signal.SIGTERM);stopped.append(pid)
    deadline=time.time()+5
    while time.time()<deadline and any(j['process'].poll() is None for j in RUNNING.values()):time.sleep(.2)
    for pid,job in RUNNING.items():
        if job['process'].poll() is None and identity(pid,job):os.killpg(pid,signal.SIGKILL)
    return stopped
def checks():
    # Pure scheduler logic; no processes, GPU mutation or arrays.
    def allow(running,device,pids,free,mem):return device not in running and not pids and free>=36864 and mem>=64*1024**3
    assert allow(set(),'cuda:0',[],40000,128*1024**3)
    assert not allow({'cuda:0'},'cuda:0',[],40000,128*1024**3)
    assert not allow(set(),'cuda:0',[999],40000,128*1024**3)
    assert allow({'cuda:0'},'cuda:1',[],40000,128*1024**3)
    return dict(passed=True,new_NN=0,one_per_GPU_and_foreign_PID_wait=True)
def run():
    attempt_start=time.time();amendment=ROOT/'RESOURCE_RECOVERY_AMENDMENT.json'
    start=read(amendment)['original_queue_start_unix'] if amendment.exists() else attempt_start
    p=read(ROOT/'PROTOCOL.json');verify(p);guard();checks()
    approval=read(ROOT/'COORDINATOR_EXECUTION_REVIEW.json');assert approval['approved_for_execution'] and approval['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
    assert read(ROOT/'ENGINEERING_CHECK.json')['passed']
    pending=[(seed,arm) for seed in SEEDS for arm in ARMS];finished=[]
    dump(ROOT/'QUEUE_STARTED.json',dict(time=start,queue_attempt_start_unix=attempt_start,pid=os.getpid(),uid=os.getuid(),wall_cap_seconds=86400,max_workers=2,one_per_GPU=True,protocol_sha256=sha(ROOT/'PROTOCOL.json')))
    while pending or RUNNING:
        assert time.time()-start<86400,'24h total cap; no automatic extension or retry'
        for pid,job in list(RUNNING.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();finished.append(dict(seed=job['seed'],arm=job['arm'],pid=pid,returncode=code));del RUNNING[pid]
            assert code==0,f'Fit failed: {finished[-1]}'
        for device in ('cuda:0','cuda:1'):
            if not pending or any(x['device']==device for x in RUNNING.values()):continue
            resource=admission(device)
            if not resource['admitted']:continue
            guard();seed,arm=pending.pop(0);log=(ROOT/f'{arm}_seed{seed}.log').open('xb')
            env=dict(os.environ,OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4');env.pop('CUDA_VISIBLE_DEVICES',None)
            proc=subprocess.Popen([str(PYTHON),str(TRAIN),'--seed',str(seed),'--arm',arm,'--device',device],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            tick=Path(f'/proc/{proc.pid}/stat').read_text().split()[21];job=dict(seed=seed,arm=arm,device=device,start_ticks=tick,process=proc,log=log);RUNNING[proc.pid]=job;assert identity(proc.pid,job)
            with (ROOT/'WORKER_LAUNCHES.jsonl').open('a') as f:f.write(json.dumps(dict(seed=seed,arm=arm,device=device,pid=proc.pid,uid=os.getuid(),start_ticks=tick,time=time.time(),resource_admission=resource))+'\n')
        update(ROOT/'QUEUE_STATUS.json',dict(time=time.time(),elapsed_seconds=time.time()-start,pending=pending,running={str(pid):dict(seed=j['seed'],arm=j['arm'],device=j['device']) for pid,j in RUNNING.items()},finished=finished))
        if pending or RUNNING:time.sleep(5)
    done=[read(ROOT/f'fits/{arm}_seed{seed}/COMPLETED.json') for seed in SEEDS for arm in ARMS]
    assert len(done)==12 and sum(x['update_positions'] for x in done)==946800 and sum(x['optimizer_updates'] for x in done)==946800 and sum(x['target_window_presentations'] for x in done)==60588960
    totals={key:sum(x['component_forward_counts'][key] for x in done) for key in done[0]['component_forward_counts']}
    for key,value in totals.items():assert value<=p['budget'][key+'_upper'],(key,value,p['budget'][key+'_upper'])
    dump(ROOT/'QUEUE_COMPLETED.json',dict(workers=finished,wall_seconds=time.time()-start,CPU_worker_seconds=sum(x['CPU_seconds'] for x in done),formal_fits=12,total_update_positions=946800,total_optimizer_updates=946800,target_window_presentations=60588960,component_forward_counts=totals,goal_complete=False,independent_review_required=True))
if __name__=='__main__':
    try:run()
    except BaseException:
        error=traceback.format_exc();stopped=stop_own();dump(ROOT/'QUEUE_FAILURE.json',dict(time=time.time(),traceback=error,own_workers_stopped=stopped,automatic_retry=False));raise
