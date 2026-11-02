"""D16: up to nine own jobs (5/4 GPUs), bounded resources and safe identity checks."""
import subprocess,traceback
from pub_01_s03_ss07_token_norm_core import *
PYTHON='/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python'
TRAIN=ROOT/'input/pub_01_s03_ss07_token_norm_train.py'
CAP={'cuda:0':5,'cuda:1':4};RUNNING={}
def identity_fields(uid,ticks,argv,expected_uid,job):
    try:return uid==expected_uid and ticks==job['start_ticks'] and argv[1]==str(TRAIN) and argv[argv.index('--seed')+1]==str(job['seed']) and argv[argv.index('--arm')+1]==job['arm'] and argv[argv.index('--device')+1]==job['device']
    except (ValueError,IndexError):return False
def identity(pid,job):
    proc=Path(f'/proc/{pid}')
    try:return identity_fields(proc.stat().st_uid,(proc/'stat').read_text().split()[21],(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),os.getuid(),job)
    except FileNotFoundError:return False
def admission(device,running,observed,free_MB,mem_available,verify):
    return (sum(j['device']==device for j in running.values())<CAP[device] and free_MB>=4096 and mem_available>=16*1024**3
        and all(pid in running and running[pid]['device']==device and verify(pid,running[pid]) for pid in observed))
def scheduler_checks():
    jobs={};seed=1024
    for device in CAP:
        for slot in range(CAP[device]):
            observed=[pid for pid,j in jobs.items() if j['device']==device]
            assert admission(device,jobs,observed,46000,1024**4,lambda p,j:True)
            jobs[len(jobs)+100]=dict(device=device,seed=seed,arm='H_full',start_ticks='123')
        assert not admission(device,jobs,[],46000,1024**4,lambda p,j:True)
    assert len(jobs)==9
    empty={};one={100:dict(device='cuda:0')}
    assert not admission('cuda:0',empty,[999],46000,1024**4,lambda p,j:True)
    assert not admission('cuda:0',one,[100],46000,1024**4,lambda p,j:False)
    assert not admission('cuda:1',one,[100],46000,1024**4,lambda p,j:True)
    assert not admission('cuda:0',empty,[],3000,1024**4,lambda p,j:True)
    assert not admission('cuda:0',empty,[],46000,8*1024**3,lambda p,j:True)
    j=dict(seed=1024,arm='H_full',device='cuda:0',start_ticks='123');args=[PYTHON,str(TRAIN),'--seed','1024','--arm','H_full','--device','cuda:0']
    assert identity_fields(1012,'123',args,1012,j)
    assert not identity_fields(1,'123',args,1012,j) and not identity_fields(1012,'124',args,1012,j)
    wrong=list(args);wrong[3]='2026';assert not identity_fields(1012,'123',wrong,1012,j)
    wrong=list(args);wrong[1]='foreign.py';assert not identity_fields(1012,'123',wrong,1012,j)
    return dict(passed=True,no_training=True,no_NN=True,nine_slots_5plus4=True,foreign_PID_rejected=True,UID_startticks_argv_mismatch_rejected=True,low_memory_or_VRAM_wait=True,no_global_GPU_setting_changes=True)
def stop_own():
    stopped=[]
    for pid,j in list(RUNNING.items()):
        p=j['p']
        if p.poll() is None:
            assert identity(pid,j),'Refuse to terminate process with changed identity'
            p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:assert identity(pid,j);p.kill();p.wait(timeout=10)
            stopped.append(dict(pid=pid,seed=j['seed'],arm=j['arm'],identity_verified=True,returncode=p.returncode))
        j['log'].close()
    return stopped
def resources(device):
    gpu=device[-1]
    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    output=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    observed=[int(x.strip()) for x in output.splitlines() if x.strip().isdigit()]
    fields={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    available=int(fields['MemAvailable'].split()[0])*1024
    return free,observed,available
def run():
    start=time.time();guard();p=read(ROOT/'PROTOCOL.json');review=read(ROOT/'COORDINATOR_EXECUTION_REVIEW.json')
    assert review['approved_for_execution'] and review['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
    assert read(ROOT/'ENGINEERING_CHECK.json')['passed'] and p['max_per_GPU']==CAP
    dump(ROOT/'QUEUE_STARTED.json',dict(time=start,pid=os.getpid(),source_sha256=sha(__file__),max_jobs=9,max_per_GPU=CAP,wall_cap_seconds=21600))
    pending=[(s,a) for s in SEEDS for a in ARMS];done=[]
    while pending or RUNNING:
        assert time.time()-start<21600,'Total6h deadline; no automatic retry/extension'
        for pid,j in list(RUNNING.items()):
            code=j['p'].poll()
            if code is None:continue
            j['log'].close();done.append(dict(seed=j['seed'],arm=j['arm'],pid=pid,returncode=code));del RUNNING[pid]
            assert code==0,f'Failed fit {done[-1]}'
        for device in CAP:
            while pending:
                free,observed,available=resources(device)
                if not admission(device,RUNNING,observed,free,available,identity):break
                guard();seed,arm=pending.pop(0);log=(ROOT/f'{arm}_seed{seed}.log').open('xb')
                env=dict(os.environ,OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4');env.pop('CUDA_VISIBLE_DEVICES',None)
                child=subprocess.Popen([PYTHON,str(TRAIN),'--seed',str(seed),'--arm',arm,'--device',device],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                proc=Path(f'/proc/{child.pid}');job=dict(seed=seed,arm=arm,device=device,start_ticks=(proc/'stat').read_text().split()[21],p=child,log=log)
                RUNNING[child.pid]=job;assert identity(child.pid,job)
                launch=dict(seed=seed,arm=arm,device=device,pid=child.pid,uid=os.getuid(),start_ticks=job['start_ticks'],time=time.time(),prelaunch_compute_pids=observed,prelaunch_free_GPU_MiB=free,prelaunch_MemAvailable_bytes=available,observed_processes_all_verified_own=True)
                with (ROOT/'WORKER_LAUNCHES.jsonl').open('a') as f:f.write(json.dumps(launch)+'\n')
        status=dict(time=time.time(),pending=pending,running={str(pid):dict(seed=j['seed'],arm=j['arm'],device=j['device']) for pid,j in RUNNING.items()},finished=done)
        tmp=ROOT/'QUEUE_STATUS.writing';tmp.write_text(json.dumps(status,indent=2));os.replace(tmp,ROOT/'QUEUE_STATUS.json')
        if pending or RUNNING:time.sleep(5)
    costs=[read(ROOT/f'fits/{a}_seed{s}/COMPLETED.json') for s in SEEDS for a in ARMS]
    assert len(done)==9 and sum(x['updates'] for x in costs)==710100 and sum(x['new_residual_NN_window_forwards'] for x in costs)==59473248
    dump(ROOT/'QUEUE_COMPLETED.json',dict(workers=done,wall_seconds=time.time()-start,CPU_worker_seconds=sum(x['CPU_seconds'] for x in costs),new_fits=9,total_updates=710100,formal_residual_NN_windows=59473248,new_encoder_head=0,max_per_GPU=CAP,goal_complete=False,independent_review_required=True))
if __name__=='__main__':
    try:run()
    except BaseException:
        failure=traceback.format_exc();stopped=stop_own();dump(ROOT/'QUEUE_FAILURE.json',dict(time=time.time(),traceback=failure,own_workers_stopped=stopped,automatic_retry=False));raise
