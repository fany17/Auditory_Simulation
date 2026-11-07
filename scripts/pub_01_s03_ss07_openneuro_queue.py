"""Bounded orchestration: prepare -> final freeze -> coordinator review ->3fixed seeds.

No automatic retries, no new fits, and no real NN until the review binds both
the final protocol and this queue's exact source. Only our children are managed.
"""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
from pub_01_s03_ss07_openneuro_transport import ROOT

PYTHON='/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python'
EVALUATOR=ROOT/'input/pub_01_s03_ss07_openneuro_evaluate_v2.py'
STATISTICS=ROOT/'input/pub_01_s03_ss07_openneuro_statistics.py'
START=time.time()


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def event(**value):
    value={'time':time.time(),'elapsed':time.time()-START,**value}
    with (ROOT/'EVALUATION_QUEUE_EVENTS.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
    p=ROOT/'EVALUATION_QUEUE_STATUS.writing';p.write_text(json.dumps(value,indent=2));os.replace(p,ROOT/'EVALUATION_QUEUE_STATUS.json')
    print(json.dumps(value),flush=True)
def guard():assert time.time()-START<8*3600,'8h orchestration cap; no unbounded polling/retry'
def quiet_wait():guard();time.sleep(10)


def stage(script,args,name,cuda=''):
    env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES=cuda)
    with (ROOT/name).open('xb') as log:
        result=subprocess.run([PYTHON,str(script),*args],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
    assert result.returncode==0,f'{name} exit{result.returncode}; no automatic retry'


def run():
    assert not (ROOT/'EVALUATION_QUEUE_STARTED.json').exists()
    dump('EVALUATION_QUEUE_STARTED.json',dict(pid=os.getpid(),time=START,source_sha256=sha(__file__),
        target_seeds=[1024,2026,31415],new_model_fits=0,real_NN_started=False))
    event(stage='WAITING_FULL_PREPROCESSING')
    while not (ROOT/'EEG_PREPARE_COMPLETED_V2.json').exists():
        assert not (ROOT/'EEG_PREPARE_FAILURE_V2.json').exists(),'Input preparation failure requires inspection'
        quiet_wait()
    completed=read(ROOT/'EEG_PREPARE_COMPLETED_V2.json')
    assert completed['passed'] and completed['records']==380 and completed['failures']==[]
    assert not (ROOT/'EEG_PREPARE_FAILURE_V2.json').exists()
    event(stage='FINAL_DATA_MODEL_MASK_FREEZE_NO_NN')
    stage(EVALUATOR,['freeze'],'EVALUATION_FINAL_FREEZE.log')
    protocol_sha=sha(ROOT/'EVALUATION_PROTOCOL.json')
    event(stage='WAITING_COORDINATOR_FINAL_FREEZE_REVIEW',protocol_sha256=protocol_sha,queue_source_sha256=sha(__file__),real_NN_started=False)
    while True:
        p=ROOT/'COORDINATOR_EVALUATION_REVIEW.json'
        if p.exists():
            review=read(p)
            assert review['approved_for_execution'] and review['protocol_sha256']==protocol_sha
            assert review['queue_source_sha256']==sha(__file__)
            break
        quiet_wait()
    event(stage='STARTING_THREE_FROZEN_SEEDS',protocol_sha256=protocol_sha)
    pending=[1024,2026,31415];running={};finished=[];launches=[]
    while pending or running:
        guard()
        for device,job in list(running.items()):
            code=job['process'].poll()
            if code is None:continue
            job['log'].close();finished.append(dict(seed=job['seed'],device=device,pid=job['process'].pid,returncode=code))
            event(stage='SEED_EXITED',seed=job['seed'],device=device,returncode=code)
            del running[device]
        for device in ['cuda:0','cuda:1']:
            if not pending or device in running:continue
            query=subprocess.check_output(['nvidia-smi',f'--id={device[-1]}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(query.strip())<8192:continue
            seed=pending.pop(0);log=(ROOT/f'EVALUATION_seed{seed}.log').open('xb')
            env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            env.pop('CUDA_VISIBLE_DEVICES',None)
            process=subprocess.Popen([PYTHON,str(EVALUATOR),'run','--seed',str(seed),'--device',device],
                stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
            proc=Path(f'/proc/{process.pid}')
            launch=dict(seed=seed,device=device,pid=process.pid,UID=os.getuid(),
                start_ticks=(proc/'stat').read_text().split()[21],argv=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(),time=time.time())
            launches.append(launch)
            with (ROOT/'EVALUATION_WORKER_LAUNCHES.jsonl').open('a') as f:f.write(json.dumps(launch)+'\n')
            running[device]=dict(seed=seed,process=process,log=log)
            event(stage='SEED_LAUNCHED',**launch)
        if pending or running:quiet_wait()
    dump('EVALUATION_ALL_WORKERS_EXITED.json',dict(workers=finished,launches=launches,automatic_retries=0))
    assert len(finished)==3 and all(j['returncode']==0 for j in finished),'A worker failed; preserve it and inspect, no automatic retry'
    event(stage='PERSON_STATISTICS')
    stage(STATISTICS,['run'],'EXTERNAL_STATISTICS.log')
    dump('EVALUATION_QUEUE_COMPLETED.json',dict(wall_seconds=time.time()-START,protocol_sha256=protocol_sha,
        seed_workers=finished,new_model_fits=0,goal_complete=False,independent_result_review_still_required=True))
    event(stage='EVALUATION_COMPLETED_NEEDS_INDEPENDENT_RESULT_REVIEW',goal_complete=False)


if __name__=='__main__':
    try:run()
    except BaseException:
        if not (ROOT/'EVALUATION_QUEUE_FAILURE.json').exists():dump('EVALUATION_QUEUE_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False))
        raise
