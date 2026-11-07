"""Approved nine fits, at most two idle GPUs, six-hour total wall cap."""
import os,subprocess,time,traceback
from pub_01_s03_ss07_full_ft_history_core import *
PYTHON='/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python'
def run():
    started=time.time();assert read(ROOT/'ENGINEERING_CHECK.json')['passed'];guard()
    dump(ROOT/'QUEUE_STARTED.json',dict(time=started,pid=os.getpid(),source_sha256=sha(__file__),total_fits=9,wall_cap_seconds=21600))
    pending=[(s,a) for s in SEEDS for a in ARMS];running={};finished=[]
    while pending or running:
        assert time.time()-started<21600,'6h total cap; preserve failures, no automatic budget extension'
        for device,job in list(running.items()):
            code=job['p'].poll()
            if code is None:continue
            job['log'].close();finished.append(dict(seed=job['seed'],arm=job['arm'],returncode=code,pid=job['p'].pid));del running[device]
            assert code==0,f'Failed fit {finished[-1]}; no auto retry or new fits after failure'
        for device in ['cuda:0','cuda:1']:
            if device in running or not pending:continue
            free=int(subprocess.check_output(['nvidia-smi',f'--id={device[-1]}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<4096:continue
            seed,arm=pending.pop(0);log=(ROOT/f'{arm}_seed{seed}.log').open('xb')
            env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4');env.pop('CUDA_VISIBLE_DEVICES',None)
            p=subprocess.Popen([PYTHON,str(ROOT/'input/pub_01_s03_ss07_full_ft_history_train.py'),'--seed',str(seed),'--arm',arm,'--device',device],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            proc=Path(f'/proc/{p.pid}');launch=dict(seed=seed,arm=arm,device=device,pid=p.pid,uid=os.getuid(),start_ticks=(proc/'stat').read_text().split()[21],time=time.time())
            with (ROOT/'WORKER_LAUNCHES.jsonl').open('a') as f:json.dump(launch,f);f.write('\n')
            running[device]=dict(seed=seed,arm=arm,p=p,log=log)
        status=dict(time=time.time(),pending=pending,running={d:dict(seed=j['seed'],arm=j['arm'],pid=j['p'].pid) for d,j in running.items()},finished=finished)
        temporary=ROOT/'QUEUE_STATUS.writing';temporary.write_text(json.dumps(status,indent=2));os.replace(temporary,ROOT/'QUEUE_STATUS.json')
        if pending or running:time.sleep(10)
    assert len(finished)==9
    dump(ROOT/'QUEUE_COMPLETED.json',dict(workers=finished,wall_seconds=time.time()-started,new_fits=9,total_updates=710100,goal_complete=False,independent_review_required=True))

if __name__=='__main__':
    try:run()
    except BaseException:
        dump(ROOT/'QUEUE_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc(),automatic_retry=False));raise
