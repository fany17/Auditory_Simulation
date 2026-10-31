"""Wait for the substantive formal gate, then assess once and review outputs."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pub_01_s03_ss06_data import ROOT,read,dump
from pub_01_s03_ss06_contract import verify_all_training,digest


def call(args,name):
    with (ROOT/(name+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT/'input',stdout=log,stderr=subprocess.STDOUT,check=True)


def worker(device,jobs):
    for variant,seed in jobs:
        call(['pub_01_s03_ss06_evaluate.py','--variant',variant,'--seed',str(seed),'--device',device],f'audit_{variant}_seed{seed}')


def main():
    dump(ROOT/'audit_waiter_started.json',{'pid':os.getpid(),'epoch':time.time(),'state':'WAIT_ALL_TRAINING_GATE','audit_read':False})
    while not (ROOT/'ready_for_audit.json').exists():
        if 'Traceback' in (ROOT/'formal_scheduler.log').read_text():raise RuntimeError('formal scheduler failure; audit remains closed')
        time.sleep(30)
    gate=verify_all_training();assert read(ROOT/'all_training_complete.json')==gate
    assert read(ROOT/'review_method_validation.json')['technical']=='PASS'
    dump(ROOT/'observer_sources.json',{name:digest(ROOT/'input'/name) for name in
        ('pub_01_s03_ss06_audit.py','pub_01_s03_ss06_review.py','pub_01_s03_ss06_review_method.py')})
    call(['pub_01_s03_ss06_data.py','audit'],'audit_data')
    protocol=read(ROOT/'FORMAL_PROTOCOL.json')
    gpu=[(v,s) for v in protocol['variants'] if v not in ('m5_full','m5_prior') for s in protocol['seeds']]
    cpu=[(v,s) for v in ('m5_full','m5_prior') for s in protocol['seeds']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs=[pool.submit(worker,'cpu',[item]) for item in cpu]
        jobs += [pool.submit(worker,f'cuda:{i}',gpu[i::2]) for i in (0,1)]
        for job in concurrent.futures.as_completed(jobs):job.result()
    dump(ROOT/'audit_assessments_complete.json',{'epoch':time.time(),'actual_assessments':39,'all_frozen_conditions_complete':True})
    call(['pub_01_s03_ss06_review.py'],'postexecution_review')
    print(json.dumps({'stage':'ACTUAL_AUDIT_AND_POST_REVIEW_COMPLETE','epoch':time.time()}),flush=True)


if __name__=='__main__':main()
