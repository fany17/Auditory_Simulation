"""Persistent SS06 formal scheduler; stops before opening any audit payload."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss06_data import ROOT,dump,read
from pub_01_s03_ss06_contract import verify_all_training,identities


def call(args,name):
    with (ROOT/(name+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-u',*args],cwd=ROOT/'input',stdout=log,stderr=subprocess.STDOUT,check=True)


def gpu_worker(index,jobs,protocol):
    for name,seed in jobs:
        args=['pub_01_s03_ss06_train.py','--variant',name,'--seed',str(seed),'--device',f'cuda:{index}',
              '--steps',str(protocol['optimizer_steps'])]
        if name.startswith('m3_'):args+=['--pretrained',str(ROOT/'formal01'/f'pretrain_seed{seed}'/'pretrained.pt')]
        call(args,f'formal_{name}_seed{seed}')


def main():
    protocol=read(ROOT/'FORMAL_PROTOCOL.json');dump(ROOT/'formal_started.json',{'epoch':time.time(),'pid':os.getpid(),'audit_read':False})
    cpu_jobs=[(name,seed) for name in ('m5_prior','m5_full') for seed in protocol['seeds']]
    gpu_jobs=[(name,seed) for name in protocol['variants'] if name not in ('m5_prior','m5_full') for seed in protocol['seeds']]
    # Independent one-thread CPU jobs overlap GPUs; explicit device cost reported.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending=[pool.submit(call,['pub_01_s03_ss06_train.py','--variant',name,'--seed',str(seed),'--device','cpu',
                    '--steps',str(protocol['optimizer_steps'])],f'formal_{name}_seed{seed}') for name,seed in cpu_jobs]
        for seed in protocol['seeds']:
            call(['pub_01_s03_ss06_train.py','--pretrain','--seed',str(seed),'--device','cuda:0','--steps',str(protocol['pretraining_steps'])],f'formal_pretrain_seed{seed}')
        pending += [pool.submit(gpu_worker,index,gpu_jobs[index::2],protocol) for index in (0,1)]
        for job in concurrent.futures.as_completed(pending):job.result()
    gate=verify_all_training();dump(ROOT/'all_training_complete.json',gate)
    # A separate pass reloads every selected checkpoint and validates real
    # maximum-validation choice and common complete record/block coverage.
    checked=verify_all_training();assert checked==gate
    dump(ROOT/'all_training_review.json',{'technical':'PASS','checkpoint_contents_checked':len(gate['checked_runs']),
        'scope':'same-agent separate pre-audit review; coordinator additionally reviewed source/metadata; not third-party scientific signoff',**identities()})
    call(['pub_01_s03_ss06_evaluate.py','--select-output-controls'],'validation_output_controls')
    dump(ROOT/'ready_for_audit.json',{'technical':'PASS','all_training_complete':True,'audit_read':False,'epoch':time.time()})
    print(json.dumps({'stage':'ALL_TRAINING_COMPLETE_AUDIT_STILL_CLOSED','runs':len(gate['checked_runs'])}),flush=True)


if __name__=='__main__':main()
