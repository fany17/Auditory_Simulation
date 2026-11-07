"""One bounded native-P FT or EMA phase; no retries and no test access."""
import argparse
import json
import os
import subprocess
import sys
import time
from pub_01_s03_ss07_baseline import ROOT,read,dump,update,sha


def main(kind,seed,device):
    assert kind in ('finetune','ema')
    phase=read(ROOT/('FINETUNE_PROTOCOL.json' if kind=='finetune' else 'EMA_PROTOCOL.json'))
    assert phase['frozen'] and phase['test_operations'] is False
    for path,digest in phase['sources'].items():assert sha(path)==digest
    key=f'finetune_seed{seed}' if kind=='finetune' else 'ema'
    dump(ROOT/f'{key}_queue_lease.json',{'pid':os.getpid(),'time':time.time(),'automatic_retries':False})
    required=[seed] if kind=='finetune' else read(ROOT/'INCREMENTAL_PROTOCOL.json')['screening_seeds']
    deadline=time.time()+7*86400
    while not all((ROOT/f'baseline/head_seed{s}/completed.json').exists() for s in required):
        if time.time()>deadline:raise RuntimeError('baseline full-head wait expired; no restart')
        for s in required:
            if 'Traceback (most recent call last)' in (ROOT/f'baseline_seed{s}.log').read_text():
                raise RuntimeError(f'baseline seed{s} failed')
        update(ROOT/f'{key}_queue_status.json',{'state':'WAIT_COMPLETE_NATIVE_HEAD','required_seeds':required,'time':time.time()})
        time.sleep(30)
    command=[sys.executable,str(ROOT/f'input/pub_01_s03_ss07_{kind}.py')]
    if kind=='finetune':command+=['--seed',str(seed),'--device',device]
    result=subprocess.run(command)
    assert result.returncode==0,f'{kind} failed; preserve evidence and request explicit review'
    if kind=='finetune':
        completed=read(ROOT/f'finetune/native_P_seed{seed}/completed.json')
        assert completed['seed']==seed and completed['test_read'] is False and completed['updates']==completed['epochs']*3945
        assert read(ROOT/f'finetune/native_P_seed{seed}/INCUMBENT_EQUIVALENCE.json')['passed']
    else:assert read(ROOT/'EMA_SELECTION.json')['test_read'] is False
    update(ROOT/f'{key}_queue_status.json',{'state':'COMPLETED_VALIDATED','time':time.time(),'test_read':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['finetune','ema']);p.add_argument('--seed',type=int,default=1024);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    try:main(a.kind,a.seed,a.device)
    except Exception as exc:
        update(ROOT/f'{a.kind}_{a.seed}_QUEUE_FAILURE.json',{'error':repr(exc),'time':time.time(),'explicit_review_required':True});raise
