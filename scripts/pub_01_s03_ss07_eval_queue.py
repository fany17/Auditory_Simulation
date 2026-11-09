"""Wait for fixed training cells, then lock/mask/evaluate once, no retries."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss07_baseline import ROOT,read,dump,update,sha


def training_ready():
    seeds=read(ROOT/'INCREMENTAL_PROTOCOL.json')['final_seeds']
    required=['FULL_SELECTED_CONDITIONS_COMPLETED.json','FULL_PRIMARY_SELECTION.json','EMA_SELECTION.json']
    required.extend(f'baseline/{stage}_seed{s}/completed.json' for s in seeds for stage in ['pretrain','head'])
    required.extend(f'finetune/native_P_seed{s}/completed.json' for s in seeds)
    return [path for path in required if not (ROOT/path).exists()]


def live_upstreams(missing):
    handles=[]
    full=read(ROOT/'FULL_QUEUE_V2_HANDOFF.json')['new_pid']
    if 'FULL_SELECTED_CONDITIONS_COMPLETED.json' in missing:
        p=Path(f'/proc/{full}/cmdline')
        assert p.exists() and b'pub_01_s03_ss07_full_queue_v2.py' in p.read_bytes(),'full controller ended without required outputs'
        handles.append(full)
    strong=read(ROOT/'STRONG_QUEUED.json')['jobs']
    for seed in read(ROOT/'INCREMENTAL_PROTOCOL.json')['final_seeds']:
        if f'finetune/native_P_seed{seed}/completed.json' not in missing:continue
        pid=strong[str(seed)];p=Path(f'/proc/{pid}/cmdline')
        assert p.exists() and b'pub_01_s03_ss07_strong_queue.py' in p.read_bytes(),'FT driver ended before completion'
        handles.append(pid)
    if 'EMA_SELECTION.json' in missing:
        pid=read(ROOT/'EMA_V2_HANDOFF_COMPLETED.json')['new_pid'];p=Path(f'/proc/{pid}/cmdline')
        assert p.exists() and b'pub_01_s03_ss07_ema_queue_v2.py' in p.read_bytes(),'EMA driver ended before required selection'
        handles.append(pid)
    return handles


def run_script(name,args=None,log=None):
    command=[sys.executable,str(ROOT/f'input/pub_01_s03_ss07_{name}.py')]+(args or [])
    if log is None:result=subprocess.run(command)
    else:
        with log.open('x') as output:result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
    return result.returncode


def worker(device,seeds):
    failures=[]
    for seed in seeds:
        exit_code=run_script('eval_seed',['--seed',str(seed),'--device',device],ROOT/f'final_eval_seed{seed}.log')
        if exit_code:failures.append({'seed':seed,'exit':exit_code,'no_retry':True})
        else:
            completed=read(ROOT/f'final_results/seed{seed}/COMPLETED.json')
            assert completed['seed']==seed and completed['all_conditions_complete']
    return failures


def main():
    protocol=read(ROOT/'EVALUATION_EXECUTOR_PROTOCOL.json');assert protocol['frozen']
    for path,digest in protocol['sources'].items():assert sha(path)==digest
    dump(ROOT/'FINAL_EVAL_QUEUE_LEASE.json',{'pid':os.getpid(),'time':time.time(),'no_automatic_retries':True})
    deadline=time.time()+14*86400
    while True:
        for name in ['SCREEN_FAILURES.jsonl','FULL_FAILURES.jsonl','FULL_QUEUE_V2_STOPPED.json','EMA_V2_QUEUE_FAILURE.json']:
            path=ROOT/name
            if path.exists() and path.stat().st_size:raise RuntimeError('upstream failure needs explicit review: '+name)
        missing=training_ready()
        if not missing:break
        handles=live_upstreams(missing)
        if time.time()>deadline:raise RuntimeError('upstream wait expired; no training restarted')
        update(ROOT/'FINAL_EVAL_QUEUE_STATUS.json',{'stage':'WAIT_ALL_FIXED_TRAINING','missing':missing,
            'actual_live_upstream_handles':handles,'time':time.time(),'test_read':False})
        time.sleep(30)
    assert run_script('final_manifest')==0
    assert run_script('final_val_bridge')==0
    assert run_script('eval_masks')==0
    from pub_01_s03_ss07_final_manifest import require_ready
    from pub_01_s03_ss07_eval_io import planned_array_budget
    manifest,_=require_ready();budget=planned_array_budget(manifest,read(ROOT/'SPARR_TEST_MASK.json')['records'])
    dump(ROOT/'FINAL_EVAL_ARRAY_BUDGET.json',budget)
    update(ROOT/'FINAL_EVAL_QUEUE_STATUS.json',{'stage':'LOCKED_AND_MASKED_EVALUATING','time':time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,'cuda:0',[1024,31415,12345]),pool.submit(worker,'cuda:1',[2026,42])]
        failures=[]
        for future in futures:failures.extend(future.result())
    if failures:
        dump(ROOT/'FINAL_EVAL_CONDITION_FAILURES.json',{'failures':failures,'explicit_review_required':True})
        raise RuntimeError('final conditions failed; never average only successful seeds')
    assert run_script('eval_aggregate')==0
    update(ROOT/'FINAL_EVAL_QUEUE_STATUS.json',{'stage':'EVALUATED_AGGREGATED_REVIEW_PENDING',
        'time':time.time(),'scientific_goal_complete':False})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update(ROOT/'FINAL_EVAL_QUEUE_FAILURE.json',{'error':repr(exc),'time':time.time(),
            'preserve_evidence_no_automatic_retries':True});raise
