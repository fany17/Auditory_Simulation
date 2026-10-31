"""Bounded SS07 train/val continuation. No test operations or fit retries.

Existing screen drivers are the ONLY screen executors. This controller waits
for every designated cell, then launches the predeclared fresh full-budget
families and the deterministic final primary/control expansion.
"""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss07_baseline import ROOT,read,dump,update,sha,disk_guard
from pub_01_s03_ss07_select import collect,screen_selection,primary_selection


def verify_sources():
    p=read(ROOT/'FULL_QUEUE_V2_PROTOCOL.json')
    assert p['frozen'] and p['test_operations'] is False
    assert p['incremental_protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
    for path,value in p['sources'].items():assert sha(path)==value
    return p


def failure(condition,seed,reason):
    with (ROOT/'FULL_FAILURES.jsonl').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        f.write(json.dumps({'condition':condition,'seed':seed,'reason':reason,'time':time.time()})+'\n')
        f.flush();os.fsync(f.fileno())
    raise RuntimeError(f'full condition {condition}/{seed} failed: {reason}; explicit review required')


def run_cell(condition,seed,device):
    verify_sources();disk_guard();folder=ROOT/f'full/{condition}_seed{seed}'
    p=read(ROOT/'INCREMENTAL_PROTOCOL.json');config=next(c for c in p['full_conditions'] if c['id']==condition)
    if (folder/'completed.json').exists():
        # Only reuse EXACT completed full cells, never screen weights.
        _,means,missing,failed=collect('full',[config],[seed])
        assert not missing and not failed and len(means)==1
        return
    if folder.exists():failure(condition,seed,'existing incomplete directory; no automatic overwrite/retry')
    log=ROOT/f'full_{condition}_seed{seed}.log'
    print(json.dumps({'start_full':condition,'seed':seed,'device':device}),flush=True)
    with log.open('x') as output:
        result=subprocess.run([sys.executable,str(ROOT/'input/pub_01_s03_ss07_train_full_v2.py'),
            '--condition',condition,'--seed',str(seed),'--device',device,'--stage','full'],
            stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:failure(condition,seed,f'exit {result.returncode}; see {log.name}')
    _,means,missing,failed=collect('full',[config],[seed])
    if missing or failed or len(means)!=1:failure(condition,seed,'exit0 without valid completed output')
    assert read(folder/'completed.json')['engineering_amendment_sha256']==sha(ROOT/'FULL_ENGINEERING_AMENDMENT.json')
    print(json.dumps({'completed_full':condition,'seed':seed}),flush=True)


def seed_worker(seed,conditions,device):
    deadline=time.time()+7*86400
    while not (ROOT/f'baseline/head_seed{seed}/cache_manifest.json').exists():
        if time.time()>deadline:raise RuntimeError('cache wait expired; no upstream restart')
        if 'Traceback (most recent call last)' in (ROOT/f'baseline_seed{seed}.log').read_text():
            raise RuntimeError(f'upstream baseline {seed} failed')
        time.sleep(30)
    for condition in conditions:run_cell(condition,seed,device)


def run_workers(jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(seed_worker,*job) for job in jobs]
        # A failure preserves all other already-running independent cells.
        outcomes=[]
        for f in futures:
            try:f.result()
            except Exception as exc:outcomes.append(repr(exc))
        if outcomes:raise RuntimeError(json.dumps(outcomes))


def main():
    verify_sources();dump(ROOT/'FULL_QUEUE_V2_LEASE.json',{'pid':os.getpid(),'time':time.time(),'automatic_restarts':False})
    p=read(ROOT/'INCREMENTAL_PROTOCOL.json');deadline=time.time()+7*86400
    while True:
        _,_,missing,failed=collect('screen',p['conditions'],p['screening_seeds'])
        if failed:raise RuntimeError('screen failures require explicit review; full stage not started')
        if not missing:break
        pids=read(ROOT/'SCREEN_QUEUED.json')['pids'].values()
        alive=any(Path(f'/proc/{pid}/cmdline').exists() and
            b'screen_seed' in Path(f'/proc/{pid}/cmdline').read_bytes() for pid in pids)
        if not alive:raise RuntimeError('all screen drivers ended with missing outputs; explicit review required')
        if time.time()>deadline:raise RuntimeError('screen wait expired; no fit retries')
        update(ROOT/'FULL_QUEUE_V2_STATUS.json',{'stage':'WAIT_ALL54_SCREEN','missing_cells':len(missing),'time':time.time()})
        time.sleep(30)
    screen_selection();selection=read(ROOT/'FULL_SELECTION.json')
    devices={1024:'cuda:0',2026:'cuda:1',31415:'cuda:0',42:'cuda:1',12345:'cuda:1'}
    run_workers([(s,selection['selected_family_condition_ids'],devices[s]) for s in p['screening_seeds']])
    primary_selection();final=read(ROOT/'FULL_PRIMARY_SELECTION.json')
    initial=set(p['screening_seeds']);controls=final['same_full_budget_control_ids']
    jobs=[(s,([] if s in initial else [final['primary']])+controls,devices[s]) for s in final['final_seeds']]
    run_workers(jobs)
    catalog={c['id']:c for c in p['full_conditions']}
    ids=[final['primary']]+controls
    rows,means,missing,failed=collect('full',[catalog[i] for i in ids],final['final_seeds'])
    assert not missing and not failed and len(rows)==len(ids)*5
    dump(ROOT/'FULL_SELECTED_CONDITIONS_COMPLETED.json',{'time':time.time(),'conditions':ids,
        'paired_seeds':final['final_seeds'],'all_cells_validated':len(rows),'validation_means':means,
        'test_read':False,'strong_finetune_and_EMA_separate_pending':True,'research_goal_complete':False})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update(ROOT/'FULL_QUEUE_V2_STOPPED.json',{'error':repr(exc),'time':time.time(),
            'explicit_review_required':True,'no_automatic_retry':True})
        raise
