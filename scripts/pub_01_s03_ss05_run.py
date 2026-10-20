"""Two current-task GPU workers: all train -> gate -> cacheholdout -> one test."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss05_data import ROOT,dump


def main():
    cfg=json.loads((ROOT/'FORMAL_PROTOCOL.json').read_text());jobs=[(v,s) for v in cfg['variants'] for s in cfg['seeds']]
    dump(ROOT/'formal_started.json',{'pid':os.getpid(),'utc_epoch':time.time(),'runs':jobs,'holdout_access':False})
    status=ROOT/'formal_status.json'
    def worker(device,subset,stage):
        for variant,seed in subset:
            print(json.dumps({'stage':stage,'variant':variant,'seed':seed,'device':device,'utc_epoch':time.time()}),flush=True)
            cmd=[sys.executable,str(ROOT/'input'/f'pub_01_s03_ss05_{stage}.py'),'--variant',variant,'--seed',str(seed),'--device',device]
            with (ROOT/f'{stage}_{variant}_seed{seed}.log').open('x') as log:subprocess.run(cmd,cwd=ROOT/'input',env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        status.write_text(json.dumps({'stage':'TRAIN15','state':'RUNNING'}))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,f'cuda:{i}',jobs[i::2],'train') for i in range(2)]
            for future in futures:future.result()
        checks=[]
        for variant,seed in jobs:
            run=ROOT/'formal01'/f'{variant}_seed{seed}';train=json.loads((run/'train_completed.json').read_text());history=json.loads((run/'history.json').read_text())
            selected=max(history,key=lambda r:r['validation_record_then_subject_macro_60s_r'])
            checks.append(train['optimizer_steps']==cfg['optimizer_steps'] and train['best_step']==selected['step'] and not train['holdout_access'] and
                [r['step'] for r in history]==list(range(cfg['validation_every_steps'],cfg['optimizer_steps']+1,cfg['validation_every_steps'])))
        assert all(checks)
        dump(ROOT/'all_training_review.json',{'technical':'PASS','runs':len(checks),'same_budget_and_val_selection':True,'holdout_signal_access':False})
        status.write_text(json.dumps({'stage':'BUILD_HOLDOUT_AFTER_FREEZE_AND_TRAIN','state':'RUNNING'}))
        with (ROOT/'holdout_data01.log').open('x') as log:subprocess.run([sys.executable,str(ROOT/'input/pub_01_s03_ss05_data.py'),'--stage','holdout'],cwd=ROOT/'input',stdout=log,stderr=subprocess.STDOUT,check=True)
        dump(ROOT/'holdout_opened.json',{'utc_epoch':time.time(),'all15_training_complete':True,'fitting_after_this_point':'FORBIDDEN'})
        status.write_text(json.dumps({'stage':'HOLDOUT15_ONCE_WITH_ALL_FIXED_HISTORY_CONDITIONS','state':'RUNNING'}))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,f'cuda:{i}',jobs[i::2],'test') for i in range(2)]
            for future in futures:future.result()
        status.write_text(json.dumps({'stage':'ACTUAL_RESULTS_COMPLETE','state':'REVIEW_REQUIRED','utc_epoch':time.time()}))
    except Exception as e:
        status.write_text(json.dumps({'stage':'FAILED','error':str(e),'utc_epoch':time.time()}));raise


if __name__=='__main__':main()
