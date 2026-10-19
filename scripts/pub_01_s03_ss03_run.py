"""Two GPU workers, all12 training→independent schedule review→locked tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss03_data import load_contract,dump,digest


def main(config):
    _,_,_,root=load_contract(config);protocol=json.loads((root/'formal_protocol.json').read_text());jobs=[(v,s) for v in protocol['variants'] for s in protocol['seeds']]
    status=root/'formal_job_status.json';dump(root/'formal_job_started.json',{'pid':os.getpid(),'utc_epoch':time.time(),'jobs':jobs,'test_started':False})
    def worker(device,jobs,stage):
        for variant,seed in jobs:
            command=[sys.executable,str(root/'input'/f'pub_01_s03_ss03_{stage}.py'),'--config',str(config.resolve()),'--variant',variant,'--seed',str(seed),'--device',device]
            record={'stage':stage,'variant':variant,'seed':seed,'device':device,'utc_epoch':time.time()}
            print(json.dumps(record),flush=True)
            with (root/f'{stage}_{variant}_seed{seed}.log').open('x') as f:
                subprocess.run(command,cwd=root/'input',env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        status.write_text(json.dumps({'stage':'TRAIN_ALL12','state':'RUNNING'}))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,f'cuda:{i}',jobs[i::2],'train') for i in range(2)]
            for future in futures:future.result()
        checks=[]
        for variant,seed in jobs:
            out=root/'formal01'/f'{variant}_seed{seed}';train=json.loads((out/'train_completed.json').read_text());history=json.loads((out/'history.json').read_text())
            expected=max(history,key=lambda r:r['validation_subject_macro_60s_r'])
            checks.append(train['optimizer_steps']==25000 and train['best_step']==expected['step'] and len(history)==10 and
                [r['step'] for r in history]==list(range(2500,25001,2500)) and not train['test_scores_computed'] and
                train['protocol_sha256']==digest(root/'formal_protocol.json') and not(out/'test01').exists())
        if not all(checks):raise RuntimeError('training-only independent schedule review failed')
        dump(root/'all_training_review.json',{'technical':'PASS','runs_checked':len(checks),'all_same_budget_and_selection':True,'test_access':False})
        dump(root/'test_opened.json',{'utc_epoch':time.time(),'all12_training_finished':True,'fitting_after_this_point':'FORBIDDEN'})
        status.write_text(json.dumps({'stage':'TEST_ALL12','state':'RUNNING'}))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,f'cuda:{i}',jobs[i::2],'test') for i in range(2)]
            for future in futures:future.result()
        status.write_text(json.dumps({'stage':'ACTUAL_RESULTS_COMPLETE','state':'INDEPENDENT_REVIEW_REQUIRED','utc_epoch':time.time()}))
    except Exception as error:
        status.write_text(json.dumps({'stage':'FAILED','error':str(error),'utc_epoch':time.time()}));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args().config)
