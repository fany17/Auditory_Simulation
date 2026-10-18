"""Durable sequential feature→prepare→train job with explicit stage status."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main(config,start,attempt,train_script,run_name):
    cfg=json.loads(config.read_text());work=Path(cfg['work'])
    if not str(work).startswith('/home/fanyu/auditory_simulation_m6a/') or cfg['patient_data']!='FORBIDDEN':
        raise RuntimeError('2203 public-only')
    stages=['features','prepare','train'];history=[]
    for stage in stages[stages.index(start):]:
        if stage=='train':
            command=[sys.executable,str(work/'input'/train_script),'--config',str(config),'--run-name',run_name]
        else:
            command=[sys.executable,str(work/'input/pub_01_s03_transformer_inputs.py'),'--config',str(config),'--stage',stage]
        record={'stage':stage,'started_utc':datetime.now(timezone.utc).isoformat(),'command':command}
        with (work/(stage+'_attempt'+attempt+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=work/'input',env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
            record.update(pid=child.pid,state='RUNNING');(work/'job_status.json').write_text(json.dumps(record,indent=2))
            code=child.wait()
        record.update(exit_code=code,state='COMPLETED' if code==0 else 'FAILED',ended_utc=datetime.now(timezone.utc).isoformat());history.append(record)
        (work/('job_history_attempt'+attempt+'.json')).write_text(json.dumps(history,indent=2));(work/'job_status.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(record),flush=True)
        if code:raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--start',choices=['features','prepare','train'],default='features');parser.add_argument('--attempt',default='01');parser.add_argument('--train-script',default='pub_01_s03_transformer_train.py');parser.add_argument('--run-name',default='training_seed1024');args=parser.parse_args();main(args.config,args.start,args.attempt,args.train_script,args.run_name)
