"""Launch one durable2203 training job, with recorded PID and exclusive ticket."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main(config, script):
    cfg=json.loads(config.read_text())
    runroot=Path(cfg['run_dir']).parent
    if not str(runroot).startswith('/home/fanyu/auditory_simulation_m6a/'):
        raise RuntimeError('2203 project path required')
    if not (Path(cfg['prepared_dir'])/'completed.json').is_file():
        raise RuntimeError('prepared freeze incomplete')
    if Path(cfg['run_dir']).exists():
        raise FileExistsError('training output exists')
    with (runroot/'train_job.json').open('x') as ticket:
        logfile=runroot/'train.log'
        command=[sys.executable,str(script.resolve()),'--config',str(config.resolve())]
        with logfile.open('x') as output:
            process=subprocess.Popen(command,cwd=script.parent,env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'),
                stdout=output,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        job={'pid':process.pid,'started_utc':datetime.now(timezone.utc).isoformat(),
            'command':command,'config':str(config.resolve()),'log':str(logfile),'run_dir':cfg['run_dir']}
        json.dump(job,ticket,indent=2)
    print(json.dumps(job),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--script',type=Path,required=True)
    args=parser.parse_args();main(args.config,args.script)
