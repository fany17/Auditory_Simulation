"""Finish this authorized SS02 run sequentially; no new experiment selection."""
import json
from pathlib import Path
import subprocess
import sys
import time

work=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01')
run=work/'training_seed1024_attempt04'
status=work/'finalize_status.json'


def stage(name,script,config,extra=()):
    status.write_text(json.dumps({'stage':name,'status':'RUNNING','utc_epoch':time.time()}))
    with (work/(name+'.log')).open('x') as output:
        subprocess.run([sys.executable,str(work/'input'/script),'--config',str(work/'input'/config),*extra],
            stdout=output,stderr=subprocess.STDOUT,check=True,cwd=work/'input')


try:
    status.write_text(json.dumps({'stage':'WAIT_CURRENT_TRAINING','status':'RUNNING'}))
    deadline=time.time()+4*3600
    while not (run/'completed.json').exists():
        if time.time()>deadline:raise RuntimeError('bounded wait exceeded; preserve all outputs')
        if not Path('/proc/3109489').exists():raise RuntimeError('training process ended without completed result')
        time.sleep(10)
    stage('contrastive_review','pub_01_s03_review_transformer_result.py','pub_01_s03_transformer_available_20261015.json',('--run-name',run.name))
    stage('envelope_readout','pub_01_s03_transformer_envelope.py','pub_01_s03_transformer_envelope_20261015.json')
    stage('envelope_review','pub_01_s03_review_transformer_envelope.py','pub_01_s03_transformer_envelope_20261015.json')
    status.write_text(json.dumps({'stage':'ALL_COMPLETED','status':'PASS_WITH_LIMITATION','utc_epoch':time.time()}))
except Exception as error:
    status.write_text(json.dumps({'stage':'FAILED','status':'REVIEW_REQUIRED','error':str(error),'utc_epoch':time.time()}))
    raise
