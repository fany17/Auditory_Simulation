"""One-shot D18 JSON-only presentation gate; never launches/changes training."""
import hashlib,json,os,resource,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
PY=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python')
PRESENT=ROOT/'reporting/pub_01_s03_ss07_preprojection_present.py'
SEEDS=(1024,2026,31415);ARMS=('H_history','C_current','T_continue','P_prediction_history')
START=time.time();CPU=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False)
def run():
    protocol=read(ROOT/'POSTPROCESS_PROTOCOL.json');assert sha(PRESENT)==protocol['presentation_source_sha256']
    assert sha(ROOT/'PROTOCOL.json')==protocol['scientific_protocol_sha256']
    deadline=read(ROOT/'RESOURCE_RECOVERY_AMENDMENT.json')['absolute_deadline_unix']+3600
    dump(ROOT/'POSTPROCESS_WAIT_STARTED.json',dict(pid=os.getpid(),uid=os.getuid(),started_unix=START,wait_handle=str(ROOT/'QUEUE_COMPLETED.json'),presentation_source_sha256=sha(PRESENT),wait_deadline_unix=deadline,new_NN=0))
    while not (ROOT/'QUEUE_COMPLETED.json').exists():
        assert not (ROOT/'QUEUE_FAILURE.json').exists(),'Frozen training queue failed; preserve it; no postprocessing or retry'
        assert time.time()<deadline,'Bounded wait ended; never restart or extend training'
        time.sleep(30)
    assert not (ROOT/'report01').exists(),'No overwrite or automatic presentation retry'
    assert all((ROOT/f'fits/{arm}_seed{seed}/COMPLETED.json').exists() for seed in SEEDS for arm in ARMS)
    assert sha(PRESENT)==protocol['presentation_source_sha256']
    command=[str(PY),str(PRESENT)]
    with (ROOT/'POSTPROCESS.stdout.log').open('xb') as log:
        child=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=False)
    assert child.returncode==0,('Presentation failed; retain report01 and log; no automatic retry',child.returncode)
    assert read(ROOT/'report01/SUMMARY.json')['formal_complete_fits']==12
    dump(ROOT/'POSTPROCESS_COMPLETED.json',dict(passed=True,command=command,returncode=child.returncode,waiter_CPU_seconds=time.process_time()-CPU,waiter_wall_seconds=time.time()-START,peak_waiter_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,timer_scope='Waiter CPU excludes child; waiter wall includes waiting for training and presentation, so not additive to queue wall or presentation wall.',presentation_cost_receipt=str(ROOT/'report01/PRESENTATION_COST.json'),new_NN=0,new_backward=0,training_processes_launched_or_changed=0,scientific_interpretation_and_figure_viewing_still_required=True))
if __name__=='__main__':
    try:run()
    except BaseException:
        dump(ROOT/'POSTPROCESS_FAILURE.json',dict(traceback=traceback.format_exc(),new_NN=0,new_backward=0,automatic_retry=False,waiter_CPU_seconds=time.process_time()-CPU,waiter_wall_seconds=time.time()-START));raise
