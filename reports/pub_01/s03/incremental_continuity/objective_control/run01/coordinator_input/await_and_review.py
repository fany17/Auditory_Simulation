import os,time,json,hashlib,subprocess,traceback
from pathlib import Path
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_objective_control_20261102')
PIN='b80f9aeffada112c012ce23f944279027d4219250fdf2999a28c4878d336506f'
PY='/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261013/env01/bin/python'
SOURCES={'pub_01_s03_ss07_objective_control_independent_review.py':'36e9eebda70a7b60d8bb1cbc1ac27785cf36cd29fe403f260b2cd94d8731385b','pub_01_s03_ss07_full_ft_history_independent_review.py':'0113a30704b0fe2f13ae0e23aa53b2527afa29cfefc3c0c784789f384229cc64'}
def read(p):return json.loads(p.read_text())
def write(name,value):
 with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 start=time.time();cpu=time.process_time();deadline=read(ROOT/'QUEUE_STARTED.json')['time']+21600
 assert sha(ROOT/'PROTOCOL.json')==PIN
 for n,h in SOURCES.items():assert sha(ROOT/'coordinator_input'/n)==h
 assert not (ROOT/'coordinator_review').exists()
 while not (ROOT/'QUEUE_COMPLETED.json').exists():
  assert not (ROOT/'QUEUE_FAILURE.json').exists(),'Formal queue failed; do not run review or retry'
  p=Path('/proc/1769088');assert p.exists(),'Training queue exited without completed evidence'
  st=(p/'stat').read_text().split()
  assert st[2]!='Z' and st[21]=='755330725' and p.stat().st_uid==1012
  assert str(ROOT/'input/pub_01_s03_ss07_objective_control_queue.py').encode() in (p/'cmdline').read_bytes()
  assert time.time()<deadline,'Frozen queue deadline exceeded'
  time.sleep(10)
 q=read(ROOT/'QUEUE_COMPLETED.json');assert len(q['workers'])==9 and all(x['returncode']==0 for x in q['workers'])
 assert sha(ROOT/'PROTOCOL.json')==PIN
 for n,h in SOURCES.items():assert sha(ROOT/'coordinator_input'/n)==h
 write('COORDINATOR_REVIEW_RUN_STARTED.json',dict(time=time.time(),runner_pid=os.getpid(),protocol_sha256=PIN,source_pins=SOURCES,max_NN_windows=631239,optimizer_updates=0,fit=0))
 rc=subprocess.run([PY,str(ROOT/'coordinator_input/pub_01_s03_ss07_objective_control_independent_review.py'),'--protocol-sha',PIN],cwd=ROOT,check=False).returncode
 assert rc==0,('Independent reviewer failed',rc)
 result=read(ROOT/'coordinator_review/REVIEW.json');assert result['passed']
 write('COORDINATOR_REVIEW_RUN_COMPLETED.json',dict(time=time.time(),runner_pid=os.getpid(),returncode=rc,review_sha256=sha(ROOT/'coordinator_review/REVIEW.json'),review_passed=True,runner_only_CPU_seconds=time.process_time()-cpu,wait_and_child_wall_seconds=time.time()-start,child_cost_record='coordinator_review/REVIEW.json',extra_NN_beyond_child=0,goal_complete=False))
if __name__=='__main__':
 try:run()
 except BaseException:
  write('COORDINATOR_REVIEW_RUN_FAILURE.json',dict(time=time.time(),runner_pid=os.getpid(),traceback=traceback.format_exc(),automatic_retry=False));raise
