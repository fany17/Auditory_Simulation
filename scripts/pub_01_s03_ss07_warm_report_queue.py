"""Bounded wait for the verified train/val controller, then report once.

Does not touch the training queue or fit/choose any neural model.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from pub_01_s03_ss07_warm import ROOT,read,dump,update,sha

def main():
    protocol=read(ROOT/'REPORT_PROTOCOL.json')
    assert protocol['sources'][str(Path(__file__))]==sha(Path(__file__))
    for p,d in protocol['sources'].items():assert sha(p)==d,p
    lease=read(ROOT/'QUEUE_LEASE.json');start=read(ROOT/'START_VERIFIED.json')
    assert lease['pid']==start['controller']['pid']==416910
    dump(ROOT/'REPORT_QUEUE_LEASE.json',{'pid':os.getpid(),'upstream_pid':lease['pid'],'time':time.time(),'no_retries':True})
    deadline=time.time()+2*86400
    while not (ROOT/'TRAIN_VAL_COMPLETED.json').exists():
        assert time.time()<deadline,'wait deadline; preserve all runs and do not restart'
        assert not (ROOT/'QUEUE_FAILURE.json').exists(),'training failed; do not report incomplete matrix'
        path=Path(f'/proc/{lease["pid"]}')
        assert path.exists(),'controller handle gone without complete marker; inspect, do not restart'
        argv=path.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
        assert argv[1]==str(ROOT/'input/pub_01_s03_ss07_warm_queue.py')
        stat=path.joinpath('stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19])
        assert ticks==start['controller']['start_ticks'],'PID was reused'
        update(ROOT/'REPORT_QUEUE_STATUS.json',{'state':'WAIT_ALL18_TRAIN_VAL','time':time.time(),'upstream_verified_live':lease['pid'],'test_read':False})
        time.sleep(20)
    for p,d in protocol['sources'].items():assert sha(p)==d,p
    result=subprocess.run([sys.executable,str(ROOT/'report_input/pub_01_s03_ss07_warm_report.py')])
    assert result.returncode==0,'report failed, preserve inputs and output; no automatic retry'
    assert read(ROOT/'results01/COMPLETED.json')['all18_reported']
    update(ROOT/'REPORT_QUEUE_STATUS.json',{'state':'COMPLETE_ALL18_REPORTED','time':time.time(),'test_read':False,'scientific_goal_achieved_claimed':False})

if __name__=='__main__':
    try:main()
    except Exception as e:
        update(ROOT/'REPORT_QUEUE_FAILURE.json',{'time':time.time(),'error':repr(e),'automatic_retries':False});raise
