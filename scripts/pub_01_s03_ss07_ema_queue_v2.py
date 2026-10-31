"""Numerically aligned EMA phase only; no training or retries."""
import json
import os
import subprocess
import sys
import time
from pub_01_s03_ss07_baseline import ROOT,read,dump,update,sha


def main():
    p=read(ROOT/'EMA_PROTOCOL_V2.json');assert p['frozen'] and p['test_operations'] is False
    for path,digest in p['sources'].items():assert sha(path)==digest
    assert p['numerical_amendment_sha256']==sha(ROOT/'EMA_NUMERICAL_AMENDMENT.json')
    dump(ROOT/'EMA_V2_QUEUE_LEASE.json',{'pid':os.getpid(),'time':time.time(),'no_automatic_retries':True})
    seeds=read(ROOT/'INCREMENTAL_PROTOCOL.json')['screening_seeds'];deadline=time.time()+7*86400
    while not all((ROOT/f'baseline/head_seed{s}/completed.json').exists() for s in seeds):
        if time.time()>deadline:raise RuntimeError('native head wait expired; no restart')
        for s in seeds:
            if 'Traceback (most recent call last)' in (ROOT/f'baseline_seed{s}.log').read_text():raise RuntimeError('native baseline failed')
        update(ROOT/'EMA_V2_QUEUE_STATUS.json',{'stage':'WAIT_COMPLETE_NATIVE_HEAD','required_seeds':seeds,'time':time.time()})
        time.sleep(30)
    assert not (ROOT/'EMA_SELECTION.json').exists() and not (ROOT/'ema_validation_v2').exists()
    result=subprocess.run([sys.executable,str(ROOT/'input/pub_01_s03_ss07_ema_v2.py')])
    assert result.returncode==0
    selected=read(ROOT/'EMA_SELECTION.json')
    assert selected['test_read'] is False and selected['EMA_protocol_sha256']==sha(ROOT/'EMA_PROTOCOL_V2.json')
    assert selected['numerical_amendment_sha256']==sha(ROOT/'EMA_NUMERICAL_AMENDMENT.json')
    update(ROOT/'EMA_V2_QUEUE_STATUS.json',{'stage':'COMPLETED_VALIDATED','time':time.time(),'test_read':False})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update(ROOT/'EMA_V2_QUEUE_FAILURE.json',{'error':repr(exc),'time':time.time(),'explicit_review_required':True});raise
