"""Bounded run02 completion monitor: review all fits, assess once, recompute.

Waits ONLY for this authorized run, never starts a new condition or budget.
Failure stops the chain; no retry of fitting or held assessment is automatic.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from pub_01_s03_ss06_matched import ROOT,SEEDS,guard_formal,read,dump,sha


def main():
    expected=[ROOT/f'formal01/{name}_seed{seed}/completed.json' for name in ('C','O','P_pretrain','P_head') for seed in SEEDS]
    deadline=time.time()+4*3600
    while not all(p.exists() for p in expected):
        if time.time()>deadline:raise RuntimeError('run02 completion wait expired; no assessment opened')
        for lane in ('C_seed1024','C_seed2026','C_seed31415','offline_reference_native'):
            lines=(ROOT/(lane+'.log')).read_text()
            if 'Traceback (most recent call last)' in lines:raise RuntimeError('training lane failed: '+lane)
        time.sleep(15)
    guard_formal()
    assert 'NATIVE03_PILOT_COMPLETED' in (ROOT/'native03_pilot.log').read_text()
    assert sha(ROOT/'input/pub_01_s03_ss06_matched_native_p.py')==read(ROOT/'P_NATIVE_AMENDMENT_02.json')['source_sha256']
    for seed in SEEDS:
        h=read(ROOT/f'formal01/P_head_seed{seed}/completed.json')
        assert h['actual_window_index_sequence']==list(range(130231))
        assert h['pretrain_checkpoint_sha256']==sha(ROOT/f'formal01/P_pretrain_seed{seed}/best_validation.pt')
    subprocess.run([sys.executable,str(ROOT/'input/pub_01_s03_ss06_matched_freeze.py'),'review'],check=True)
    assert read(ROOT/'ALL_TRAINING_REVIEW.json')['passed']
    dump(ROOT/'ASSESSMENT_OPENED.json',{'epoch':time.time(),'latest_completed_mtime':max(p.stat().st_mtime for p in expected),
        'protocol_sha256':sha(ROOT/'FORMAL_PROTOCOL.json'),'P_amendment_sha256':sha(ROOT/'P_NATIVE_AMENDMENT_02.json'),
        'gate_passed_before_opening':True,'assessment_count':1,'kind':'POST-HOC INTERNAL EXPLORATORY'})
    subprocess.run([sys.executable,str(ROOT/'input/pub_01_s03_ss06_matched.py'),'audit','--device','cuda:0'],check=True)
    subprocess.run([sys.executable,str(ROOT/'input/pub_01_s03_ss06_matched_review.py')],check=True)
    print('RUN02_ASSESSMENT_AND_REVIEW_COMPLETED',flush=True)

if __name__=='__main__':main()
