"""Two-GPU bounded warm train/val queue; no retry, test or downstream job."""
import os
import subprocess
import sys
import time
from pub_01_s03_ss07_warm import ROOT,read,dump,update,sha,disk_guard

def main():
    protocol=read(ROOT/'WARM_PROTOCOL.json');assert protocol['frozen'] and not protocol['test_operations']
    assert read(ROOT/'INDEPENDENT_CODE_REVIEW.json')['decision']=='PASS_FOR_TRAIN_VAL'
    for path,digest in protocol['sources'].items():assert sha(path)==digest,path
    dump(ROOT/'QUEUE_LEASE.json',{'pid':os.getpid(),'time':time.time(),'automatic_retries':False,'protocol_sha256':sha(ROOT/'WARM_PROTOCOL.json')})
    cells=[(c,s) for c in protocol['new_fit_conditions'] for s in protocol['seeds']]
    assert len(cells)==protocol['new_fits']<=18
    live={};completed=[];counter=0
    while cells or live:
        for gpu,(process,condition,seed,log) in list(live.items()):
            rc=process.poll()
            if rc is None:continue
            log.close();del live[gpu]
            assert rc==0,f'{condition}/{seed} failed, preserve all evidence; no retry'
            result=read(ROOT/f'fits/{condition}_seed{seed}/completed.json')
            assert result['epochs']==5 and result['updates']==19725 and result['test_read'] is False
            completed.append({'condition':condition,'seed':seed})
        for gpu in (0,1):
            if gpu in live or not cells:continue
            disk_guard();condition,seed=cells.pop(0)
            path=ROOT/f'{condition}_seed{seed}.log';log=path.open('x')
            command=[sys.executable,str(ROOT/'input/pub_01_s03_ss07_warm.py'),'--condition',condition,'--seed',str(seed),'--device',f'cuda:{gpu}']
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            live[gpu]=(process,condition,seed,log);counter+=1
        update(ROOT/'QUEUE_STATUS.json',{'state':'RUNNING' if cells or live else 'TRAIN_VAL_COMPLETED','time':time.time(),
            'completed_cells':completed,'remaining_not_started':len(cells),'launched':counter,
            'live':[{'GPU':g,'pid':v[0].pid,'condition':v[1],'seed':v[2]} for g,v in live.items()],'test_read':False})
        if cells or live:time.sleep(20)
    assert len(completed)==protocol['new_fits']
    dump(ROOT/'TRAIN_VAL_COMPLETED.json',{'time':time.time(),'cells':completed,'new_fits':len(completed),
        'total_updates':len(completed)*19725,'supervised_windows':len(completed)*1262270,'test_read':False,
        'scientific_goal_achieved_claimed':False,'protocol_sha256':sha(ROOT/'WARM_PROTOCOL.json')})

if __name__=='__main__':
    try:main()
    except Exception as e:
        update(ROOT/'QUEUE_FAILURE.json',{'time':time.time(),'error':repr(e),'automatic_retries':False,'preserve_live_children':True})
        raise
