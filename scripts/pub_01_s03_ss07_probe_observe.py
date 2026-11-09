"""Post-launch read-only supplement; does not change the frozen D4 process."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_conditional_history_probe_20261101')
OLD=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01')
def read(p):return json.loads(p.read_text())
def dump(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2)
protocol=read(ROOT/'PROTOCOL.json');launch=read(ROOT/'LAUNCH.json');pid=launch['pid'];proc=Path(f'/proc/{pid}')
assert proc.stat().st_uid==os.getuid()
assert proc.joinpath('cmdline').read_bytes().decode().replace(chr(0),' ')==launch['proc_cmdline']
env=dict(v.split('=',1) for v in proc.joinpath('environ').read_bytes().decode().split(chr(0)) if '=' in v)
threads={k:env.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
assert all(v is not None and 0<int(v)<=8 for v in threads.values())
inputs={};lengths=[]
for phase,rr in protocol['records'].items():
    cache=OLD/f'envelope_readout_seed1024/{phase}_envelope.npy';x=np.load(cache,mmap_mode='r')
    assert x.shape==({'train':252454,'val':30500}[phase],1,192)
    for p in [cache]+[Path(r['envelope_path']) for r in rr]:
        if str(p) in inputs:continue
        v=np.load(p,mmap_mode='r');s=p.stat()
        inputs[str(p)]={'shape':list(v.shape),'dtype':str(v.dtype),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
    for r in rr:
        shape=inputs[r['envelope_path']]['shape'];n=int(np.prod(shape));assert n>=r['common_samples']
        lengths.append(dict(phase=phase,record=r['record'],target_samples=n,common_samples=r['common_samples'],windows=r['windows']))
dump(ROOT/'POST_LAUNCH_INPUT_RESOURCE_SUPPLEMENT.json',dict(time=time.time(),timing='POST_LAUNCH, not a pre-run freeze; seed1024 first Gram was in progress',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),formal_pid=pid,formal_source_sha=launch['source_sha256'],thread_environment=threads,
    target_inputs=inputs,record_length_checks=lengths,changes_to_formal_code_protocol_scores=False,
    cost_limitations='Per-lambda main times combine coefficient formation and prediction/score, exclude shared eigendecomposition; those subphases not separately measured. Process CPU sampled externally. No replay to invent missing detail.'))
fields=['time','elapsed_from_launch','stage','seed','arm','cpu_seconds','rss_bytes','process_threads','GPU_MiB','finished_solution_files']
observations=[];startticks=None;laststate=None;lastcpu=0.
with (ROOT/'RESOURCE_OBSERVATIONS.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    while proc.exists():
        try:
            stat=proc.joinpath('stat').read_text().split();state=stat[2]
            if state=='Z':break
            tick=stat[21]
            if startticks is None:startticks=tick
            assert tick==startticks,'PID reused'
            lastcpu=(int(stat[13])+int(stat[14]))/os.sysconf('SC_CLK_TCK')
            s=read(ROOT/'STATUS.json');v=dict(time=time.time(),elapsed_from_launch=time.time()-launch['time'],stage=s.get('stage'),seed=s.get('seed'),arm=s.get('arm'),
                cpu_seconds=lastcpu,rss_bytes=int(stat[23])*os.sysconf('SC_PAGE_SIZE'),process_threads=int(stat[19]),GPU_MiB=None,
                finished_solution_files=len(list(ROOT.glob('seed*/*_METRICS.json'))))
            gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
            values=[int(line.split(',')[1].strip()) for line in gpu.splitlines() if line.split(',')[0].strip()==str(pid)]
            if values:v['GPU_MiB']=max(values)
            writer.writerow(v);f.flush();observations.append(v)
            key=(v['stage'],v['seed'],v['arm'])
            if key!=laststate:print(json.dumps(v),flush=True);laststate=key
            if (ROOT/'COMPLETED.json').exists() or (ROOT/'FAILURE.json').exists():break
            time.sleep(5)
        except FileNotFoundError:break
for p,v in inputs.items():
    st=Path(p).stat();assert st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],p
dump(ROOT/'RESOURCE_OBSERVATION_COMPLETED.json',dict(time=time.time(),samples=len(observations),last_observed_process_CPU_seconds=lastcpu,
    sample_peak_RSS_bytes=max((r['rss_bytes'] for r in observations),default=0),sample_peak_GPU_MiB=max((r['GPU_MiB'] or 0 for r in observations),default=0),
    observed_GPU_within_1GiB=all(r['GPU_MiB'] is None or r['GPU_MiB']<=1024 for r in observations),target_inputs_unchanged_since_supplement=True,
    limitations='Sampling begins after launch; not a complete continuous resource trace. CPU final sample is a lower bound when process exited between samples.',formal_completed=(ROOT/'COMPLETED.json').exists()))
