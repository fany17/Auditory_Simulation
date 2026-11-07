"""2203-only fixed audio preparation and descriptive, non-predictive overview."""
import csv
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pub_01_s03_ss07_openneuro_transport import ROOT
from pub_01_s03_ss07_openneuro_adapter import envelope64, GammatoneEnvelope, ResamplePoly


def dump(name, value):
    with (ROOT/name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    started=time.time()
    pre=json.loads((ROOT/'ACQUISITION_PREFLIGHT_V2_COMPLETED.json').read_text())
    assert len(pre['objects'])==21 and pre['failures']==[]
    assert not (ROOT/'ACQUISITION_PREFLIGHT_V2_FAILURE.json').exists()
    assert '"stage": "COMPLETED"' in (ROOT/'ACQUISITION_PREFLIGHT_V2.log').read_text()
    rows=sorted([r for r in pre['objects'] if r['kind']=='audio'], key=lambda r:r['path'])
    assert len(rows)==20
    dump('AUDIO_PREPARE_STARTED.json',dict(pid=os.getpid(),time=started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),NN_forwards=0))
    for name in ['targets','figures']:(ROOT/name).mkdir(exist_ok=True)
    sources=[]
    for cls in [GammatoneEnvelope,ResamplePoly]:
        p=Path(inspect.getfile(cls))
        sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    output=[]
    for row in rows:
        assert time.time()-started<4*3600
        assert shutil.disk_usage(ROOT).free>=20*1024**3
        p=ROOT/'source_audio'/Path(row['path']).name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        wave,rate=sf.read(p,dtype='float32',always_2d=True)
        assert np.isfinite(wave).all()
        difference=wave[:,0].astype('f8')-wave[:,1]
        mono=wave.mean(1,dtype=np.float32)
        env,details=envelope64(p)
        expected=(len(wave)*64+rate-1)//rate
        assert len(env)==expected
        target=ROOT/'targets'/p.with_suffix('.npy').name
        with target.open('xb') as f:np.save(f,env,allow_pickle=False)
        a=dict(path=row['path'],native_rate=rate,native_frames=len(wave),duration_seconds=len(wave)/rate,
            channels=wave.shape[1],identical_channels=bool(np.array_equal(wave[:,0],wave[:,1])),
            maximum_LR_difference=float(np.abs(difference).max()),
            different_sample_fraction=float(np.count_nonzero(difference)/len(wave)),
            LR_difference_RMS=float(np.sqrt(np.mean(difference**2))),
            left_RMS=float(np.sqrt(np.mean(wave[:,0].astype('f8')**2))),
            right_RMS=float(np.sqrt(np.mean(wave[:,1].astype('f8')**2))),
            mono_RMS=float(np.sqrt(np.mean(mono.astype('f8')**2))),
            target_samples64=len(env),target_range=[float(env.min()),float(env.max())],
            target_mean=float(env.mean(dtype='f8')),target_std=float(env.std(dtype='f8')),
            target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),**details)
        output.append(a)
        print(json.dumps(dict(stage='AUDIO_TARGET_READY',done=len(output),file=p.name,seconds=time.time()-started)),flush=True)
        if len(output)==1:
            fig,axes=plt.subplots(2,1,figsize=(10,5),layout='constrained')
            # Static examples only; no raw array embedded in local reports.
            axes[0].plot(np.arange(0,min(len(mono),10*rate),44)/rate,mono[:10*rate:44],lw=.5)
            axes[0].set(xlabel='Time (s)',ylabel='Mono amplitude',title='audio01: first 10 seconds, fixed arithmetic channel mean')
            axes[1].plot(np.arange(len(env))/64,env,lw=.6)
            axes[1].set(xlabel='Time (s)',ylabel='Envelope (sum, a.u.)',title='28-band gammatone |x|^0.6 sum; resample_poly 64 Hz')
            fig.savefig(ROOT/'figures/audio01_fixed_envelope.png',dpi=150);plt.close(fig)
    fields=[k for k in output[0] if not isinstance(output[0][k],(dict,list))]
    with (ROOT/'AUDIO_OVERVIEW.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(output)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    x=np.arange(1,21)
    axes[0].bar(x,[r['duration_seconds'] for r in output],color='#457b9d');axes[0].set(xlabel='Audio segment',ylabel='Duration (s)',ylim=(0,220))
    axes[1].bar(x,[r['different_sample_fraction'] for r in output],color='#e9c46a');axes[1].set(xlabel='Audio segment',ylabel='Fraction L != R',ylim=(0,1))
    fig.suptitle('One shared narrative split into 20 segments; not 20 independent stories')
    fig.savefig(ROOT/'figures/audio_structure.png',dpi=150);plt.close(fig)
    dump('AUDIO_PREPARE_COMPLETED.json',dict(passed=True,audios=output,source_files=sources,
        fixed_mono='librosa default arithmetic average; not selected by model score',
        wall_seconds=time.time()-started,CPU_seconds=time.process_time(),
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,NN_forwards=0,
        EEG_unit_status='UNRESOLVED; audio processing independent of EEG scale'))


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        dump('AUDIO_PREPARE_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
