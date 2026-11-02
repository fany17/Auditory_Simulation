"""Describe released canary scale without choosing a correction or calling models."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pub_01_s03_ss07_openneuro_transport import ROOT
from pub_01_s03_ss07_openneuro_adapter import SOURCE_NAMES,geometry,decode_microvolt,eeg64


def main():
    start=time.time()
    info=json.loads((ROOT/'HEADERS_AND_SIZES_V2.json').read_text())
    base='sub-001/eeg/sub-001_task-listening_run-01'
    header=next(r for r in info['headers'] if r['path']==base+'_eeg.vhdr')
    table=next(r for r in info['channel_table_mapping'] if r['path']==base+'_channels.tsv')
    rows=list(csv.DictReader((ROOT/'metadata/channel_tables'/f"{table['git_blob_sha']}.tsv").open(),delimiter='\t'))
    assert tuple(r['name'] for r in rows)==SOURCE_NAMES
    good=[r['name'] for r in rows if r['status']=='good']
    bad=[r['name'] for r in rows if r['status']=='bad']
    assert len(good)+len(bad)==128
    path=ROOT/'canary_raw'/Path(base+'_eeg.eeg').name
    payload=path.read_bytes()
    raw=decode_microvolt(payload,header,good)
    matrix,geom=geometry(good)
    processed=eeg64(payload,header,good,matrix)
    oldpath=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01/prepared/train_only_normalization.npz')
    norm=np.load(oldpath,allow_pickle=False)
    z=(processed.T.astype('f8')-norm['mean'])/norm['std']
    def stats(x):
        x=np.asarray(x,'f8');s=x.std(axis=1)
        return dict(shape=list(x.shape),minimum=float(x.min()),maximum=float(x.max()),
            channel_std_min_median_max=[float(s.min()),float(np.median(s)),float(s.max())])
    j=dict(raw_as_declared=stats(raw),processed_as_declared=stats(processed),
        old_TRAIN_std=[float(norm['std'].min()),float(np.median(norm['std'])),float(norm['std'].max())],
        normalized_input_absolute_quantiles=np.quantile(abs(z),[.5,.95,.99,1]).tolist(),
        excluded_only_curator_bad=bad,geometry=geom,source_path=str(path),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        interpretation='Header-declared microvolt scale; physical calibration unresolved. Reader equality is not unit ground truth.',
        decision='No correction inferred from amplitude, TRAIN statistics or target/model correlation; no NN.',
        NN_forwards=0,wall_seconds=time.time()-start)
    (ROOT/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(11,6.5),layout='constrained')
    axes[0,0].plot(np.arange(5120)/512,raw[0,:5120],lw=.6)
    axes[0,0].set(xlabel='Time (s)',ylabel='Header-declared microvolt',title='Canary A1: raw first 10 s; physical scale unresolved')
    axes[0,1].plot(np.arange(640)/64,processed[0,:640],lw=.6)
    axes[0,1].set(xlabel='Time (s)',ylabel='Header-declared microvolt',title='Fixed spline / high-pass / CAR / 64 Hz; no fitted scale')
    axes[1,0].semilogy(np.arange(64),processed.astype('f8').std(1),'o',ms=3,label='Canary processed')
    axes[1,0].semilogy(np.arange(64),norm['std'],'o',ms=3,label='Saved TRAIN std')
    axes[1,0].set(xlabel='Target64 channel index',ylabel='Std (declared microvolt)',title='Different datasets / preprocessing; not a scale estimator');axes[1,0].legend(fontsize=8)
    freq,power=signal.welch(processed.astype('f8'),fs=64,nperseg=256,axis=1)
    axes[1,1].semilogy(freq[1:],np.median(power,axis=0)[1:])
    axes[1,1].set(xlabel='Frequency (Hz)',ylabel='PSD in declared units squared / Hz',title='Descriptive processed spectrum; not a model score')
    fig.savefig(ROOT/'figures/EEG_unit_diagnostic.png',dpi=150);plt.close(fig)
    with (ROOT/'EEG_UNIT_DIAGNOSTIC.json').open('x') as f:json.dump(j,f,indent=2,allow_nan=False)
    print(json.dumps(j),flush=True)


if __name__=='__main__':main()
