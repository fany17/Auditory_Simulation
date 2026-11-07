"""Authorized no-prediction acquisition:20 WAVs + one retained EEG unit canary,2203 only."""
import concurrent.futures as futures
import hashlib,io,json,os,shutil,time,urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102')


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def obtain(row,folder):
    path=folder/Path(row['path']).name
    if path.exists():
        data=path.read_bytes()
        assert len(data)==row['declared_bytes'] and hashlib.sha256(data).hexdigest()==row['payload_sha256']
        return data
    assert shutil.disk_usage(ROOT).free-row['declared_bytes']>=20*1024**3
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())+row['declared_bytes']<=3*1024**3
    request=urllib.request.Request(row['public_url'],headers={'User-Agent':'Auditory-Simulation-public-research/1.0'})
    with urllib.request.urlopen(request,timeout=120) as response:
        data=response.read(row['declared_bytes']+1)
    assert len(data)==row['declared_bytes']
    assert hashlib.sha256(data).hexdigest()==row['payload_sha256']
    with path.open('xb') as stream:stream.write(data)
    return data


def main():
    start=time.time()
    assert not (ROOT/'ACQUISITION_PREFLIGHT_STARTED.json').exists()
    dump(ROOT/'ACQUISITION_PREFLIGHT_STARTED.json',dict(pid=os.getpid(),time=start,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='20authorized audio bodies and one EEG canary; no model/target-stat fit'))
    objects=json.loads((ROOT/'ANNEX_CATALOG_V2.json').read_text())['objects']
    audios=[r for r in objects if r['path'].endswith('.wav')]
    assert len(audios)==20
    audiofolder=ROOT/'source_audio';audiofolder.mkdir(exist_ok=False)
    canary=ROOT/'canary_raw';canary.mkdir(exist_ok=False)
    rows=[]
    def audio(row):
        data=obtain(row,audiofolder)
        info=sf.info(io.BytesIO(data))
        return dict(path=row['path'],bytes=len(data),sha256=row['payload_sha256'],
            frames=info.frames,sampling_rate=info.samplerate,channels=info.channels,
            subtype=info.subtype,duration_seconds=info.frames/info.samplerate)
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={pool.submit(audio,row):row for row in audios}
        for f in futures.as_completed(pending):
            result=f.result();rows.append(result)
            print(json.dumps(dict(stage='audio_verified',done=len(rows),total=20,path=result['path'])),flush=True)
    eeg=next(r for r in objects if r['path']=='sub-001/eeg/sub-001_task-listening_run-01_eeg.eeg')
    data=obtain(eeg,canary)
    for ext in ('.vhdr','.vmrk'):
        name=Path(eeg['path']).with_suffix(ext).name
        with (canary/name).open('xb') as f:f.write((ROOT/'metadata/payload_headers'/name).read_bytes())
    values=np.frombuffer(data,dtype='<f4').reshape(-1,128)
    # Header resolution0.1microvolt for all128 checked independently below with MNE.
    expected_V=values[:1024].T.astype(np.float64)*1e-7
    import mne
    raw=mne.io.read_raw_brainvision(str(canary/Path(eeg['path']).with_suffix('.vhdr').name),preload=False,verbose='ERROR')
    decoded=raw.get_data(start=0,stop=1024)
    difference=float(abs(decoded-expected_V).max())
    assert difference<=1e-18
    assert raw.n_times==len(values) and raw.info['sfreq']==512 and len(raw.ch_names)==128
    physical=values.astype(np.float64)*.1
    assert np.isfinite(physical).all()
    # No normalization fitted: only compare physical-unit description to frozen source TRAIN.
    old=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01/prepared/train_only_normalization.npz')
    norm=np.load(old,allow_pickle=False)
    summary=dict(audio=sorted(rows,key=lambda r:r['path']),EEG_canary=dict(
        path=eeg['path'],bytes=len(data),sha256=eeg['payload_sha256'],frames=len(values),
        sampling_rate=512,channels=128,MNE_version=mne.__version__,
        MNE_vs_declared_float32_resolution_to_V_max_abs=difference,
        conversion_to_microvolt='.eeg stored float32 times0.1; not raw float times1e6',
        microvolt_range=[float(physical.min()),float(physical.max())],
        channel_std_microvolt_min_median_max=[float(physical.std(0).min()),float(np.median(physical.std(0))),float(physical.std(0).max())]),
        original_TRAIN_normalization=dict(path=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
          mean_shape=list(norm['mean'].shape),std_shape=list(norm['std'].shape),
          std_min_median_max=[float(norm['std'].min()),float(np.median(norm['std'])),float(norm['std'].max())],
          source_units='Published source LoadEEGNumpy unit_multiplier1e6; source derivatives/model norm interpreted in microvolt, no new-data rescaling or stats fit'),
        root_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        free_disk_bytes=shutil.disk_usage(ROOT).free,wall_seconds=time.time()-start,
        NN_forwards=0,new_fit=0,new_lag_search=False)
    dump(ROOT/'ACQUISITION_PREFLIGHT_COMPLETED.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if not (ROOT/'ACQUISITION_PREFLIGHT_FAILURE.json').exists():
            dump(ROOT/'ACQUISITION_PREFLIGHT_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
