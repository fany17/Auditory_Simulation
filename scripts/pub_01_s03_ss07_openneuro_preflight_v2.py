"""Preflight attempt02: pinned Range download; reuse verified files, concurrent EEG unit canary."""
import concurrent.futures as futures
import hashlib,io,json,os,shutil,time
from pathlib import Path
import numpy as np
import soundfile as sf
from pub_01_s03_ss07_openneuro_transport import ROOT,obtain_verified


def dump(name,value):
    with (ROOT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def audio(row):
    data=obtain_verified(row,ROOT/'source_audio')
    info=sf.info(io.BytesIO(data))
    wave,rate=sf.read(io.BytesIO(data),dtype='float32',always_2d=True)
    assert rate==info.samplerate and len(wave)==info.frames and np.isfinite(wave).all()
    return dict(kind='audio',path=row['path'],bytes=len(data),sha256=row['payload_sha256'],
        frames=info.frames,sampling_rate=rate,channels=info.channels,subtype=info.subtype,
        channels_sample_identical=bool(all(np.array_equal(wave[:,0],wave[:,j]) for j in range(1,info.channels))),
        duration_seconds=info.frames/rate)


def canary(row):
    folder=ROOT/'canary_raw'
    data=obtain_verified(row,folder)
    for ext in ('.vhdr','.vmrk'):
        name=Path(row['path']).with_suffix(ext).name;source=(ROOT/'metadata/payload_headers'/name).read_bytes();path=folder/name
        if path.exists():assert path.read_bytes()==source
        else:
            with path.open('xb') as f:f.write(source)
    values=np.frombuffer(data,dtype='<f4').reshape(-1,128)
    import mne
    raw=mne.io.read_raw_brainvision(str(folder/Path(row['path']).with_suffix('.vhdr').name),preload=False,verbose='ERROR')
    decoded=raw.get_data(start=0,stop=1024)
    expected=values[:1024].T.astype(np.float64)*1e-7
    difference=float(abs(decoded-expected).max())
    assert difference<=1e-18 and raw.n_times==len(values) and raw.info['sfreq']==512
    physical=values.astype(np.float64)*.1
    assert np.isfinite(physical).all()
    std=physical.std(0)
    old=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01/prepared/train_only_normalization.npz')
    norm=np.load(old,allow_pickle=False)
    return dict(kind='EEG_canary',path=row['path'],bytes=len(data),sha256=row['payload_sha256'],
        frames=len(values),sampling_rate=512,channels=128,MNE_version=mne.__version__,
        MNE_vs_declared_float32_resolution_to_V_max_abs=difference,
        conversion_to_microvolt='.eeg stored float32 times0.1, not times1e6',
        microvolt_range=[float(physical.min()),float(physical.max())],
        channel_std_microvolt_min_median_max=[float(std.min()),float(np.median(std)),float(std.max())],
        original_TRAIN_normalization=dict(path=str(old),sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
            mean_shape=list(norm['mean'].shape),std_shape=list(norm['std'].shape),
            std_min_median_max=[float(norm['std'].min()),float(np.median(norm['std'])),float(norm['std'].max())],
            units='source LoadEEGNumpy unit_multiplier1e6, original saved microvolt normalization reused'),
        no_new_norm_fit=True)


def main():
    started=time.time()
    assert not (ROOT/'ACQUISITION_PREFLIGHT_V2_STARTED.json').exists()
    dump('ACQUISITION_PREFLIGHT_V2_STARTED.json',dict(pid=os.getpid(),time=started,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_model_scores=True))
    objects=json.loads((ROOT/'ANNEX_CATALOG_V2.json').read_text())['objects']
    audios=[r for r in objects if r['path'].endswith('.wav')]
    eeg=next(r for r in objects if r['path']=='sub-001/eeg/sub-001_task-listening_run-01_eeg.eeg')
    for name in ('source_audio','canary_raw'):(ROOT/name).mkdir(exist_ok=True)
    results=[];failures=[]
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(canary,eeg):eeg}
        pending.update({pool.submit(audio,row):row for row in audios})
        for f in futures.as_completed(pending):
            try:
                value=f.result();results.append(value)
                if value['kind']=='EEG_canary':dump('EEG_UNIT_CANARY.json',value)
                print(json.dumps(dict(stage='verified_object',done=len(results),total=21,path=value['path'])),flush=True)
            except Exception as e:
                failures.append(dict(path=pending[f]['path'],error=repr(e)))
                print(json.dumps(dict(stage='object_failure',path=pending[f]['path'],error=repr(e))),flush=True)
    summary=dict(objects=results,failures=failures,root_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        free_disk_bytes=shutil.disk_usage(ROOT).free,wall_seconds=time.time()-started,NN_forwards=0,new_fit=0)
    dump('ACQUISITION_PREFLIGHT_V2_COMPLETED.json',summary)
    assert len(results)==21 and not failures
    print(json.dumps(dict(stage='COMPLETED',objects=21,wall_seconds=summary['wall_seconds'],NN_forwards=0)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        if not (ROOT/'ACQUISITION_PREFLIGHT_V2_FAILURE.json').exists():
            dump('ACQUISITION_PREFLIGHT_V2_FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
