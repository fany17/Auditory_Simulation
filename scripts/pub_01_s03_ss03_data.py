"""Strict SS02-aligned raw EEG windows for SS03; server2203 only."""
import hashlib
import json
from pathlib import Path
import numpy as np

PHASES=('train','val','test_seen','test_heldout')


def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(config):
    cfg=json.loads(Path(config).read_text());root=Path(cfg['server_root'])
    if cfg['patient_data']!='FORBIDDEN' or str(root)!='/home/fanyu/auditory_simulation_m6a':raise RuntimeError('public2203 only')
    source=root/cfg['upstream_public_available'];readout=root/cfg['upstream_frozen_encoder_readout'];out=Path(cfg['new_output_root'])
    if not str(out).startswith(str(root)+'/pub_01/s03_architecture_'):raise RuntimeError('new output boundary')
    return cfg,source,readout,out


def prepare(config):
    cfg,source,readout,root=load_contract(config);out=root/'cache01';out.mkdir(exist_ok=False)
    manifest=json.loads((readout/'manifest_frozen.json').read_text());records=manifest['records']
    prepared=json.loads((source/'prepared/manifest_frozen.json').read_text())
    assert [(r['phase'],r['record'],r['common_samples']) for r in records]==[(r['phase'],r['record'],r['common_samples']) for r in prepared['records']]
    assert prepared['normalization_fit_phase']=='train_only'
    story={p:{r['story'] for r in records if r['phase']==p} for p in PHASES}
    people={p:{r['subject'] for r in records if r['phase']==p} for p in PHASES}
    assert not(story['train']&story['val'] or story['train']&story['test_seen'] or story['val']&story['test_seen'])
    assert not people['train']&people['test_heldout'];assert people['test_seen']<=people['train']
    scaling=np.load(source/'prepared/train_only_normalization.npz');mean,std=scaling['mean'],scaling['std']
    assert mean.shape==std.shape==(64,) and np.isfinite(mean).all() and (std>0).all()
    counts=manifest['phase_windows'];offsets={p:0 for p in PHASES}
    targets={p:np.load(readout/(p+'_envelope.npy'),mmap_mode='r') for p in PHASES}
    cache={p:np.lib.format.open_memmap(out/(p+'_eeg.npy'),mode='w+',dtype='float32',shape=(counts[p],192,64)) for p in PHASES}
    for p in PHASES:assert targets[p].shape==(counts[p],1,192) and targets[p].dtype==np.float32
    audit=[]
    for ri,r in enumerate(records):
        p=r['phase'];n=r['readout_windows'];assert r['phase_offset']==offsets[p]
        assert n==(r['common_samples']-192)//96+1
        for key in ('eeg_path','envelope_path'):assert str(r[key]).startswith(str(cfg['server_root'])+'/')
        raw=np.load(r['eeg_path'],mmap_mode='r');assert raw.ndim==2 and raw.shape.count(64)==1
        eeg=raw.T if raw.shape[0]==64 else raw;env=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)
        assert min(len(eeg),len(env))>=r['common_samples']
        maximum_error=0.
        for first in range(0,n,256):
            ids=np.arange(first,min(first+256,n));starts=ids*96
            x=np.stack([(np.asarray(eeg[s:s+192],dtype=np.float32)-mean)/std for s in starts]).astype(np.float32)
            truth=np.stack([env[s:s+192] for s in starts]).astype(np.float32)
            frozen=np.asarray(targets[p][offsets[p]+first:offsets[p]+first+len(ids),0])
            assert np.isfinite(x).all() and np.isfinite(frozen).all()
            maximum_error=max(maximum_error,float(np.max(np.abs(truth-frozen))))
            assert np.array_equal(truth,frozen),'frozen target differs from exact raw-record coordinate'
            cache[p][offsets[p]+first:offsets[p]+first+len(ids)]=x
        audit.append(dict(r,alignment_max_absolute_error=maximum_error,eeg_orientation='time_by64',target_reference=str(readout/(p+'_envelope.npy'))))
        offsets[p]+=n
        if ri%25==0:print(json.dumps({'prepared_records':ri+1,'total_records':len(records)}),flush=True)
    assert offsets==counts
    for a in cache.values():a.flush()
    checks={'frozen_records_identical':True,'source_target_bit_equal_float32':True,'exact_story_split_preserved':True,
        'heldout_person_isolated':True,'train_only_scaling_reused':True,'no_transformer_encoder_used':True,'test_performance_not_computed':True}
    output={'task':cfg['task'],'records':audit,'phase_windows':counts,'phase_subjects':{p:sorted(people[p]) for p in PHASES},
        'checks':checks,'physical_sync':'UNKNOWN','config':cfg,
        'metadata_source_sha256':{str(x):digest(x) for x in (Path(config),readout/'manifest_frozen.json',source/'prepared/manifest_frozen.json',source/'prepared/train_only_normalization.npz')},
        'hash_scope':'explicit SS03 source/config/contract identity only; no bulk raw-data checksum audit'}
    dump(out/'manifest.json',output);dump(out/'completed.json',{'ready':True,'scientific_test_scores_computed':False})
    print(json.dumps({'status':'INPUT_ALIGNMENT_PASS','phase_windows':counts,'records':len(records)}),flush=True)


def arrays(config,phase,device=None):
    import torch
    cfg,source,readout,out=load_contract(config)
    if not (out/'cache01/completed.json').exists():raise RuntimeError('input freeze incomplete')
    x=np.load(out/'cache01'/(phase+'_eeg.npy'),mmap_mode='r');y=np.load(readout/(phase+'_envelope.npy'),mmap_mode='r')
    if device is None:return x,y.transpose(0,2,1)
    return torch.from_numpy(x.copy()).to(device),torch.from_numpy(y.transpose(0,2,1).copy()).to(device)
