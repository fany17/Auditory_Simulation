"""D8 read-only TRAIN/VAL inventory: JSON, file metadata, NPY headers only."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
P=BASE/'s03_incremental_continuity_20261101'
OLD=BASE/'s03_strong_baselines_20261015/transformer_available01'
ROOT=BASE/'s03_finetuned_readout_transfer_20261101'
SEEDS=(1024,2026,31415)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def header(p):
    with p.open('rb') as f:
        version=np.lib.format.read_magic(f)
        shape,order,dtype=np.lib.format._read_array_header(f,version)
        offset=f.tell()
    st=p.stat()
    assert not dtype.hasobject
    return dict(shape=list(shape),dtype=str(dtype),fortran_order=order,header_bytes=offset,bytes=st.st_size,mtime_ns=st.st_mtime_ns)

ROOT.mkdir(exist_ok=True)
out=ROOT/'INPUT_INVENTORY.json';assert not out.exists()
manifest=read(OLD/'envelope_readout_seed1024/manifest_frozen.json')
records={phase:[r for r in manifest['records'] if r['phase']==phase] for phase in ('train','val')}
counts={};inputs={};sources={str(Path(__file__)):sha(Path(__file__))}
for phase,rr in records.items():
    offset=0
    for r in rr:
        assert r['phase_offset']==offset
        assert r['readout_windows']==(r['common_samples']-192)//96+1
        offset+=r['readout_windows']
        for key in ('eeg_path','envelope_path'):
            path=Path(r[key]);inputs[str(path)]=header(path)
    assert offset=={'train':252454,'val':30500}[phase]
    inputs[str(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy')]=header(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy')
    counts[phase]=dict(records=len(rr),people=len({r['subject'] for r in rr}),windows=offset,fit_i_ge4_windows=offset-4*len(rr),global64_batches=math.ceil(offset/64))
seeds=[]
for seed in SEEDS:
    folder=P/f'finetune/native_P_seed{seed}';done=read(folder/'completed.json')
    assert done['seed']==seed and done['selected_epoch']==4 and done['epochs']==5
    arrays=list(folder.rglob('*.npy'))+list(folder.rglob('*.npz'))
    assert not arrays,'Unexpected FT array assets; reconsider reuse before encoding'
    cp=folder/'best_validation.pt'
    pins=[cp]+[folder/name for name in ('completed.json','resolved_config.json','history.json','INCUMBENT_EQUIVALENCE.json')]
    sources.update({str(path):sha(path) for path in pins})
    native=P/f'baseline/head_seed{seed}'
    native_headers={phase:header(native/f'{phase}_latent.npy') for phase in counts}
    seeds.append(dict(seed=seed,checkpoint=str(cp),checkpoint_bytes=cp.stat().st_size,checkpoint_sha256=sources[str(cp)],selected_epoch=done['selected_epoch'],best_validation_negative_Pearson=done['best_validation_negative_Pearson'],prior_extra_epochs=done['epochs'],prior_extra_updates=done['updates'],prior_target_windows=done['target_window_presentations'],prior_seconds=done['seconds'],FT_TRAIN_VAL_arrays=0,native_latents_not_FT=native_headers))
for path in [OLD/'envelope_readout_seed1024/manifest_frozen.json',OLD/'prepared/train_only_normalization.npz',P/'FINETUNE_PROTOCOL.json',P/'INCREMENTAL_PROTOCOL.json',P/'block_eligibility/val.json']:
    sources[str(path)]=sha(path)
roots=[p for p in BASE.iterdir() if p.is_dir() and p.name.startswith('s03_') and p.name.endswith('_20261101') and p!=ROOT]
existing={p.name:sum(f.stat().st_size for f in p.rglob('*.npy')) for p in roots}
assert len(existing)==7
N=sum(c['windows'] for c in counts.values());phases=len(counts);seeds_n=len(SEEDS)
latent=seeds_n*N*8*192*4;pred=seeds_n*N*192*4
# NumPy format v1, standard shape/dtype header here occupies 128 bytes per file.
header_bytes=seeds_n*phases*2*128
new_npy=latent+pred+header_bytes
max_derived=7*1024**3;reserve=20*1024**3;cap=25*1024**3;free=shutil.disk_usage(ROOT).free
# Per seed full [Z1536,P192,Zsquare1536] moment/three transforms/cross matrices;
# each of 3 lambdas: Ponly192x192, C1728x192, Ccap3264x192 + free intercept.
moment_bytes=seeds_n*8*(3264**2+3264*192+3264+192+1)
coef_bytes=seeds_n*len((.0001,.01,1))*8*((192+1728+3264)*192+3*192)
budget=dict(existing_npy_bytes_by_root=existing,existing_npy_bytes=sum(existing.values()),new_latent_payload_bytes=latent,new_window_P_payload_bytes=pred,new_npy_header_bytes=header_bytes,projected_total_new_npy_bytes=new_npy,projected_combined_npy_bytes=sum(existing.values())+new_npy,array_cap_bytes=cap,array_cap_headroom_bytes=cap-sum(existing.values())-new_npy,new_root_cap_bytes=max_derived,disk_free_bytes=free,free_after_full_root_cap_bytes=free-max_derived,minimum_free_reserve_bytes=reserve,estimated_moments_uncompressed_bytes=moment_bytes,estimated_coefficients_uncompressed_bytes=coef_bytes,new_primary_encoder_windows=seeds_n*N,new_primary_global64_batches=seeds_n*sum(c['global64_batches'] for c in counts.values()),bridge_extra_max_windows=seeds_n*2*64,primary_single_pass_per_seed_phase=True)
assert budget['array_cap_headroom_bytes']>0 and free-max_derived>=reserve
result=dict(scope='TRAIN_VAL_JSON_FILE_METADATA_NPY_HEADERS_ONLY_NO_ARRAY_VALUES_MODEL_OR_SCORING',time=time.time(),seeds=seeds,counts=counts,budget=budget,inputs=inputs,source_pins=sources,precision=read(P/'INCREMENTAL_PROTOCOL.json')['precision'],no_test_DTU=True,no_new_cache_fit_or_forward=True,goal_complete=False)
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','source_pins')},indent=2))
