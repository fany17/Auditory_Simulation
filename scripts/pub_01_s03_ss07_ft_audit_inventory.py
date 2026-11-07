"""D10 only JSON/source identity and NPY headers; no new test value reads."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');P=BASE/'s03_incremental_continuity_20261101';D8=BASE/'s03_finetuned_readout_transfer_20261101';D9=BASE/'s03_physical_objective_20261101';ROOT=BASE/'s03_finetuned_internal_audit_20261101'
SEEDS=(1024,2026,31415)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def header(p):
    with p.open('rb') as f:
        version=np.lib.format.read_magic(f);shape,order,dtype=np.lib.format._read_array_header(f,version);offset=f.tell()
    st=p.stat();assert not dtype.hasobject
    return dict(shape=list(shape),dtype=str(dtype),fortran_order=order,header_bytes=offset,bytes=st.st_size,mtime_ns=st.st_mtime_ns)
ROOT.mkdir(exist_ok=True);assert not (ROOT/'INPUT_INVENTORY.json').exists()
mask=read(P/'SPARR_TEST_MASK.json');manifest=read(P/'FINAL_CHECKPOINT_MANIFEST.json');raw=read(P/'independent_raw_review/RAW_INPUT_REVIEW.json');assert raw['passed']
rows=mask['records'];assert len(rows)==144
counts={}
for phase in ('test_seen','test_heldout'):
    rr=[r for r in rows if r['phase']==phase];N=sum((r['common_samples']-192)//96+1 for r in rr);offset=0
    for row in rr:assert row['phase_offset']==offset;offset+=(row['common_samples']-192)//96+1
    blocks=[b for rm in mask['record_masks'] if rm['record'] in {r['record'] for r in rr} for b in rm['blocks'] if b['eligible']]
    counts[phase]=dict(records=len(rr),people=len({r['subject'] for r in rr}),families=len({r['source_family'] for r in rr}),windows=N,eligible_blocks=len(blocks),global64_batches=math.ceil(N/64),first_batch=64,tail_batch=N%64 or 64,first4_fallback_windows=4*len(rr))
inputs={};sources={str(Path(__file__)):sha(Path(__file__))};seeds=[]
for row in rows:
    for key in ('eeg_path','envelope_path'):inputs[row[key]]=header(Path(row[key]))
for seed in SEEDS:
    entry=manifest['entries'][f'P_FT:{seed}'];cp=Path(entry['path']);assert sha(cp)==entry['sha256']
    native=read(P/f'final_latent/seed{seed}/cache_manifest.json');assert native['complete'] and native['encoder_sha256']==manifest['encoders'][str(seed)]['sha256']
    assert native['encoder_sha256']!=entry['sha256']
    oldscore=P/f'final_results/seed{seed}/INTERNAL_P_FT.json';score=read(oldscore);assert score['complete'] and len(score['segments'])==2044
    frozen=[D8/f'seed{seed}/TRAIN_TRANSFORMS.npz',D8/f'seed{seed}/P_FT_BASELINE.json']
    frozen.extend(D8/f'seed{seed}/{arm}_lambda{lam:g}_COEFFICIENTS.npz' for arm,lam in (('Ponly',.01),('C',.0001),('Ccap',.01)))
    frozen.extend(D9/f'seed{seed}/{arm}_COEFFICIENTS.npz' for arm in ('W','F'))
    pins=[cp,oldscore,P/f'finetune/native_P_seed{seed}/completed.json',P/f'final_latent/seed{seed}/cache_manifest.json']+frozen
    sources.update({str(path):sha(path) for path in pins})
    for row in rows:
        path=P/f'final_results/seed{seed}/predictions/internal/P_FT/{row["record"]}.npy';info=header(path);assert info['shape']==[row['common_samples']] and info['dtype']=='float64';inputs[str(path)]=info
    for kind in ('latent','P'):
        path=D8/f'seed{seed}/val_{kind}.npy';inputs[str(path)]=header(path)
    proofs=[r for r in raw['records'] if r['seed']==seed and r['condition']=='P_FT' and r['dataset']=='Sparr']
    assert len(proofs)==3 and max(r['maximum_absolute_waveform_difference'] for r in proofs)<=1e-6
    seeds.append(dict(seed=seed,checkpoint=str(cp),checkpoint_sha256=entry['sha256'],selected_epoch=entry['selected_epoch'],existing_native_test_latents_not_FT=True,original_FT_fused_records=144,original_FT_window_P_saved=False,FT_test_latent_available=False,prior_actual_FT_raw_Sparr_bridges=proofs))
for path in [P/'SPARR_TEST_MASK.json',P/'FINAL_CHECKPOINT_MANIFEST.json',P/'INCREMENTAL_PROTOCOL.json',P/'independent_raw_review/RAW_INPUT_REVIEW.json',D8/'PROTOCOL.json',D8/'SUMMARY.json',D8/'COORDINATOR_RESULT_REVIEW.json',D9/'PROTOCOL.json',D9/'COORDINATOR_RESULT_REVIEW.json',BASE/'s03_current_readout_internal_audit_20261101/COORDINATOR_HISTORICAL_FT_REFERENCE.json']+[P/'input'/name for name in ('pub_01_s03_ss07_eval_io.py','pub_01_s03_ss07_eval_common.py','pub_01_s03_ss07_finetune.py')]:sources[str(path)]=sha(path)
N=sum(c['windows'] for c in counts.values());new_payload=len(SEEDS)*N*(8*192+192)*4;headers=len(SEEDS)*2*len(counts)*128
existing=sum(path.stat().st_size for root in BASE.glob('s03_*_20261101') for path in root.rglob('*.npy'));free=shutil.disk_usage(ROOT).free;cap=25*1024**3;rootcap=2*1024**3
budget=dict(existing_NPY_bytes=existing,new_Z_P_payload_bytes=new_payload,new_NPY_headers_bytes=headers,new_NPY_bytes=new_payload+headers,combined_NPY_bytes=existing+new_payload+headers,array_cap_bytes=cap,array_cap_headroom_bytes=cap-existing-new_payload-headers,root_cap_bytes=rootcap,disk_free_bytes=free,free_after_full_root_cap_bytes=free-rootcap,minimum_free_reserve_bytes=20*1024**3,main_encoder_head_windows=3*N,main_encoder_batches=3*sum(c['global64_batches'] for c in counts.values()),extra_VAL_capture_gold_windows=300,extra_original_Decoder_test_windows=3*sum(c['first_batch']+c['tail_batch'] for c in counts.values()))
assert budget['combined_NPY_bytes']<cap and budget['free_after_full_root_cap_bytes']>=20*1024**3
result=dict(scope='METADATA_ONLY_NO_TEST_ARRAY_VALUES_MODEL_FORWARD_NEW_SCORING_OR_FIT',time=time.time(),counts=counts,seeds=seeds,budget=budget,inputs=inputs,source_pins=sources,raw_recipe='SparrRecords.raw_windows np.stack(...).astype(float32) then predict_raw_phase global64 perphase including mixed-record batches; restack pending, no_grad/eval/grad flags/TF32 original',new_ft_cache_proposal='Per-phase contiguous Z/P 12float32 NPY; no learned fused copies',old_test_label='POST-HOC INTERNAL EXPLORATORY',no_DTU=True,goal_complete=False)
with (ROOT/'INPUT_INVENTORY.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','source_pins','seeds')},indent=2))
