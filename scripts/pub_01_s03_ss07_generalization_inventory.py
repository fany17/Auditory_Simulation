"""D7 metadata/header inventory only; no array values, model or scores read."""
import json
from pathlib import Path
import shutil
import time
import numpy as np

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
OLD=BASE/'s03_incremental_continuity_20261101'
ROOT=BASE/'s03_current_readout_internal_audit_20261101'
SEEDS=(1024,2026,31415)
def read(p):return json.loads(p.read_text())
mask=read(OLD/'SPARR_TEST_MASK.json');rows=mask['records'];assert len(rows)==144
manifest=read(OLD/'FINAL_CHECKPOINT_MANIFEST.json');ready=read(OLD/'EVALUATION_READY.json')
inventory=[];inputs={};errors=[]
for seed in SEEDS:
    cache=read(OLD/f'final_latent/seed{seed}/cache_manifest.json')
    assert cache['complete'] and not cache['read_failures'] and len(cache['records'])==144
    assert cache['encoder_sha256']==manifest['encoders'][str(seed)]['sha256']
    for row in rows:
        record=row['record'];windows=(row['common_samples']-192)//96+1
        paths={'frozen_test_latent':Path(cache['records'][record]),'original_P_fused':OLD/f'final_results/seed{seed}/predictions/internal/native_P/{record}.npy'}
        for kind,p in paths.items():
            try:
                arr=np.load(p,mmap_mode='r',allow_pickle=False)
                shape=(windows,8,192) if kind=='frozen_test_latent' else (row['common_samples'],)
                assert arr.shape==shape
                st=p.stat();inputs[str(p)]={'kind':kind,'shape':list(arr.shape),'dtype':str(arr.dtype),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
            except Exception as exc:errors.append(dict(seed=seed,record=record,kind=kind,error=repr(exc)))
    inventory.append(dict(seed=seed,latent_cache_records=len(cache['records']),encoder_sha256=cache['encoder_sha256'],original_P_checkpoint=manifest['entries'][f'native_P:{seed}']['path'],
        original_eval_head_context='pub_01_s03_ss07_eval_common.predict_windows / aligned_chunks(global phase offset); head only on frozen cache; no encoder needed'))
phase_counts={}
for phase in ('test_seen','test_heldout'):
    rr=[r for r in rows if r['phase']==phase];names={r['record'] for r in rr}
    blocks=[b for rm in mask['record_masks'] if rm['record'] in names for b in rm['blocks'] if b['eligible']]
    phase_counts[phase]=dict(records=len(rr),people=len({r['subject'] for r in rr}),source_families=len({r['source_family'] for r in rr}),
        windows=sum((r['common_samples']-192)//96+1 for r in rr),eligible_blocks=len(blocks),common_samples=sum(r['common_samples'] for r in rr),
        first4_fallback_windows=4*len(rr))
array_bytes=sum(p.stat().st_size for root in [OLD,BASE/'s03_conditional_history_probe_20261101',BASE/'s03_history_interaction_probe_20261101',BASE/'s03_readout_attribution_20261101'] for p in root.rglob('*.npy'))
new_P_window_bytes=3*sum(v['windows'] for v in phase_counts.values())*192*4
result=dict(time=time.time(),scope='READ_ONLY_JSON_AND_NPY_HEADERS_NO_VALUES_NO_MODEL_NO_NEW_SCORING',seeds=SEEDS,phase_counts=phase_counts,seeds_inventory=inventory,inputs=inputs,errors=errors,
    old_test_label='POST-HOC INTERNAL EXPLORATORY; all these records already evaluated in round01',old_mask_path=str(OLD/'SPARR_TEST_MASK.json'),
    old_ready_path=str(OLD/'EVALUATION_READY.json'),existing_frozen_latents_reusable=not errors,encoder_forward_needed=False,
    original_P_persisted='fused only; no persisted window P in original final_results. Can replay head only from validated existing latent and bridge every original fused P record.',
    existing_npy_bytes_across_SS07_roots=array_bytes,proposed_new_P_window_cache_bytes=new_P_window_bytes,proposed_combined_npy_bytes=array_bytes+new_P_window_bytes,
    array_cap_bytes=25*1024**3,disk_free_bytes=shutil.disk_usage(ROOT).free,no_DTU_or_new_payload=True)
with (ROOT/'INPUT_INVENTORY.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))
