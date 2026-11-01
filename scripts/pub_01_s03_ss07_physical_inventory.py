"""D9 plan inventory: JSON/file metadata and device capacity only, no values."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
D8=BASE/'s03_finetuned_readout_transfer_20261101';ROOT=BASE/'s03_physical_objective_20261101'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=read(D8/'PROTOCOL.json');s=read(D8/'SUMMARY.json')
assert read(D8/'COORDINATOR_RESULT_REVIEW.json')['status']=='PASS_WITH_LIMITATIONS'
assert next(v for v in s['selection'] if v['arm']=='Ccap')['selected_lambda']==.01
records=p['records']['train'];fit=sum(r['windows']-4 for r in records)
assert len(records)==426 and fit==250750 and all(r['windows']>4 for r in records)
counts=dict(TRAIN_records=len(records),TRAIN_people=len({r['subject'] for r in records}),all_windows=sum(r['windows'] for r in records),fit_windows=fit,
    common_physical_target_points=sum((r['windows']-4)*96+94 for r in records),active_positive_Hann_contributions=fit*190,fixed_first4_context_contributions=len(records)*94,
    combined_positive_Hann_contributions=fit*190+len(records)*94,free_readout_parameters=3264*190+190,frozen_endpoint_parameters=2*(3264+1),paired_output_systems=96)
inputs={};sources={str(Path(__file__)):sha(Path(__file__))}
for seed in (1024,2026,31415):
    folder=D8/f'seed{seed}'
    for name in ('TRAIN_TRANSFORMS.npz','TRAIN_MOMENTS.npz','Ccap_lambda0.01_COEFFICIENTS.npz','P_FT_BASELINE.json','FT_VAL_BRIDGE.json','CACHE_COMPLETED.json'):
        path=folder/name;sources[str(path)]=sha(path)
    for phase in ('train','val'):
        for kind in ('latent','P'):
            path=folder/f'{phase}_{kind}.npy';st=path.stat();inputs[str(path)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
for name in ('PROTOCOL.json','SUMMARY.json','COORDINATOR_RESULT_REVIEW.json'):sources[str(D8/name)]=sha(D8/name)
npy=sum(path.stat().st_size for root in BASE.glob('s03_*_20261101') for path in root.rglob('*.npy'))
result=dict(scope='PLAN_METADATA_ONLY_NO_ARRAY_VALUE_NO_NEW_SOLVE_FORWARD_OR_SCORE',time=time.time(),counts=counts,source_pins=sources,inputs=inputs,
    existing_all_SS07_npy_bytes=npy,proposed_new_npy_bytes=0,new_root_cap_bytes=256*1024**2,disk_free_bytes=shutil.disk_usage(BASE).free,
    GPU_metadata=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).splitlines(),
    GPU_full_TRAIN_feature_bytes=counts['all_windows']*3264*8,GPU_full_TRAIN_window_value_bytes=counts['all_windows']*192*8,lambda_D8_sum_output=.01,lambda_scalar_mean=.01/192,
    max_fits=6,max_CG_iterations_per_fit=200,max_Hessian_products_per_fit=214,encoder_head_forward=0,old_test_DTU=False,goal_complete=False)
ROOT.mkdir(exist_ok=True)
with (ROOT/'INPUT_INVENTORY.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ('source_pins','inputs')},indent=2))
