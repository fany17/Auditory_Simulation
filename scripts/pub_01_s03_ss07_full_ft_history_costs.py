"""Assemble measured and inherited cost references; no models or signal values."""
from pathlib import Path
import json,hashlib,shutil,platform
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_history_20261102');BASE=ROOT.parent
def read(p):return json.loads(p.read_text())
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
entries=[]
for p in sorted((ROOT/'fits').glob('*/COMPLETED.json')):entries.append(dict(stage='formal_'+p.parent.name,inherited=False,reference=ref(p),measurements=read(p)))
for name,label,nested in [('ENGINEERING_CHECK.json','successful_disposable_engineering',False),
    ('coordinator_review/REVIEW.json','independent_selected_and_older_review',True),
    ('coordinator_last_review/REVIEW.json','independent_all_last19_review',True),
    ('COORDINATOR_REVIEW_TRAIN_BRIDGE.json','nonzero_TRAIN_CPU_GPU_bridge',False),
    ('COORDINATOR_TARGET_MAPPING.json','complete_target_mapping',False),
    ('COORDINATOR_REVIEW_SOURCE_CHECK.json','synthetic_manual_GRU_and_Hann_check',False)]:
    p=ROOT/name;j=read(p);m=j['cost'] if nested else {k:v for k,v in j.items() if k in ['wall_seconds','CPU_seconds','peak_RSS_KiB','peak_GPU_allocated_bytes','temporary_discarded_optimizer_steps','temporary_residual_NN_window_forwards','residual_NN_window_forwards_CPU','residual_NN_window_forwards_GPU','synthetic_GRU_sequences','synthetic_GRU_steps_per_sequence','NN_forwards']}
    entries.append(dict(stage=label,inherited=False,reference=ref(p),measurements=m))
p=ROOT/'engineering_attempt01/FAILURE_RECORD.json';entries.append(dict(stage='failed_engineering_attempt01',inherited=False,reference=ref(p),measurements=read(p),counts_path_derived_not_runtime_counter=True))
P=BASE/'s03_incremental_continuity_20261101'
for seed in [1024,2026,31415]:
    for label,folder,key in [('pretraining','baseline/pretrain_seed','seconds'),('head_including_cache','baseline/head_seed','seconds_including_cache'),('strong_FT','finetune/native_P_seed','seconds')]:
        p=P/f'{folder}{seed}/completed.json';j=read(p)
        entries.append(dict(stage=f'inherited_{label}_seed{seed}',inherited=True,reference=ref(p),measurements=dict(wall_seconds=j[key],
            CPU_seconds=j.get('CPU_seconds'),epochs=j['epochs'],updates=j['updates'],selected_epoch=j['selected_epoch'],
            target_window_presentations=j.get('target_window_presentations'),stochastic_actual_presentations=j.get('stochastic_actual_presentations'),
            sample_presentations=j.get('sample_presentations'),target_sample_presentations=j.get('target_sample_presentations')),missing_measurements='Unrecorded fields remain null, not zero'))
p=BASE/'s03_finetuned_readout_transfer_20261101/COST.json';j=read(p)
entries.append(dict(stage='inherited_D8_FT_cache_calibration_and_27_ridge_fits',inherited=True,reference=ref(p),measurements={k:v for k,v in j.items() if k!='stages'},all_stage_details_in_referenced_manifest=True))
entries.append(dict(stage='unmeasured_overhead_and_upstream_cost',inherited=True,measurements=dict(wall_seconds=None,CPU_seconds=None),
    scope='Original published preprocessing/upstream XLSR pretraining/complete historic SS07 search; new inventory/source retrieval/report preparation not fully instrumented',unknown_is_not_zero=True))
ledger=dict(entries=entries,units=dict(wall_seconds='seconds',CPU_seconds='seconds',peak_RSS_KiB='KiB',peak_GPU_allocated_bytes='bytes, excludes driver/context'),
    formal_updates=710100,discarded_engineering_updates=18,formal_new_residual_NN_windows=50522928,
    successful_engineering_windows=92046,failed_engineering_windows_path_derived=30169,independent_review_windows=360708+270531+96,
    all_counted_full_residual_windows_including_path_derived_failure=50522928+92046+30169+360708+270531+96,
    synthetic_short_GRU_sequences_separate=7,new_encoder_head_windows=0,new_external_access=False,
    actual_parallel_queue_wall_seconds=read(ROOT/'QUEUE_COMPLETED.json')['wall_seconds'],parallel_fit_wall_times_not_summed=True,
    ledger_not_complete_historic_search_cost=True)
with (ROOT/'report01/COST_LEDGER.json').open('x') as f:json.dump(ledger,f,indent=2,allow_nan=False)
cache=Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102/caches')
snapshot=dict(new_run_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),new_run_cap_bytes=128*1024**2,
    home_free_bytes=shutil.disk_usage(ROOT).free,sata_free_bytes=shutil.disk_usage(cache).free,each_free_reserve_bytes=20*1024**3,
    no_new_history_disk_arrays=True,formal_queue_complete=True,all_worker_returncodes_zero=True)
assert snapshot['new_run_bytes']<128*1024**2 and min(snapshot['home_free_bytes'],snapshot['sata_free_bytes'])>20*1024**3
with (ROOT/'FINAL_RESOURCE_SNAPSHOT.json').open('x') as f:json.dump(snapshot,f,indent=2)
print(dict(full_residual_window_count=ledger['all_counted_full_residual_windows_including_path_derived_failure'],resources=snapshot),flush=True)
