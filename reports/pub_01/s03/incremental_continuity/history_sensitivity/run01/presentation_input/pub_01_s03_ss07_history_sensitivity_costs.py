"""D17 final existing-JSON/stat ledger; 0 new NN/backward."""
import json,time,resource,shutil
from pathlib import Path
WALL=time.time();CPU=time.process_time()
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_sensitivity_20261102')
def read(p):return json.loads(p.read_text())
done=read(ROOT/'COMPLETED.json');assert done['passed'] and done['residual_NN_windows']==384 and done['history_input_backward_calls']==8
stages=[]
for name,path in [('prepare_v1_source_stat_and_synthetic_axis_checks','prepare_v1/PREPARED.json'),('prepare_v2_review_metadata_and_grouping_refreeze','PREPARED.json'),('fixed48_four_checkpoints_two_conditions','COMPLETED.json'),('JSON_only_tables','PRESENTATION_COST.json')]:
    d=read(ROOT/path);stages.append(dict(stage=name,NN=d.get('residual_NN_windows',0),backward=d.get('history_input_backward_calls',0),fits=0,optimizer_updates=0,new_encoder_head=0,CPU_seconds=d['CPU_seconds'],wall_seconds=d['wall_seconds'],peak_RSS_KiB=d['peak_RSS_KiB'],peak_GPU_allocated_bytes=d.get('peak_GPU_allocated_bytes',0)))
files=[x for x in ROOT.rglob('*') if x.is_file()]
limits=dict(root_bytes=sum(x.stat().st_size for x in files),root_cap_bytes=8*1024**2,home_free_bytes=shutil.disk_usage(ROOT).free,sata_free_bytes=shutil.disk_usage('/sata5/fanyu/auditory_simulation_m6a').free,each_disk_reserve_bytes=20*1024**3,new_array_or_weight_files=[str(x.relative_to(ROOT)) for x in files if x.suffix in ('.npy','.npz','.pt')])
assert limits['root_bytes']<limits['root_cap_bytes'] and min(limits['home_free_bytes'],limits['sata_free_bytes'])>=limits['each_disk_reserve_bytes'] and not limits['new_array_or_weight_files']
stages.append(dict(stage='JSON_only_cost_ledger',NN=0,backward=0,fits=0,optimizer_updates=0,new_encoder_head=0,CPU_seconds=time.process_time()-CPU,wall_seconds=time.time()-WALL,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=0))
ledger=dict(stages=stages,total_new_residual_NN=sum(x['NN'] for x in stages),total_input_backward_calls=sum(x['backward'] for x in stages),fits=0,optimizer_updates=0,new_encoder_head=0,listed_instrumented_CPU_seconds=sum(x['CPU_seconds'] for x in stages),resource=limits,notes=['Prepare timers include lightweight source/checkpoint hashes/stat checks and synthetic scalar checks; no cache values or NN. Their preceding imports are excluded.','Diagnostic run timer includes verification, bounded emptyGPU wait, checkpoint/subset loading, diagnostics and backward; preceding imports and final receipt serialization excluded.','384NN windows and8 local input-gradient backward calls are separate cost measures; no optimizer or encoder/head. Additional hook statistic arithmetic is included in runtime, not relabeled as zero arithmetic.','Administrative review, SSH/copy, local archive and timers before imports/after final serialization not comprehensively instrumented; no FLOP equality claim.','Inherited D16/D15 and earlier costs remain provenance, not new D17 computation. Independent review of source and scalar receipts is not an additional neural replay.'],goal_complete=False)
assert ledger['total_new_residual_NN']==384 and ledger['total_input_backward_calls']==8
(ROOT/'COST_LEDGER.json').write_text(json.dumps(ledger,indent=2,allow_nan=False));print(json.dumps(ledger,allow_nan=False))
