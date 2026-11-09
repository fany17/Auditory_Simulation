"""D16 final stage ledger: frozen JSON/stat only; zero model or array reads."""
import json,time,shutil,resource
from pathlib import Path
WALL=time.time();CPU=time.process_time()
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_token_normalization_20261102')
def read(p):return json.loads(p.read_text())
done=read(ROOT/'QUEUE_COMPLETED.json');review=read(ROOT/'coordinator_review/REVIEW.json');assert review['passed']
fits=[read(p) for p in sorted(ROOT.glob('fits/*/COMPLETED.json'))];assert len(fits)==9
v1=read(ROOT/'engineering_v1/ENGINEERING_CHECK.json');bridge=read(ROOT/'REFERENCE_BRIDGE.json');refreeze=read(ROOT/'REFREEZE_RECORD.json');presentation=read(ROOT/'PRESENTATION_COST.json')
assert v1['NN_windows']==528 and bridge['NN_windows']==288 and refreeze['new_NN']==0
stages=[dict(stage='formal_nine_fits_including_prespecified_evaluation',NN=sum(x['new_residual_NN_window_forwards'] for x in fits),optimizer_updates=sum(x['updates'] for x in fits),fits=9,CPU_seconds=sum(x['CPU_seconds'] for x in fits),wall_seconds=done['wall_seconds'],wall_definition='Concurrent queue elapsed, not worker elapsed sum',peak_RSS_KiB=max(x['peak_RSS_KiB'] for x in fits),peak_GPU_allocated_bytes=max(x['peak_GPU_allocated_bytes'] for x in fits))]
for name,d,nn in [('original_v1_engineering',v1,528),('minimal_independent_output_bridge',bridge,288),('evidence_refreeze_no_NN',refreeze,0),('coordinator_independent_manual_neural_metrics_review',review['cost'],review['cost']['residual_NN_window_forwards']),('JSON_only_tables_and_five_figures',presentation,0)]:
    stages.append(dict(stage=name,NN=nn,optimizer_updates=0,fits=0,CPU_seconds=d['CPU_seconds'],wall_seconds=d['wall_seconds'],peak_RSS_KiB=d.get('peak_RSS_KiB'),peak_GPU_allocated_bytes=d.get('peak_GPU_allocated_bytes',0)))
assert stages[0]['NN']==59473248 and stages[0]['optimizer_updates']==710100
files=list(ROOT.rglob('*'));sizes=[]
for p in files:
    try:
        if p.is_file():sizes.append(p.stat().st_size)
    except FileNotFoundError:continue
limits=dict(time=time.time(),root_bytes=sum(sizes),root_cap_bytes=160*1024**2,inflight_reserve_during_training=28*1024**2,home_free_bytes=shutil.disk_usage(ROOT).free,sata_free_bytes=shutil.disk_usage('/sata5/fanyu/auditory_simulation_m6a').free,each_disk_reserve=20*1024**3,checkpoint_files=len(list(ROOT.glob('fits/*/*.pt'))),new_disk_arrays=[str(p.relative_to(ROOT)) for p in files if p.suffix in ('.npy','.npz')],transport_tar_in_root=any(p.suffix=='.tar' for p in files))
assert limits['root_bytes']<limits['root_cap_bytes'] and min(limits['home_free_bytes'],limits['sata_free_bytes'])>=limits['each_disk_reserve'] and not limits['new_disk_arrays'] and not limits['transport_tar_in_root'] and limits['checkpoint_files']==18
stages.append(dict(stage='JSON_only_cost_ledger',NN=0,optimizer_updates=0,fits=0,CPU_seconds=time.process_time()-CPU,wall_seconds=time.time()-WALL,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,peak_GPU_allocated_bytes=0,timer_scope='Excludes final serialization and standard-library imports'))
result=dict(stages=stages,total_known_residual_NN=sum(s['NN'] for s in stages),formal_residual_NN=59473248,engineering_residual_NN=816,engineering_GPU_residual_NN=672,engineering_manual_CPU_residual_NN=144,independent_residual_NN=review['cost']['residual_NN_window_forwards'],new_encoder_head=0,total_learning_fits=9,total_optimizer_updates=710100,listed_instrumented_CPU_seconds=sum(s['CPU_seconds'] for s in stages),resource=limits,notes=['ENGINEERING_CHECK v2 top-level timing still describes original528 only; bridge288 and refreeze0 costs are separate actual measurements','Queue wall and worker CPU answer different questions; parallelism is not method compute reduction','CPU sum covers listed measured scopes, not SSH, approval waiting, source copying, local archive generation, process startup before timers or all administrative time','D15 storage estimate was inherited, not newly executed or counted as D16 compute','Identical parameter counts do not imply identical operator/FLOP costs; normalization adds arithmetic and changes the represented function','Historical D12/D13/D14/D15 costs remain provenance and are not relabeled as D16 computation'],goal_complete=False)
(ROOT/'COST_LEDGER.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,allow_nan=False))
