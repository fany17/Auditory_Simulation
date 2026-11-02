"""D17 JSON-only tables; zero neural computation, standard library only."""
import json,csv,time,resource
from pathlib import Path
WALL=time.time();CPU=time.process_time()
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_sensitivity_20261102')
def read(p):return json.loads(p.read_text())
def write_csv(path,rows):
    with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
done=read(ROOT/'COMPLETED.json');assert done['passed'] and done['residual_NN_windows']==384 and done['history_input_backward_calls']==8 and done['fits']==0
out=ROOT/'tables';out.mkdir(exist_ok=False)
rows=[];summary=[];gradients=[]
for seed,kind in [(2026,'selected'),(1024,'last'),(2026,'last'),(31415,'last')]:
    item=read(ROOT/f'H_seed{seed}_{kind}.json');a,b=item['conditions'];assert a['condition']=='near' and b['condition']=='older'
    ident=dict(seed=seed,checkpoint_kind=kind,epoch=item['checkpoint_epoch'])
    for node,diff in item['near_minus_older'].items():
        rows.append(dict(**ident,node=node,near_global_RMS=a['nodes'][node]['RMS'],older_global_RMS=b['nodes'][node]['RMS'],near_across_window_centered_RMS=a['nodes'][node]['across_windows_centered_RMS'],older_across_window_centered_RMS=b['nodes'][node]['across_windows_centered_RMS'],**diff))
    for c in (a,b):gradients.append(dict(**ident,condition=c['condition'],**c['history_input_gradient']))
    r=dict(**ident)
    for node in ['input_history','token_before_norm','token_after_norm','GRU_last_h','film_gain_delta','film_bias_delta','film_modulated_current','residual','final_prediction_float64']:
        diff=item['near_minus_older'][node]
        for key in ['difference_RMS','relative_difference_RMS','exactly_equal_element_fraction','exactly_equal_window_count']:r[node+'__'+key]=diff[key]
    for c in (a,b):
        for key in ['nonzero_scalars','L2_norm','max_abs']:r[c['condition']+'__input_gradient__'+key]=c['history_input_gradient'][key]
    summary.append(r)
write_csv(out/'NODE_COMPARISONS.csv',rows);write_csv(out/'CHECKPOINT_SUMMARY.csv',summary);write_csv(out/'INPUT_GRADIENTS.csv',gradients)
(out/'SUMMARY.json').write_text(json.dumps(dict(checkpoints=summary,gradient_rows=gradients,node_rows=len(rows),new_NN=0,goal_complete=False),indent=2,allow_nan=False))
(ROOT/'PRESENTATION_COST.json').write_text(json.dumps(dict(NN=0,backward=0,fits=0,optimizer_updates=0,new_encoder_head=0,CPU_seconds=time.process_time()-CPU,wall_seconds=time.time()-WALL,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,GPU_bytes=0,timer_scope='After standard-library imports; before receipt serialization'),indent=2,allow_nan=False))
print(json.dumps(dict(passed=True,node_rows=len(rows),checkpoint_rows=4,gradient_rows=8,new_NN=0)))
