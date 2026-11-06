"""D22 independent aggregation of frozen saved scores, no model computation."""
import json, hashlib, math, time
from pathlib import Path
from collections import defaultdict

root=Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_output_query_20261106')
assert Path(__file__).resolve().parent==root
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
start=time.monotonic(); cpu=time.process_time()
sources={}
def source(path):
    value=read(path);sources[str(path)]=sha(path);return value
assert source(root/'COORDINATOR_FIT_AUDIT_BATCH01.json')['passed']
assert source(root/'COORDINATOR_FUNCTIONAL_REPLAY.json')['passed']
done=source(root/'fit/COMPLETED.json');assert done['passed']
def mean(x):return math.fsum(x)/len(x)
def decompose(anchor,metric):
    reference={(b['record'],b['start']):b for b in anchor['blocks']}
    groups=defaultdict(list)
    for b in metric['blocks']:
        a=reference[b['record'],b['start']]
        assert (b['subject'],b['samples'])==(a['subject'],a['samples'])
        groups[b['subject']].append((b['start']<1344,b['r']-a['r']))
    assert len(groups)==49 and sum(map(len,groups.values()))==752
    parts={}
    for label,which in [('mixed_prefix',True),('pure_active',False)]:
        grouped=[[v for flag,v in bs if flag==which] for bs in groups.values()]
        assert all(grouped)
        parts[label]=dict(blocks=sum(map(len,grouped)),macro_within_group_delta=mean([mean(bs) for bs in grouped]),
                         primary_contribution=mean([math.fsum(v for flag,v in bs if flag==which)/len(bs) for bs in groups.values()]))
    delta=metric['r']-anchor['r']
    assert abs(sum(v['primary_contribution'] for v in parts.values())-delta)<1e-12
    return dict(delta_r=delta,parts=parts,active_MSE=metric['active_MSE'],active_MSE_delta=metric['active_MSE']-anchor['active_MSE'])
def people_delta(a,b):
    ref={p['subject']:p['r'] for p in b['people']}
    ds=[p['r']-ref[p['subject']] for p in a['people']]
    assert len(ds)==49
    return dict(positive=sum(v>0 for v in ds),negative=sum(v<0 for v in ds),equal=sum(v==0 for v in ds),mean_delta=mean(ds))
outputs=[]
for seed in (1024,2026,31415):
    r=source(root/'fit'/f'seed{seed}_P_past_prediction'/'RESULT.json')
    anchor=source(root/'fit'/f'seed{seed}_ANCHOR.json')['anchor']['val']
    d20=source(root.parent/'s03_recent_prediction_context_20261106'/'fit'/f'seed{seed}_P_past_prediction'/'RESULT.json')
    ridge=source(root.parent/'s03_output_ridge_20261106'/'fit'/f'seed{seed}_METRICS.json')
    fkey=ridge['selection']['F']['selected'];f=ridge['anchor'] if fkey is None else ridge['models'][fkey]['val']
    b=r['selected'];g=d20['selected']
    assert abs(b['r']-done['seeds'][str(seed)]['selected_r'])<1e-12
    best=max(r['epochs'][1:],key=lambda e:e['val']['r'])
    same=[x['val']['r']-y['val']['r'] for x,y in zip(r['epochs'][1:],d20['epochs'][1:])]
    assert len(same)==20
    out=dict(seed=seed,anchor_r=anchor['r'],selected_epoch=r['selected_epoch'],selected_r=b['r'],
             D20_r=g['r'],F_r=f['r'],B_minus_D20=b['r']-g['r'],B_minus_F=b['r']-f['r'],
             last_r=r['epochs'][-1]['val']['r'],last_train_active_MSE=r['epochs'][-1]['train']['active_MSE'],
             best_trained_epoch=best['epoch'],best_trained_r=best['val']['r'],
             same_epoch_B_wins=sum(v>0 for v in same),epochs_above_anchor=sum(x['val']['r']>anchor['r'] for x in r['epochs'][1:]),
             B=decompose(anchor,b),D20=decompose(anchor,g),F=decompose(anchor,f),
             people_B_vs_anchor=people_delta(b,anchor),people_B_vs_D20=people_delta(b,g),
             paired_far_r=r['val_pair']['older']['r'],paired_near_minus_far=r['val_pair']['near']['r']-r['val_pair']['older']['r'])
    outputs.append(out)
summary={key:mean([r[key] for r in outputs]) for key in ('anchor_r','selected_r','D20_r','F_r','B_minus_D20','B_minus_F','last_r','last_train_active_MSE','paired_near_minus_far')}
summary['B_minus_anchor']=summary['selected_r']-summary['anchor_r']
summary['same_epoch_B_wins']=sum(r['same_epoch_B_wins'] for r in outputs)
summary['B_pure_active_contribution']=mean([r['B']['parts']['pure_active']['primary_contribution'] for r in outputs])
summary['B_mixed_prefix_contribution']=mean([r['B']['parts']['mixed_prefix']['primary_contribution'] for r in outputs])
summary['B_active_MSE']=mean([r['B']['active_MSE'] for r in outputs])
out=dict(passed=True,source_sha256=sha(Path(__file__)),sources=sources,results=outputs,summary=summary,
         scope='Independent descriptive aggregation of original selected saved blocks and all epoch metrics; no new model selection, significance, raw score calculation, or external test.',
         CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,NN_forwards=0,fits=0,goal_complete=False)
with (root/'COORDINATOR_SCIENTIFIC_REVIEW.json').open('x') as stream:json.dump(out,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(passed=True,summary=summary,results=outputs,CPU_seconds=out['CPU_seconds'],wall_seconds=out['wall_seconds'])))
