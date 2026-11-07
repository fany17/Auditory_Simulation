"""D23 independent frozen-score comparison, without new model execution or selection."""
import json, hashlib, math, time
from pathlib import Path
from collections import defaultdict

root=Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_memoryless_output_20261106')
assert Path(__file__).resolve().parent==root
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
start=time.monotonic(); cpu=time.process_time(); sources={}
def source(p):
    x=read(p);sources[str(p)]=sha(p);return x
assert source(root/'COORDINATOR_FIT_AUDIT_BATCH01.json')['passed']
assert source(root/'COORDINATOR_FUNCTIONAL_REPLAY.json')['passed']
done=source(root/'fit/COMPLETED.json');assert done['passed'] and not done['history_intervention']
def mean(xs):return math.fsum(xs)/len(xs)
def decompose(anchor,metric):
    base={(r['record'],r['start']):r for r in anchor['blocks']}
    people=defaultdict(list)
    for b in metric['blocks']:
        a=base[b['record'],b['start']]
        assert (b['samples'],b['subject'])==(a['samples'],a['subject'])
        people[b['subject']].append((b['start']<1344,b['r']-a['r']))
    assert len(people)==49 and sum(map(len,people.values()))==752
    parts={}
    for name,mixed in [('mixed_prefix',True),('pure_active',False)]:
        groups=[[v for flag,v in bs if flag==mixed] for bs in people.values()]
        assert all(groups)
        parts[name]=dict(blocks=sum(map(len,groups)),macro_within_group_delta=mean([mean(bs) for bs in groups]),
                         primary_contribution=mean([math.fsum(v for flag,v in bs if flag==mixed)/len(bs) for bs in people.values()]))
    delta=metric['r']-anchor['r']
    assert abs(sum(v['primary_contribution'] for v in parts.values())-delta)<1e-12
    return dict(delta_r=delta,parts=parts,active_MSE=metric['active_MSE'],active_MSE_delta=metric['active_MSE']-anchor['active_MSE'])
def people_delta(a,b):
    ref={p['subject']:p['r'] for p in b['people']};ds=[p['r']-ref[p['subject']] for p in a['people']]
    assert len(ds)==49
    return dict(positive=sum(x>0 for x in ds),negative=sum(x<0 for x in ds),equal=sum(x==0 for x in ds),mean_delta=mean(ds))
results=[]
for seed in (1024,2026,31415):
    role=f'seed{seed}_P_past_prediction'
    s=source(root/'fit'/role/'RESULT.json')
    b=source(root.parent/'s03_output_query_20261106'/'fit'/role/'RESULT.json')
    anchor=source(root/'fit'/f'seed{seed}_ANCHOR.json')['anchor']['val']
    ridge=source(root.parent/'s03_output_ridge_20261106'/'fit'/f'seed{seed}_METRICS.json')
    assert s['val_pair']['near']==s['val_pair']['older']
    assert abs(s['selected']['r']-done['seeds'][str(seed)]['selected_r'])<1e-12
    same=[x['val']['r']-y['val']['r'] for x,y in zip(b['epochs'][1:],s['epochs'][1:])]
    assert len(same)==20
    best=max(s['epochs'][1:],key=lambda e:e['val']['r'])
    row=dict(seed=seed,anchor_r=anchor['r'],S_selected_epoch=s['selected_epoch'],S_r=s['selected']['r'],B_r=b['selected']['r'],
             B_minus_S=b['selected']['r']-s['selected']['r'],S_last_r=s['epochs'][-1]['val']['r'],
             S_best_trained_epoch=best['epoch'],S_best_trained_r=best['val']['r'],
             S_epochs_above_anchor=sum(e['val']['r']>anchor['r'] for e in s['epochs'][1:]),same_epoch_B_wins=sum(v>0 for v in same),
             S_last_train_active_MSE=s['epochs'][-1]['train']['active_MSE'],S=decompose(anchor,s['selected']),B=decompose(anchor,b['selected']),
             S_vs_anchor_people=people_delta(s['selected'],anchor),B_vs_S_people=people_delta(b['selected'],s['selected']),
             duplicate_slots_exact=True,A_r=ridge['selection']['A']['r'],F_r=ridge['selection']['F']['r'])
    results.append(row)
summary={k:mean([r[k] for r in results]) for k in ('anchor_r','S_r','B_r','B_minus_S','S_last_r','S_best_trained_r','S_last_train_active_MSE','A_r','F_r')}
summary.update(S_minus_anchor=summary['S_r']-summary['anchor_r'],same_epoch_B_wins=sum(r['same_epoch_B_wins'] for r in results),
               S_pure_active_contribution=mean([r['S']['parts']['pure_active']['primary_contribution'] for r in results]),
               S_mixed_prefix_contribution=mean([r['S']['parts']['mixed_prefix']['primary_contribution'] for r in results]),
               S_selected_active_MSE=mean([r['S']['active_MSE'] for r in results]))
out=dict(passed=True,source_sha256=sha(Path(__file__)),sources=sources,results=results,summary=summary,
         scope='Saved-score descriptive comparison of fixed D23 S vs frozen D22 B and D21 controls. Primary endpoint and selection unchanged; split contributions retain original person denominators. No new fit, forward, significance, or independent test.',
         CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,NN_forwards=0,fits=0,goal_complete=False)
with (root/'COORDINATOR_SCIENTIFIC_REVIEW.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=True,summary=summary,results=results,CPU_seconds=out['CPU_seconds'],wall_seconds=out['wall_seconds'])))
