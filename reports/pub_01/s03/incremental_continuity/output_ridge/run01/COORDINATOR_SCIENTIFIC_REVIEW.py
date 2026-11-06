"""Saved-block scientific audit: prefix and active contributions, no predictions or fits."""
import json, hashlib, math, time
from pathlib import Path
from collections import defaultdict

root=Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_output_ridge_20261106')
assert Path(__file__).resolve().parent==root
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
start=time.monotonic(); cpu=time.process_time()
audit=read(root/'COORDINATOR_TERMINAL_AUDIT.json'); assert audit['passed']
sources={str(root/'COORDINATOR_TERMINAL_AUDIT.json'):sha(root/'COORDINATOR_TERMINAL_AUDIT.json')}
outputs=[]
def mean(xs):
    return math.fsum(xs)/len(xs)

def decompose(anchor,metric):
    baseline={(r['record'],r['start']):r for r in anchor['blocks']}
    people=defaultdict(list)
    for b in metric['blocks']:
        a=baseline[(b['record'],b['start'])]
        assert b['samples']==a['samples'] and b['subject']==a['subject']
        people[b['subject']].append((b['start']<1344,b['r']-a['r']))
    assert len(people)==49
    full=mean([mean([v for _,v in bs]) for bs in people.values()])
    assert abs(full-(metric['r']-anchor['r']))<1e-12
    pieces={}
    for label,mixed in [('mixed_prefix',True),('pure_active',False)]:
        contribution=mean([math.fsum(v for flag,v in bs if flag==mixed)/len(bs) for bs in people.values()])
        by_person=[[v for flag,v in bs if flag==mixed] for bs in people.values()]
        assert all(by_person)
        all_values=[v for bs in by_person for v in bs]
        pieces[label]=dict(blocks=len(all_values),macro_within_group_delta=mean([mean(bs) for bs in by_person]),
                           unweighted_block_delta=mean(all_values),contribution_to_primary_delta=contribution,
                           max_abs_block_delta=max(map(abs,all_values)),people_positive=sum(mean(bs)>0 for bs in by_person))
    assert abs(sum(v['contribution_to_primary_delta'] for v in pieces.values())-full)<1e-12
    assert pieces['mixed_prefix']['blocks']==49 and pieces['pure_active']['blocks']==703
    return dict(primary_delta=full,parts=pieces,active_MSE=metric['active_MSE'],active_MSE_delta=metric['active_MSE']-anchor['active_MSE'])

for seed in (1024,2026,31415):
    path=root/'fit'/f'seed{seed}_METRICS.json'; x=read(path); sources[str(path)]=sha(path)
    path=root.parent/'s03_recent_prediction_context_20261106'/'fit'/f'seed{seed}_P_past_prediction'/'RESULT.json'
    d20=read(path); sources[str(path)]=sha(path)
    selected={k:x['models'][x['selection'][k]['selected']] for k in ('A','F')}
    assert all(v['val']['r']<d20['selected']['r'] for v in x['models'].values())
    result=dict(seed=seed,anchor_r=x['anchor']['r'],anchor_train_MSE=x['train_anchor_active_MSE'],anchor_val_MSE=x['anchor']['active_MSE'],
                D20=decompose(x['anchor'],d20['selected']),grid_below_D20=6)
    for role in ('A','F'):
        result[role]=decompose(x['anchor'],selected[role]['val'])
        result[role]['train_MSE']=selected[role]['train_active_MSE']
        result[role]['r']=selected[role]['val']['r']
    outputs.append(result)
summary={role:{'mean_delta':mean([v[role]['primary_delta'] for v in outputs]),
               'mixed_prefix_contribution':mean([v[role]['parts']['mixed_prefix']['contribution_to_primary_delta'] for v in outputs]),
               'pure_active_contribution':mean([v[role]['parts']['pure_active']['contribution_to_primary_delta'] for v in outputs]),
               'pure_active_macro_delta':mean([v[role]['parts']['pure_active']['macro_within_group_delta'] for v in outputs]),
               'active_MSE_mean':mean([v[role]['active_MSE'] for v in outputs])} for role in ('A','F','D20')}
out=dict(passed=True,source_sha256=sha(Path(__file__)),sources=sources,results=outputs,summary=summary,
         scope='Descriptive decomposition of frozen selected-model saved blocks; preserves original macro-person weights and original endpoint. No new selection, model output, fit, external test, or significance claim.',
         CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,NN_forwards=0,fits=0,goal_complete=False)
with (root/'COORDINATOR_SCIENTIFIC_REVIEW.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(passed=True,summary=summary,CPU_seconds=out['CPU_seconds'],wall_seconds=out['wall_seconds'])))
