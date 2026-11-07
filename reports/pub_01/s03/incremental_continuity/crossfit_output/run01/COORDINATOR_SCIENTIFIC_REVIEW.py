"""Independent descriptive D24 synthesis from all twelve saved fits only."""
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_source_family_crossfit_20261106')
SEEDS = (1024, 2026, 31415)
ARMS = ('B_IS', 'S_IS', 'B_OOF', 'S_OOF')
PAIRS = (('B_OOF','B_IS'), ('B_OOF','S_OOF'), ('S_OOF','S_IS'), ('B_IS','S_IS'))
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
started, cpu = time.time(), time.process_time()
assert read(ROOT/'downstream_queue_v03/TERMINAL.json')['passed']
assert read(ROOT/'coordinator_downstream_audit_v02/COMPLETED.json')['passed']
assert read(ROOT/'COORDINATOR_DOWNSTREAM_FUNCTIONAL_REPLAY.json')['passed']
fits, provenance, per_seed = {}, {}, {}
for seed in SEEDS:
    row = {}
    for arm in ARMS:
        p = ROOT/f'downstream_formal/seed{seed}/{arm}/seed{seed}_P_past_prediction/RESULT.json'
        result = read(p)
        fits[seed,arm] = result
        provenance[str(p)] = sha(p)
        assert result['seed'] == seed and [x['epoch'] for x in result['epochs']] == list(range(-1,20))
        anchor = result['epochs'][0]['val']
        chosen, last = result['selected'], result['epochs'][-1]
        row[arm] = dict(selected_epoch=result['selected_epoch'], selected_r=chosen['r'],
            anchor_r=anchor['r'], selected_minus_anchor=chosen['r']-anchor['r'], last_r=last['val']['r'],
            last_minus_anchor=last['val']['r']-anchor['r'], selected_active_MSE=chosen['active_MSE'],
            anchor_active_MSE=anchor['active_MSE'], last_active_MSE=last['val']['active_MSE'],
            input_TRAIN_anchor_r=result['epochs'][0]['train']['r'],
            input_TRAIN_anchor_active_MSE=result['epochs'][0]['train']['active_MSE'],
            last_TRAIN_r=last['train']['r'], last_TRAIN_active_MSE=last['train']['active_MSE'],
            selected_near_minus_older=result['val_pair']['near']['r']-result['val_pair']['older']['r'],
            epochs_above_anchor=sum(x['val']['r'] > anchor['r'] for x in result['epochs'][1:]))
    assert len({row[arm]['anchor_r'] for arm in ARMS}) == 1
    for lane in ('IS','OOF'):
        assert fits[seed,f'B_{lane}']['epochs'][0] == fits[seed,f'S_{lane}']['epochs'][0]
    per_seed[str(seed)] = row
summary = {}
for arm in ARMS:
    fields = [key for key in per_seed[str(SEEDS[0])][arm] if key not in ('selected_epoch','epochs_above_anchor')]
    summary[arm] = {key:statistics.mean(per_seed[str(seed)][arm][key] for seed in SEEDS) for key in fields}
    summary[arm]['selected_epochs'] = [per_seed[str(seed)][arm]['selected_epoch'] for seed in SEEDS]
    summary[arm]['epochs_above_anchor_total_of_60'] = sum(per_seed[str(seed)][arm]['epochs_above_anchor'] for seed in SEEDS)
    summary[arm]['selected_r_seed_sample_std'] = statistics.stdev(per_seed[str(seed)][arm]['selected_r'] for seed in SEEDS)
comparisons = {}
for left,right in PAIRS:
    seeds = []
    for seed in SEEDS:
        a,b = fits[seed,left], fits[seed,right]
        pa = {x['subject']:x for x in a['selected']['people']}
        pb = {x['subject']:x for x in b['selected']['people']}
        assert set(pa) == set(pb) and len(pa) == 49
        people = {key:pa[key]['r']-pb[key]['r'] for key in pa}
        selected = a['selected']['r']-b['selected']['r']
        assert abs(statistics.mean(people.values())-selected) < 1e-12
        seeds.append(dict(seed=seed, selected_r_difference=selected,
            last_r_difference=a['epochs'][-1]['val']['r']-b['epochs'][-1]['val']['r'],
            selected_active_MSE_difference=a['selected']['active_MSE']-b['selected']['active_MSE'],
            same_epoch_r_wins_of_20=sum(x['val']['r'] > y['val']['r'] for x,y in zip(a['epochs'][1:],b['epochs'][1:])),
            subjects_improved_of_49=sum(value > 0 for value in people.values()), subject_r_differences=people))
    comparisons[f'{left}_minus_{right}'] = dict(per_seed=seeds,
        selected_r_difference_mean=statistics.mean(x['selected_r_difference'] for x in seeds),
        last_r_difference_mean=statistics.mean(x['last_r_difference'] for x in seeds),
        selected_active_MSE_difference_mean=statistics.mean(x['selected_active_MSE_difference'] for x in seeds),
        selected_positive_seeds=sum(x['selected_r_difference'] > 0 for x in seeds),
        last_positive_seeds=sum(x['last_r_difference'] > 0 for x in seeds),
        same_epoch_r_wins_of_60=sum(x['same_epoch_r_wins_of_20'] for x in seeds))
output = dict(passed=True, source_sha256=sha(Path(__file__)), result_sources=provenance,
    per_seed=per_seed, summary=summary, comparisons=comparisons,
    scope='Descriptive reuse of existing development scores only. Three seeds, same 49-person validation cohort and fixed full-TRAIN anchor. No independent generalization or confirmatory test.',
    qualifications=['B versus S changes both history input and parameter count (9441 versus 3361).',
        'OOF versus IS routes each TRAIN record through an excluded versus included source-family model; TRAIN normalizers are lane-specific. OOF is a known stacking safeguard, not evidence of architectural novelty.',
        'All validation subjects were present in TRAIN; this development cohort has been used repeatedly.',
        'Upstream models train on two of three source-family folds; validation input remains the frozen full-TRAIN upstream anchor.',
        'Selected and last/complete trajectories are both retained; no selection or training changed after observing these scores.'],
    new_neural_forward=0, new_prediction_or_target_arrays_read=0, new_optimizer_updates=0,
    CPU_seconds=time.process_time()-cpu, wall_seconds=time.time()-started)
with (ROOT/'COORDINATOR_SCIENTIFIC_REVIEW.json').open('x') as stream:
    json.dump(output,stream,indent=2,allow_nan=False)
print(json.dumps(dict(summary=summary,comparisons={k:{x:y for x,y in v.items() if x!='per_seed'} for k,v in comparisons.items()})))
