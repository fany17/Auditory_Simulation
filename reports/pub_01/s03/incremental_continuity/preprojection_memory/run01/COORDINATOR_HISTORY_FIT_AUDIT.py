"""Read-only paired saved-metric audit; execute only on 2203 after H completes."""
import argparse
import gzip
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
sys.path.insert(0, str(ROOT))
from COORDINATOR_COMPLETED_FIT_AUDIT import audit as completed_audit


def audit(seed):
    wall, cpu = time.time(), time.process_time()
    name = f'H_history_seed{seed}'
    basic = completed_audit(name)
    folder = ROOT / 'fits' / name

    def read(filename):
        data = (folder / filename).read_bytes()
        return json.loads(gzip.decompress(data) if filename.endswith('.gz') else data)

    selected = read('SELECTED_METRICS.json.gz')
    older = read('OLDER_METRICS.json.gz')
    checks, max_error = 0, 0.0

    def close(a, b):
        nonlocal checks, max_error
        assert math.isfinite(a) and math.isfinite(b)
        error = abs(a - b)
        assert error <= 1e-10, (a, b, error)
        checks += 1
        max_error = max(max_error, error)

    def mean(values):
        values = list(values)
        assert values
        return math.fsum(values) / len(values)

    assert len(selected['segments']) == len(older['segments']) == 752
    for key in ('people', 'blocks', 'records', 'windows'):
        assert selected['phases']['val'][key] == older['phases']['val'][key]
    assert selected['target_window_counts'] == older['target_window_counts']
    for near, far in zip(selected['segments'], older['segments']):
        for key in ('record', 'phase', 'subject', 'story', 'source_family', 'start', 'samples', 'family_mapping_basis'):
            assert near[key] == far[key], (key, near[key], far[key])
    people = defaultdict(list)
    for block in older['segments']:
        people[block['subject']].append(block)
        close(block['weighted_single_MSE'] - block['MSE'], block['disagreement'])
    assert len(people) == len(older['people']) == 49
    for person in older['people']:
        blocks = people[person['subject']]
        assert person['blocks'] == len(blocks)
        for field in ('r', 'MSE', 'weighted_single_MSE', 'disagreement'):
            close(person[field], mean(block[field] for block in blocks))
    for field in ('r', 'MSE', 'weighted_single_MSE', 'disagreement'):
        close(older['phases']['val'][field], mean(mean(block[field] for block in blocks) for blocks in people.values()))
    for family in older['source_family_descriptive']:
        groups = defaultdict(list)
        for block in older['segments']:
            if block['source_family'] == family['source_family']:
                groups[block['subject']].append(block['r'])
        assert family['people'] == len(groups) and family['blocks'] == sum(map(len, groups.values()))
        close(family['r'], mean(mean(values) for values in groups.values()))
    near_people = {p['subject']: p['r'] for p in selected['people']}
    far_people = {p['subject']: p['r'] for p in older['people']}
    assert near_people.keys() == far_people.keys()
    deltas = {subject: near_people[subject] - far_people[subject] for subject in sorted(near_people)}
    delta = selected['phases']['val']['r'] - older['phases']['val']['r']
    close(delta, mean(deltas.values()))
    trained = int(basic['selected_epoch'] >= 0)
    current = 24 * 252454 + (20 + trained) * 30500
    attention = 24 * 248620 + (20 + 2 * trained) * (30500 - 49 * 9)
    expected = dict(current_encoder_prefix_windows=current,
                    current_projection_windows=current + trained * 30500,
                    current_head_windows=current + trained * 30500,
                    frozen_encoder_prefix_windows=2 * attention,
                    attention_query_windows=attention, prior_projection_windows=0)
    assert basic['component_forward_counts'] == expected
    if not trained:
        assert selected == older
    return dict(fit=name, passed=True, basic_audit=basic,
                paired_scalar_comparisons=checks, paired_maximum_absolute_error=max_error,
                older_metric_sha256=hashlib.sha256((folder/'OLDER_METRICS.json.gz').read_bytes()).hexdigest(),
                audit_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                     (ROOT/'COORDINATOR_COMPLETED_FIT_AUDIT.py', ROOT/'COORDINATOR_HISTORY_FIT_AUDIT.py')},
                near_validation_r=selected['phases']['val']['r'], older_validation_r=older['phases']['val']['r'],
                near_minus_older_r=delta, per_person_near_minus_older_r=deltas,
                positive_people=sum(value > 0 for value in deltas.values()),
                selected_incumbent_structural_equality=not bool(trained),
                scope='One seed, saved block/person metrics and invoked-count formulas; paired current-query reuse verified in source, not independently replayed',
                limitations='Repeated-development VAL; older intervention changes content and recency with the same position slots, not pure delay; no complete-matrix, independent-test, or history-benefit claim',
                audit_CPU_seconds=time.process_time()-cpu, audit_wall_seconds=time.time()-wall,
                audit_NN_forwards=0, audit_backwards=0, audit_optimizer_updates=0, goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    print(json.dumps(audit(parser.parse_args().seed), allow_nan=False))
