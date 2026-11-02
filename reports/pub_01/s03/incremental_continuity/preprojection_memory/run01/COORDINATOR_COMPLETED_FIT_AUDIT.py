"""Read-only D18 completed-fit audit, standard-library metadata only on 2203.

This recomputes reported aggregation from saved block metrics, not predictions.
It adds no neural execution and does not certify an independent test or benefit.
"""
import argparse
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
PROTOCOL_SHA = '255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632'


def audit(name):
    wall, cpu = time.time(), time.process_time()
    folder = ROOT / 'fits' / name
    assert folder.is_dir() and (folder / 'COMPLETED.json').is_file()
    assert not (folder / 'FAILURE.json').exists()
    hashes, checks, max_error = {}, 0, 0.0

    def read(filename, lines=False):
        data = (folder / filename).read_bytes()
        hashes[filename] = hashlib.sha256(data).hexdigest()
        if filename.endswith('.gz'):
            data = gzip.decompress(data)
        return [json.loads(x) for x in data.splitlines()] if lines else json.loads(data)

    def close(a, b, tolerance=1e-10):
        nonlocal checks, max_error
        assert math.isfinite(a) and math.isfinite(b)
        error = abs(a - b)
        assert error <= tolerance, (a, b, error)
        max_error = max(max_error, error)
        checks += 1

    def mean(values):
        values = list(values)
        assert values
        return math.fsum(values) / len(values)

    completed = read('COMPLETED.json')
    started = read('STARTED.json')
    history = read('HISTORY.jsonl', True)
    permutations = read('PERMUTATIONS.jsonl', True)
    incumbent = read('INCUMBENT.json.gz')
    selected = read('SELECTED_METRICS.json.gz')
    counts = read('COMPONENT_COUNTS.json')
    protocol_data = (ROOT / 'PROTOCOL.json').read_bytes()
    assert hashlib.sha256(protocol_data).hexdigest() == PROTOCOL_SHA
    protocol = json.loads(protocol_data)
    assert started['protocol_sha256'] == completed['protocol_sha256'] == PROTOCOL_SHA
    assert started['seed'] == completed['seed'] and started['arm'] == completed['arm']
    assert name == f"{completed['arm']}_seed{completed['seed']}"
    assert len(history) == len(permutations) == completed['epochs'] == 20
    for epoch, (row, permutation) in enumerate(zip(history, permutations)):
        assert row['epoch'] == permutation['epoch'] == epoch
        assert row['optimizer_updates'] == row['update_positions'] == (epoch + 1) * 3945
        assert row['empty_positions'] == 0
        assert row['target_TRAIN_windows'] == permutation['windows'] == 252454
        assert row['effective_TRAIN_windows'] == 248620
        assert permutation['fallback_presentations'] == 3834
        frozen = [x for x in protocol['permutations'] if x['seed'] == completed['seed'] and x['epoch'] == epoch]
        assert len(frozen) == 1 and frozen[0]['sha256'] == permutation['sha256']
    assert completed['optimizer_updates'] == completed['update_positions'] == 78900
    assert completed['empty_positions'] == 0
    assert completed['target_window_presentations'] == 20 * 252454
    assert completed['effective_TRAIN_presentations'] == 20 * 248620
    assert completed['registered_trainable_parameters'] == protocol['registered_parameters'][completed['arm']]
    best = incumbent['selection_reference_r_exact_from_frozen_D16']
    best_epoch = -1
    for row in history:
        if row['validation']['r'] > best:
            best, best_epoch = row['validation']['r'], row['epoch']
        assert row['selected_epoch_so_far'] == best_epoch
    assert completed['selected_epoch'] == best_epoch
    close(completed['selected_validation_r'], best)
    close(selected['phases']['val']['r'], best)
    close(incumbent['phases']['val']['r'], incumbent['selection_reference_r_exact_from_frozen_D16'])
    phase = selected['phases']['val']
    people = selected['people']
    segments = selected['segments']
    assert len(people) == phase['people'] == 49
    assert len(segments) == phase['blocks'] == 752
    assert phase['records'] == 49 and phase['windows'] == 30500
    grouped = defaultdict(list)
    for block in segments:
        assert block['phase'] == 'val'
        grouped[block['subject']].append(block)
        close(block['weighted_single_MSE'] - block['MSE'], block['disagreement'])
    assert len(grouped) == 49
    for person in people:
        blocks = grouped[person['subject']]
        assert person['blocks'] == len(blocks)
        for field in ('r', 'MSE', 'weighted_single_MSE', 'disagreement'):
            close(person[field], mean(b[field] for b in blocks))
    for field in ('r', 'MSE', 'weighted_single_MSE', 'disagreement'):
        close(phase[field], mean(mean(b[field] for b in blocks) for blocks in grouped.values()))
    for family in selected['source_family_descriptive']:
        subset = defaultdict(list)
        for block in segments:
            if block['source_family'] == family['source_family']:
                subset[block['subject']].append(block['r'])
        assert family['people'] == len(subset)
        assert family['blocks'] == sum(map(len, subset.values()))
        close(family['r'], mean(mean(values) for values in subset.values()))
    for key, total in counts['total'].items():
        assert total == sum(row[key] for row in counts['stages'])
        assert total == completed['component_forward_counts'][key]
    if completed['arm'] == 'C_current':
        current = 24 * 252454 + (20 + int(best_epoch >= 0)) * 30500
        effective = 24 * 248620 + (20 + int(best_epoch >= 0)) * (30500 - 49 * 9)
        for field in ('current_encoder_prefix_windows', 'current_projection_windows', 'current_head_windows'):
            assert counts['total'][field] == current
        for field in ('frozen_encoder_prefix_windows', 'attention_query_windows'):
            assert counts['total'][field] == effective
        assert counts['total']['prior_projection_windows'] == 0
    return dict(fit=name, passed=True, protocol_sha256=PROTOCOL_SHA,
                scope='One completed fit; saved-metric reaggregation and budget/selection metadata only; no prediction replay or matrix-level conclusion',
                scalar_comparisons=checks, maximum_absolute_error=max_error,
                selected_epoch=best_epoch, incumbent_validation_r=incumbent['phases']['val']['r'],
                selected_validation_r=best, last_validation_r=history[-1]['validation']['r'],
                optimizer_updates=completed['optimizer_updates'], component_forward_counts=counts['total'],
                input_sha256=hashes, audit_NN_forwards=0, audit_backwards=0, audit_optimizer_updates=0,
                audit_CPU_seconds=time.process_time()-cpu, audit_wall_seconds=time.time()-wall,
                limitations='Seen-participant, mixed-source, repeatedly developed VAL; checkpoint contents and predictions not independently replayed; other fits pending',
                goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fit')
    print(json.dumps(audit(parser.parse_args().fit), allow_nan=False))
