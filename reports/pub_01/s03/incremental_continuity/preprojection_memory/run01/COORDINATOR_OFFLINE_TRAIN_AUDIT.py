"""Read-only saved fixed-checkpoint metric audit on 2203; no model/data replay."""
import argparse
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
PROTOCOL = '255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632'
EPOCHS = (-1, 0, 4, 9, 19)
FIELDS = ('r', 'MSE', 'weighted_single_MSE', 'disagreement')


def audit(name):
    wall, cpu = time.time(), time.process_time()
    folder = ROOT/'fits'/name
    completed_bytes = (folder/'COMPLETED.json').read_bytes()
    completed = json.loads(completed_bytes)
    assert not (folder/'FAILURE.json').exists()
    assert completed['protocol_sha256'] == PROTOCOL
    assert hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest() == PROTOCOL
    history_bytes = (folder/'HISTORY.jsonl').read_bytes()
    history = [json.loads(line) for line in history_bytes.splitlines()]
    assert len(history) == 20 and [row['epoch'] for row in history] == list(range(20))
    incumbent_bytes = (folder/'INCUMBENT.json.gz').read_bytes()
    incumbent = json.loads(gzip.decompress(incumbent_bytes))
    checks, max_error = 0, 0.0
    hashes, curves, coordinates, prefixes, anchor_counts = {}, [], None, None, None

    def close(actual, expected):
        nonlocal checks, max_error
        assert math.isfinite(actual) and math.isfinite(expected)
        error = abs(actual-expected)
        assert error <= 1e-10*max(1.0, abs(expected)), (actual, expected, error)
        checks += 1
        max_error = max(max_error, error)

    def mean(values):
        values = list(values)
        assert values
        return math.fsum(values)/len(values)

    for epoch in EPOCHS:
        filename = f'OFFLINE_TRAIN_epoch{epoch}.json.gz'
        raw = (folder/filename).read_bytes()
        hashes[filename] = hashlib.sha256(raw).hexdigest()
        metric = json.loads(gzip.decompress(raw))
        assert metric['parameters_fixed'] is True and metric['new_prediction_disk_arrays'] == 0
        assert metric['weighted'] is None and set(metric['phases']) == {'train'}
        phase = metric['phases']['train']
        records, segments, people = metric['records'], metric['segments'], metric['people']
        assert len(records) == phase['records'] == 426
        assert len(segments) == phase['blocks'] == 6104
        assert len(people) == phase['people'] == 70
        assert sum(row['windows'] for row in records) == phase['windows'] == 252454
        assert segments == [block for row in records for block in row['segments']]
        identity_keys = ('record', 'phase', 'subject', 'story', 'source_family', 'start', 'samples', 'family_mapping_basis')
        current_coordinates = [tuple(block[key] for key in identity_keys) for block in segments]
        current_prefixes = {row['record']: row['boundary_tail']['prefix'] for row in records}
        if epoch == -1:
            coordinates, prefixes, anchor_counts = current_coordinates, current_prefixes, metric['target_window_counts']
        else:
            assert current_coordinates == coordinates
            assert current_prefixes == prefixes, 'The unchanged fallback-only prefix must retain its metrics'
            assert metric['target_window_counts'] == anchor_counts
        assert anchor_counts['windows'] == 252454
        groups = defaultdict(list)
        for block in segments:
            assert block['phase'] == 'train'
            groups[block['subject']].append(block)
            close(block['weighted_single_MSE']-block['MSE'], block['disagreement'])
        assert set(groups) == {person['subject'] for person in people}
        for person in people:
            blocks = groups[person['subject']]
            assert person['blocks'] == len(blocks)
            assert person['records'] == len({block['record'] for block in blocks})
            for field in FIELDS:
                close(person[field], mean(block[field] for block in blocks))
        for field in FIELDS:
            close(phase[field], mean(mean(block[field] for block in blocks) for blocks in groups.values()))
        for family in metric['source_family_descriptive']:
            family_groups = defaultdict(list)
            for block in segments:
                if block['source_family'] == family['source_family']:
                    family_groups[block['subject']].append(block['r'])
            assert family['people'] == len(family_groups)
            assert family['blocks'] == sum(map(len, family_groups.values()))
            close(family['r'], mean(mean(values) for values in family_groups.values()))
        assert len(metric['source_family_descriptive']) == 37
        windows = phase['windows']
        undefined = sum(row['window_r_undefined'] for row in records)
        assert undefined == phase['window_r_undefined']
        assert phase['window_r_defined'] == windows-undefined
        if undefined:
            assert phase['r3s'] is None
        else:
            close(phase['r3s'], math.fsum(row['r3s']*row['windows'] for row in records)/windows)
        close(phase['window_MSE'], math.fsum(row['window_MSE']*row['windows'] for row in records)/windows)
        pairs = sum(row['overlap_pairs'] for row in records)
        assert pairs == phase['overlap_pairs'] == windows-len(records)
        assert phase['valid_normalized_pairs'] == sum(row['valid_normalized_pairs'] for row in records)
        close(phase['overlap_MSE'], math.fsum(row['overlap_SSE_mean_sum'] for row in records)/pairs)
        denominator = math.fsum(row['normalized_overlap_den'] for row in records)
        if denominator > 0:
            close(phase['overlap_NRMSE'], math.sqrt(math.fsum(row['normalized_overlap_num'] for row in records)/denominator))
        else:
            assert phase['overlap_NRMSE'] is None
        close(metric['window_prediction_mean'], math.fsum(row['window_prediction_mean']*row['windows'] for row in records)/windows)
        close(metric['window_prediction_RMS'], math.sqrt(math.fsum(row['window_prediction_RMS']**2*row['windows'] for row in records)/windows))
        for key in ('maximum_window_abs', 'maximum_fused_abs'):
            close(metric[key], max(row['boundary_tail'][key] for row in records))
        close(metric['Hann_identity_max_abs'], max(row['Hann_identity_max_abs'] for row in records))
        assert metric['boundary_tail_records'] == [dict(record=row['record'], **row['boundary_tail']) for row in records]
        assert math.isfinite(metric['negative_cosine']) and -1-1e-12 <= metric['negative_cosine'] <= 1+1e-12
        val = incumbent['phases']['val'] if epoch == -1 else history[epoch]['validation']
        curves.append(dict(epoch=epoch, train_fixed=phase, val_same_epoch=val,
                           negative_cosine=metric['negative_cosine'],
                           window_prediction_mean=metric['window_prediction_mean'],
                           window_prediction_RMS=metric['window_prediction_RMS'],
                           maximum_window_abs=metric['maximum_window_abs'],
                           maximum_fused_abs=metric['maximum_fused_abs'],
                           Hann_identity_max_abs=metric['Hann_identity_max_abs'],
                           train_minus_val_r=phase['r']-val['r']))
    return dict(fit=name, passed=True, protocol_sha256=PROTOCOL,
                fixed_epochs=list(EPOCHS), scalar_comparisons=checks, maximum_absolute_error=max_error,
                selected_epoch=completed['selected_epoch'],
                selected_epoch_has_fixed_TRAIN=completed['selected_epoch'] in EPOCHS,
                fallback_prefix_unchanged_all_fixed_epochs=True,
                target_counts_unchanged=anchor_counts, curves=curves,
                input_sha256=dict(offline=hashes,
                                  completed=hashlib.sha256(completed_bytes).hexdigest(),
                                  history=hashlib.sha256(history_bytes).hexdigest(),
                                  incumbent=hashlib.sha256(incumbent_bytes).hexdigest()),
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                audit_CPU_seconds=time.process_time()-cpu, audit_wall_seconds=time.time()-wall,
                audit_NN_forwards=0, audit_backwards=0, audit_optimizer_updates=0,
                scope='Saved fixed-checkpoint TRAIN block/person and auxiliary aggregation only; no prediction/target replay. Same-epoch VAL copied from existing frozen history. Online training trajectory is not fixed-checkpoint performance.',
                limitations='Only fixed epochs -1/0/4/9/19 exist; no interpolation or selected-epoch TRAIN claim if missing. TRAIN/VAL differ in participants/records/sources, so their gap alone does not identify a mechanism.',
                goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fits', nargs='+')
    args = parser.parse_args()
    print(json.dumps([audit(name) for name in args.fits], allow_nan=False))
