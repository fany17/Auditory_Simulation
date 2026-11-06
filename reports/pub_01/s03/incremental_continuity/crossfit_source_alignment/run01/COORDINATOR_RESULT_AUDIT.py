"""Independently aggregate frozen D25 scalar results; execute only on 2203."""
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_crossfit_source_alignment_20261106')
SEEDS = (1024, 2026, 31415)
EPOCHS = (0, 19)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    assert math.isfinite(actual) and math.isfinite(expected), label
    assert abs(actual - expected) <= 2e-12, (label, actual, expected)


def main():
    started, cpu = time.monotonic(), time.process_time()
    paths = {name: ROOT / name for name in (
        'COMPLETED.json', 'FORMAL_GATE.json', 'VAL_ROUTING_METADATA.json',
        'ROUTING_AND_BUDGET_CANDIDATE.json', 'RECORD_SCORES.csv', 'RESULTS.csv',
    )}
    done, gate = read(paths['COMPLETED.json']), read(paths['FORMAL_GATE.json'])
    assert done['passed'] and done['no_fit']
    assert done['new_optimizer_updates'] == done['new_statistics_fits'] == done['old_test_reads'] == 0
    assert done['gate_sha256'] == digest(paths['FORMAL_GATE.json'])
    for path, expected in gate['code_and_JSON'].items():
        assert digest(Path(path)) == expected, path
    for group in ('OOF_fixed_weights', 'upstream_payloads'):
        for path, pin in gate[group].items():
            assert digest(Path(path)) == pin['sha256'], path
    route = read(paths['VAL_ROUTING_METADATA.json'])['records']
    candidate = read(paths['ROUTING_AND_BUDGET_CANDIDATE.json'])
    full_score_sources = {}
    for fixed in candidate['weights']:
        source = Path(fixed['path']).parents[1] / 'RESULT.json'
        history = read(source)['epochs']
        assert [x['epoch'] for x in history] == list(range(-1, 20))
        point = history[fixed['epoch'] + 1]
        close(fixed['full_anchor_VAL_score'], history[0]['val']['r'], 'saved full anchor')
        close(fixed['full_model_VAL_score'], point['val']['r'], 'saved fixed full model')
        close(fixed['full_model_VAL_active_MSE'], point['val']['active_MSE'], 'saved fixed full MSE')
        full_score_sources[str(source)] = digest(source)
    assert len(route) == len({r['record'] for r in route}) == len({r['subject'] for r in route}) == 49
    subsets = {
        'all': route,
        'shared': [r for r in route if r['shared_train_family']],
        'unseen': [r for r in route if not r['shared_train_family']],
    }
    assert len(subsets['shared']) == 29 and len(subsets['unseen']) == 20
    with paths['RECORD_SCORES.csv'].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with paths['RESULTS.csv'].open(newline='') as stream:
        reported = list(csv.DictReader(stream))
    assert len(rows) == 1335 and len(reported) == 36
    actual = {}
    for r in rows:
        key = (int(r['seed']), r['variant'], int(r['epoch']) if r['epoch'] else None,
               r['record'], int(r['fold']))
        assert key not in actual, key
        actual[key] = r
    reported_map = {}
    for r in reported:
        key = (int(r['seed']), r['variant'], int(r['epoch']), r['group'])
        assert key not in reported_map
        reported_map[key] = r

    def aggregate(seed, variant, epoch, subset):
        people, mse, sse, points = [], [], 0., 0
        for record in subset:
            peers = [actual[seed, variant, epoch, record['record'], f] for f in record['folds']]
            for peer in peers:
                assert peer['subject'] == record['subject'] and peer['family'] == record['source_family']
                assert int(peer['active_points']) == record['active_points']
            # Each person has one record. Fold averaging occurs within this record,
            # without multiplying its person weight or averaging predicted waveforms.
            people.append(mean(float(p['r']) for p in peers))
            mse.append(mean(float(p['MSE']) for p in peers))
            sse += record['active_points'] * mean(float(p['active_MSE']) for p in peers)
            points += record['active_points']
        return dict(r=mean(people), MSE=mean(mse), active_MSE=sse / points,
                    active_points=points, people=len(subset), blocks=sum(r['VAL_blocks'] for r in subset))

    aggregates, comparisons = [], []
    for seed in SEEDS:
        for group, subset in subsets.items():
            anchor = aggregate(seed, 'anchor', None, subset)
            for epoch in EPOCHS:
                arms = {v: aggregate(seed, v, epoch, subset) for v in ('B', 'S')}
                for variant, metrics in arms.items():
                    official = reported_map[seed, variant, epoch, group]
                    close(float(official['fold_anchor_r']), anchor['r'], 'anchor r')
                    close(float(official['fold_anchor_active_MSE']), anchor['active_MSE'], 'anchor MSE')
                    close(float(official['fold_model_r']), metrics['r'], 'model r')
                    close(float(official['fold_model_active_MSE']), metrics['active_MSE'], 'model MSE')
                    difference = metrics['r'] - anchor['r']
                    close(float(official['fold_delta_r']), difference, 'fold delta')
                    for name in ('people', 'blocks', 'active_points'):
                        assert int(official[name]) == metrics[name], name
                    full = next(x for x in candidate['weights'] if
                                (x['seed'], x['variant'], x['epoch']) == (seed, variant, epoch))
                    if group == 'all':
                        full_delta = full['full_model_VAL_score'] - full['full_anchor_VAL_score']
                        close(float(official['full_r']), full['full_model_VAL_score'], 'full model')
                        close(float(official['full_delta_r']), full_delta, 'full delta')
                        close(float(official['delta_of_deltas']), difference - full_delta, 'difference of differences')
                    else:
                        assert all(official[name] == '' for name in ('full_r', 'full_delta_r', 'delta_of_deltas'))
                    aggregates.append(dict(seed=seed, group=group, epoch=epoch, variant=variant,
                                           anchor=anchor, model=metrics))
                b, s = arms['B'], arms['S']
                full = next(x for x in candidate['weights'] if
                            (x['seed'], x['variant'], x['epoch']) == (seed, 'B', epoch))
                comparisons.append(dict(seed=seed, epoch=epoch, group=group,
                    B_minus_fold_anchor=b['r'] - anchor['r'], S_minus_fold_anchor=s['r'] - anchor['r'],
                    B_minus_S=b['r'] - s['r'],
                    B_minus_full_strong_anchor=b['r'] - full['full_anchor_VAL_score'] if group == 'all' else None,
                    B_delta_change=b['r'] - anchor['r'] -
                        (full['full_model_VAL_score'] - full['full_anchor_VAL_score']) if group == 'all' else None))
    counters = done['counters']
    for name, expected in dict(encoder_windows=170634, encoder_batches=2673,
                              downstream_active_points=64189524, logical_fold_readout_routes=36,
                              current_stem_points=64189524, context_output_points=64189524).items():
        assert counters[name] == expected, (name, counters.get(name), expected)
    all_six = [c for c in comparisons if c['group'] == 'all']
    summary = {key: dict(mean=mean(c[key] for c in all_six),
                        positive_fixed_seed_epoch_pairs=sum(c[key] > 0 for c in all_six), total=6)
               for key in ('B_minus_fold_anchor', 'S_minus_fold_anchor', 'B_minus_S',
                           'B_minus_full_strong_anchor', 'B_delta_change')}
    result = dict(passed=True, source_sha256=digest(Path(__file__)),
        inputs={str(p): digest(p) for p in paths.values()}, full_score_sources=full_score_sources,
        aggregates=aggregates,
        comparisons=comparisons, summary=summary, actual_counters=counters,
        new_neural_forward=0, new_signal_arrays=0, new_fits=0,
        CPU_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-started,
        scope='Independent aggregation of all fixed D25 scalar record/fold results and verification of frozen identities. Does not replace bounded functional checks of encoder, readout or block scoring.',
        scientific_limitations=[
            'Repeated development VAL; no independent confirmation or ICML/NeurIPS-level completion.',
            'Same weights evaluated at fixed epochs 0 and 19, without selection or fallback.',
            'B/S have different function classes and parameter counts.',
            'Fold models differ from full-TRAIN models in fit data and prediction strength; source mismatch is not uniquely identified.',
            'A gain over a weaker fold anchor is not by itself a gain over the strong full-TRAIN baseline.',
        ])
    with (ROOT/'COORDINATOR_RESULT_AUDIT.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, summary=summary), indent=2))


if __name__ == '__main__':
    main()
