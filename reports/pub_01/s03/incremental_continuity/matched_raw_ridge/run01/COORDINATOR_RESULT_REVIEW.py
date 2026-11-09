"""Independent D28 saved-statistics/score audit. Run only on server2203.

No signal reads, new regression fit, eigendecomposition, neural forward or update.
The numerical work is restricted to already saved moments, coefficients and scores.
"""
import csv
import hashlib
import json
import math
import os
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_matched_raw_ridge_20261107')
F = ROOT / 'formal'
ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, .01, .1, 1., 10., 100.)
PROTOCOL_SHA = 'b9942886d0ddaa90919e7c9409dfd057003714dfe3bbb21b6b8427df54299500'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as stored:
        return {k: stored[k] for k in stored.files}


def mean(values):
    return math.fsum(values) / len(values)


def aggregate(blocks):
    people = defaultdict(list)
    for b in blocks:
        people[b['subject']].append(b)
    return {k: mean([mean([b[k] for b in rows]) for rows in people.values()])
            for k in ('r', 'MSE')}


def compare(a, b, tolerance=1e-12):
    assert all(math.isclose(a[k], b[k], abs_tol=tolerance, rel_tol=0.) for k in ('r', 'MSE')), (a, b)


def stamp(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def audit():
    assert ROOT.is_dir() and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started, cpu = time.monotonic(), time.process_time()
    assert sha(ROOT / 'PROTOCOL.json') == PROTOCOL_SHA
    gate, inv, complete = read(ROOT / 'PROTOCOL.json'), read(ROOT / 'INVENTORY.json'), read(F / 'COMPLETED.json')
    assert gate['inventory_sha256'] == sha(ROOT / 'INVENTORY.json')
    assert all(sha(p) == s for p, s in gate['sources'].items())
    assert all(stamp(p) == s for p, s in inv['inputs'].items())
    assert complete['passed'] and complete['protocol_sha256'] == PROTOCOL_SHA
    assert complete['TRAIN_records'] == 426 and complete['TRAIN_points'] == 24297126
    assert complete['per_record_G_moment_calls'] == 6176
    assert complete['eigensystems'] == 50 and complete['ridge_candidates'] == 450
    assert complete['VAL_candidate_blocks'] == 13536 and complete['VAL_eligible_blocks'] == 752
    assert all(complete[k] == 0 for k in ('gpu', 'new_neural_forwards', 'new_neural_optimizations', 'old_test_reads'))
    fp = {'protocol_sha256': PROTOCOL_SHA, 'inventory_sha256': sha(ROOT / 'INVENTORY.json')}
    train_by_id = {r['record']: r for r in inv['train']}
    assert len(train_by_id) == 426
    accumulated, ledger = None, []
    moment_fields = ('n', 'gram', 'x', 'xy', 'y', 'yy')
    assert len(list((F / 'stats').glob('*.npz'))) == 71
    for person, expected_ids in inv['groups'].items():
        a = archive(F / 'stats' / (person + '.npz'))
        meta = json.loads(str(a['metadata']))
        ids = [r['record'] for r in meta['record_ledger']]
        assert meta['fingerprint'] == fp and ids == expected_ids
        assert int(a['n']) == sum(train_by_id[i]['common_samples'] for i in ids)
        assert all(r['target_times'] == train_by_id[r['record']]['common_samples'] for r in meta['record_ledger'])
        ledger.extend(meta['record_ledger'])
        if accumulated is None:
            accumulated = {k: np.zeros_like(a[k]) for k in moment_fields}
            accumulated['minimum'] = np.full(2112, np.inf)
            accumulated['maximum'] = np.full(2112, -np.inf)
        for k in moment_fields:
            accumulated[k] += a[k]
        accumulated['minimum'] = np.minimum(accumulated['minimum'], a['minimum'])
        accumulated['maximum'] = np.maximum(accumulated['maximum'], a['maximum'])
    g = archive(F / 'stats/G.npz')
    assert all(np.array_equal(g[k], v) for k, v in accumulated.items())
    assert ledger == read(F / 'MOMENT_LEDGER.json')['record_ledger']
    assert [r['record'] for r in ledger] == [r['record'] for r in inv['train']]
    del accumulated, g, a

    fits = ['G', *inv['personal_subjects']]
    assert len(fits) == 50 and len(list((F / 'coefficients').glob('*.npz'))) == 50
    selected = read(F / 'SELECTED_VAL.json')
    selected_coefficients = archive(F / 'SELECTED_COEFFICIENTS.npz')
    kkt_max, coefficient_costs, const_counts = 0., [], {}
    for fit in fits:
        source = F / 'stats' / (fit + '.npz')
        a = archive(source)
        c = archive(F / 'coefficients' / (fit + '.npz'))
        m = json.loads(str(c['metadata']))
        assert m['fingerprint'] == fp and m['stats_stamp'] == stamp(source)
        n = int(a['n'])
        assert m['TRAIN_samples'] == n and m['decompositions'] == 1 and m['ridge_candidates'] == 9
        assert np.array_equal(c['alphas'], ALPHAS) and c['weights'].shape == (2112, 9)
        mu, ym = a['x'] / n, float(a['y']) / n
        gram = np.tril(a['gram'])
        gram = gram + np.tril(gram, -1).T
        cov = gram / n - np.outer(mu, mu)
        constant = a['minimum'] == a['maximum']
        xs = np.sqrt(np.where(constant, 1., cov.diagonal()))
        ys = math.sqrt(float(a['yy']) / n - ym * ym)
        assert np.array_equal(c['constant'], constant)
        assert np.array_equal(c['design_mean'], mu) and np.array_equal(c['design_scale'], xs)
        assert float(c['target_mean']) == ym and float(c['target_scale']) == ys
        cov[constant, :] = 0.
        cov[:, constant] = 0.
        corr = cov / xs[:, None] / xs[None, :]
        cross = (a['xy'] / n - mu * ym) / xs / ys
        cross[constant] = 0.
        w = c['weights']
        residual = np.max(np.abs(corr @ w + w * np.asarray(ALPHAS) - cross[:, None]), axis=0)
        assert np.isfinite(residual).all() and residual.max() < 1e-7 * max(1., float(abs(cross).max()))
        assert np.allclose(c['raw_weights'], ys * w / xs[:, None], rtol=0., atol=1e-12)
        assert np.allclose(c['raw_intercepts'], ym - mu @ c['raw_weights'], rtol=0., atol=1e-12)
        kkt_max = max(kkt_max, float(residual.max()))
        const_counts[fit] = int(constant.sum())
        coefficient_costs.append({k: m[k] for k in ('fit', 'CPU_seconds', 'elapsed_seconds', 'TRAIN_samples')})
        alpha = selected['alpha_by_fit'][fit]
        j = ALPHAS.index(alpha)
        for key in ('design_mean', 'design_scale', 'target_mean', 'target_scale', 'constant'):
            assert np.array_equal(selected_coefficients[f'{fit}_{key}'], c[key])
        for key in ('weights', 'raw_weights'):
            assert np.array_equal(selected_coefficients[f'{fit}_{key}'], c[key][:, j])
        assert selected_coefficients[f'{fit}_raw_intercept'] == c['raw_intercepts'][j]
        assert selected_coefficients[f'{fit}_alpha'] == alpha
    del a, c, gram, cov, corr, w, selected_coefficients

    mask_path = next(p for p in gate['sources'] if p.endswith('block_eligibility/val.json'))
    mask = read(mask_path)
    support = {(b['record'], int(b['start']), int(b['samples'])) for b in mask['blocks'] if b['eligible']}
    assert len(support) == 752
    val_meta = {r['record']: r for r in inv['val']}
    all_rows = list(csv.DictReader((F / 'VAL_BLOCKS.csv').open(newline='')))
    assert len(all_rows) == 13536
    combinations, grid = defaultdict(list), defaultdict(list)
    for b in all_rows:
        b['alpha'], b['r'], b['MSE'] = (float(b[k]) for k in ('alpha', 'r', 'MSE'))
        assert b['alpha'] in ALPHAS and b['kind'] in ('G', 'I')
        assert all(math.isfinite(b[k]) for k in ('r', 'MSE')) and -1 <= b['r'] <= 1 and b['MSE'] >= 0
        original = val_meta[b['record']]
        assert all(b[k] == original[k] for k in ('subject', 'story', 'source_family'))
        combinations[b['kind'], b['alpha']].append(b)
        grid['G' if b['kind'] == 'G' else b['subject'], b['alpha']].append(b)
    assert len(combinations) == 18 and len(grid) == 450
    for rows in combinations.values():
        keys = [(b['record'], int(b['start']), int(b['samples'])) for b in rows]
        assert len(keys) == 752 and set(keys) == support
    rebuilt = {key: aggregate(rows) for key, rows in grid.items()}
    candidates = list(csv.DictReader((F / 'CANDIDATE_SCORES.csv').open(newline='')))
    assert len(candidates) == 450
    native = {}
    for b in candidates:
        key = b['fit'], float(b['alpha'])
        assert key not in native
        native[key] = {k: float(b[k]) for k in ('r', 'MSE')}
        compare(rebuilt[key], native[key])
    assert set(native) == set(rebuilt)
    for fit in fits:
        chosen = selected['alpha_by_fit'][fit]
        assert max(ALPHAS, key=lambda a: native[fit, a]['r']) == chosen
        assert rebuilt[fit, chosen]['r'] >= max(rebuilt[fit, a]['r'] for a in ALPHAS) - 1e-12
    selected_rows = [b for b in all_rows if b['alpha'] == selected['alpha_by_fit']['G' if b['kind'] == 'G' else b['subject']]]
    metrics, strata, per_person = {}, {}, {}
    families = set(inv['train_source_families'])
    for kind in ('G', 'I'):
        rows = [b for b in selected_rows if b['kind'] == kind]
        assert len(rows) == 752 and len({b['subject'] for b in rows}) == 49
        metrics[kind] = aggregate(rows)
        compare(metrics[kind], selected['metrics'][kind])
        compare(metrics[kind], complete['scores'][kind])
        per_person[kind] = {s: aggregate([b for b in rows if b['subject'] == s]) for s in inv['personal_subjects']}
        strata[kind] = {}
        for field in ('story', 'family'):
            names = sorted({b['story'] for b in rows}) if field == 'story' else ('shared', 'unseen')
            for name in names:
                sub = [b for b in rows if (b['story'] == name if field == 'story' else
                       (b['source_family'] in families) == (name == 'shared'))]
                value = aggregate(sub)
                compare(value, selected['metrics'][kind]['by_' + field][name])
                strata[kind][field + ':' + name] = dict(value, people=len({b['subject'] for b in sub}), blocks=len(sub))
    audit_cost = dict(wall_seconds=time.monotonic() - started, CPU_seconds=time.process_time() - cpu,
                      peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      actual_threadpools=threadpool_info(), signal_reads=0, new_fits=0, eigendecompositions=0,
                      neural_forwards=0, coefficient_matrix_residual_checks=450)
    assert audit_cost['peak_RSS_bytes'] < 2 * 1024**3
    assert all(p['num_threads'] <= 4 for p in audit_cost['actual_threadpools'])
    result = dict(passed=True, protocol_sha256=PROTOCOL_SHA, review_source_sha256=sha(__file__),
                  completed_sha256=sha(F / 'COMPLETED.json'), score_csv_sha256=sha(F / 'VAL_BLOCKS.csv'),
                  source_hashes_verified=len(gate['sources']), input_stat_identities_verified=len(inv['inputs']),
                  subject_stats=70, global_raw_moments_exact_sum=True, physical_TRAIN_points=24297126,
                  distinct_fits=50, coefficient_solutions_checked=450, normal_equation_max_residual=kkt_max,
                  candidate_blocks_checked=13536, exact_original_support=True, metrics=metrics,
                  per_person=per_person, strata=strata, constant_feature_counts=const_counts,
                  chosen_alpha=selected['alpha_by_fit'],
                  alpha_endpoint_fits=[s for s in fits if selected['alpha_by_fit'][s] in (ALPHAS[0], ALPHAS[-1])],
                  coefficient_CPU_seconds=math.fsum(c['CPU_seconds'] for c in coefficient_costs),
                  coefficient_wall_seconds=math.fsum(c['elapsed_seconds'] for c in coefficient_costs),
                  audit_cost=audit_cost,
                  scope='Saved sufficient-statistics, coefficients and score audit; no independent raw-signal prediction replay. Repeatedly developed internal VAL, not independent confirmation or SOTA. G shared and I personal selection differ. Startup CPU-affinity deviation remains disclosed in runtime review.')
    target = ROOT / 'COORDINATOR_RESULT_REVIEW.json'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'metrics', 'normal_equation_max_residual', 'alpha_endpoint_fits', 'audit_cost')}))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        audit()
