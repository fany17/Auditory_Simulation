"""Coordinator numerical review: cached Z/P only; no NN or learned fits.

Run solely on server2203. Reconstruct the fixed readouts with separately
implemented algebra, sample-level overlap accumulation, block/person aggregation
and exact sign enumeration. Do not import the evaluator or its scoring helpers.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_openneuro_confirmation_20261102')
BASE = ROOT.parent
D8 = BASE / 's03_finetuned_readout_transfer_20261101'
D9 = BASE / 's03_physical_objective_20261101'
SEEDS = (1024, 2026, 31415)
ROLES = ('P_FT_star', 'Ccap', 'W', 'F')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def close(actual, reference, label, atol=1e-9, rtol=1e-9):
    if actual is None or reference is None:
        assert actual is None and reference is None, (label, actual, reference)
        return
    assert np.isfinite(actual) and np.isfinite(reference), label
    assert abs(actual-reference) <= atol+rtol*max(abs(actual), abs(reference)), (
        label, actual, reference)


def correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    aa = a-a.mean()
    bb = b-b.mean()
    return float(np.sum(aa*bb)/np.sqrt(np.sum(aa*aa)*np.sum(bb*bb)))


def transformed(x, stats, dtype):
    out = (np.asarray(x, dtype=dtype)-np.asarray(stats['mean'], dtype=dtype))
    out /= np.asarray(stats['std'], dtype=dtype)
    out[:, stats['constant']] = 0
    return out


def predict(z, p, calibration, stats, coefficients, dtype=np.float64):
    base = dtype(calibration['a'])*p.astype(dtype)+dtype(calibration['b'])
    result = base.copy()
    if coefficients is None:
        return result
    latent = transformed(z[4:].reshape(-1, 1536), stats[0], dtype)
    pred = transformed(base[4:], stats[1], dtype)
    square = transformed(latent*latent, stats[2], dtype)
    latent /= np.sqrt(dtype(1536))
    pred /= np.sqrt(dtype(192))
    square /= np.sqrt(dtype(1536))
    co = coefficients['coef'].astype(dtype)
    bias = coefficients['bias'].astype(dtype)
    if dtype == np.float64:
        # Independent split-matrix reconstruction of the primary dense product.
        residual = latent @ co[:1536]
        residual += pred @ co[1536:1728]
        residual += square @ co[1728:]
        residual += bias
    else:
        # FP32 diagnostic specifies the dense arithmetic order, not split BLAS.
        features = np.concatenate([latent, pred, square], axis=1)
        residual = features @ co + bias
    result[4:] += residual
    assert result.dtype == dtype and np.array_equal(result[:4], base[:4])
    return result


def reconstruct(pred, target, dtype=np.float64):
    pred = np.asarray(pred, dtype=dtype)
    n = len(target)
    index = np.arange(len(pred))[:, None]*96 + np.arange(192)[None, :]
    taper = np.hanning(192).astype(dtype)
    weight = np.zeros(n, dtype=dtype)
    numerator = np.zeros(n, dtype=dtype)
    np.add.at(weight, index.ravel(), np.broadcast_to(taper, pred.shape).ravel())
    np.add.at(numerator, index.ravel(), (pred*taper).ravel())
    covered = weight > 0
    waveform = numerator.copy()
    waveform[covered] /= weight[covered]
    return waveform, covered, index, weight, taper


def inspect_record(pred, target, row):
    target = np.asarray(target, dtype=np.float64)[:row['length']]
    wave, covered, index, weight, taper = reconstruct(pred, target)
    window_target = target[index]
    single = np.zeros(len(target), dtype=np.float64)
    variance = np.zeros(len(target), dtype=np.float64)
    np.add.at(single, index.ravel(), ((pred-window_target)**2*taper).ravel())
    np.add.at(variance, index.ravel(), ((pred-wave[index])**2*taper).ravel())
    single[covered] /= weight[covered]
    variance[covered] /= weight[covered]
    error = (wave-target)**2
    assert int(covered.sum()) == row['covered']

    def span(start, stop):
        mask = covered[start:stop]
        if not mask.any():
            return dict(covered=0, r=None, MSE=None, weighted_single_MSE=None, disagreement=None)
        sources = pred[(index >= start) & (index < stop) & (taper[None, :] > 0)]
        constant = np.ptp(sources) == 0
        return dict(covered=int(mask.sum()),
                    r=None if constant else correlation(wave[start:stop][mask], target[start:stop][mask]),
                    MSE=float(error[start:stop][mask].mean()),
                    weighted_single_MSE=float(single[start:stop][mask].mean()),
                    disagreement=float(variance[start:stop][mask].mean()))

    blocks = [dict(start=b['start'], stop=b['stop'], **span(b['start'], b['stop']))
              for b in row['blocks'] if b['eligible']]
    tail = span(row['tail']['start'], len(target))
    if not row['tail']['eligible_r']:
        tail['r'] = None
    r3 = [correlation(p, y) for p, y in zip(pred, window_target)]
    last = (len(pred)-1)*96
    result = dict(blocks=blocks, whole_record=span(0, len(target)), tail=tail,
                  window_MSE=float(np.mean((pred-window_target)**2)),
                  r3s=None if any(x is None for x in r3) else float(np.mean(r3)),
                  defined_window_r=sum(x is not None for x in r3),
                  undefined_window_r=sum(x is None for x in r3),
                  uniform_overlap_MSE=float(np.mean((pred[:-1, 96:]-pred[1:, :96])**2)),
                  max_abs_window_prediction=float(np.max(np.abs(pred))),
                  max_abs_fused_prediction=float(np.max(np.abs(wave[covered]))),
                  first4_window_MSE=float(np.mean((pred[:4]-window_target[:4])**2)),
                  prefix_before_learning=span(0, min(385, len(target))),
                  first_learning_overlap=span(385, min(480, len(target))),
                  final_single_window_tail=span(last+96, min(last+191, len(target))))
    return result, wave, covered


def sign_test(d):
    d = np.asarray(d, dtype=np.float64)
    assert 1 <= len(d) <= 19 and np.isfinite(d).all()
    observed = abs(np.sum(d)/len(d))
    count = 0
    for start in range(0, 2**len(d), 8192):
        values = np.arange(start, min(start+8192, 2**len(d)), dtype=np.uint32)
        signed = np.zeros(len(values), dtype=np.float64)
        for col, value in enumerate(d):
            signed += np.where((values >> col) & 1, -value, value)
        count += int(np.sum(np.abs(signed/len(d)) >= observed-1e-15))
    return count / (2**len(d))


def check():
    rng = np.random.default_rng(7781)
    y = rng.normal(size=8000)
    starts = np.arange(0, len(y)-191, 96)
    p = np.stack([y[s:s+192] for s in starts])
    wave, mask, _, _, _ = reconstruct(p, y)
    assert np.max(np.abs(wave[mask]-y[mask])) < 1e-14
    row = dict(length=len(y), covered=int(mask.sum()),
               blocks=[dict(start=s, stop=s+3840, eligible=True) for s in (0, 3840)],
               tail=dict(start=7680, eligible_r=bool(mask[7680:].sum() >= 192)))
    shifted, _, _ = inspect_record(p+.4, y, row)
    for block in shifted['blocks']:
        close(block['r'], 1., 'synthetic additive offset preserves correlation', atol=1e-14)
        close(block['MSE'], .16, 'synthetic known physical error', atol=1e-14)
        assert block['disagreement'] < 1e-28
    constant, _, _ = inspect_record(np.full_like(p, .1), y, row)
    assert all(b['r'] is None for b in constant['blocks'])
    z = rng.normal(size=(7, 8, 192)).astype(np.float32)
    raw = rng.normal(size=(7, 192)).astype(np.float32)
    stats = [dict(mean=np.full(d, mean), std=np.full(d, std), constant=np.zeros(d, bool))
             for d, mean, std in [(1536, .2, 2.), (192, .1, 4.), (1536, 1., 3.)]]
    co = dict(coef=np.zeros((3264, 192)), bias=np.zeros(192))
    co['coef'][0, 5], co['coef'][1536, 5], co['coef'][1728, 5] = 1., 2., 3.
    for dtype, tolerance in [(np.float64, 1e-14), (np.float32, 1e-6)]:
        prediction = predict(z, raw, dict(a=.2, b=.3), stats, co, dtype)
        expected = .2*raw.astype(np.float64)+.3
        a = (z[4:, 0, 0].astype(np.float64)-.2)/2.
        expected[4:, 5] += a/np.sqrt(1536) + 2.*((expected[4:, 0]-.1)/4.)/np.sqrt(192)
        expected[4:, 5] += 3.*((a*a-1.)/3.)/np.sqrt(1536)
        assert np.max(np.abs(prediction-expected)) < tolerance
    for d in ([1., 2., 3.], [1., -1.], [0., 0., 0.], [.1, -.2, .3, .5]):
        expected = sum(abs(np.mean(np.array(d)*sgn)) >= abs(np.mean(d))-1e-15
                       for sgn in itertools.product([-1, 1], repeat=len(d))) / (2**len(d))
        assert sign_test(d) == expected
    assert correlation(np.ones(10), np.arange(10)) is None
    print(json.dumps(dict(independent_overlap_reconstruction=True,
                         independent_sign_enumeration=True, constant_prediction=True,
                         known_offset_block_metrics=True, known_sparse_readout_both_precisions=True,
                         real_NN=0, real_EEG_read=False)))


def run():
    started = time.time()
    protocol = read(ROOT/'EVALUATION_PROTOCOL.json')
    summary = read(ROOT/'EXTERNAL_SUMMARY.json')
    assert tuple(protocol['seeds']) == SEEDS and tuple(protocol['roles']) == ROLES
    for path, digest in protocol['source_pins'].items():
        assert sha(path) == digest, path
    data_pins = {}
    for row in protocol['rows']:
        for kind in ['EEG', 'target']:
            path, digest = row[f'{kind}_path'], row[f'{kind}_sha256']
            assert path not in data_pins or data_pins[path] == digest
            data_pins[path] = digest
    for path, digest in data_pins.items():
        assert sha(path) == digest, path
    cache_count = block_count = precision_count = record_count = 0
    person = {}
    block_map = {}
    scalar_records = {role: [] for role in ROLES}
    input_pins = {'EVALUATION_PROTOCOL.json': sha(ROOT/'EVALUATION_PROTOCOL.json'),
                  'EXTERNAL_SUMMARY.json': sha(ROOT/'EXTERNAL_SUMMARY.json')}
    by_subject = sorted({r['subject'] for r in protocol['rows']})
    for seed in SEEDS:
        folder = ROOT/f'evaluation_seed{seed}'
        cache = read(folder/'CACHE_PATHS.json')
        assert cache['source_protocol_sha256'] == input_pins['EVALUATION_PROTOCOL.json']
        assert sha(cache['Z']) == cache['Z_sha256'] and sha(cache['P']) == cache['P_sha256']
        cache_count += 2
        z = np.load(cache['Z'], mmap_mode='r', allow_pickle=False)
        p = np.load(cache['P'], mmap_mode='r', allow_pickle=False)
        assert z.shape == (protocol['windows'], 8, 192) and z.dtype == np.float32
        assert p.shape == (protocol['windows'], 192) and p.dtype == np.float32
        train_stats = np.load(D8/f'seed{seed}/TRAIN_TRANSFORMS.npz', allow_pickle=False)
        stats = [{k: train_stats[f'{k}_{i}'] for k in ('mean', 'std', 'constant')} for i in range(3)]
        calibration = read(D8/f'seed{seed}/P_FT_BASELINE.json')['calibration']
        for role in ROLES:
            metrics_path = folder/f'{role}_METRICS.json'
            report = read(metrics_path)
            input_pins[str(metrics_path)] = sha(metrics_path)
            # A failed endpoint needs an explicit failure review, never a silent subset PASS.
            assert report['passed'] and not report['failures'], f'{seed}/{role}: endpoint requires failure review'
            lookup = {r['record']: r for r in report['records']}
            assert len(lookup) == len(protocol['rows']) == len(report['records'])
            coefficients = None
            if role != 'P_FT_star':
                path = (D8/f'seed{seed}/Ccap_lambda0.01_COEFFICIENTS.npz'
                        if role == 'Ccap' else D9/f'seed{seed}/{role}_COEFFICIENTS.npz')
                coefficients = np.load(path, allow_pickle=False)
            for row in protocol['rows']:
                a, b = row['offset'], row['offset']+row['windows']
                y = np.load(row['target_path'], allow_pickle=False)[:row['length']]
                pred = predict(z[a:b], p[a:b], calibration, stats, coefficients)
                own, wave, mask = inspect_record(pred, y, row)
                scalar_records[role].append((row['windows'], own))
                ref = lookup[row['record']]
                assert len(own['blocks']) == len(ref['blocks'])
                for field in ['window_MSE', 'r3s', 'uniform_overlap_MSE', 'max_abs_window_prediction',
                              'max_abs_fused_prediction', 'first4_window_MSE']:
                    close(own[field], ref[field], (seed, role, row['record'], field))
                for field in ['defined_window_r', 'undefined_window_r']:
                    assert own[field] == ref[field]
                for field in ['whole_record', 'tail', 'prefix_before_learning',
                              'first_learning_overlap', 'final_single_window_tail']:
                    for metric in ['r', 'MSE', 'weighted_single_MSE', 'disagreement']:
                        close(own[field][metric], ref[field][metric], (seed, role, row['record'], field, metric))
                for got, expected in zip(own['blocks'], ref['blocks']):
                    assert (got['start'], got['stop'], got['covered']) == (
                        expected['start'], expected['stop'], expected['covered'])
                    for metric in ['r', 'MSE', 'weighted_single_MSE', 'disagreement']:
                        close(got[metric], expected[metric], (seed, role, row['record'], got['start'], metric))
                    block_map.setdefault((role, seed, row['subject']), []).append((row['record'], got))
                    block_count += 1
                if role in ('W', 'F'):
                    diagnostics = [d for d in report['precision_diagnostics'] if d['record'] == row['record']]
                    assert len(diagnostics) == 1, 'Missing precision diagnostic requires explicit review'
                    ref32 = diagnostics[0]
                    p32 = predict(z[a:b], p[a:b], calibration, stats, coefficients, dtype=np.float32)
                    w32, keep32, _, _, _ = reconstruct(p32, y, dtype=np.float32)
                    assert np.array_equal(keep32, mask)
                    yy = y.astype(np.float64)
                    close(float(np.max(np.abs(w32[mask].astype(np.float64)-wave[mask]))),
                          ref32['whole_record_waveform_max_abs'], 'FP32 waveform diagnostic', atol=1e-7, rtol=1e-6)
                    close(float(np.mean((w32[mask].astype(np.float64)-yy[mask])**2)),
                          ref32['fused_MSE32'], 'FP32 MSE diagnostic', atol=1e-7, rtol=1e-6)
                    for br in ref32['blocks']:
                        sl = slice(br['start'], br['stop'])
                        close(correlation(w32[sl][mask[sl]], yy[sl][mask[sl]]),
                              br['r32'], 'FP32 block r', atol=1e-8, rtol=1e-8)
                    precision_count += 1
                record_count += 1
            print(json.dumps(dict(stage='RECONSTRUCTED', seed=seed, role=role,
                                  records=record_count, blocks=block_count)), flush=True)
    for role in ROLES:
        for subject in by_subject:
            per_seed = []
            for seed in SEEDS:
                rows = block_map[role, seed, subject]
                per_seed.append(np.mean([x['r'] for _, x in rows]))
            person[role, subject] = float(np.mean(per_seed))
        ref = next(r for r in summary['models'] if r['role'] == role)
        close(float(np.mean([person[role, s] for s in by_subject])), ref['r'], (role, 'person macro'))
        clip_people = []
        for subject in by_subject:
            seed_clips = []
            for seed in SEEDS:
                clips = {}
                for record, block in block_map[role, seed, subject]:
                    clips.setdefault(record, []).append(block['r'])
                seed_clips.append(np.mean([np.mean(v) for v in clips.values()]))
            clip_people.append(np.mean(seed_clips))
        close(float(np.mean(clip_people)), ref['equal_clip_r'], (role, 'equal clip secondary'))
        window_count = sum(n for n, _ in scalar_records[role])
        close(float(sum(n*v['window_MSE'] for n, v in scalar_records[role])/window_count),
              ref['window_MSE'], (role, 'window MSE aggregate'))
        undefined = sum(v['undefined_window_r'] for _, v in scalar_records[role])
        assert window_count == ref['windows'] and undefined == ref['undefined_3s_windows']
        r3 = None if undefined else float(sum(n*v['r3s'] for n, v in scalar_records[role])/window_count)
        close(r3, ref['r3s'], (role, 'window r aggregate'))
        for metric in ['MSE', 'weighted_single_MSE', 'disagreement']:
            value = np.mean([np.mean([np.mean([b[metric] for _, b in block_map[role, seed, subject]])
                                     for seed in SEEDS]) for subject in by_subject])
            close(float(value), ref[metric], (role, metric))
    pvalues = []
    rng = np.random.default_rng(20261102)
    indices = rng.integers(0, len(by_subject), size=(20000, len(by_subject)))
    for i, (a, b) in enumerate((('Ccap', 'P_FT_star'), ('F', 'W'))):
        difference = np.array([person[a, s]-person[b, s] for s in by_subject])
        ref = summary['primary'][i]['inference']
        assert ref is not None
        pvalue = sign_test(difference)
        close(pvalue, ref['p'], 'exact signflip', atol=0, rtol=0)
        close(float(difference.mean()), ref['mean_difference'], 'primary difference')
        interval = np.quantile(difference[indices].mean(axis=1), [.0125, .9875])
        for got, expected in zip(interval, ref['Bonferroni_family95_approximate_CI']):
            close(float(got), expected, 'person bootstrap CI')
        pvalues.append(pvalue)
    order = sorted(range(2), key=lambda i: (pvalues[i], i))
    adjusted = [None, None]
    adjusted[order[0]] = min(1., 2*pvalues[order[0]])
    adjusted[order[1]] = min(1., max(adjusted[order[0]], pvalues[order[1]]))
    for got, item in zip(adjusted, summary['primary']):
        close(got, item['inference']['p_Holm2'], 'Holm2', atol=0, rtol=0)
    result = dict(passed=True, reviewer='Independent coordinator implementation; no evaluator/scoring imports',
                  source_sha256=sha(__file__), source_pins_verified=len(protocol['source_pins']),
                  distinct_EEG_and_target_hashes_verified=len(data_pins),
                  cache_hashes_verified=cache_count, role_seed_records_rebuilt=record_count,
                  role_seed_blocks_rebuilt=block_count, full_FP32_records_rebuilt=precision_count,
                  primary_statistics_independently_recomputed=2, input_pins=input_pins,
                  wall_seconds=time.time()-started, CPU_seconds=time.process_time(),
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  real_NN=0, new_model_fits=0, scientific_goal_complete=False)
    with (ROOT/'COORDINATOR_INDEPENDENT_NUMERICAL_REVIEW.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['self-check', 'run'])
    args = parser.parse_args()
    assert os.name == 'posix' and ROOT.is_dir(), 'Scientific computation only on server2203'
    check() if args.mode == 'self-check' else run()

