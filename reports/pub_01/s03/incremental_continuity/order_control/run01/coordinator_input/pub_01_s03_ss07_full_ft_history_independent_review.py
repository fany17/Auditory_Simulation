"""Coordinator audit: raw state dict -> manual GRU -> independent Hann metrics.

Only original VAL caches/targets. CPU, four threads; no imported project code,
encoder, model training, external data, or new checkpoint selection. Execute once
after the nine frozen fits complete. Tolerances fixed before reviewing outcomes.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE = Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT = BASE / 's03_full_ft_history_20261102'
D8 = BASE / 's03_finetuned_readout_transfer_20261101'
OUT = ROOT / 'coordinator_review'
SEEDS = (1024, 2026, 31415)
ARMS = ('H_full', 'C_match', 'P_prior')
PIN = 'b8e3552f33557be86609aa244c6d837ef63d9d765f518b5e3df58b46d89289f3'
TOLERANCES = dict(r_absolute=2e-6, mse_absolute=2e-6,
                  mse_relative=2e-5, other_absolute=2e-5, other_relative=5e-5)
COUNT = dict(residual_NN_window_forwards=0, new_encoder_head_forwards=0,
             optimizer_updates=0, actual_input_sha256_bytes=0)
CHECKS = dict(comparisons=0, maximum_r_difference=0., maximum_other_difference=0.)
DIFFERENCES = []


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def compare(label, actual, expected):
    if isinstance(expected, dict):
        for k, v in expected.items():
            if k in actual:
                compare(label + '/' + k, actual[k], v)
        return
    if expected is None or isinstance(expected, (str, bool)):
        assert actual == expected, (label, actual, expected)
        return
    if isinstance(expected, (int, np.integer)):
        assert actual == expected, (label, actual, expected)
        return
    if not isinstance(expected, (float, np.floating)):
        return
    assert math.isfinite(actual) and math.isfinite(expected), label
    error = abs(actual - expected)
    key = label.rsplit('/', 1)[-1]
    if key in ('r', 'r3s'):
        limit = TOLERANCES['r_absolute']
        CHECKS['maximum_r_difference'] = max(CHECKS['maximum_r_difference'], error)
    elif 'MSE' in key or key == 'disagreement':
        limit = TOLERANCES['mse_absolute'] + TOLERANCES['mse_relative'] * abs(expected)
    else:
        limit = TOLERANCES['other_absolute'] + TOLERANCES['other_relative'] * abs(expected)
    CHECKS['comparisons'] += 1
    CHECKS['maximum_other_difference'] = max(CHECKS['maximum_other_difference'], error)
    if error > limit:
        DIFFERENCES.append(dict(field=label, actual=actual, expected=expected,
                                abs_difference=error, limit=limit))


def manual_gru(state, x):
    h = torch.zeros((len(x), 32), dtype=torch.float32)
    for k in range(2):
        ir, iz, inn = F.linear(x[:, k], state['gru.weight_ih_l0'], state['gru.bias_ih_l0']).chunk(3, 1)
        hr, hz, hn = F.linear(h, state['gru.weight_hh_l0'], state['gru.bias_hh_l0']).chunk(3, 1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        proposal = torch.tanh(inn + reset * hn)
        h = (1 - update) * proposal + update * h
    return h


def manual_residual(state, current, history):
    """Independent functional implementation, including explicit GRU equations."""
    b = len(current)
    x = torch.from_numpy(history).reshape(b * 2, history.shape[2], 192)
    x = F.gelu(F.conv1d(x, state['token.0.weight'], state['token.0.bias'], padding=4))
    x = F.gelu(F.conv1d(x, state['token.2.weight'], state['token.2.bias'], padding=4))
    x = F.gelu(F.linear(x.flatten(1), state['token.5.weight'], state['token.5.bias'])).reshape(b, 2, 32)
    h = manual_gru(state, x)
    q = F.gelu(F.linear(torch.from_numpy(current).flatten(1), state['current.1.weight'], state['current.1.bias']))
    gamma, beta = F.linear(h, state['film.weight'], state['film.bias']).chunk(2, 1)
    q = q * (1 + .1 * torch.tanh(gamma)) + .1 * beta
    return F.linear(q, state['output.weight'], state['output.bias']).numpy()


def transformed(x, st):
    y = (x - st['mean']) / st['std']
    y[..., st['constant']] = 0
    return y


def inputs(seed, rows):
    folder = D8 / f'seed{seed}'
    cal = read(folder / 'P_FT_BASELINE.json')['calibration']
    with np.load(folder / 'TRAIN_TRANSFORMS.npz') as f:
        st = [{k: f[f'{k}_{i}'].copy() for k in ('mean', 'std', 'constant')} for i in range(3)]
    with np.load(folder / 'Ccap_lambda0.01_COEFFICIENTS.npz') as f:
        coef, bias = f['coef'].copy(), f['bias'].copy()
    raw = np.load(folder / 'val_latent.npy', mmap_mode='r')
    p = np.load(folder / 'val_P.npy', mmap_mode='r')
    z = np.empty(raw.shape, dtype='f4')
    anchor = np.empty(p.shape, dtype='f8')
    for row in rows:
        offset, n = row['offset'], row['windows']
        for a in range(offset, offset + n, 256):
            b = min(a + 256, offset + n)
            zz = transformed(np.asarray(raw[a:b], dtype='f8').reshape(b - a, 1536), st[0])
            z[a:b] = zz.reshape(-1, 8, 192)
            anchor[a:b] = cal['a'] * np.asarray(p[a:b], dtype='f8') + cal['b']
            first = max(a, offset + 4)
            if first < b:
                latent = zz[first - a:]
                features = np.concatenate((latent / np.sqrt(1536),
                    transformed(anchor[first:b], st[1]) / np.sqrt(192),
                    transformed(latent * latent, st[2]) / np.sqrt(1536)), axis=1)
                anchor[first:b] += features @ coef + bias
    prior = transformed(anchor, st[1]).astype('f4')
    return z, anchor, prior


@torch.no_grad()
def predictions(state, arm, rows, z, anchor, prior, older=False):
    out = anchor.copy()
    for row in rows:
        o, n = row['offset'], row['windows']
        for first in range(9, n, 128):
            local = np.arange(first, min(first + 128, n))
            ix = o + local
            current = z[ix].copy()
            if arm == 'C_match':
                history = np.repeat(current[:, None], 2, axis=1)
            else:
                lag0, lag1 = (9, 7) if older else (5, 3)
                hi = np.column_stack((ix - lag0, ix - lag1))
                assert np.all(hi >= o) and np.all(hi < o + n)
                history = prior[hi, None, :].copy() if arm == 'P_prior' else z[hi].copy()
            residual = manual_residual(state, current, history)
            COUNT['residual_NN_window_forwards'] += len(ix)
            out[ix] += residual.astype('f8') * .1
        assert np.array_equal(out[o:o + 9], anchor[o:o + 9])
    assert np.isfinite(out).all()
    return out


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return float(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)))


def independent_measure(pred, row, blocks):
    """Pairwise half-window fusion, independently of production overlap loops."""
    target = np.asarray(np.load(row['envelope_path'], mmap_mode='r'), dtype='f8').reshape(-1)
    n, w = row['common_samples'], row['windows']
    h = .5 - .5 * np.cos(2 * np.pi * np.arange(192) / 191)
    left, right = pred[:-1, 96:], pred[1:, :96]
    alpha = h[96:] / (h[96:] + h[:96])
    middle = left * alpha + right * (1 - alpha)
    wave = np.zeros(n)
    covered = (w + 1) * 96
    wave[:96], wave[96:w * 96], wave[w * 96:covered] = pred[0, :96], middle.reshape(-1), pred[-1, 96:]
    keep = np.zeros(n, dtype=bool)
    keep[1:covered - 1] = True
    wave[~keep] = 0
    error = (wave - target[:n]) ** 2
    disagreement = np.zeros(n)
    disagreement[96:w * 96] = (alpha * (1 - alpha) * (left - right) ** 2).reshape(-1)
    weighted = error + disagreement
    yw = np.lib.stride_tricks.sliding_window_view(target[:n], 192)[::96][:w].astype('f4').astype('f8')
    centered, tc = pred - pred.mean(1, keepdims=True), yw - yw.mean(1, keepdims=True)
    defined = (np.ptp(pred, axis=1) > 0) & (np.ptp(yw, axis=1) > 0)
    r3 = np.sum(centered[defined] * tc[defined], axis=1) / np.sqrt(np.sum(centered[defined] ** 2, axis=1) * np.sum(tc[defined] ** 2, axis=1))
    pair_mse = np.mean((left - right) ** 2, axis=1)
    denominator = (np.var(left, axis=1) + np.var(right, axis=1)) / 2
    valid = denominator > 0
    meta = {k: row[k] for k in ('record', 'subject', 'story', 'phase')}
    meta['source_family'] = row.get('source_family', row['story'])
    segments = []
    for block in blocks:
        s = block['start']
        k = keep[s:s + 3840]
        assert len(k) == 3840 and k.sum() == block['samples']
        if not block['eligible']:
            continue
        segments.append(dict(**meta, start=s, samples=int(k.sum()), r=corr(wave[s:s + 3840][k], target[s:s + 3840][k]),
            MSE=float(error[s:s + 3840][k].mean()), weighted_single_MSE=float(weighted[s:s + 3840][k].mean()),
            disagreement=float(disagreement[s:s + 3840][k].mean())))
    def region(first, last):
        k = keep[first:last]
        pp, tt = wave[first:last][k], target[first:last][k]
        return dict(samples=len(pp), MSE=float(np.mean((pp - tt) ** 2)) if len(pp) else None,
                    r=corr(pp, tt) if len(pp) > 1 and np.ptp(pp) > 0 and np.ptp(tt) > 0 else None)
    record = dict(**meta, windows=w, r3s=float(r3.mean()) if defined.all() else None,
        window_r_defined=int(defined.sum()), window_r_undefined=int((~defined).sum()),
        target_constant_windows=int((np.ptp(yw, axis=1) == 0).sum()),
        prediction_constant_windows=int((np.ptp(pred, axis=1) == 0).sum()),
        window_MSE=float(np.mean((pred - yw) ** 2)), covered_samples=int(keep.sum()),
        overlap_pairs=w - 1, valid_normalized_pairs=int(valid.sum()), overlap_SSE_mean_sum=float(pair_mse.sum()),
        normalized_overlap_num=float(pair_mse[valid].sum()), normalized_overlap_den=float(denominator[valid].sum()),
        overlap_MSE=float(pair_mse.mean()), overlap_NRMSE=float(np.sqrt(pair_mse[valid].sum() / denominator[valid].sum())), segments=segments)
    boundary = dict(record=row['record'], prefix=region(0, 864), first_learning_overlap=region(864, 960),
        tail=region(n // 3840 * 3840, n), last_single_window=region(w * 96, min(n, (w + 1) * 96)),
        maximum_window_abs=float(np.abs(pred).max()), maximum_fused_abs=float(np.abs(wave[keep]).max()))
    return record, boundary


def aggregate(records):
    segments = [s for r in records for s in r['segments']]
    people = []
    keys = ('r', 'MSE', 'weighted_single_MSE', 'disagreement')
    for subject in sorted({b['subject'] for b in segments}):
        bb = [b for b in segments if b['subject'] == subject]
        people.append(dict(subject=subject, phase='val', blocks=len(bb), records=len({b['record'] for b in bb}),
                           **{k: float(np.mean([b[k] for b in bb])) for k in keys}))
    windows = sum(r['windows'] for r in records)
    pairs = sum(r['overlap_pairs'] for r in records)
    undefined = sum(r['window_r_undefined'] for r in records)
    phase = dict(people=len(people), records=len(records), blocks=len(segments), windows=windows,
        window_r_undefined=undefined, window_r_defined=windows - undefined,
        **{k: float(np.mean([p[k] for p in people])) for k in keys},
        r3s=sum(r['r3s'] * r['windows'] for r in records) / windows if undefined == 0 else None,
        window_MSE=sum(r['window_MSE'] * r['windows'] for r in records) / windows,
        overlap_MSE=sum(r['overlap_SSE_mean_sum'] for r in records) / pairs, overlap_pairs=pairs,
        valid_normalized_pairs=sum(r['valid_normalized_pairs'] for r in records),
        overlap_NRMSE=float(np.sqrt(sum(r['normalized_overlap_num'] for r in records) / sum(r['normalized_overlap_den'] for r in records))))
    warm = [b for b in segments if b['start'] >= 3840]
    subjects = sorted({b['subject'] for b in warm})
    aux = dict(people=len(subjects), blocks=len(warm), r=float(np.mean([np.mean([b['r'] for b in warm if b['subject'] == s]) for s in subjects])))
    return dict(phases=dict(val=phase), people=people, segments=segments, records=records, fixed_warm_auxiliary=aux)


def score(pred, rows, mask):
    records, boundaries = [], []
    for row in rows:
        o, n = row['offset'], row['windows']
        rr, bb = independent_measure(pred[o:o + n], row, [b for b in mask if b['record'] == row['record']])
        records.append(rr)
        boundaries.append(bb)
    result = aggregate(records)
    assert result['phases']['val']['blocks'] == 752 and result['phases']['val']['people'] == 49
    result['boundary_tail_records'] = boundaries
    return result


def compare_metrics(label, own, original):
    compare(label + '/phase', own['phases']['val'], original['phases']['val'])
    compare(label + '/warm', own['fixed_warm_auxiliary'], original['fixed_warm_auxiliary'])
    for table, fields in [('people', ('subject',)), ('segments', ('record', 'start')),
                          ('records', ('record',)), ('boundary_tail_records', ('record',))]:
        actual = {tuple(r[k] for k in fields): r for r in own[table]}
        expected = {tuple(r[k] for k in fields): r for r in original[table]}
        assert set(actual) == set(expected), (label, table)
        for key in actual:
            compare(f'{label}/{table}/{key}', actual[key], expected[key])


def run():
    started, cpu = time.time(), time.process_time()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    protocol = read(ROOT / 'PROTOCOL.json')
    assert sha(ROOT / 'PROTOCOL.json') == PIN
    assert (ROOT / 'QUEUE_COMPLETED.json').exists() and not (ROOT / 'QUEUE_FAILURE.json').exists()
    OUT.mkdir(exist_ok=False)
    write(OUT / 'STARTED.json', dict(time=started, pid=os.getpid(), source_sha256=sha(__file__), protocol_sha256=PIN,
        tolerances=TOLERANCES, device='CPU', threads=4, expected_new_residual_forwards=12 * 30059,
        old_encoder_head_or_external_access=False))
    for path, digest in protocol['source_pins'].items():
        assert sha(path) == digest, path
    for item in protocol['cache_metadata']:
        p = Path(item['path'])
        st = p.stat()
        assert st.st_size == item['bytes'] and st.st_mtime_ns == item['mtime_ns'], str(p)
        assert sha(p) == item['sha256'], str(p)
        assert p.stat().st_mtime_ns == st.st_mtime_ns
        COUNT['actual_input_sha256_bytes'] += st.st_size
    d8 = read(D8 / 'PROTOCOL.json')
    rows, mask = d8['records']['val'], d8['mask']['blocks']
    summary, ledger = [], []
    for seed in SEEDS:
        z, anchor, prior = inputs(seed, rows)
        anchor_metrics = score(anchor, rows, mask)
        compare_metrics(f'anchor/{seed}', anchor_metrics, read(ROOT / f'fits/H_full_seed{seed}/INCUMBENT.json'))
        write(OUT / f'anchor_seed{seed}.json', anchor_metrics)
        for arm in ARMS:
            folder = ROOT / f'fits/{arm}_seed{seed}'
            completed = read(folder / 'COMPLETED.json')
            hist = [json.loads(s) for s in (folder / 'HISTORY.jsonl').read_text().splitlines()]
            assert len(hist) == 20 and [v['epoch'] for v in hist] == list(range(20))
            assert [v['updates'] for v in hist] == [3945 * (k + 1) for k in range(20)]
            for h in hist:
                assert h['effective_TRAIN_windows'] == 248620 and h['full_target_windows'] == 252454
            expected_epoch = int(np.argmax([read(folder / 'INCUMBENT.json')['phases']['val']['r']] + [v['validation']['r'] for v in hist])) - 1
            checkpoint = torch.load(folder / 'best_validation.pt', map_location='cpu', weights_only=True)
            assert expected_epoch == checkpoint['epoch'] == completed['selected_epoch']
            assert checkpoint['seed'] == seed and checkpoint['arm'] == arm
            assert completed['updates'] == 78900 and completed['epochs'] == 20
            assert completed['target_window_presentations'] == 252454 * 20
            assert completed['effective_residual_TRAIN_presentations'] == 248620 * 20
            assert completed['new_residual_NN_window_forwards'] == 248620 * 20 + 30059 * (22 if arm == 'H_full' else 21)
            assert completed['new_encoder_head_forwards'] == 0 and completed['test_or_external_access'] is False
            assert sum(v.numel() for v in checkpoint['model'].values()) == (164784 if arm == 'P_prior' else 165792)
            start = read(folder / 'STARTED.json')
            assert start['protocol_sha256'] == PIN
            ledger.append(dict(**completed, checkpoint_sha256=sha(folder / 'best_validation.pt'), history_sha256=sha(folder / 'HISTORY.jsonl')))
            for older in ([False, True] if arm == 'H_full' else [False]):
                label = f'{arm}_seed{seed}' + ('_older' if older else '')
                pred = predictions(checkpoint['model'], arm, rows, z, anchor, prior, older)
                metrics = score(pred, rows, mask)
                original = read(folder / ('OLDER_METRICS.json' if older else 'SELECTED_METRICS.json'))
                compare_metrics(label, metrics, original)
                write(OUT / f'{label}.json', metrics)
                summary.append(dict(seed=seed, arm=arm, older=older, selected_epoch=expected_epoch,
                                    r=metrics['phases']['val']['r'], anchor_r=anchor_metrics['phases']['val']['r'],
                                    warm_r=metrics['fixed_warm_auxiliary']['r'], MSE=metrics['phases']['val']['MSE'],
                                    window_MSE=metrics['phases']['val']['window_MSE']))
                print(json.dumps(dict(finished=label, comparisons=CHECKS['comparisons'], failed_comparisons=len(DIFFERENCES))), flush=True)
        del z, anchor, prior
    assert COUNT['residual_NN_window_forwards'] == 12 * 30059
    development = []
    for seed in SEEDS:
        rr = {(r['arm'], r['older']): r for r in summary if r['seed'] == seed}
        h = rr[('H_full', False)]
        development.append(dict(seed=seed, H_minus_anchor=h['r'] - h['anchor_r'],
            H_minus_current=h['r'] - rr[('C_match', False)]['r'], H_minus_prior=h['r'] - rr[('P_prior', False)]['r'],
            true_minus_older=h['r'] - rr[('H_full', True)]['r']))
    write(OUT / 'REVIEW.json', dict(passed=not DIFFERENCES, protocol_sha256=PIN, source_sha256=sha(__file__),
        tolerances=TOLERANCES, checks=CHECKS, failed_comparisons=DIFFERENCES, fits=ledger, metrics=summary,
        development_conditions=development, development_gate_met=all(v > 0 for d in development for k, v in d.items() if k != 'seed'),
        cost=dict(**COUNT, wall_seconds=time.time() - started, CPU_seconds=time.process_time() - cpu,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, GPU_bytes=0),
        limitations=['Repeated development VAL; no independent confirmation.', 'Inputs include full FT windows; no online-causal claim.',
                     'Older intervention changes content and recency.', 'Review validates numerical execution, not publication novelty.'],
        goal_complete=False))
    with (OUT / 'SUMMARY.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    assert not DIFFERENCES, f'{len(DIFFERENCES)} numerical comparison failures; do not call PASS'


@torch.no_grad()
def self_check():
    started, cpu = time.time(), time.process_time()
    torch.set_num_threads(4)
    torch.manual_seed(84071)
    module = torch.nn.GRU(32, 32, batch_first=True)
    x = torch.randn(7, 2, 32)
    state = {'gru.' + k: v for k, v in module.state_dict().items()}
    expected = module(x)[0][:, -1]
    actual = manual_gru(state, x)
    error = float((expected - actual).abs().max())
    assert error < 1e-6
    # Exact-window target gives zero errors under independent pairwise fusion.
    target = np.arange(4200, dtype='f8') / 4200
    pred = np.lib.stride_tricks.sliding_window_view(target, 192)[::96].copy()
    n, w = len(target), len(pred)
    hann = np.hanning(192)
    reference = np.zeros(n)
    weight = np.zeros(n)
    for i, win in enumerate(pred):
        reference[i * 96:i * 96 + 192] += hann * win
        weight[i * 96:i * 96 + 192] += hann
    k = weight > 1e-8
    reference[k] /= weight[k]
    alpha = hann[96:] / (hann[96:] + hann[:96])
    middle = pred[:-1, 96:] * alpha + pred[1:, :96] * (1 - alpha)
    own = np.concatenate((pred[0, :96], middle.reshape(-1), pred[-1, 96:]))
    fusion_error = float(np.max(np.abs(own[1:-1] - reference[1:(w + 1) * 96 - 1])))
    assert fusion_error < 1e-12
    result = dict(source_sha256=sha(__file__), passed=True, manual_GRU_max_abs=error,
        synthetic_GRU_sequences=7, synthetic_GRU_steps_per_sequence=2,
        independent_pairwise_Hann_max_abs=fusion_error, scientific_input_access=False,
        CPU_seconds=time.process_time() - cpu, wall_seconds=time.time() - started)
    write(ROOT / 'COORDINATOR_REVIEW_SOURCE_CHECK.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    assert not (args.run and args.self_check)
    if args.self_check:
        self_check()
    if args.run:
        try:
            run()
        except BaseException:
            if OUT.exists() and not (OUT / 'FAILURE.json').exists():
                write(OUT / 'FAILURE.json', dict(time=time.time(), traceback=traceback.format_exc(), counters=COUNT))
            raise
