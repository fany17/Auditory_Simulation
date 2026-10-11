#!/usr/bin/env python3
"""Frozen initial public EEG envelope decoding; execute only on server2203.

Signals/model parameters stay on server. Only summary CSV/JSON/static PNG leave.
Explicit lag convention: envelope[t] <- EEG[t + lag + shift].
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh, solve


def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys or ['status'])
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class Score:
    def __init__(self):
        self.n = 0
        self.sy = self.sp = self.syy = self.spp = self.syp = 0.0

    def update(self, y, p):
        assert y.shape == p.shape and np.isfinite(y).all() and np.isfinite(p).all()
        self.n += len(y)
        self.sy += float(y.sum()); self.sp += float(p.sum())
        self.syy += float(y @ y); self.spp += float(p @ p); self.syp += float(y @ p)

    def metrics(self):
        if self.n < 2:
            raise ValueError('insufficient score samples')
        vy = max(0.0, self.syy - self.sy ** 2 / self.n)
        vp = max(0.0, self.spp - self.sp ** 2 / self.n)
        cov = self.syp - self.sy * self.sp / self.n
        r = cov / np.sqrt(vy * vp) if min(vy, vp) > 1e-12 else None
        r2 = 1 - (self.syy + self.spp - 2 * self.syp) / vy if vy > 1e-12 else None
        # Target was standardized only by training; zero is train-mean prediction.
        baseline = 1 - self.syy / vy if vy > 1e-12 else None
        return {'samples': self.n, 'r': r, 'r2': r2, 'train_mean_r2': baseline}


def macro(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


class Stats:
    def __init__(self, p):
        self.n = 0
        self.sx = np.zeros(p); self.sxx = np.zeros((p, p)); self.sxy = np.zeros(p)
        self.sy = self.syy = 0.0

    def update(self, x, y):
        self.n += len(y)
        self.sx += x.sum(axis=0); self.sxx += x.T @ x; self.sxy += x.T @ y
        self.sy += float(y.sum()); self.syy += float(y @ y)

    def fit(self, alphas):
        if self.n <= len(self.sx):
            raise ValueError('insufficient training samples')
        mx = self.sx / self.n; my = self.sy / self.n
        sx = np.sqrt(np.maximum(np.diag(self.sxx) / self.n - mx * mx, 0))
        sy = np.sqrt(max(self.syy / self.n - my * my, 0))
        sx[sx < 1e-12] = 1
        if sy < 1e-12:
            raise ValueError('constant training target')
        gram = (self.sxx / self.n - np.outer(mx, mx)) / sx[:, None] / sx[None, :]
        gram = (gram + gram.T) / 2
        cross = (self.sxy / self.n - mx * my) / sx / sy
        eig, vec = eigh(gram)
        if eig.min() < -1e-6:
            raise ValueError('invalid negative Gram eigenvalue')
        eig = np.maximum(eig, 0)
        weights = np.column_stack([vec @ ((vec.T @ cross) / (eig + a)) for a in alphas])
        return mx, sx, my, sy, weights


def load_record(row):
    eeg = np.load(row['eeg_path'], mmap_mode='r', allow_pickle=False)
    env = np.load(row['envelope_path'], mmap_mode='r', allow_pickle=False)
    if eeg.ndim != 2:
        raise ValueError('EEG must be 2D')
    if eeg.shape[0] == 64:
        eeg = eeg.T
    if eeg.shape[1] != 64:
        raise ValueError('EEG must have 64 channels')
    if env.ndim == 2:
        if env.shape[0] <= 256 and env.shape[1] > env.shape[0]:
            env = env.T
        env = np.asarray(env, dtype=np.float64).mean(axis=1)
    elif env.ndim != 1:
        raise ValueError('unsupported envelope shape')
    n = min(int(row['samples']), len(eeg), len(env))
    if n != int(row['common_samples']) or float(row['sample_rate_hz']) != 64:
        raise ValueError('upstream common-overlap/rate mismatch')
    return np.asarray(eeg[:n], dtype=np.float64), np.asarray(env[:n], dtype=np.float64)


def support(eeg, env, all_lags, shift):
    n = min(len(eeg), len(env))
    t = np.arange(max(0, -min(all_lags)), n - shift - max(all_lags), dtype=np.int64)
    finite = np.isfinite(env[t])
    for offset in (0, shift):
        for lag in all_lags:
            finite &= np.isfinite(eeg[t + offset + lag]).all(axis=1)
    return t[finite], len(t) - int(finite.sum())


def design(eeg, t, lags, shift):
    return np.concatenate([eeg[t + shift + lag] for lag in lags], axis=1)


def preflight(c):
    splits = read_csv(c['split_path']); samples = read_csv(c['sample_path']); index = read_csv(c['index_path'])
    indexes = {r['source_recording']: r for r in index}
    samplemap = {r['source_recording']: r for r in samples}
    ready = [r for r in splits if r['record_status'] == 'READY' and r['selected_in_mode'] == 'True']
    subjects = sorted({r['subject'] for r in ready})
    assert len(subjects) == c['scope']['expected_upstream_subjects']
    assert len(ready) == c['scope']['expected_upstream_records']
    chosen = subjects[:c['scope']['subject_count']]
    rows = []
    for r in ready:
        if r['subject'] in chosen:
            base = dict(indexes[r['source_recording']]); base.update(r)
            base['common_samples'] = samplemap[r['source_recording']]['common_samples']
            assert base['status'].startswith('PASS') and base['split'] in ('train', 'val', 'test')
            assert base['subject'] == indexes[r['source_recording']]['subject']
            rows.append(base)
    for subject in chosen:
        groups = {k: {r['source_family'] for r in rows if r['subject'] == subject and r['split'] == k}
                  for k in ('train', 'val', 'test')}
        assert all(groups.values())
        assert not(groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test'])
    return chosen, sorted(rows, key=lambda r: (r['subject'], r['source_recording']))


def self_test():
    eeg = np.arange(100 * 2, dtype=float).reshape(100, 2); env = np.arange(100, dtype=float)
    t, dropped = support(eeg, env, [0, 2, 4], 13)
    assert np.array_equal(t, np.arange(83)) and dropped == 0
    x = design(eeg, t, [0, 2, 4], 13)
    assert np.array_equal(x[0], eeg[[13, 15, 17]].ravel())
    assert np.array_equal(x[-1], eeg[[95, 97, 99]].ravel())
    eeg[15, 0] = np.nan
    finite, dropped = support(eeg, env, [0, 2, 4], 13)
    assert 0 not in finite and dropped > 0
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1000, 8)); y = x @ rng.normal(size=8) + 0.1 * rng.normal(size=1000)
    stats = Stats(8)
    stats.update(x[:333], y[:333]); stats.update(x[333:], y[333:])
    mx, sx, my, sy, weights = stats.fit([0.01, 1])
    z = (x - mx) / sx; yz = (y - my) / sy
    reference = solve(z.T @ z / len(y) + 0.01 * np.eye(8), z.T @ yz / len(y), assume_a='pos')
    assert np.allclose(weights[:, 0], reference, atol=1e-10)
    score = Score(); p = z @ weights[:, 0]
    score.update(yz[:333], p[:333]); score.update(yz[333:], p[333:])
    m = score.metrics()
    assert np.isclose(m['r'], np.corrcoef(yz, p)[0, 1])
    assert np.isclose(m['r2'], 1 - np.sum((yz - p) ** 2) / np.sum((yz - yz.mean()) ** 2))
    # Future-response signal can recover current envelope with the explicit backward lag.
    target = rng.normal(size=500)
    signal = np.zeros((500, 2)); signal[2:, 0] = target[:-2]
    ti, _ = support(signal, target, [0, 2], 13)
    assert np.corrcoef(design(signal, ti, [2], 0)[:, 0], target[ti])[0, 1] > 0.999999
    return {'status': 'PASS', 'tests': ['lag direction', 'noncircular edge', 'shared finite mask', 'streaming ridge vs direct solve', 'streaming metrics vs direct formulas', 'synthetic delayed response']}


def fit_subject(subject, rows, c, out, capture=False):
    alphas = c['ridge']['alpha_candidates']
    variants = [(model, cond, lags, shift) for model, lags in c['models'].items() for cond, shift in c['conditions'].items()]
    all_lags = c['models']['backward_trf']; max_shift = max(c['conditions'].values())
    loaded = []; manifest = []; skips = []
    for row in rows:
        try:
            eeg, env = load_record(row)
            t, nonfinite = support(eeg, env, all_lags, max_shift)
            if len(t) < 2:
                raise ValueError('insufficient common finite support')
            loaded.append((row, eeg, env, t))
            manifest.append({'subject': subject, 'source_recording': row['source_recording'], 'raw_story': row['raw_story'],
                             'source_family': row['source_family'], 'split': row['split'], 'tail_flag': row['tail_flag'],
                             'common_samples': len(env), 'support_start_index': int(t[0]), 'support_stop_exclusive': len(env)-max_shift-max(all_lags),
                             'common_finite_samples': len(t), 'nonfinite_dropped': nonfinite,
                             'trimmed_boundary_samples': max_shift + max(all_lags)})
        except Exception as ex:
            skips.append({'subject': subject, 'source_recording': row['source_recording'], 'stage': 'load/support', 'error': repr(ex)})
    if {row['split'] for row, *_ in loaded} != {'train', 'val', 'test'}:
        raise ValueError('missing usable split after read failures')
    results = []; records = []; selections = []; parameters = {}; example = {}
    for model, cond, lags, shift in variants:
        print(f'{subject} {model} {cond} train', flush=True)
        stats = Stats(64 * len(lags))
        for row, eeg, env, t in loaded:
            if row['split'] != 'train':
                continue
            for start in range(0, len(t), 4096):
                ti = t[start:start+4096]
                stats.update(design(eeg, ti, lags, shift), env[ti])
        mx, sx, my, sy, weights = stats.fit(alphas)
        val = [[] for _ in alphas]
        for row, eeg, env, t in loaded:
            if row['split'] != 'val':
                continue
            scores = [Score() for _ in alphas]
            for start in range(0, len(t), 4096):
                ti = t[start:start+4096]
                pred = ((design(eeg, ti, lags, shift) - mx) / sx) @ weights
                y = (env[ti] - my) / sy
                for i in range(len(alphas)):
                    scores[i].update(y, pred[:, i])
            for i, score in enumerate(scores):
                val[i].append(score.metrics()['r'])
        v = [macro(values) for values in val]
        if not any(value is not None for value in v):
            raise ValueError('all validation correlations undefined')
        best = int(np.argmax([value if value is not None else -np.inf for value in v]))
        for i, alpha in enumerate(alphas):
            selections.append({'subject': subject, 'model': model, 'condition': cond, 'alpha': alpha,
                               'validation_record_macro_r': v[i], 'selected': i == best})
        weight = weights[:, best]
        parameters[f'{model}_{cond}_weights'] = weight
        parameters[f'{model}_{cond}_mean_x'] = mx; parameters[f'{model}_{cond}_std_x'] = sx
        parameters[f'{model}_{cond}_mean_y'] = my; parameters[f'{model}_{cond}_std_y'] = sy
        pooled = Score(); per = []
        first_test = True
        for row, eeg, env, t in loaded:
            if row['split'] != 'test':
                continue
            score = Score()
            for start in range(0, len(t), 4096):
                ti = t[start:start+4096]
                pred = ((design(eeg, ti, lags, shift) - mx) / sx) @ weight
                y = (env[ti] - my) / sy
                score.update(y, pred); pooled.update(y, pred)
                if capture and first_test and start == 0:
                    keep = min(1280, len(ti))
                    example[(model, cond)] = (ti[:keep]/64, y[:keep], pred[:keep], row['source_recording'])
            first_test = False
            m = score.metrics(); per.append(m)
            records.append({'subject': subject, 'model': model, 'condition': cond, 'source_recording': row['source_recording'],
                            'source_family': row['source_family'], 'alpha': alphas[best], **m})
        pm = pooled.metrics()
        results.append({'subject': subject, 'model': model, 'condition': cond, 'status': 'PASS',
                        'alpha': alphas[best], 'design_columns': len(mx), 'train_samples': stats.n,
                        'val_samples': sum(len(t) for row, _, _, t in loaded if row['split']=='val'),
                        'test_samples': pm['samples'], 'test_records': len(per), 'validation_record_macro_r': v[best],
                        'test_record_macro_r': macro([m['r'] for m in per]), 'test_record_macro_r2': macro([m['r2'] for m in per]),
                        'test_pooled_r': pm['r'], 'test_pooled_r2': pm['r2'],
                        'train_mean_record_macro_r2': macro([m['train_mean_r2'] for m in per]),
                        'undefined_r_records': sum(m['r'] is None for m in per)})
    np.savez(out / 'server_only' / f'{subject}_parameters.npz', **parameters)
    return results, records, selections, manifest, skips, example


def plots(out, subjects, results, example):
    colors = {'matched': '#1976B9', 'null_shift13s': '#C76B39'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, model in enumerate(('backward_trf', 'zero_lag_ridge')):
        for row, (metric, label) in enumerate([('test_record_macro_r', 'Test record-macro Pearson r'), ('test_record_macro_r2', 'Test record-macro R2')]):
            ax = axes[row, col]
            for cond in colors:
                vals = {r['subject']: r.get(metric) for r in results if r.get('model') == model and r.get('condition') == cond}
                ax.plot(range(len(subjects)), [vals.get(s, np.nan) for s in subjects], 'o-', label=cond, color=colors[cond])
            ax.axhline(0, color='0.6', lw=0.8)
            ax.set(xticks=range(len(subjects)), xticklabels=[s.removeprefix('sub-') for s in subjects], xlabel='Subject ID (fixed first 10)', ylabel=label, title=model)
            ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / 'performance.png', dpi=160); plt.close(fig)
    if example:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
        for ax, model in zip(axes, ('backward_trf', 'zero_lag_ridge')):
            t, y, p, name = example[(model, 'matched')]
            ax.plot(t, y, color='0.2', lw=1, label='Actual envelope (train standardized)')
            for cond in colors:
                t, _, p, _ = example[(model, cond)]
                ax.plot(t, p, color=colors[cond], lw=1, label=cond)
            ax.set(title=f'{model}: fixed first test record, first 20 s', xlabel='Publisher story time (s)', ylabel='Train standardized amplitude')
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(name, fontsize=9)
        fig.savefig(out / 'prediction_example.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path); parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true'); parser.add_argument('--subject', help='predeclared first-subject reproduction only')
    args = parser.parse_args()
    tests = self_test()
    if args.self_test:
        print(json.dumps(tests)); return
    if platform.system() != 'Linux' or not Path('/home/fanyu/auditory_simulation_m6a').is_dir():
        raise RuntimeError('server2203 project runtime required; no local signal processing')
    start = time.time(); c = json.loads(args.config.read_text(encoding='utf-8-sig'))
    subjects, rows = preflight(c)
    if args.subject:
        if args.subject != subjects[0]:
            raise ValueError('rerun only predeclared first selected subject')
        subjects = [args.subject]; rows = [r for r in rows if r['subject'] == args.subject]
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'server_only').mkdir()
    shutil.copy2(args.config, args.out / 'config_snapshot.json')
    shutil.copy2(__file__, args.out / Path(__file__).name)
    write_json(args.out/'technical_self_test.json', tests)
    write_json(args.out/'scope_frozen.json', {'subjects': subjects, 'records': len(rows), 'frozen_before_signal_fit_utc': datetime.now(timezone.utc).isoformat(), 'patient_data': 'FORBIDDEN'})
    write_csv(args.out/'split_manifest.csv', [{k:r[k] for k in ('subject','source_recording','source_family','raw_story','split','tail_flag')} for r in rows])
    results = []; records = []; selections = []; manifest = []; skips = []; failures = []; examples = {}
    for subject in subjects:
        try:
            sr, pr, ar, sm, sk, ex = fit_subject(subject, [r for r in rows if r['subject']==subject], c, args.out, capture=subject==subjects[0])
            results.extend(sr); records.extend(pr); selections.extend(ar); manifest.extend(sm); skips.extend(sk); examples.update(ex)
            print(f'COMPLETED {subject}: '+json.dumps(sr), flush=True)
        except Exception as exc:
            failures.append({'subject': subject, 'status': 'FAIL', 'error': repr(exc)})
            print(f'FAILED {subject} {exc!r}', flush=True)
        write_csv(args.out/'metrics_per_subject.csv', results)
    write_csv(args.out/'metrics_per_record.csv', records); write_csv(args.out/'alpha_selection.csv', selections)
    write_csv(args.out/'sample_manifest.csv', manifest); write_csv(args.out/'skipped_records.csv', skips)
    write_csv(args.out/'failed_subjects.csv', failures)
    summary = []
    deltas = []
    for model in c['models']:
        for cond in c['conditions']:
            selected = [r for r in results if r['model']==model and r['condition']==cond]
            summary.append({'model':model,'condition':cond,'subjects':len(selected),
                            **{k:macro([r[k] for r in selected]) for k in ('test_record_macro_r','test_record_macro_r2','test_pooled_r','test_pooled_r2')},
                            'test_samples':sum(r['test_samples'] for r in selected)})
        bysub = defaultdict(dict)
        for r in results:
            if r['model']==model:
                bysub[r['subject']][r['condition']] = r
        for subject, values in bysub.items():
            a = values['matched']; b = values['null_shift13s']
            assert a['test_samples'] == b['test_samples'] and a['train_samples'] == b['train_samples']
            deltas.append({'subject':subject,'model':model,'delta_r_matched_minus_null':a['test_record_macro_r']-b['test_record_macro_r'],
                           'delta_r2_matched_minus_null':a['test_record_macro_r2']-b['test_record_macro_r2']})
    for subject in subjects:
        vals = [r for r in results if r['subject']==subject]
        if vals:
            assert len({r['test_samples'] for r in vals}) == 1
    write_csv(args.out/'paired_deltas.csv', deltas)
    plots(args.out, subjects, results, examples)
    write_json(args.out/'run_summary.json', {'run_id':c['run_id'],'status':'PASS_WITH_LIMITATION' if not failures else 'PARTIAL_WITH_FAILURES',
              'scope_subjects':subjects,'scope_records':len(rows),'usable_records':len(manifest),'skipped_records':len(skips),
              'failed_subjects':failures,'summary':summary,'elapsed_seconds':time.time()-start,
              'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
              'finished_utc':datetime.now(timezone.utc).isoformat(),'same_support_checked':True,'split_family_disjoint_checked':True,
              'limitations':['initial first-10 within-subject cross-source-family only','publisher origin not physical synchronization','offline future EEG up to 500ms','single descriptive mismatch, no significance test','no semantic/audio waveform reconstruction','no deep EEG decoder trained']})
    print('FINISHED '+str(args.out), flush=True)


if __name__ == '__main__':
    main()
