"""D21 saved-fit audit and bounded direct-bin replay; no fit or new test."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
import sys, json, hashlib, math, time, signal, resource
from pathlib import Path
from collections import defaultdict

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_output_ridge_20261106')
D19 = ROOT.parent / 's03_physical_stream_20261106'
D20 = ROOT.parent / 's03_recent_prediction_context_20261106'
PROTOCOL = 'c5c6ab5ced4b8811a3fdfa96fd6c786ed53e182e71d7c82e7098a1b825fcff0c'
SOURCE = '2a21690924aae786a1d46cb5ad68b1a4702170d33e4719cb9934d8c09d67abcc'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
wall, cpu = time.monotonic(), time.process_time()
count = dict(fits=0, NN_forwards=0, producer_linear_points=0, independent_linear_points=0,
             nominal_kernel_points=0, frozen_readout_windows=0, saved_blocks=0, normal_equations=0)

def save(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def failure(kind, error, trace):
    import traceback
    save('COORDINATOR_TERMINAL_AUDIT_FAILURE.json', dict(passed=False, error=repr(error),
         counters=count, source_sha256=sha(__file__), CPU_seconds=time.process_time()-cpu,
         wall_seconds=time.monotonic()-wall))
    traceback.print_exception(kind, error, trace)

sys.excepthook = failure
assert Path(__file__).resolve().parent == ROOT
assert sha(ROOT / 'PROTOCOL.json') == PROTOCOL
protocol = read(ROOT / 'PROTOCOL.json')
assert sha(ROOT / 'input/pub_01_s03_ss07_output_ridge.py') == SOURCE
for path, digest in protocol['frozen_pins'].items():
    assert sha(path) == digest, path
done = read(ROOT / 'fit/COMPLETED.json')
assert done['passed'] and done['protocol_sha256'] == PROTOCOL
assert not (ROOT / 'fit/FAILURE.json').exists()
proc = Path('/proc') / str(done['process']['pid'])
if proc.exists():
    ticks = int((proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19])
    assert ticks != done['process']['linux_startticks'], 'Formal writer remains live'
for key, n in protocol['formal_budget'].items():
    assert done['counters'].get(key, 0) == n, key
metadata = read(D19 / 'PREPARATION_METADATA.json')
for path, digest in metadata['provenance'].items():
    assert sha(path) == digest
for path, stamp in metadata['inputs'].items():
    s = Path(path).stat()
    assert (s.st_size, s.st_mtime_ns) == (stamp['bytes'], stamp['mtime_ns'])

def timeout(*_):
    raise TimeoutError('Bounded D21 audit exceeded 180 seconds')

signal.signal(signal.SIGALRM, timeout)
signal.alarm(180)
import numpy as np
sys.path.insert(0, str(D19 / 'input'))
sys.path.insert(0, str(ROOT / 'input'))
import pub_01_s03_ss07_physical_stream_core as core
import pub_01_s03_ss07_output_ridge as production

errors = defaultdict(float)
def close(label, a, b, tolerance=1e-10):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    err = float(np.max(np.abs(a-b))) if a.size else 0.0
    errors[label] = max(errors[label], err)
    assert err <= tolerance, (label, err, tolerance)

expected = {(b['record'], b['start']): b for b in metadata['phases']['val']['blocks']}
assert len(expected) == 752
def aggregate(metric):
    assert len(metric['blocks']) == 752
    assert {(b['record'], b['start']) for b in metric['blocks']} == set(expected)
    people = defaultdict(list)
    for block in metric['blocks']:
        assert block['samples'] == expected[block['record'], block['start']]['samples']
        assert math.isfinite(block['r']) and -1 <= block['r'] <= 1
        assert math.isfinite(block['MSE']) and block['MSE'] >= 0
        people[block['subject']].append(block)
    assert len(people) == 49 and len(metric['people']) == 49
    saved = {p['subject']: p for p in metric['people']}
    assert set(saved) == set(people)
    rr, mm = [], []
    for subject, blocks in people.items():
        r = math.fsum(b['r'] for b in blocks) / len(blocks)
        mse = math.fsum(b['MSE'] for b in blocks) / len(blocks)
        close('person_r', r, saved[subject]['r'])
        close('person_MSE', mse, saved[subject]['MSE'])
        assert saved[subject]['blocks'] == len(blocks)
        rr.append(r); mm.append(mse)
    close('macro_r', math.fsum(rr)/49, metric['r'])
    close('macro_MSE', math.fsum(mm)/49, metric['MSE'])
    assert metric['active_points'] == 2866799
    assert math.isfinite(metric['active_MSE']) and metric['active_MSE'] >= 0
    assert math.isfinite(metric['full_record_r']) and -1 <= metric['full_record_r'] <= 1
    count['saved_blocks'] += len(metric['blocks'])

rows = []
chosen = [metadata['phases']['val']['records'][i] for i in (0, -1)]
assert chosen[0]['record'] != chosen[1]['record']
for seed in (1024, 2026, 31415):
    metric_path = ROOT / 'fit' / f'seed{seed}_METRICS.json'
    coeff_path = ROOT / 'fit' / f'seed{seed}_COEFFICIENTS_AND_GRAM.npz'
    result = read(metric_path)
    assert result['seed'] == seed
    aggregate(result['anchor'])
    anchor_reference = read(D20 / 'fit' / f'seed{seed}_ANCHOR.json')['anchor']['val']
    for label in ('r', 'active_MSE', 'MSE'):
        close('anchor_' + label, result['anchor'][label], anchor_reference[label])
    norm = read(D20 / 'fit' / f'seed{seed}_TRAIN_STATS.json')['P_past_prediction']
    mu, sd = np.float32(norm['mean']), np.float32(norm['std'])
    assert not norm['constant'] and sd > 0
    with np.load(coeff_path, allow_pickle=False) as f:
        a = {k: f[k].copy() for k in f.files}
    n = int(a['n'])
    assert n == 23703510 and a['xx'].shape == (97, 97)
    assert all(np.isfinite(v).all() for v in a.values())
    close('gram_symmetric', a['xx'], a['xx'].T, 1e-7)
    close('train_anchor_MSE', float(a['rr'])/n, result['train_anchor_active_MSE'])
    assert set(result['models']) == {f'{role}_{lam:g}' for role in ('A','F') for lam in (1.,.01,.0001)}
    for key, model in result['models'].items():
        d = 1 if model['role'] == 'A' else 97
        lam = model['penalty']
        assert key == f"{model['role']}_{lam:g}" and lam in (1., .01, .0001)
        beta = a[key + '_beta']; intercept = float(a[key + '_intercept'])
        assert beta.shape == (d,)
        close('intercept_saved', intercept, model['intercept'])
        # Augmented raw moments independently include the free intercept.
        moments = np.empty((d+1,d+1)); moments[0,0] = 1
        moments[0,1:] = moments[1:,0] = a['x'][:d]/(n*np.sqrt(d))
        moments[1:,1:] = a['xx'][:d,:d]/(n*d)
        cross = np.r_[float(a['r'])/n, a['xr'][:d]/(n*np.sqrt(d))]
        theta = np.r_[intercept,beta]
        penalty = np.r_[0., np.full(d, lam)]
        residual = moments@theta - cross + penalty*theta
        close('normal_equation', residual, np.zeros(d+1), 1e-10)
        mse = float(a['rr'])/n - 2*float(theta@cross) + float(theta@moments@theta)
        close('train_MSE', mse, model['train_active_MSE'])
        close('objective', mse + lam*float(beta@beta), model['objective'])
        assert mse >= -1e-12
        count['normal_equations'] += 1
        slope = beta / (np.sqrt(d)*float(sd))
        kernel = np.zeros(385); kernel[-1] = 1+slope[0]
        if d == 97: kernel[:-1] = np.repeat(slope[1:]/4, 4)
        close('kernel', kernel, a[key+'_physical_kernel'])
        close('kernel_intercept', intercept-float(mu)*math.fsum(slope), a[key+'_physical_intercept'])
        close('current_gain', kernel[-1], model['current_gain'])
        close('past_gain_sum', kernel[:-1].sum(), model['past_gain_sum'])
        close('past_gain_L1', np.abs(kernel[:-1]).sum(), model['past_gain_L1'])
        aggregate(model['val'])
    selections = {}
    for role in ('A','F'):
        best, best_r = None, result['anchor']['r']
        for lam in (1.,.01,.0001):
            key = f'{role}_{lam:g}'
            if result['models'][key]['val']['r'] > best_r:
                best, best_r = key, result['models'][key]['val']['r']
        assert result['selection'][role]['selected'] == best
        assert done['selections'][str(seed)][role]['selected'] == best
        close('selected_r', best_r, result['selection'][role]['r'])
        close('selected_delta', best_r-result['anchor']['r'], result['selection'][role]['delta_r'])
        selections[role] = dict(selected=best,r=best_r,delta_r=best_r-result['anchor']['r'])
        pm = result['anchor'] if best is None else result['models'][best]['val']
        ar = {p['subject']:p['r'] for p in result['anchor']['people']}
        selections[role]['people_positive'] = sum(p['r']>ar[p['subject']] for p in pm['people'])
        selections[role]['people_negative'] = sum(p['r']<ar[p['subject']] for p in pm['people'])

    # Fixed metadata endpoints; no new scientific metric or candidate selection.
    files = core.assets(seed)
    z = np.load(files['folder']/'val_latent.npy', mmap_mode='r',allow_pickle=False)
    p = np.load(files['folder']/'val_P.npy', mmap_mode='r',allow_pickle=False)
    readout = core.frozen_assets(seed)
    for row in chosen:
        first, stop = row['offset'], row['offset']+row['windows']
        b, keep = core.legacy.fuse(core.anchor_frames(z[first:stop],p[first:stop],readout),row['common_samples'])
        b = b[:row['covered_stop']]
        count['frozen_readout_windows'] += row['windows']
        spans = [(1344,1408),(len(b)-64,len(b))]
        points = np.concatenate([np.arange(lo,hi) for lo,hi in spans])
        assert len(set(points)) == 128
        assert keep[points].all()
        rolling = core.rolling_mean(b,4)
        xp = np.concatenate([production.features(b,rolling,lo,hi,(mu,sd),np) for lo,hi in spans])
        explicit = np.asarray([np.r_[b[t],b[t-384:t].reshape(96,4).mean(axis=1,dtype=np.float64)] for t in points],dtype=np.float32)
        xr = ((explicit-mu)/sd).astype(np.float64)
        close('explicit_bins', xp,xr,2e-6)
        for key, model in result['models'].items():
            d=1 if model['role']=='A' else 97
            beta=a[key+'_beta']; intercept=float(a[key+'_intercept'])
            actual=b[points]+intercept+xp[:,:d]@(beta/np.sqrt(d))
            manual=np.asarray([b[t]+intercept+math.fsum(float(x)*float(w)/math.sqrt(d) for x,w in zip(xrow[:d],beta)) for t,xrow in zip(points,xr)])
            close('functional_prediction',actual,manual,2e-6)
            count['producer_linear_points'] += len(points)
            count['independent_linear_points'] += len(points)
            nominal=np.asarray([float(a[key+'_physical_intercept'])+math.fsum(float(x)*float(w) for x,w in zip(b[t-384:t+1],a[key+'_physical_kernel'])) for t in points])
            # Report the rounding discrepancy; nominal FIR is not the exact FP32 function.
            errors['nominal_kernel_rounding'] = max(errors['nominal_kernel_rounding'],float(np.max(np.abs(actual-nominal))))
            count['nominal_kernel_points'] += len(points)
    rows.append(dict(seed=seed,anchor_r=result['anchor']['r'],selection=selections,
                     metrics_sha256=sha(metric_path),coefficients_sha256=sha(coeff_path)))
    del z,p,readout,a

assert count['producer_linear_points']==count['independent_linear_points']==count['nominal_kernel_points']==4608
assert count['frozen_readout_windows']==3756 and count['normal_equations']==18
assert count['saved_blocks']==15792
d20 = read(D20/'COORDINATOR_SCIENTIFIC_REVIEW.json')
assert d20['passed']
means={role:math.fsum(r['selection'][role]['r'] for r in rows)/3 for role in ('A','F')}
comparison=dict(mean_r=means,anchor_mean_r=math.fsum(r['anchor_r'] for r in rows)/3,
                D20_mean_r=d20['mean_r'],F_minus_A=means['F']-means['A'],F_minus_D20=means['F']-d20['mean_r'])
out=dict(passed=True,source_sha256=sha(__file__),protocol_sha256=PROTOCOL,
         completion_sha256=sha(ROOT/'fit/COMPLETED.json'),selected_records=[r['record'] for r in chosen],
         results=rows,comparison=comparison,max_errors=dict(errors),counters=count,
         CPU_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
         peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         scope='Full saved budget/18 normal equations/coefficients/primary block-to-person aggregation/selection audit. Fixed first-last VAL record endpoints: 4608 outputs per implementation, no new scoring/fit. Frozen Ccap reconstruction reused, no raw neural replay. Does not independently rebuild full TRAIN Gram or all raw VAL predictions/full-record metric.',
         goal_complete=False)
save('COORDINATOR_TERMINAL_AUDIT.json',out)
signal.alarm(0)
print(json.dumps(out,allow_nan=False))
