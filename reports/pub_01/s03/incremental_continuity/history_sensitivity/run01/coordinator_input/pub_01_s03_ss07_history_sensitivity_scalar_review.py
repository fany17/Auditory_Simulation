"""D17 coordinator scalar-consistency review on2203; zero neural replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_history_sensitivity_20261102')
SHAPES = {
    'input_history': [48, 2, 8, 192], 'token_before_norm': [48, 2, 32],
    'token_after_norm': [48, 2, 32], 'GRU_states': [48, 2, 32],
    'GRU_last_h': [48, 32], 'film_gamma': [48, 32], 'film_beta': [48, 32],
    'film_tanh_gamma': [48, 32], 'film_gain_delta': [48, 32],
    'film_bias_delta': [48, 32], 'current_q': [48, 32],
    'film_modulated_current': [48, 32], 'residual': [48, 192],
    'final_prediction_float64': [48, 192],
}
CHECKS = 0


def check(ok, label):
    global CHECKS
    CHECKS += 1
    assert ok, label


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=1e-12, rtol=1e-10):
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def run(pin):
    start, cpu = time.time(), time.process_time()
    check(sha(ROOT / 'PROTOCOL.json') == pin, 'protocol')
    p = read(ROOT / 'PROTOCOL.json')
    for path, digest in p['source_pins'].items():
        check(sha(Path(path)) == digest, path)
    for item in p['checkpoints']:
        check(sha(Path(item['path'])) == item['sha256'], item['path'])
    done = read(ROOT / 'COMPLETED.json')
    check(done['passed'] and done['protocol_sha256'] == pin, 'completion')
    check(done['residual_NN_windows'] == 384 and done['forward_calls'] == 8
          and done['history_input_backward_calls'] == 8, 'fixed computation counts')
    check(done['optimizer_updates'] == done['fits'] == done['new_encoder_head'] == 0,
          'no fitting or encoder forward')
    summaries = []
    for cp in p['checkpoints']:
        item = read(ROOT / f"H_seed{cp['seed']}_{cp['kind']}.json")
        check(item['seed'] == cp['seed'] and item['checkpoint_kind'] == cp['kind']
              and item['checkpoint_epoch'] == cp['epoch'], 'checkpoint identity')
        check(item['metadata'] == p['metadata'], 'same predetermined metadata')
        check(item['effective_windows'] == 48 and item['metadata_windows'] == 64,
              'window counts')
        check(item['current_and_anchor_tensor_equal'] and item['weights_exactly_unchanged'],
              'runtime assertions recorded')
        conditions = {x['condition']: x for x in item['conditions']}
        check(set(conditions) == {'near', 'older'}, 'conditions')
        for label, row in conditions.items():
            check(set(row['nodes']) == set(SHAPES), 'all nodes')
            for name, x in row['nodes'].items():
                check(x['shape'] == SHAPES[name], name + ' shape')
                r, m, c = x['RMS'], x['max_abs'], x['across_windows_centered_RMS']
                check(all(math.isfinite(v) and v >= 0 for v in (r, m, c)), name + ' finite')
                check(c <= r + 1e-12 and r <= m + 1e-12, name + ' RMS bounds')
                if 'per_token_position_across_windows_centered_RMS' in x:
                    a, b = x['per_token_position_across_windows_centered_RMS']
                    check(close(c * c, (a * a + b * b) / 2), name + ' token aggregation')
            g = row['history_input_gradient']
            n, norm, maximum = g['nonzero_scalars'], g['L2_norm'], g['max_abs']
            check(g['finite'] and g['scalars'] == 147456 and 0 <= n <= g['scalars'], 'gradient count')
            check(math.isfinite(norm) and math.isfinite(maximum) and 0 <= maximum <= norm + 1e-12,
                  'gradient magnitude')
            check(norm <= math.sqrt(n) * maximum + 1e-12, 'gradient norm upper bound')
            check((n == 0) == (norm == maximum == 0), 'zero gradient consistency')
        differences = item['near_minus_older']
        check(set(differences) == set(SHAPES), 'all node differences')
        for name, d in differences.items():
            a, b = [conditions[x]['nodes'][name] for x in ('near', 'older')]
            rms, maximum = d['difference_RMS'], d['difference_max_abs']
            check(math.isfinite(rms) and math.isfinite(maximum) and 0 <= rms <= maximum + 1e-12,
                  name + ' difference bounds')
            check(rms <= a['RMS'] + b['RMS'] + 1e-12, name + ' triangle RMS')
            check(maximum <= a['max_abs'] + b['max_abs'] + 1e-12, name + ' triangle max')
            for key, source, ratio in [
                ('pooled_global_RMS', 'RMS', 'relative_difference_RMS'),
                ('pooled_across_window_centered_RMS', 'across_windows_centered_RMS',
                 'relative_to_across_window_centered_RMS')]:
                den = math.sqrt((a[source] ** 2 + b[source] ** 2) / 2)
                check(close(d[key], den), name + ' pooled denominator')
                check(d[ratio] is None if den == 0 else close(d[ratio], rms / den), name + ' ratio/null')
            lo, median, hi = [d['per_window_difference_RMS_' + x] for x in ('min', 'median', 'max')]
            check(0 <= lo <= median <= hi and lo <= rms + 1e-12 and rms <= hi + 1e-12,
                  name + ' per-window bounds')
            fraction, count = d['exactly_equal_element_fraction'], d['exactly_equal_window_count']
            check(0 <= fraction <= 1 and 0 <= count <= 48 and count <= fraction * 48 + 1e-12,
                  name + ' equality counts')
            elements = math.prod(SHAPES[name])
            check(abs(fraction * elements - round(fraction * elements)) < 1e-8,
                  name + ' equality fraction integral')
            check((maximum == 0) == (rms == 0) == (fraction == 1) == (count == 48),
                  name + ' exact equality')
        check(differences['current_q']['exactly_equal_element_fraction'] == 1, 'fixed current control')
        for key in ('difference_RMS', 'difference_max_abs', 'per_window_difference_RMS_min',
                    'per_window_difference_RMS_median', 'per_window_difference_RMS_max'):
            check(close(differences['final_prediction_float64'][key], .1 * differences['residual'][key]),
                  'anchor plus .1 residual ' + key)
        for target, source in [('film_gain_delta', 'film_tanh_gamma'), ('film_bias_delta', 'film_beta')]:
            scale = sum(conditions[x]['nodes'][source]['max_abs'] for x in ('near', 'older'))
            tolerance = 4 * 2 ** -23 * .1 * scale + 1e-12
            for key in ('difference_RMS', 'difference_max_abs'):
                check(close(differences[target][key], .1 * differences[source][key], atol=tolerance),
                      target + ' float32 scaling ' + key)
        summaries.append(dict(seed=cp['seed'],kind=cp['kind'],epoch=cp['epoch'],
            residual_difference_RMS=differences['residual']['difference_RMS'],
            prediction_difference_RMS=differences['final_prediction_float64']['difference_RMS']))
    result = dict(passed=True, protocol_sha256=pin, source_sha256=sha(Path(__file__)),
        comparisons=CHECKS, summaries=summaries, new_NN=0, new_backward=0, fits=0, optimizer_updates=0,
        CPU_seconds=time.process_time() - cpu, wall_seconds=time.time() - start,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limits=['Source, metadata and scalar arithmetic consistency review only; no independent neural or activation replay',
                'Runtime unchanged-input/weight assertions inspected, not independently replayed',
                'Fixed48 windows from one subject; one input-gradient direction, not global sensitivity'], goal_complete=False)
    with (ROOT / 'COORDINATOR_SCALAR_REVIEW.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha', required=True)
    run(parser.parse_args().protocol_sha)
