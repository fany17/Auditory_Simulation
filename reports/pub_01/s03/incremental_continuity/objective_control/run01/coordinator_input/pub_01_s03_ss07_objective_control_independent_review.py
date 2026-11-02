"""D15 predeclared CPU numerical review after all nine fits, on 2203 only.

Manual functional GRU/pairwise Hann from the independent D12 reviewer. No
production imports, fitting, new checkpoint selection, test, or external data.
Pass the reviewed protocol SHA explicitly; never infer it from a changed file.
"""
import argparse
import hashlib
import json
import resource
import time
import numpy as np
import torch
import pub_01_s03_ss07_full_ft_history_independent_review as a

ROOT = a.BASE / 's03_full_ft_objective_control_20261102'
OUT = ROOT / 'coordinator_review'


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def audit_offline(folder, counts):
    for epoch in (-1, 0, 4, 9, 19):
        d = a.read(folder / f'OFFLINE_TRAIN_epoch{epoch}.json')
        assert d['parameters_fixed'] and d['windows'] == 252454
        assert len(d['people']) == 70 and len(d['blocks']) == 6104
        assert np.isfinite(d['window_MSE']) and d['window_MSE'] >= 0
        assert -1.000000000001 <= d['negative_cosine'] <= 1.000000000001
        assert d['target_window_counts'] == counts
        for person in d['people']:
            bb = [b for b in d['blocks'] if b['subject'] == person['subject']]
            assert person['blocks'] == len(bb)
            for k in ('r', 'MSE'):
                assert abs(np.mean([b[k] for b in bb]) - person[k]) < 1e-12
        for k in ('r', 'MSE'):
            assert abs(np.mean([p[k] for p in d['people']]) - d[f'physical60s_{k}']) < 1e-12
        records = d['amplitude_overlap_records']
        assert len(records) == 426 and sum(r['windows'] for r in records) == 252454
        assert d['overlap_pairs'] == 252454 - 426
        for key in ('maximum_window_abs', 'maximum_fused_abs'):
            assert d[key] == max(r[key] for r in records)
        for key in ('window_prediction_mean', 'window_prediction_RMS'):
            power = 2 if key.endswith('RMS') else 1
            mean = sum(r['windows'] * r[key] ** power for r in records) / 252454
            actual = mean ** (1 / power)
            assert np.isclose(actual, d[key], rtol=1e-12, atol=1e-12)
        overlap = sum((r['windows'] - 1) * r['overlap_MSE'] for r in records) / d['overlap_pairs']
        assert np.isclose(overlap, d['overlap_MSE'], rtol=1e-12, atol=1e-12)
        for key in counts:
            if key != 'windows':
                assert sum(r['target_' + key] for r in records) == counts[key]
    return 5


def run(pin):
    start, cpu = time.time(), time.process_time()
    torch.set_num_threads(4)
    assert len(pin) == 64 and a.sha(ROOT / 'PROTOCOL.json') == pin
    done = a.read(ROOT / 'QUEUE_COMPLETED.json')
    assert done['formal_residual_NN_windows'] == 59473248
    assert done['total_updates'] == 710100 and len(done['workers']) == 9
    assert all(x['returncode'] == 0 for x in done['workers'])
    p = a.read(ROOT / 'PROTOCOL.json')
    for path, digest in p['source_pins'].items():
        assert a.sha(path) == digest, path
    OUT.mkdir(exist_ok=False)
    a.write(OUT / 'STARTED.json', dict(protocol_sha256=pin,
        source_sha256=a.sha(__file__), helper_sha256=a.sha(a.__file__),
        device='CPU', threads=4, max_NN_windows=21 * 30059,
        evaluations='All9 selected/all9 last19; reuse identical states; H selected older3',
        new_encoder_head=0, optimizer_updates=0, learning_model_fits=0,
        tolerances=a.TOLERANCES))
    frozen = a.read(a.D8 / 'PROTOCOL.json')
    rows, mask = frozen['records']['val'], frozen['mask']['blocks']
    summaries, offline_files, permutation_logs, gradient_logs, reused = [], 0, 0, 0, 0
    for seed in a.SEEDS:
        z, anchor, prior = a.inputs(seed, rows)
        expected = {x['epoch']: x['sha256'] for x in p['permutations'] if x['seed'] == seed}
        assert len(expected) == 20
        first_batches = {}
        for epoch, digest in expected.items():
            order = np.random.default_rng(np.random.SeedSequence([seed, 20261102, epoch])).permutation(252454)
            assert np.array_equal(np.sort(order), np.arange(252454))
            assert hashlib.sha256(order.astype('<i8').tobytes()).hexdigest() == digest
            first_batches[epoch] = order[:64].tolist()
        for arm in a.ARMS:
            folder = ROOT / f'fits/{arm}_seed{seed}'
            completed = a.read(folder / 'COMPLETED.json')
            hist = [json.loads(l) for l in (folder / 'HISTORY.jsonl').read_text().splitlines()]
            orders = [json.loads(l) for l in (folder / 'PERMUTATIONS.jsonl').read_text().splitlines()]
            gradients = [json.loads(l) for l in (folder / 'FIRST_BATCH_GRADIENTS.jsonl').read_text().splitlines()]
            assert len(gradients) == 20
            for epoch, item in enumerate(gradients):
                assert item['epoch'] == epoch and item['updates_before_step'] == 3945 * epoch
                assert item['batch_global_indices'] == first_batches[epoch]
                assert item['effective_windows'] + item['fallback_windows'] == 64
                assert item['new_forward_windows'] == item['extra_optimizer_updates'] == 0
                assert set(item['modules']) == {'token', 'gru', 'film', 'current', 'output'}
                for module, value in item['modules'].items():
                    assert value['nonzero_scalars'] >= 0 and np.isfinite(value['L2_norm']) and value['L2_norm'] >= 0
                    if epoch == 0 and module != 'output':
                        assert value['nonzero_scalars'] == value['L2_norm'] == 0
                gradient_logs += 1
            assert len(hist) == len(orders) == 20 and completed['updates'] == 78900
            for epoch, (record, order) in enumerate(zip(hist, orders)):
                assert record['epoch'] == order['epoch'] == epoch
                assert record['updates'] == 3945 * (epoch + 1)
                assert order['sha256'] == expected[epoch] and order['fallback_presentations'] == 3834
                assert record['effective_TRAIN_windows'] == 248620 and record['full_target_windows'] == 252454
                permutation_logs += 1
            incumbent = a.read(folder / 'INCUMBENT.json')['phases']['val']['r']
            selected = int(np.argmax([incumbent] + [h['validation']['r'] for h in hist])) - 1
            assert selected == completed['selected_epoch']
            ss = torch.load(folder / 'best_validation.pt', map_location='cpu', weights_only=True)
            ll = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
            for cp in (ss, ll):
                assert cp['seed'] == seed and cp['arm'] == arm
            assert ss['epoch'] == selected and ll['epoch'] == 19
            pred = a.predictions(ss['model'], arm, rows, z, anchor, prior)
            metrics = a.score(pred, rows, mask)
            a.compare_metrics(f'{arm}/{seed}/selected', metrics, a.read(folder / 'SELECTED_METRICS.json'))
            a.write(OUT / f'{arm}_seed{seed}_selected.json', metrics)
            if state_hash(ss['model']) == state_hash(ll['model']):
                last = metrics
                reused += 1
            else:
                last_pred = a.predictions(ll['model'], arm, rows, z, anchor, prior)
                last = a.score(last_pred, rows, mask)
                del last_pred
            a.compare(f'{arm}/{seed}/last19', last['phases']['val'], hist[-1]['validation'])
            a.write(OUT / f'{arm}_seed{seed}_last19.json', last)
            if arm == 'H_full':
                older = a.predictions(ss['model'], arm, rows, z, anchor, prior, older=True)
                om = a.score(older, rows, mask)
                a.compare_metrics(f'{arm}/{seed}/older', om, a.read(folder / 'OLDER_METRICS.json'))
                a.write(OUT / f'{arm}_seed{seed}_older.json', om)
                del older
            offline_files += audit_offline(folder, p['target_window_counts']['train'])
            summaries.append(dict(seed=seed, arm=arm, selected_epoch=selected,
                selected_r=metrics['phases']['val']['r'], last19_r=last['phases']['val']['r'],
                selected_state_sha256=state_hash(ss['model']), last_state_sha256=state_hash(ll['model'])))
            print(json.dumps(summaries[-1]), flush=True)
            del ss, ll, pred, metrics, last
        del z, anchor, prior
    assert a.COUNT['residual_NN_window_forwards'] == (21 - reused) * 30059
    result = dict(passed=not a.DIFFERENCES, source_sha256=a.sha(__file__),
        helper_sha256=a.sha(a.__file__), protocol_sha256=pin, checks=a.CHECKS,
        failures=a.DIFFERENCES, summaries=summaries,
        all180_actual_permutation_logs_checked=permutation_logs,
        all180_existing_backward_gradient_logs_checked=gradient_logs,
        offline_TRAIN_aggregation_files=offline_files, selected_equals_last_reused=reused,
        cost=dict(**a.COUNT, CPU_seconds=time.process_time() - cpu,
            wall_seconds=time.time() - start, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            GPU_bytes=0),
        limits=['No second complete TRAIN neural forward for objective/window MSE/amplitude diagnostics',
            'No replay of unretained intermediate epochs',
            'Original repeatedly viewed VAL, not independent confirmation',
            'Known objective adaptation is not architectural novelty'], goal_complete=False)
    a.write(OUT / 'REVIEW.json', result)
    assert result['passed'], a.DIFFERENCES


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha', required=True)
    run(parser.parse_args().protocol_sha)
