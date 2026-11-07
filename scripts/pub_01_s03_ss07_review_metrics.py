"""Read-only independent metric replay AFTER the entire fixed evaluation.

Does not validate neural prediction correctness: that is a separate raw-input
replay. Reopens original targets and every saved fused prediction on2203.
Never loads actual test/DTU data before both completed-summary and ready gates.
"""
import json
import time
import numpy as np
from scipy.io import loadmat
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha
from pub_01_s03_ss07_final_manifest import require_ready
from pub_01_s03_ss07_review_math import (
    signal_metrics, pearson, support, crossed, family_sensitivity, external,
    linear_quantiles, population)

KINDS = ['reset', 'content_blind', 'shuffled', 'older_outside_fused_support',
         'same_person_wrong_family', 'same_story_wrong_person']


class Checks:
    def __init__(self):
        self.count = 0; self.max_absolute = {}; self.max_relative = {}

    def close(self, name, actual, expected, relative=False):
        a = np.asarray(actual, dtype=np.float64); b = np.asarray(expected, dtype=np.float64)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), name
        difference = float(np.max(np.abs(a - b), initial=0))
        scaled = float(np.max(np.abs(a - b) / np.maximum(1., np.abs(b)), initial=0))
        self.max_absolute[name] = max(difference, self.max_absolute.get(name, 0.))
        self.max_relative[name] = max(scaled, self.max_relative.get(name, 0.))
        assert (scaled if relative else difference) <= 1e-10, (name, difference, scaled)
        self.count += 1


def gated():
    # Read this metadata BEFORE require_ready, checkpoint loading, target arrays,
    # or signal/MAT access. A partial seed is never reviewed as full evidence.
    complete = read(ROOT / 'final_summary/COMPLETED.json')
    assert complete['all_fixed_cells_aggregated']
    assert not complete['scientific_goal_achieved_claimed']
    manifest, ready = require_ready()
    assert complete['checkpoint_manifest_sha256'] == sha(ROOT / 'FINAL_CHECKPOINT_MANIFEST.json')
    for seed in manifest['seeds']:
        cell = read(ROOT / f'final_results/seed{seed}/COMPLETED.json')
        assert cell['all_conditions_complete'] and cell['seed'] == seed
        assert cell['checkpoint_manifest_sha256'] == complete['checkpoint_manifest_sha256']
    protocol = read(ROOT / 'INDEPENDENT_REVIEW_PROTOCOL.json')
    assert protocol['frozen'] and protocol['metric_absolute_tolerance'] == 1e-10
    assert protocol['executor_protocol_sha256'] == sha(ROOT / 'EVALUATION_EXECUTOR_PROTOCOL.json')
    for path, digest in dict(protocol['sources'], **protocol['proofs']).items():
        assert sha(path) == digest
    return manifest, ready


def keymap(rows, fields):
    mapping = {tuple(row[f] for f in fields): row for row in rows}
    assert len(mapping) == len(rows), 'duplicate keys'
    return mapping


def paired_values(results, conditions, seeds, keys=None):
    canonical = None; metadata = None; all_values = []
    for condition in conditions:
        seed_values = []
        for seed in seeds:
            mapping = keymap(results[(condition, seed)], ['record', 'start'])
            if keys is not None:
                assert set(keys).issubset(mapping)
                mapping = {key: mapping[key] for key in keys}
            if canonical is None:
                canonical = sorted(mapping); metadata = [mapping[k] for k in canonical]
                assert canonical
            assert sorted(mapping) == canonical
            for key, meta in zip(canonical, metadata):
                assert all(mapping[key][f] == meta[f] for f in ['phase', 'subject', 'story', 'source_family', 'samples'])
            seed_values.append([mapping[key]['r'] for key in canonical])
        all_values.append(seed_values)
    return np.asarray(all_values), metadata


def absolute(rows):
    phases = {}
    for phase in ['test_seen', 'test_heldout']:
        people = sorted({r['subject'] for r in rows if r['phase'] == phase})
        if people:
            phases[phase] = sum(sum(r['r'] for r in rows if r['phase'] == phase and r['subject'] == person)
                / sum(r['phase'] == phase and r['subject'] == person for r in rows) for person in people) / len(people)
    return phases, (2 * phases['test_seen'] + phases['test_heldout']) / 3 if len(phases) == 2 else None


def internal_replay(manifest, checks):
    mask = read(ROOT / 'SPARR_TEST_MASK.json')
    masks = keymap(mask['record_masks'], ['record'])
    pairs = read(ROOT / 'INTERVENTION_PAIR_MASK.json')['pairs']
    all_keys = {(m['record'], b['start']) for m in mask['record_masks'] for b in m['blocks'] if b['eligible']}
    results = {}; cell_counts = []; signal_checks = 0
    for seed in manifest['seeds']:
        ids = list(manifest['main_model_ids'])
        if seed in manifest['screening_seeds']:
            ids += sorted(set(manifest['exploratory_family_ids']) - set(ids))
        ids += ['history__' + kind for kind in KINDS]
        stored = {}
        for condition in ids:
            cell = read(ROOT / f'final_results/seed{seed}/INTERNAL_{condition}.json')
            assert cell['seed'] == seed and cell['condition'] == condition and cell['complete'] and not cell['failures']
            mapping = keymap(cell['segments'], ['record', 'start'])
            kind = condition.removeprefix('history__')
            expected = {tuple(key) for key in pairs[kind]['eligible_keys']} if kind in pairs else all_keys
            assert set(mapping) == expected, 'missing/extra frozen keys'
            stored[condition] = (cell, mapping); results[(condition, seed)] = []
        for row in mask['records']:
            record = row['record']; n = row['common_samples']; covered = support(n)
            truth = np.asarray(np.load(row['envelope_path'], mmap_mode='r')).reshape(-1)[:n]
            record_mask = masks[(record,)]; assert record_mask['common_samples'] == n
            native = None
            for condition in ids:
                cell, mapping = stored[condition]
                expected = [key for key in mapping if key[0] == record]
                if not expected:
                    continue  # only frozen ineligible donor records
                prediction = np.load(ROOT / f'final_results/seed{seed}/predictions/internal/{condition}/{record}.npy', mmap_mode='r')
                assert prediction.dtype == np.float64 and prediction.shape == (n,) and np.isfinite(prediction).all()
                if condition == 'native_P': native = np.array(prediction)
                if condition == 'P_EMA' and manifest['EMA_alias']['tau_seconds'] == 0:
                    assert np.array_equal(prediction, native), 'tau0 must be an exact saved waveform alias'
                for block in record_mask['blocks']:
                    start = block['start']; keep = covered[start:start + 3840]
                    target = truth[start:start + 3840][keep]
                    reason = ('insufficient_overlap_support' if len(target) < 3800 else
                        'nonfinite_target' if not np.isfinite(target).all() else
                        'constant_target' if np.std(target) == 0 else None)
                    assert block['samples'] == len(target) and block['eligible'] == (reason is None) and block['reason'] == reason
                    if not block['eligible']: continue
                    source = mapping[(record, start)]
                    assert all(source[f] == row[f] for f in ['phase', 'subject', 'story', 'source_family'])
                    assert source['samples'] == len(target)
                    calculated = signal_metrics(prediction[start:start + 3840][keep], target)
                    for name, value in calculated.items():
                        checks.close('internal_' + name, value, source[name], relative=name == 'MSE_descriptive')
                    results[(condition, seed)].append(dict(source, **calculated)); signal_checks += 1
        for condition in ids:
            cell, mapping = stored[condition]; rows = results[(condition, seed)]
            assert len(rows) == len(mapping)
            phases, weighted = absolute(rows)
            assert set(phases) == set(cell['summary']['phase_macro_r'])
            for phase, value in phases.items(): checks.close('absolute_phase', value, cell['summary']['phase_macro_r'][phase])
            if weighted is None: assert cell['summary']['weighted_r'] is None
            else: checks.close('absolute_weighted', weighted, cell['summary']['weighted_r'])
            cell_counts.append({'seed': seed, 'condition': condition, 'blocks': len(rows), 'records': len({r['record'] for r in rows})})
    return results, {'cells': cell_counts, 'all_block_metrics_replayed': signal_checks}


def check_summary(diff, blocks, names, stored, checks):
    assert stored['M'] == len(names) and stored['draws'] == 20000
    point, samples = crossed(diff, blocks)
    sensitivity = family_sensitivity(diff, blocks)
    rows = stored['contrasts']; assert [r['contrast'] for r in rows] == names
    for i, row in enumerate(rows):
        checks.close('internal_seed_differences', point[i, :, 0], row['seed_paired_weighted_differences'])
        checks.close('internal_point', point[i, :, 0].mean(), row['weighted_mean_difference'])
        checks.close('internal_95', linear_quantiles(samples[i, :, 0], [.025, .975]), row['weighted_95_approx_bootstrap_interval'])
        tail = .05 / (2 * len(names))
        checks.close('internal_Bonferroni', linear_quantiles(samples[i, :, 0], [tail, 1-tail]), row['weighted_Bonferroni_M_approx_bootstrap_interval'])
        checks.close('internal_family_sensitivity', linear_quantiles(sensitivity[i, :, 0], [.025, .975]), row['weighted_crossed_Exp_sensitivity_interval_not_calibrated_CI'])
        for j, phase in [(1, 'seen'), (2, 'heldout')]:
            checks.close('internal_support_point', point[i, :, j].mean(), row['supporting_endpoints'][phase]['mean_difference'])
            checks.close('internal_support_95', linear_quantiles(samples[i, :, j], [.025, .975]), row['supporting_endpoints'][phase]['exploratory_95_interval'])


def internal_statistics(manifest, results, checks):
    ids = manifest['main_model_ids']; seeds = manifest['seeds']; primary = ids.index(manifest['primary'])
    values, blocks = paired_values(results, ids, seeds)
    contrasts = manifest['contrast_ids_and_comparators']
    diff = np.stack([values[primary] - values[ids.index(other)] for name, other in contrasts])
    check_summary(diff, blocks, [name for name, other in contrasts], read(ROOT / 'final_summary/INTERNAL_MAIN.json'), checks)
    ids = ['native_P'] + manifest['exploratory_family_ids']
    values, blocks = paired_values(results, ids, manifest['screening_seeds'])
    check_summary(values[1:] - values[:1], blocks, [i + '-native_P' for i in ids[1:]],
                  read(ROOT / 'final_summary/INTERNAL_FIVE_FAMILY_EXPLORATORY.json'), checks)
    pair_masks = read(ROOT / 'INTERVENTION_PAIR_MASK.json')['pairs']
    stored = read(ROOT / 'final_summary/INTERNAL_HISTORY_DIAGNOSTICS.json')['diagnostics']
    assert [row['intervention'] for row in stored] == KINDS
    for kind, row in zip(KINDS, stored):
        keys = [tuple(k) for k in pair_masks[kind]['eligible_keys']] if kind in pair_masks else None
        values, blocks = paired_values(results, [manifest['primary'], 'history__' + kind], seeds, keys)
        assert {tuple(k) for k in row['eligible_keys']} == {(b['record'], b['start']) for b in blocks}
        for phase in ['test_seen', 'test_heldout']:
            selected = [b for b in blocks if b['phase'] == phase]
            assert row['coverage'][phase] == {'people': len({b['subject'] for b in selected}),
                'records': len({b['record'] for b in selected}), 'blocks': len(selected)}
        if all(row['coverage'][p]['people'] for p in ['test_seen', 'test_heldout']):
            point, samples = crossed(values[:1] - values[1:], blocks)
            sensitivity = family_sensitivity(values[:1] - values[1:], blocks)
            checks.close('history_seeds', point[0, :, 0], row['true_minus_intervention_seed_weighted_differences'])
            checks.close('history_point', point[0, :, 0].mean(), row['weighted_mean_difference'])
            checks.close('history_95', linear_quantiles(samples[0, :, 0], [.025, .975]), row['exploratory_95_approx_interval'])
            checks.close('history_sensitivity', linear_quantiles(sensitivity[0, :, 0], [.025, .975]), row['crossed_Exp_sensitivity_not_CI'])
            for seed, saved in zip(seeds, row['matched_true_subset_absolute']):
                selected = [r for r in results[(manifest['primary'], seed)] if (r['record'], r['start']) in {(b['record'], b['start']) for b in blocks}]
                phases, weighted = absolute(selected)
                checks.close('history_matched_true_absolute', weighted, saved['weighted_r'])
        else:
            assert 'weighted_endpoint_unavailable_reason' in row and 'weighted_mean_difference' not in row


def external_replay(manifest, checks):
    ids = manifest['main_model_ids']; seeds = manifest['seeds']
    values = np.empty((len(ids), 5, 18, 60)); accuracies = np.empty_like(values)
    cells = {}
    for ci, condition in enumerate(ids):
        for si, seed in enumerate(seeds):
            cell = read(ROOT / f'final_results/seed{seed}/EXTERNAL_{condition}.json')
            assert cell['complete'] and not cell['failures'] and cell['condition'] == condition and cell['seed'] == seed
            mapping = keymap(cell['trials'], ['subject', 'trial'])
            assert set(mapping) == {(f'S{p}', t) for p in range(1, 19) for t in range(1, 61)}
            cells[(ci, si)] = mapping
    covered = support(3200); assert np.flatnonzero(covered).tolist() == list(range(1, 3167))
    for person in range(1, 19):
        data = loadmat(ROOT / f'external_dtu/extracted/S{person}_data_preproc.mat', simplify_cells=True)['data']
        for trial in range(1, 61):
            key = f'S{person}_trial{trial:02d}'
            a = np.asarray(data['wavA'][trial-1], dtype=np.float64)[covered]
            b = np.asarray(data['wavB'][trial-1], dtype=np.float64)[covered]
            for si, seed in enumerate(seeds):
                native = None
                for ci, condition in enumerate(ids):
                    prediction = np.load(ROOT / f'final_results/seed{seed}/predictions/external/{condition}/{key}.npy', mmap_mode='r')
                    assert prediction.shape == (3200,) and prediction.dtype == np.float64 and np.isfinite(prediction).all()
                    if condition == 'native_P': native = np.array(prediction)
                    if condition == 'P_EMA' and manifest['EMA_alias']['tau_seconds'] == 0:
                        assert np.array_equal(native, prediction)
                    ra = pearson(prediction[covered], a); rb = pearson(prediction[covered], b)
                    accuracy = 1. if ra > rb else .5 if ra == rb else 0.
                    row = cells[(ci, si)][(f'S{person}', trial)]
                    assert row['samples'] == 3166
                    checks.close('DTU_rA', ra, row['rA']); checks.close('DTU_rB', rb, row['rB'])
                    assert accuracy == row['aux_accuracy']
                    values[ci, si, person-1, trial-1] = ra
                    accuracies[ci, si, person-1, trial-1] = accuracy
    saved = read(ROOT / 'final_summary/EXTERNAL_DTU_MAIN.json')
    for row in saved['absolute_per_seed']:
        ci = ids.index(row['condition']); si = seeds.index(row['seed'])
        checks.close('DTU_absolute', values[ci, si].mean(), row['attended_r'])
        checks.close('DTU_person_absolute', values[ci, si].mean(1), row['per_person_attended_r'])
        checks.close('DTU_accuracy', accuracies[ci, si].mean(), row['aux_accuracy_descriptive'])
        checks.close('DTU_person_accuracy', accuracies[ci, si].mean(1), row['per_person_aux_accuracy'])
    contrasts = manifest['contrast_ids_and_comparators']; primary = ids.index(manifest['primary'])
    differences = np.stack([values[primary] - values[ids.index(other)] for name, other in contrasts])
    point, samples = external(differences)
    assert saved['draws'] == 20000 and saved['M'] == len(contrasts)
    assert [r['contrast'] for r in saved['contrasts']] == [c[0] for c in contrasts]
    for i, row in enumerate(saved['contrasts']):
        checks.close('DTU_point', point[i].mean(), row['mean_r_difference'])
        checks.close('DTU_seed_diff', point[i].mean(1), row['paired_seed_r_differences'])
        checks.close('DTU_person_diff', point[i].mean(0), row['paired_person_r_differences'])
        checks.close('DTU_95', linear_quantiles(samples[i], [.025, .975]), row['approx_95_interval'])
        tail = .05 / (2 * len(contrasts))
        checks.close('DTU_Bonferroni', linear_quantiles(samples[i], [tail, 1-tail]), row['Bonferroni_M_approx_interval'])
        leave_one_out = [(point[i].sum() - point[i, :, p].sum()) / (5 * 17) for p in range(18)]
        checks.close('DTU_LOPO', leave_one_out, row['leave_one_person_out_descriptive'])
    return {'cells': len(ids) * 5, 'all_trial_pairs_replayed': len(ids) * 5 * 1080}


def main():
    manifest, ready = gated()  # No actual prediction/target loads before this.
    destination = ROOT / 'independent_review'; destination.mkdir(exist_ok=False)
    checks = Checks(); started = time.time()
    try:
        results, internal_info = internal_replay(manifest, checks)
        internal_statistics(manifest, results, checks)
        external_info = external_replay(manifest, checks)
        report = {'passed': True, 'checks': checks.count, 'maximum_absolute_difference': checks.max_absolute,
            'maximum_scaled_difference': checks.max_relative, 'internal': internal_info, 'external': external_info,
            'seconds': time.time()-started, 'all_actual_targets_and_fused_predictions_stayed_on_server': True,
            'independent_centered_dot_product_and_frequency_matrix_implementation': True,
            'same_prespecified_rng_draw_order_20k_intervals_and_original_stratum_weights': True,
            'does_not_validate_neural_prediction_correctness': True,
            'raw_input_neural_replay_required_separately': True, 'scientific_goal_complete': False,
            'checkpoint_manifest_sha256': sha(ROOT / 'FINAL_CHECKPOINT_MANIFEST.json'),
            'ready_sha256': sha(ROOT / 'EVALUATION_READY.json'),
            'source_sha256': {name: sha(ROOT / 'input' / name) for name in [
                'pub_01_s03_ss07_review_math.py', 'pub_01_s03_ss07_review_metrics.py']}}
        dump(destination / 'METRICS_REVIEW.json', report)
        print(json.dumps({'metric_review_passed': True, 'checks': checks.count}))
    except Exception as error:
        dump(destination / 'METRICS_REVIEW_FAILURE.json', {'error': repr(error), 'checks_completed': checks.count,
            'maximum_absolute_difference': checks.max_absolute, 'seconds': time.time()-started,
            'no_predictions_or_original_results_modified': True})
        raise


if __name__ == '__main__':
    main()
