"""Independent recorded-VAL/selection/budget review of the first15 full cells.

No fits, arrays, checkpoints, test or external outcomes; execute on2203.
Checks the already frozen choice; never selects a different actual model.
"""
import csv
import json
import math
import time
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha


def main():
    primary = read(ROOT/'FULL_PRIMARY_SELECTION.json'); widths = read(ROOT/'FULL_SELECTION.json')
    protocol = read(ROOT/'INCREMENTAL_PROTOCOL.json')
    assert primary['frozen'] and not primary['test_read'] and not widths['screen_weights_reused']
    assert primary['parent_full_selection_sha256'] == sha(ROOT/'FULL_SELECTION.json')
    seeds = protocol['screening_seeds']; assert seeds == [1024, 2026, 31415]
    catalog = {c['id']: c for c in protocol['full_conditions']}
    candidates = [catalog[id_] for id_ in widths['selected_family_condition_ids']]
    assert len(candidates) == 5 and {c['family'] for c in candidates} == {'m1','m2','m3','m4','m5'}
    rows = []; means = {}; reference_keys = None; maximum_mean_error = 0.
    for config in candidates:
        seed_scores = []
        for seed in seeds:
            folder = ROOT/f'full/{config["id"]}_seed{seed}'
            cell = read(folder/'completed.json'); val = read(folder/'validation_macro.json')
            history = read(folder/'history.json'); resolved = read(folder/'resolved_config.json')
            assert cell['seed'] == seed and cell['condition'] == config and cell['stage'] == 'full' and not cell['test_read']
            assert cell['protocol_sha256'] == sha(ROOT/'INCREMENTAL_PROTOCOL.json')
            assert cell['engineering_amendment_sha256'] == sha(ROOT/'FULL_ENGINEERING_AMENDMENT.json')
            assert resolved['encoder_sha256'] == read(ROOT/f'baseline/head_seed{seed}/cache_manifest.json')['encoder_sha256']
            assert len(history) == cell['epochs'] <= 250 and cell['updates'] == 3945*cell['epochs']
            values = [r['validation_negative_Pearson'] for r in history]
            assert values and all(math.isfinite(v) for v in values)
            best_index = min(range(len(values)), key=lambda i: (values[i], i))
            assert best_index == cell['selected_epoch'] and values[best_index] == cell['best_validation_negative_Pearson']
            best = math.inf; stale = 0
            for index, h in enumerate(history):
                value = values[index]
                if value < best: best = value; stale = 0
                else: stale += 1
                assert h['epoch'] == index and h['updates'] == (index+1)*3945 and h['stale'] == stale
            if len(history) < 250: assert stale == 10
            window_count = cell['supervised_window_presentations']; sample_count = cell['supervised_target_samples']
            assert window_count == 252454*len(history) and sample_count == window_count*192
            expected_teachers = (252454-2*426)*len(history) if config['family'] == 'm3' else 0
            assert cell['teacher_presentations'] == expected_teachers
            blocks = val['segments']; people = val['people']
            assert len(blocks) == 752 and len(people) == 49
            keys = {(r['record'], r['start']): (r['subject'], r['story'], r['samples']) for r in blocks}
            assert len(keys) == 752
            if reference_keys is None: reference_keys = keys
            assert keys == reference_keys
            subject_means = []
            recorded = {r['subject']: r['r'] for r in people}; assert len(recorded) == 49
            assert set(recorded) == {r['subject'] for r in blocks}
            for subject in sorted(recorded):
                rs = [r['r'] for r in blocks if r['subject'] == subject]
                value = math.fsum(rs)/len(rs); assert abs(value-recorded[subject]) <= 1e-12
                subject_means.append(value)
            macro = math.fsum(subject_means)/49
            assert val['summary'] == cell['validation_macro']
            error = abs(macro-cell['validation_macro']['macro_r']); assert error <= 1e-12
            maximum_mean_error = max(maximum_mean_error, error)
            score = cell['validation_macro']['macro_r']; seed_scores.append(score)
            rows.append({'condition': config['id'], 'seed': seed, 'validation60s_person_macro_r': score,
                'independent_recorded_block_person_macro_r': macro, 'epochs': len(history), 'updates': cell['updates'],
                'selected_epoch_0based': best_index, 'stopping': 'PATIENCE10' if len(history)<250 else 'EPOCH_CEILING250',
                'supervised_window_presentations': window_count, 'supervised_target_sample_presentations': sample_count,
                'aux_teacher_presentations': cell['teacher_presentations']})
        means[config['id']] = math.fsum(seed_scores)/3
    assert means == primary['full_family_validation_means']
    family_order = ['m1', 'm2', 'm3', 'm4', 'm5']
    expected = max(candidates, key=lambda c: (means[c['id']], -c['width'], -family_order.index(c['family'])))
    assert expected['id'] == primary['primary']
    family = expected['family']; width = expected['width']
    controls = [f'{family}_current_h{width}', f'static_{family}_h{width}', f'output_prior_h{width}']
    if family == 'm3': controls.append(f'm3_reconstruct_h{width}')
    if family == 'm4': controls.append('static_alignment')
    if family == 'm5': controls.append(f'm5_prior_h{width}')
    assert controls == primary['same_full_budget_control_ids']
    assert all(id_ in widths['approved_condition_ids'] for id_ in controls)
    assert primary['final_seeds'] == [1024,2026,31415,42,12345]
    assert primary['extra_seeds_fixed_before_any_test'] and primary['five_seed_primary_not_reselected_after_extra_seeds']
    destination = ROOT/'primary_selection_review01'; destination.mkdir(exist_ok=False)
    with (destination/'FULL_INITIAL15_BUDGET.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {'passed': True, 'time': time.time(), 'initial_full_cells': 15,
        'recorded_VAL_blocks_reaggregated': 15*752, 'people_means_checked': 15*49,
        'maximum_macro_reaggregation_difference': maximum_mean_error, 'three_seed_means': means,
        'selected_primary_matches_frozen_rule': primary['primary'], 'matched_controls_match_preapproved_rule': controls,
        'final_seeds_prespecified': primary['final_seeds'], 'never_reselected_after_extra_seeds': True,
        'expected_total_unique_full_cells': 15+2+5*len(controls), 'new_full_cells_after_initial15': 2+5*len(controls),
        'full_selection_sha256': sha(ROOT/'FULL_SELECTION.json'),
        'primary_selection_sha256': sha(ROOT/'FULL_PRIMARY_SELECTION.json'),
        'source_sha256': sha(ROOT/'input/pub_01_s03_ss07_primary_review.py'),
        'recorded_metrics_and_history_only_not_raw_prediction_review': True,
        'no_test_external_or_checkpoint_payload_reads': True, 'no_scientific_superiority_or_goal_complete_claim': True}
    dump(destination/'PRIMARY_SELECTION_REVIEW.json', result)
    print(json.dumps({'primary_review': 'PASS', 'primary': primary['primary'],
        'controls': controls, 'unique_full_cells': result['expected_total_unique_full_cells']}))


if __name__ == '__main__': main()
