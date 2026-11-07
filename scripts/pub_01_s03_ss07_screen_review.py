"""Independent metadata-only review after all54 screen and width lock.

Re-average every stored VAL block/person and recompute the fixed width rule.
No signal, checkpoint, test, DTU, training, or new model choice is performed.
Run on2203; this checks recorded metrics, not correctness of NN predictions.
"""
import csv
import json
import math
import time
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha


def main():
    selection = read(ROOT/'FULL_SELECTION.json')
    protocol = read(ROOT/'INCREMENTAL_PROTOCOL.json')
    assert selection['frozen'] and selection['test_read'] is False and not selection['screen_weights_reused']
    assert selection['parent_protocol_sha256'] == sha(ROOT/'INCREMENTAL_PROTOCOL.json')
    seeds = protocol['screening_seeds']; configs = protocol['conditions']
    assert seeds == [1024, 2026, 31415] and len(configs) == 18
    all_rows = []; scores = {}; canonical = None; max_person = 0.; max_macro = 0.
    for config in configs:
        cell_values = []
        for seed in seeds:
            folder = ROOT/f'screen/{config["id"]}_seed{seed}'
            cell = read(folder/'completed.json'); val = read(folder/'validation_macro.json')
            assert cell['seed'] == seed and cell['condition'] == config and cell['stage'] == 'screen'
            assert cell['protocol_sha256'] == selection['parent_protocol_sha256'] and not cell['test_read']
            assert cell['epochs'] <= 12 and cell['updates'] == cell['epochs']*3945
            assert val['checkpoint_selection'].startswith('original fullval3s')
            blocks = val['segments']; people = val['people']
            assert len(blocks) == 752 and len(people) == 49
            keys = {(b['record'], b['start']): (b['subject'], b['story'], b['samples']) for b in blocks}
            assert len(keys) == 752
            if canonical is None: canonical = keys
            assert keys == canonical, 'coverage/metadata differ across screen cells'
            person_map = {p['subject']: p['r'] for p in people}; assert len(person_map) == 49
            assert set(person_map) == {b['subject'] for b in blocks}
            recalculated = []
            for person in sorted(person_map):
                values = [b['r'] for b in blocks if b['subject'] == person]
                assert all(math.isfinite(v) for v in values)
                value = math.fsum(values)/len(values)
                difference = abs(value-person_map[person]); assert difference <= 1e-12
                max_person = max(max_person, difference); recalculated.append(value)
            macro = math.fsum(recalculated)/49
            reported = cell['validation_macro']; assert reported['subjects'] == 49 and reported['blocks'] == 752
            assert val['summary'] == reported
            difference = abs(macro-reported['macro_r']); assert difference <= 1e-12
            max_macro = max(max_macro, difference)
            cell_values.append(reported['macro_r'])
            windows = cell['supervised_window_presentations']; samples = cell['supervised_target_samples']
            assert windows == 252454*cell['epochs'] and samples == 192*windows
            all_rows.append({'condition': config['id'], 'seed': seed,
                'validation_macro_r': reported['macro_r'], 'independent_block_person_macro_r': macro,
                'epochs': cell['epochs'], 'updates': cell['updates'], 'selected_epoch_0based': cell['selected_epoch'],
                'supervised_window_presentations': windows, 'supervised_target_sample_presentations': samples,
                'validation_people': 49, 'validation_blocks': 752})
        # Original completed values preserve the exact frozen floating-point
        # ranking/tie rule; independently reaveraged values check their meaning.
        scores[config['id']] = math.fsum(cell_values)/3
    assert len(all_rows) == 54 and scores == selection['ranking_means_all18']
    widths = []; chosen = []
    for family in ['m1', 'm2', 'm3', 'm4', 'm5']:
        a = scores[f'{family}_h16']; b = scores[f'{family}_h32']
        width = 16 if a >= b else 32
        condition = f'{family}_h{width}'; chosen.append(condition)
        widths.append({'family': family, 'width16_mean_three_seed_validation60s_person_r': a,
            'width32_mean_three_seed_validation60s_person_r': b, 'selected_width': width,
            'selected_condition': condition, 'selection_uses_test': False,
            'full_training_starts_fresh_not_screen_weights': True})
    assert chosen == selection['selected_family_condition_ids']
    # The selector's complete ledger is checked by key and every reported value.
    with (ROOT/'SCREEN_LEDGER.csv').open(newline='') as f: ledger = list(csv.DictReader(f))
    assert len(ledger) == 54
    keyed = {(r['condition'], int(r['seed'])): r for r in ledger}; assert len(keyed) == 54
    for row in all_rows:
        stored = keyed[(row['condition'], row['seed'])]
        assert float(stored['validation_macro_r']) == row['validation_macro_r']
        assert int(stored['updates']) == row['updates'] and int(stored['selected_epoch']) == row['selected_epoch_0based']
        assert int(stored['target_presentations']) == row['supervised_target_sample_presentations']
    destination = ROOT/'screen_selection_review01'; destination.mkdir(exist_ok=False)
    for name, rows in [('SCREEN_WIDTHS.csv', widths), ('SCREEN_INDEPENDENT_VAL.csv', all_rows)]:
        with (destination/name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(destination/'SCREEN_SELECTION_REVIEW.json', {'passed': True, 'time': time.time(),
        'all54_successful_cells_and_ledger_checked': True, 'all_recorded_block_correlations_reaveraged': 54*752,
        'all_recorded_person_means_checked': 54*49, 'maximum_person_mean_difference': max_person,
        'maximum_macro_mean_difference': max_macro, 'fixed_widths_recomputed': widths,
        'all18_ranking_means_exactly_match': True, 'ties_use_smaller_width': True,
        'original_ledger_target_presentations_column_is_sample_not_window_count': True,
        'independent_CSV_uses_explicit_window_and_sample_units': True,
        'parent_protocol_sha256': selection['parent_protocol_sha256'],
        'full_selection_sha256': sha(ROOT/'FULL_SELECTION.json'),
        'source_sha256': sha(ROOT/'input/pub_01_s03_ss07_screen_review.py'),
        'no_actual_signals_checkpoints_test_or_DTU_payload_loaded': True,
        'not_a_raw_input_prediction_or_correlation_review': True,
        'validation_only_no_performance_superiority_or_goal_completion_claim': True})
    print(json.dumps({'screen_selection_review_passed': True, 'selected_conditions': chosen,
        'recorded_VAL_blocks_reaveraged': 54*752, 'maximum_macro_difference': max_macro}))


if __name__ == '__main__': main()
