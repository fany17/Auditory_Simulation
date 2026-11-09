"""Synthetic-only rendering/guard proof on2203; no actual outcomes read."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from pub_01_s03_ss07_baseline import ROOT, dump, sha
import pub_01_s03_ss07_result_figures as actor


def fixtures(m=7):
    assert m in (6, 7)
    family = 'm3' if m == 7 else 'm2'
    seeds = [1024, 2026, 31415, 42, 12345]; primary = family+'_h16'
    ids = ['native_P', 'P_EMA', 'P_FT', primary, family+'_current_h16', 'static_'+family+'_h16', 'output_prior_h16']
    if m == 7: ids.append('m3_reconstruct_h16')
    comparators = [ids[i] for i in [0, 4, 5, 6, 1, 2]+([7] if m == 7 else [])]
    comparisons = [('primary-'+id_, id_) for id_ in comparators]
    families = [f'm{i}_h16' for i in range(1, 6)]
    manifest = {'seeds': seeds, 'main_model_ids': ids, 'primary': primary,
        'contrast_ids_and_comparators': comparisons, 'exploratory_family_ids': families, 'EMA_alias': {'tau_seconds': 0}}
    def internal_row(name, i, n=5):
        point = .03 if i == 0 else .004*(i-3)
        # First mean lies OUTSIDE its interval: valid presentation must neither
        # fail on a negative errorbar length nor replace/recenter the interval.
        bounds = [.01, .02] if i == 0 else [point-.012, point+.01]
        return {'contrast': name, 'weighted_mean_difference': point,
            'weighted_Bonferroni_M_approx_bootstrap_interval': bounds,
            'seed_paired_weighted_differences': [point+.001*(s-(n-1)/2) for s in range(n)]}
    main_rows = [internal_row(name, i) for i, (name, comparator) in enumerate(comparisons)]
    # EMA tau0 duplicate is deliberately retained, not merged away.
    main_rows[4] = dict(main_rows[0], contrast=comparisons[4][0])
    external_rows = []
    for i, (name, comparator) in enumerate(comparisons):
        point = .002*(i-3)
        external_rows.append({'contrast': name, 'mean_r_difference': point,
            'Bonferroni_M_approx_interval': [point-.008, point+.011],
            'paired_seed_r_differences': [point+.001*(s-2) for s in range(5)]})
    external_rows[4] = dict(external_rows[0], contrast=comparisons[4][0])
    internal_absolute = []; external_absolute = []
    for ci, condition in enumerate(ids):
        for si, seed in enumerate(seeds):
            actual_index = 0 if condition == 'P_EMA' else ci
            internal_absolute.append({'condition': condition, 'seed': seed, 'weighted_r': .1+.002*actual_index+.001*si})
            external_absolute.append({'condition': condition, 'seed': seed, 'attended_r': .04+.001*actual_index+.002*si})
    # Extra three-seed family rows in INTERNAL_MAIN must not leak into the
    # all-main-five-seed absolute panel or cause false missing-seed failures.
    for condition in [f for f in families if f not in ids]:
        for seed in seeds[:3]: internal_absolute.append({'condition': condition, 'seed': seed, 'weighted_r': .09})
    internal = {'primary': primary, 'M': m, 'draws': 20000, 'contrasts': main_rows, 'absolute_per_seed': internal_absolute}
    external = {'primary': primary, 'M': m, 'draws': 20000, 'contrasts': external_rows, 'absolute_per_seed': external_absolute}
    exploratory = {'M': 5, 'contrasts': [internal_row(id_+'-native_P', i, 3) for i, id_ in enumerate(families)]}
    history = []
    for i, kind in enumerate(actor.KINDS):
        missing = kind == 'same_person_wrong_family'
        row = {'intervention': kind, 'coverage': {'test_seen': {'people': 3, 'records': 4, 'blocks': 16},
            'test_heldout': {'people': 0 if missing else 2, 'records': 0 if missing else 3, 'blocks': 0 if missing else 12}}}
        if missing: row['weighted_endpoint_unavailable_reason'] = 'synthetic subset lacks heldout stratum'
        else: row.update({'weighted_mean_difference': .003*(i-2), 'exploratory_95_approx_interval': [-.01, .02],
                         'crossed_Exp_sensitivity_not_CI': [-.015, .018]})
        history.append(row)
    totals = {stage: {'cells': 5 if stage in ['shared_pretrain','native_head','finetune'] else 54 if stage == 'screen' else 32,
        'optimizer_updates': 1000*(i+1), 'actual_seconds_including_any_reported_cache': 100*(i+1),
        'supervised_window_presentations': 0 if stage == 'shared_pretrain' else 10000*(i+1),
        'aux_teacher_presentations': 100 if stage == 'full' else 0} for i, stage in enumerate(actor.STAGES)}
    documents = dict(zip(actor.FILES, [internal, external, exploratory, {'diagnostics': history}, {'stage_totals': totals}]))
    return manifest, documents


def main():
    root = Path(tempfile.mkdtemp(prefix='synthetic_result_figures_', dir=ROOT/'engineering_history'))
    manifest, documents = fixtures(); original = copy.deepcopy(documents)
    result = actor.render(root/'presentation', manifest, documents, 'SYNTHETIC ENGINEERING FIXTURE')
    assert documents == original, 'renderer mutated source summaries'
    assert result['main_M'] == 7 and result['main_seed_dots'] == 70 and result['absolute_seed_dots'] == 80
    assert len(result['main_contrasts']) == len(set(result['main_contrasts'])) == 7
    assert result['history_missing_endpoints_preserved'] == ['same_person_wrong_family']
    assert result['fixed_main_model_order'] == manifest['main_model_ids']
    text = (root/'presentation/NUMERICAL_SUMMARY.md').read_text()
    assert 'SYNTHETIC ENGINEERING FIXTURE' in text
    assert '[0.010000, 0.020000]' in text and '0.030000' in text
    assert len(list((root/'presentation').glob('*.png'))) == 5
    for name in actor.FIGURES:
        p = root/'presentation'/name; assert p.stat().st_size > 10000
        assert p.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
    manifest6, documents6 = fixtures(6); original6 = copy.deepcopy(documents6)
    result6 = actor.render(root/'presentation_M6', manifest6, documents6, 'SYNTHETIC M6 FIXTURE')
    assert documents6 == original6 and result6['main_M'] == 6
    assert result6['main_seed_dots'] == 60 and result6['absolute_seed_dots'] == 70
    assert len(list((root/'presentation_M6').glob('*.png'))) == 5
    rejected = []
    bad = copy.deepcopy(documents); bad['INTERNAL_MAIN.json']['contrasts'].reverse()
    try: actor.render(root/'bad_order', manifest, bad, 'SYNTHETIC REJECT')
    except AssertionError: rejected.append('contrast_order_mismatch')
    else: raise AssertionError('reordered test contrasts silently accepted')
    bad = copy.deepcopy(documents); bad['EXTERNAL_DTU_MAIN.json']['contrasts'][4]['mean_r_difference'] += .001
    try: actor.render(root/'bad_alias', manifest, bad, 'SYNTHETIC REJECT')
    except AssertionError: rejected.append('tau0_alias_metric_changed')
    else: raise AssertionError('changed tau0 alias silently presented')
    bad = copy.deepcopy(documents); bad['INTERNAL_MAIN.json']['absolute_per_seed'].pop(0)
    try: actor.render(root/'bad_missing_seed', manifest, bad, 'SYNTHETIC REJECT')
    except KeyError: rejected.append('missing_main_seed')
    else: raise AssertionError('missing seed silently accepted')
    bad = copy.deepcopy(documents); bad['COSTS.json']['stage_totals']['unknown_extra_stage'] = {}
    try: actor.render(root/'bad_cost', manifest, bad, 'SYNTHETIC REJECT')
    except AssertionError: rejected.append('unaccounted_cost_stage')
    else: raise AssertionError('unknown costs silently excluded')
    calls = []
    def forbidden(*a, **k): calls.append('premature'); raise AssertionError('payload before complete review')
    with patch.object(actor, 'read', return_value={'technical_review_passed': False}), \
         patch.object(actor, 'gated', forbidden), patch.object(actor, 'render', forbidden):
        try: actor.main()
        except AssertionError: pass
        else: raise AssertionError('incomplete actual review accepted')
    assert not calls
    dump(ROOT/'RESULT_FIGURE_ENGINEERING_CHECK.json', {'passed': True, 'synthetic_only': True,
        'fixture_path': str(root), 'rendering': result, 'source_summaries_unchanged': documents == original,
        'M6_rendering': result6, 'M6_and_M7_both_synthetic_paths_exercised': True,
        'interval_outside_mean_retained_not_recentered': True, 'tau0_duplicate_contrast_retained': True,
        'missing_history_stratum_not_plotted_as_zero_or_reweighted': True,
        'rejected': rejected, 'premature_gated_or_renderer_calls': 0,
        'no_actual_test_or_external_scores_loaded': True,
        'source_sha256': {name: sha(ROOT/'input'/name) for name in [
            'pub_01_s03_ss07_result_figures.py', 'pub_01_s03_ss07_result_figures_check.py']}})
    print(json.dumps({'result_figure_engineering': 'PASS', 'fixture_path': str(root), 'no_actual_results': True}))


if __name__ == '__main__': main()
