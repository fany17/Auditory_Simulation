"""Synthetic-only independent review proof; execute on2203, no EEG loads."""
import json
from unittest.mock import patch
import numpy as np
from pub_01_s03_ss07_baseline import ROOT, dump, sha
import pub_01_s03_ss07_review_math as review
import pub_01_s03_ss07_review_metrics as actor
from pub_01_s03_ss07_statistics import crossed_seed_person, crossed_family_sensitivity
from pub_01_s03_ss07_dtu_statistics import paired
from pub_01_s03_ss07_eval_common import correlation, fuse, summarize_absolute


def main():
    rng = np.random.default_rng(8721); checks = actor.Checks()
    metadata = []
    for phase, prefix, people in [('test_seen', 'a', 7), ('test_heldout', 'b', 3)]:
        for person in range(people):
            for block in range(1 + person % 4):
                metadata.append({'record': f'{prefix}{person}', 'subject': f'{prefix}{person}',
                    'phase': phase, 'start': 3840 * block, 'story': f's{block % 2}',
                    'source_family': f'f{(person + block) % 5}'})
    differences = rng.normal(.02, .03, size=(7, 5, len(metadata)))
    expected_point, expected_samples = crossed_seed_person(differences, metadata)
    actual_point, actual_samples = review.crossed(differences, metadata)
    checks.close('synthetic_crossed_points', actual_point, expected_point)
    checks.close('synthetic_crossed_all20k_samples', actual_samples, expected_samples)
    checks.close('synthetic_family_all20k_samples', review.family_sensitivity(differences, metadata),
                 crossed_family_sensitivity(differences, metadata))
    external_values = rng.normal(.02, .03, size=(7, 5, 18, 60))
    external_reference = paired(external_values, [f'c{i}' for i in range(7)])
    person, samples = review.external(external_values)
    for i, row in enumerate(external_reference['contrasts']):
        checks.close('synthetic_external_point', person[i].mean(), row['mean_r_difference'])
        checks.close('synthetic_external_95', review.linear_quantiles(samples[i], [.025, .975]), row['approx_95_interval'])
        tail = .05 / 14
        checks.close('synthetic_external_Bonferroni', review.linear_quantiles(samples[i], [tail, 1-tail]), row['Bonferroni_M_approx_interval'])
    for size in [2, 3, 100, 20000]:
        values = rng.normal(size=size)
        checks.close('synthetic_quantiles', review.linear_quantiles(values, [0, .05/14, .025, .5, .975, 1-.05/14, 1]),
                     np.quantile(values, [0, .05/14, .025, .5, .975, 1-.05/14, 1]))
    for length in [192, 3200, 3840, 7737]:
        _, expected = fuse(rng.normal(size=((length-192)//96+1, 192)), length)
        assert np.array_equal(review.support(length), expected)
    for length in [3, 3166, 3840]:
        a = rng.normal(size=length); b = rng.normal(size=length)
        checks.close('synthetic_pearson', review.pearson(a, b), correlation(a, b))
        checks.close('synthetic_pearson_positive', review.pearson(a, a), 1.)
        checks.close('synthetic_pearson_negative', review.pearson(a, -a), -1.)
    rejected = []
    for name, call in [
        ('constant_prediction', lambda: review.pearson(np.ones(3), np.arange(3))),
        ('nonfinite_target', lambda: review.pearson(np.arange(3), np.array([0, 1, np.nan]))),
        ('duplicate_block', lambda: review.population(differences[:, :, :2], [metadata[0], metadata[0]])),
        ('missing_stratum', lambda: review.population(differences[:, :, :1], metadata[:1])),
        ('mismatch_keys', lambda: actor.paired_values({('c', 1): [dict(metadata[0], r=.2, samples=3800)],
            ('c', 2): [dict(metadata[-1], r=.2, samples=3800)]}, ['c'], [1, 2])),
    ]:
        try: call()
        except AssertionError: rejected.append(name)
        else: raise AssertionError('did not reject ' + name)
    for length, value in [(3840, .1), (3166, .3), (3, .1)]:
        constant = np.full(length, value); varying = np.arange(length, dtype=np.float64)**2
        for side, a, b in [('prediction', constant, varying), ('target', varying, constant)]:
            try: review.pearson(a, b)
            except AssertionError: rejected.append(f'exact_nonbinary_constant_{side}_{length}_{value}')
            else: raise AssertionError('nonbinary constant was accepted')
    # Actual ready is absent, but this check mocks the first metadata read so it
    # remains valid even if the scientific queue completes in parallel.
    loads = {'arrays': 0, 'MAT': 0, 'ready': 0}
    def forbidden_array(*a, **k): loads['arrays'] += 1; raise RuntimeError('premature payload')
    def forbidden_mat(*a, **k): loads['MAT'] += 1; raise RuntimeError('premature MAT')
    def forbidden_ready(): loads['ready'] += 1; raise RuntimeError('ready called before completed gate')
    with patch.object(actor, 'read', return_value={'all_fixed_cells_aggregated': False}), \
         patch.object(actor.np, 'load', forbidden_array), patch.object(actor, 'loadmat', forbidden_mat), \
         patch.object(actor, 'require_ready', forbidden_ready):
        try: actor.main()
        except AssertionError: pass
        else: raise AssertionError('incomplete summary accepted')
    assert loads == {'arrays': 0, 'MAT': 0, 'ready': 0}
    dump(ROOT / 'INDEPENDENT_METRIC_MATH_CHECK.json', {'passed': True, 'synthetic_only': True,
        'checks': checks.count, 'maximum_absolute_difference': checks.max_absolute,
        'every_internal20k_resample_and_family_weight_sample_compared': True,
        'external20k_intervals_compared': True, 'manual_quantile_extremes_checked': True,
        'reject_checks': rejected, 'incomplete_summary_payload_loads': loads,
        'actual_test_or_DTU_signal_reads': 0,
        'source_sha256': {name: sha(ROOT/'input'/name) for name in [
            'pub_01_s03_ss07_review_math.py', 'pub_01_s03_ss07_review_metrics.py',
            'pub_01_s03_ss07_review_math_check.py']}})
    print(json.dumps({'synthetic_independent_metric_math': 'PASS', 'checks': checks.count}))


if __name__ == '__main__': main()
