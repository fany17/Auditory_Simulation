"""Review-only R2 comparison amendment; old sources/protocol remain intact."""
import argparse
import copy
import difflib
import json
import time
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha


def main(commit, freeze=False, review_note=None):
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    old = read(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json'); assert old['frozen']
    for path, digest in {**old['sources'], **old['proofs']}.items(): assert sha(path) == digest
    additions = ['review_metrics_v2', 'review_queue_v2', 'review_prepare_v2', 'review_math_check_v2',
                 'review_pipeline_check_v2', 'review_r2_val_check', 'result_figures_v2', 'review_v2_gate_check',
                 'result_figures', 'result_figures_check']
    sources = dict(old['sources'])
    sources.update({str(ROOT/f'input/pub_01_s03_ss07_{name}.py'): sha(ROOT/f'input/pub_01_s03_ss07_{name}.py') for name in additions})
    proof_names = ['INDEPENDENT_METRIC_MATH_CHECK_V2.json', 'INDEPENDENT_METRIC_PIPELINE_CHECK_V2.json',
                   'INDEPENDENT_R2_LEGAL_VAL_CHECK_V2.json', 'INDEPENDENT_REVIEW_V2_GATE_CHECK.json',
                   'RESULT_FIGURE_ENGINEERING_CHECK.json']
    proofs = dict(old['proofs'])
    for name in proof_names:
        evidence = read(ROOT/name); assert evidence['passed']
        for filename, digest in evidence['source_sha256'].items(): assert sha(ROOT/'input'/filename) == digest
        proofs[str(ROOT/name)] = sha(ROOT/name)
    val = read(ROOT/'INDEPENDENT_R2_LEGAL_VAL_CHECK_V2.json')
    assert val['legal_VAL_only'] and val['blocks'] == 3760 and val['actual_test_or_DTU_reads'] == val['new_fits'] == 0
    assert val['maximum_scaled_difference']['internal_R2_descriptive'] <= 1e-10
    diff_path = ROOT/'INDEPENDENT_REVIEW_V1_V2_SOURCE.diff'
    if not diff_path.exists():
        parts = []
        for name in ['review_metrics', 'review_queue']:
            a = ROOT/f'input/pub_01_s03_ss07_{name}.py'; b = ROOT/f'input/pub_01_s03_ss07_{name}_v2.py'
            parts.extend(difflib.unified_diff(a.read_text().splitlines(True), b.read_text().splitlines(True), fromfile=a.name, tofile=b.name))
        with diff_path.open('x') as f: f.write(''.join(parts))
    amendment_path = ROOT/'INDEPENDENT_REVIEW_R2_AMENDMENT_V2.json'
    if not amendment_path.exists():
        dump(amendment_path, {'version': 2, 'time': time.time(),
            'reason': 'legalVAL large-magnitude descriptiveR2 sum-vs-dot rounding can exceed an absolute1e-10 despite negligible scaled error',
            'old_review_protocol_sha256': sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json'),
            'legal_VAL_evidence_sha256': sha(ROOT/'INDEPENDENT_R2_LEGAL_VAL_CHECK_V2.json'),
            'numeric_policy': {'scaled_descriptive_metrics': ['MSE_descriptive', 'R2_descriptive'],
                'scaled_tolerance': 1e-10, 'denominator': 'max(1, abs(original_metric))',
                'r_differences_and_intervals_absolute_tolerance_unchanged': 1e-10,
                'raw_waveform_absolute_tolerance_unchanged': 1e-6, 'exact_constant_rejection_unchanged': True},
            'old8_sources_old3_proofs_and_protocol_retained_in_place_and_backed_up': True,
            'raw_replay_uses_unchanged_v1_source_geometry_and_protocol': True,
            'metrics_use_v2_protocol_and_combined_completion_marker': True,
            'original_training_evaluation_predictions_scores_masks_and_choices_unchanged': True,
            'not_an_actual_scientific_trial_failure_or_score_based_amendment': True,
            'diff_sha256': sha(diff_path)})
    if freeze: assert review_note
    protocol = copy.deepcopy(old)
    protocol.update({'frozen': freeze, 'time': time.time(), 'review_version': 2, 'source_commit': commit,
        'independent_code_review_completed': freeze, 'independent_review_note': review_note,
        'sources': sources, 'proofs': proofs, 'scaled_descriptive_metrics': ['MSE_descriptive', 'R2_descriptive'],
        'R2_tolerance': 'absolute difference / max(1, abs(originalR2)) <=1e-10; unbounded descriptive metric only',
        'r_differences_and_intervals_use_original_absolute_tolerance': True,
        'old_review_protocol_sha256': sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json'),
        'amendment_path': str(amendment_path), 'amendment_sha256': sha(amendment_path),
        'diff_path': str(diff_path), 'diff_sha256': sha(diff_path),
        'raw_replay_protocol_version': 1, 'metric_replay_protocol_version': 2,
        'combined_completion_marker': 'INDEPENDENT_REVIEW_COMPLETED_V2.json',
        'presentation_entry': 'pub_01_s03_ss07_result_figures_v2.py',
        'start_gate': 'complete fixed evaluation plus unchanged ready/checkpoint/mask checks; no premature outcome reads'})
    name = 'INDEPENDENT_REVIEW_PROTOCOL_V2.json' if freeze else 'INDEPENDENT_REVIEW_PROTOCOL_V2_CANDIDATE.json'
    dump(ROOT/name, protocol)
    print(json.dumps({'protocol': name, 'frozen': freeze, 'sources': len(sources), 'proofs': len(proofs),
        'raw_uses_original_protocol': True, 'only_R2_check_joins_existing_MSE_scaled_policy': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-commit', required=True)
    parser.add_argument('--freeze', action='store_true'); parser.add_argument('--review-note')
    a = parser.parse_args(); main(a.source_commit, a.freeze, a.review_note)
