"""Freeze a reviewed read-only protocol; metadata/code/proofs only, no data."""
import argparse
import json
import time
from pub_01_s03_ss07_baseline import ROOT, OLD, read, dump, sha
from pub_01_s03_ss07_review_raw import choose_sparr


def main(commit, freeze=False, review_note=None):
    assert len(commit) == 40 and all(c in '0123456789abcdef' for c in commit)
    sources = ['review_math', 'review_metrics', 'review_raw', 'review_queue',
               'review_math_check', 'review_raw_check', 'review_pipeline_check', 'review_prepare']
    source_map = {str(ROOT/f'input/pub_01_s03_ss07_{name}.py'):
                  sha(ROOT/f'input/pub_01_s03_ss07_{name}.py') for name in sources}
    proofs = ['INDEPENDENT_METRIC_MATH_CHECK.json', 'INDEPENDENT_METRIC_PIPELINE_CHECK.json',
              'INDEPENDENT_RAW_INPUT_VAL_CHECK.json']
    for name in proofs:
        proof = read(ROOT/name); assert proof['passed']
        for filename, digest in proof['source_sha256'].items():
            assert sha(ROOT/'input'/filename) == digest, 'stale proof: '+filename
    rejection = read(ROOT/proofs[0])['reject_checks']
    assert sum(name.startswith('exact_nonbinary_constant_') for name in rejection) == 6
    metadata = OLD/'envelope_readout_seed1024/manifest_frozen.json'
    rows = read(metadata)['records']; assert len(rows) == 619
    chosen = choose_sparr(rows)
    assert len(chosen) <= 4 and len(set(chosen)) == len(chosen)
    original_protocol = sha(ROOT/'EVALUATION_EXECUTOR_PROTOCOL.json')
    assert original_protocol == '24fffddda54be91ca3dea9c5024e903ef88f92d06331e22fbf93603b6aef56a4'
    if freeze:
        assert review_note, 'record the actual independent code-review evidence'
    result = {'frozen': freeze, 'time': time.time(), 'source_commit': commit,
        'independent_code_review_completed': freeze, 'independent_review_note': review_note,
        'sources': source_map, 'proofs': {str(ROOT/name): sha(ROOT/name) for name in proofs},
        'executor_protocol_sha256': original_protocol,
        'upstream_final_evaluation_not_modified': True,
        'start_gate': 'complete full five-seed fixed evaluation AND require_ready before any actual test/DTU load',
        'Sparr_metadata_only_rule': 'per phase lexicographic first and first non64-aligned origin; deduplicate',
        'Sparr_records': chosen, 'Sparr_manifest_path': str(metadata), 'Sparr_manifest_sha256': sha(metadata),
        'DTU_representative_subjects': ['S1', 'S11', 'S18'], 'DTU_representative_trials': [1, 60],
        'S11_rationale': 'prespecified input-scale outlier, not selected using model outcomes',
        'raw_scope': 'all five seeds/all main models on the fixed selected records/trials',
        'raw_neural_architecture_and_forward_reused': True,
        'raw_independence': 'separate readers, checkpoint roles, B256 encoder, raw global64 FT neighboring batch fill, adapter, fusion',
        'raw_limitations': 'representative not exhaustive; no independent neural arithmetic or history-intervention raw replay',
        'waveform_absolute_tolerance': 1e-6, 'metric_absolute_tolerance': 1e-10,
        'MSE_tolerance': 'absolute difference divided by max(1, abs(original MSE)) <=1e-10',
        'exact_nonbinary_constant_prediction_or_target': 'technical review failure; no epsilon/zero-r substitution/exclusion',
        'metric_scope': 'all saved fused predictions vs original targets; all internal models/history/family cells; all external1080/main/5seed',
        'statistics': 'all main M6/7, exploratory M5, matched-history subsets and Exp sensitivity; external separate family; same20k RNG chronology',
        'math_independence': 'centered dot Pearson, frequency/weight matrices and einsum; sorted explicit linear quantiles',
        'no_new_model_choice_threshold_lag_exclusion_fitting_or_download': True,
        'all_signal_reads_and_computation_only_server2203': True,
        'no_automatic_retry_or_original_prediction_overwrite': True,
        'technical_pass_does_not_complete_scientific_goal': True}
    name = 'INDEPENDENT_REVIEW_PROTOCOL.json' if freeze else 'INDEPENDENT_REVIEW_PROTOCOL_CANDIDATE.json'
    dump(ROOT/name, result)
    print(json.dumps({'protocol': name, 'frozen': freeze, 'Sparr_records': chosen, 'sources': len(source_map)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source-commit', required=True)
    parser.add_argument('--freeze', action='store_true'); parser.add_argument('--review-note')
    args = parser.parse_args(); main(args.source_commit, args.freeze, args.review_note)
