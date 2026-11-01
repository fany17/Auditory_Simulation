"""Synthetic metadata gates only; no actual test/external outcome reads."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
from pub_01_s03_ss07_baseline import ROOT, dump, sha
import pub_01_s03_ss07_result_figures_v2 as actor


def main():
    records = {
        'INDEPENDENT_REVIEW_COMPLETED_V2.json': {'technical_review_passed': True, 'review_version': 2,
            'scientific_goal_complete': False, 'protocol_sha256': 'v2', 'old_raw_protocol_sha256': 'v1'},
        'RAW_INPUT_REVIEW.json': {'passed': True, 'review_protocol_sha256': 'v1', 'checkpoint_manifest_sha256': 'cp'},
        'METRICS_REVIEW.json': {'passed': True, 'review_version': 2, 'review_protocol_sha256': 'v2', 'checkpoint_manifest_sha256': 'cp'}}
    def digest(path):
        return {'INDEPENDENT_REVIEW_PROTOCOL_V2.json': 'v2', 'INDEPENDENT_REVIEW_PROTOCOL.json': 'v1',
                'FINAL_CHECKPOINT_MANIFEST.json': 'cp'}[Path(path).name]
    calls = []
    def load(path):
        name = Path(path).name
        assert name in records, 'gate tried to load an outcome summary'
        calls.append(name); return copy.deepcopy(records[name])
    with patch.object(actor, 'read', load), patch.object(actor, 'sha', digest), \
         patch.object(actor, 'gated', return_value=({'synthetic_manifest_only': True}, {})):
        assert actor.reviewed_gate()['synthetic_manifest_only']
        rejected = []
        for name, field, value, label in [
            ('RAW_INPUT_REVIEW.json', 'review_protocol_sha256', 'v2', 'raw_wrong_protocol'),
            ('METRICS_REVIEW.json', 'review_protocol_sha256', 'v1', 'metrics_wrong_protocol'),
            ('METRICS_REVIEW.json', 'review_version', 1, 'old_metrics_not_accepted_as_v2'),
            ('RAW_INPUT_REVIEW.json', 'checkpoint_manifest_sha256', 'other_cp', 'mismatched_checkpoints'),
        ]:
            old = records[name][field]; records[name][field] = value
            try: actor.reviewed_gate()
            except AssertionError: rejected.append(label)
            else: raise AssertionError('mixed protocol accepted: '+label)
            finally: records[name][field] = old
    early = []
    def forbidden(*a, **k): early.append('premature'); raise RuntimeError('premature')
    with patch.object(actor, 'read', return_value={'technical_review_passed': False}), \
         patch.object(actor, 'gated', forbidden), patch.object(actor, 'render', forbidden):
        try: actor.main()
        except AssertionError: pass
        else: raise AssertionError('incomplete v2 review accepted')
    assert not early
    dump(ROOT/'INDEPENDENT_REVIEW_V2_GATE_CHECK.json', {'passed': True, 'synthetic_only': True,
        'unchanged_raw_v1_and_metrics_v2_combination_checked': True,
        'rejected_mixed_protocols': rejected, 'premature_gated_and_renderer_calls': 0,
        'actual_outcome_summaries_loaded': 0,
        'source_sha256': {name: sha(ROOT/'input'/name) for name in [
            'pub_01_s03_ss07_result_figures_v2.py', 'pub_01_s03_ss07_review_v2_gate_check.py']}})
    print(json.dumps({'v2_result_entry_gate': 'PASS', 'mixed_protocol_rejects': len(rejected)}))


if __name__ == '__main__': main()
