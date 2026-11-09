"""Entire independent metric caller on isolated SYNTHETIC signals only.

All five seeds, 18x60 external keys, internal donor subset with a missing
stratum, and production aggregation. External prediction arrays are generated
lazily by a mocked loader; these are engineering fixtures, not real trials.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import numpy as np
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha
from pub_01_s03_ss07_eval_masks import record_mask
from pub_01_s03_ss07_eval_common import score_record, summarize_absolute, correlation
import pub_01_s03_ss07_eval_aggregate as aggregate
import pub_01_s03_ss07_review_metrics as actor


def main():
    fixture = Path(tempfile.mkdtemp(prefix='synthetic_independent_review_', dir=ROOT/'engineering_history'))
    (fixture/'input').mkdir()
    for name in ['pub_01_s03_ss07_review_math.py', 'pub_01_s03_ss07_review_metrics.py']:
        shutil.copy2(ROOT/'input'/name, fixture/'input'/name)
    seeds = [1024, 2026, 31415, 42, 12345]; primary = 'm2_h16'
    controls = ['m2_current_h16', 'static_m2_h16', 'output_prior_h16']
    ids = ['native_P', 'P_EMA', 'P_FT', primary] + controls
    exploratory = [f'm{i}_h16' for i in range(1, 6)]
    contrasts = [('primary-native_P', 'native_P'), ('primary-current', controls[0]),
        ('primary-static', controls[1]), ('primary-output_prior', controls[2]),
        ('primary-P_EMA', 'P_EMA'), ('primary-P_FT', 'P_FT')]
    manifest = {'seeds': seeds, 'screening_seeds': seeds[:3], 'primary': primary,
        'main_model_ids': ids, 'exploratory_family_ids': exploratory,
        'contrast_ids_and_comparators': contrasts, 'EMA_alias': {'tau_seconds': 0},
        'training_costs': [{'stage': stage, 'kind': 'synthetic', 'seed': seeds[i % 5],
            'training': {'epochs': 1, 'updates': 1, 'seconds': .01}}
            for stage, count in [('screen', 54), ('shared_pretrain', 5), ('full', 32)] for i in range(count)]}
    dump(fixture/'FINAL_CHECKPOINT_MANIFEST.json', {'synthetic_only': True})
    dump(fixture/'EVALUATION_READY.json', {'synthetic_only': True})
    dump(fixture/'DTU_PROTOCOL.json', {'synthetic_only': True})
    rows = []; masks = []; truth_by_record = {}
    for i in range(4):
        n = 3840*(1+i % 2); target = np.sin(np.arange(n)/13+i*.11)+.1*np.cos(np.arange(n)/31)
        path = fixture/f'synthetic_target{i}.npy'; np.save(path, target)
        row = {'record': f'r{i}', 'subject': f'p{i}', 'story': f's{i % 2}', 'source_family': f'f{i % 2}',
            'phase': 'test_seen' if i < 2 else 'test_heldout', 'common_samples': n, 'envelope_path': str(path)}
        rows.append(row); masks.append(record_mask(row, target)); truth_by_record[f'r{i}'] = target
    dump(fixture/'SPARR_TEST_MASK.json', {'records': rows, 'record_masks': masks})
    all_keys = [[m['record'], b['start']] for m in masks for b in m['blocks'] if b['eligible']]
    pairs = {'same_person_wrong_family': {'eligible_keys': [key for key in all_keys if key[0] < 'r2']},
             'same_story_wrong_person': {'eligible_keys': all_keys}}
    dump(fixture/'INTERVENTION_PAIR_MASK.json', {'pairs': pairs})
    conditions = ids + sorted(set(exploratory)-set(ids)) + ['history__'+kind for kind in actor.KINDS]
    grid = np.arange(3200, dtype=np.float64)
    external_data = {}
    for p in range(1, 19):
        a = [np.sin(grid/13+p*.03+t*.009)+.2*np.cos(grid/37) for t in range(1, 61)]
        b = [np.cos(grid/17+p*.02+t*.013) for t in range(1, 61)]
        external_data[p] = {'data': {'wavA': a, 'wavB': b}}
    def external_prediction(ci, si, person, trial):
        if ids[ci] == 'P_EMA': ci = 0
        a = external_data[person]['data']['wavA'][trial-1]
        b = external_data[person]['data']['wavB'][trial-1]
        return a*(.3+.03*ci)+np.sin(grid/11+.1*si+.04*ci+person*.002)+.15*b
    covered = np.zeros(3200, dtype=bool); covered[1:3167] = True
    for si, seed in enumerate(seeds):
        folder = fixture/f'final_results/seed{seed}'; folder.mkdir(parents=True)
        actual_ids = ids + (sorted(set(exploratory)-set(ids)) if si < 3 else []) + ['history__'+kind for kind in actor.KINDS]
        for condition in actual_ids:
            condition_index = conditions.index('native_P' if condition == 'P_EMA' else condition)
            segments = []
            for row, mask in zip(rows, masks):
                kind = condition.removeprefix('history__')
                eligible = pairs[kind]['eligible_keys'] if kind in pairs else all_keys
                if not any(key[0] == row['record'] for key in eligible): continue
                n = row['common_samples']; count = (n-192)//96+1
                t = np.arange(192)[None, :] + 96*np.arange(count)[:, None]
                windows = (np.sin(t/13+int(row['record'][1:])*.11)*(.3+.03*condition_index)
                    + np.cos(t/11+.05*condition_index+.03*si)).astype(np.float32)
                scores, fused = score_record(windows, truth_by_record[row['record']], row, mask)
                segments.extend(scores)
                path = folder/f'predictions/internal/{condition}/{row["record"]}.npy'
                path.parent.mkdir(parents=True, exist_ok=True); np.save(path, fused)
            dump(folder/f'INTERNAL_{condition}.json', {'seed': seed, 'condition': condition, 'segments': segments,
                'complete': True, 'failures': [], 'summary': summarize_absolute(segments)})
        for ci, condition in enumerate(ids):
            trials = []
            for person in range(1, 19):
                for trial in range(1, 61):
                    prediction = external_prediction(ci, si, person, trial)[covered]
                    a = external_data[person]['data']['wavA'][trial-1][covered]
                    b = external_data[person]['data']['wavB'][trial-1][covered]
                    ra = correlation(prediction, a); rb = correlation(prediction, b)
                    trials.append({'subject': f'S{person}', 'trial': trial, 'rA': ra, 'rB': rb,
                        'aux_accuracy': 1. if ra > rb else .5 if ra == rb else 0., 'samples': 3166})
            dump(folder/f'EXTERNAL_{condition}.json', {'seed': seed, 'condition': condition, 'trials': trials,
                'complete': True, 'failures': [], 'DTU_protocol_sha256': sha(fixture/'DTU_PROTOCOL.json')})
        dump(folder/'COMPLETED.json', {'seed': seed, 'all_conditions_complete': True,
            'checkpoint_manifest_sha256': sha(fixture/'FINAL_CHECKPOINT_MANIFEST.json')})
    with patch.object(aggregate, 'ROOT', fixture), patch.object(aggregate, 'require_ready', return_value=(manifest, {})):
        aggregate.main()
    original_load = np.load; loads = {'synthetic_external_predictions': 0, 'synthetic_MAT_subjects': 0}
    def load_array(path, *args, **kwargs):
        path = Path(path); assert path.is_relative_to(fixture), 'fixture tried to load nonfixture signal'
        if '/predictions/external/' in str(path):
            loads['synthetic_external_predictions'] += 1
            seed = int(path.parts[path.parts.index('final_results')+1][4:])
            person, trial = path.stem.split('_trial')
            return external_prediction(ids.index(path.parent.name), seeds.index(seed), int(person[1:]), int(trial))
        return original_load(path, *args, **kwargs)
    def load_mat(path, **kwargs):
        path = Path(path); assert path.is_relative_to(fixture)
        loads['synthetic_MAT_subjects'] += 1
        return external_data[int(path.name.split('_')[0][1:])]
    with patch.object(actor, 'ROOT', fixture), patch.object(actor, 'gated', return_value=(manifest, {})), \
         patch.object(actor.np, 'load', load_array), patch.object(actor, 'loadmat', load_mat):
        actor.main()
        proof = read(fixture/'independent_review/METRICS_REVIEW.json'); assert proof['passed']
        original_read = actor.read
        def bad_read(path):
            value = copy.deepcopy(original_read(path))
            if str(path).endswith('seed1024/INTERNAL_native_P.json'):
                value['segments'][0]['r'] += .001
            return value
        with patch.object(actor, 'read', bad_read):
            try: actor.internal_replay(manifest, actor.Checks())
            except AssertionError: pass
            else: raise AssertionError('corrupt r silently accepted')
    assert loads['synthetic_external_predictions'] == 5*len(ids)*1080
    assert loads['synthetic_MAT_subjects'] == 18
    dump(ROOT/'INDEPENDENT_METRIC_PIPELINE_CHECK.json', {'passed': True, 'synthetic_only': True,
        'fixture_path': str(fixture), 'synthetic_loader_calls': loads,
        'internal_cells': len(proof['internal']['cells']), 'all_block_metrics': proof['internal']['all_block_metrics_replayed'],
        'checks': proof['checks'], 'maximum_absolute_difference': proof['maximum_absolute_difference'],
        'all_six_history_diagnostics_including_missing_stratum_no_renormalization': True,
        'production_full_aggregation_then_independent_metric_actor': True, 'corrupt_r_rejected': True,
        'no_actual_test_or_DTU_payload_read': True,
        'source_sha256': {name: sha(ROOT/'input'/name) for name in ['pub_01_s03_ss07_review_math.py',
            'pub_01_s03_ss07_review_metrics.py', 'pub_01_s03_ss07_review_pipeline_check.py']}})
    print(json.dumps({'synthetic_independent_metric_pipeline': 'PASS', 'checks': proof['checks']}))


if __name__ == '__main__': main()
