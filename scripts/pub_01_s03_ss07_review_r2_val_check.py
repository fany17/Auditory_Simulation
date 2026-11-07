"""All five completed native heads, legal VAL only; no fits/test/DTU.

Persist original sum-vs-independent-dot R2 rounding and exact v1/v2 checks.
Signals and predictions remain on2203; only lightweight evidence is written.
"""
import json
import time
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT, read, dump, sha
from pub_01_s03_ss07_train import NativeHead, LatentData
from pub_01_s03_ss07_eval_common import predict_windows, fuse
from pub_01_s03_ss07_review_math import signal_metrics
from pub_01_s03_ss07_review_metrics import Checks as OldChecks
from pub_01_s03_ss07_review_metrics_v2 import Checks


def main():
    device = 'cuda:0'; torch.set_num_threads(2); torch.cuda.set_device(device)
    precision = read(ROOT/'INCREMENTAL_PROTOCOL.json')['precision']
    torch.backends.cuda.matmul.allow_tf32 = precision['TF32_matmul']
    torch.backends.cudnn.allow_tf32 = precision['TF32_cudnn']
    old = OldChecks(); new = Checks(); failures = []; per_seed = []; started = time.time()
    for seed in [1024, 2026, 31415, 42, 12345]:
        folder = ROOT/f'baseline/head_seed{seed}'; complete = read(folder/'completed.json')
        assert complete['seed'] == seed and complete['epochs'] == 250
        data = LatentData(seed, 'val', device); assert len(data.rows) == 49
        checkpoint = torch.load(folder/'best_validation.pt', map_location=device, weights_only=True)
        assert checkpoint['seed'] == seed and checkpoint['epoch'] == complete['selected_epoch']
        model = NativeHead().to(device).eval(); model.head.load_state_dict(checkpoint['head'], strict=True)
        masks = {(b['record'], b['start']): b for b in data.fixed_mask['blocks']}
        count = 0; max_absolute = 0.; max_scaled = 0.; r_max = 0.; values = []
        for row in data.rows:
            z = data.x[row['offset']:row['offset']+row['windows']]
            windows = predict_windows(model, z, row['record'], global_offset=row['offset'])
            prediction, covered = fuse(windows, row['common_samples'])
            truth = np.load(row['envelope_path'], mmap_mode='r').reshape(-1)
            for start in range(0, row['common_samples']-3840+1, 3840):
                mask = masks[(row['record'], start)]
                if not mask['eligible']: continue
                keep = covered[start:start+3840]
                assert keep.sum() == mask['samples'] >= 3800
                a = prediction[start:start+3840][keep]
                b = np.asarray(truth[start:start+3840], dtype=np.float64)[keep]
                production = {'r': float(np.corrcoef(a, b)[0, 1]),
                    'MSE_descriptive': float(np.mean((a-b)**2)),
                    'R2_descriptive': float(1-np.sum((a-b)**2)/np.sum((b-b.mean())**2))}
                independent = signal_metrics(a, b)
                for name in production:
                    label = 'internal_'+name
                    new.close(label, independent[name], production[name],
                        relative=name in ['MSE_descriptive', 'R2_descriptive'])
                    try: old.close(label, independent[name], production[name], relative=name == 'MSE_descriptive')
                    except AssertionError:
                        assert name == 'R2_descriptive', 'unexpected v1 failure outside descriptiveR2'
                        difference = abs(independent[name]-production[name])
                        failures.append({'seed': seed, 'record': row['record'], 'start': start,
                            'samples': int(keep.sum()), 'production_R2': production[name], 'independent_R2': independent[name],
                            'absolute_difference': difference, 'scaled_difference': difference/max(1., abs(production[name])),
                            'interpretation': 'legalVAL numerical comparison counterexample, not scientific trial failure'})
                difference = abs(independent['R2_descriptive']-production['R2_descriptive'])
                max_absolute = max(max_absolute, difference)
                max_scaled = max(max_scaled, difference/max(1., abs(production['R2_descriptive'])))
                r_max = max(r_max, abs(independent['r']-production['r']))
                values.append(production['R2_descriptive']); count += 1
        assert count == 752
        per_seed.append({'seed': seed, 'records': 49, 'blocks': count,
            'max_R2_absolute_difference': max_absolute, 'max_R2_scaled_difference': max_scaled,
            'max_r_absolute_difference': r_max, 'production_R2_range': [min(values), max(values)],
            'checkpoint_path': str(folder/'best_validation.pt'), 'selected_epoch': checkpoint['epoch']})
        del data, model; torch.cuda.empty_cache()
    assert new.count == 3760*3
    report = {'passed': True, 'legal_VAL_only': True, 'new_fits': 0, 'actual_test_or_DTU_reads': 0,
        'seeds': per_seed, 'records': 245, 'blocks': 3760, 'comparisons': new.count,
        'old_absolute_R2_gate_rejections': failures,
        'old_strict_rejections_are_numerical_evidence_not_actual_scientific_failures': True,
        'maximum_absolute_difference': new.max_absolute, 'maximum_scaled_difference': new.max_relative,
        'R2_MSE_scaled_tolerance': 1e-10, 'r_absolute_tolerance_unchanged': 1e-10,
        'production_outputs_and_original_sources_unchanged': True, 'seconds': time.time()-started,
        'source_sha256': {name: sha(ROOT/'input'/name) for name in [
            'pub_01_s03_ss07_review_r2_val_check.py', 'pub_01_s03_ss07_review_math.py',
            'pub_01_s03_ss07_review_metrics.py', 'pub_01_s03_ss07_review_metrics_v2.py']}}
    dump(ROOT/'INDEPENDENT_R2_LEGAL_VAL_CHECK_V2.json', report)
    print(json.dumps({'legal_VAL_R2_check': 'PASS', 'blocks': 3760,
        'old_absolute_gate_rejections': len(failures), 'maximum_R2_scaled_difference': new.max_relative['internal_R2_descriptive']}))


if __name__ == '__main__': main()
