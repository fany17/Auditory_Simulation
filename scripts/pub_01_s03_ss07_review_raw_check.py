"""Legal VAL + synthetic adapter/raw batching proof, no test/DTU reads."""
import json
from unittest.mock import patch
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT, OLD, PREPARED, read, dump, sha, native, cfg
from pub_01_s03_ss07_train import NativeHead, make
from pub_01_s03_ss07_finetune import RawWindows, Decoder
from pub_01_s03_ss07_eval_common import predict_windows, predict_raw_phase, fuse
from pub_01_s03_ss07_dtu_adapter import native_windows
import pub_01_s03_ss07_review_raw as review


def main():
    device = 'cuda:0'; torch.set_num_threads(2); torch.cuda.set_device(device)
    precision = read(ROOT/'INCREMENTAL_PROTOCOL.json')['precision']
    torch.backends.cuda.matmul.allow_tf32 = precision['TF32_matmul']
    torch.backends.cudnn.allow_tf32 = precision['TF32_cudnn']
    all_rows = read(OLD/'envelope_readout_seed1024/manifest_frozen.json')['records']
    rows = [r for r in all_rows if r['phase'] == 'val']
    assert len(rows) == 49 and rows[1]['phase_offset'] % 64
    selected = rows[:2]; scaler = np.load(PREPARED/'train_only_normalization.npz')
    mean, std = scaler['mean'], scaler['std']; raw_data = RawWindows('val')
    model = native().TransformerCLIP(cfg())
    model.load_state_dict(torch.load(ROOT/'baseline/pretrain_seed1024/best_validation.pt', map_location='cpu', weights_only=True)['model'])
    encoder = model.eeg.to(device).eval(); head = NativeHead().to(device).eval()
    head.head.load_state_dict(torch.load(ROOT/'baseline/head_seed1024/best_validation.pt', map_location=device, weights_only=True)['head'])
    cache = np.load(ROOT/'baseline/head_seed1024/val_latent.npy', mmap_mode='r'); evidence = []
    folder = ROOT/'screen/m2_h16_seed1024'; config = read(folder/'completed.json')['condition']
    residual = make(config, 1024).to(device).eval()
    residual.load_state_dict(torch.load(folder/'best_validation.pt', map_location=device, weights_only=True)['model'])
    for row in selected:
        offset = row['phase_offset']; raw = review.raw_windows(row, mean, std)
        reference_raw = raw_data.batch(offset, offset+len(raw), 'cpu')[0].numpy()
        assert np.array_equal(raw, reference_raw)
        latent = review.encoded(encoder, raw, device)
        latent_error = float(np.max(np.abs(latent-cache[offset:offset+len(raw)]))); assert latent_error <= 1e-6
        z = torch.from_numpy(latent).to(device)
        for name, fitted in [('native_P', head), ('actual_nonzero_M2', residual)]:
            expected = predict_windows(fitted, z, row['record'], global_offset=offset)
            actual = review.residual_windows(fitted, z, row['record'], offset)
            error = float(np.max(np.abs(actual-expected))); assert error == 0
            a, keep = review.fused(actual, row['common_samples']); b, covered = fuse(expected, row['common_samples'])
            fusion_error = float(np.max(np.abs(a-b))); assert fusion_error <= 1e-6 and np.array_equal(keep, covered)
            evidence.append({'record': row['record'], 'model': name, 'fresh_encoder_vs_cache_max': latent_error,
                'window_output_max': error, 'fusion_max': fusion_error, 'global_offset': offset})
    # Completed actual FT seed1024; no active mutable checkpoint is loaded.
    assert read(ROOT/'finetune/native_P_seed1024/completed.json')['seed'] == 1024
    decoder = Decoder(native().TransformerCLIP(cfg()).eeg).to(device).eval()
    saved = torch.load(ROOT/'finetune/native_P_seed1024/best_validation.pt', map_location=device, weights_only=True)
    assert saved['seed'] == 1024; decoder.load_state_dict(saved['model'], strict=True)
    prefix = rows[:3]  # third fills the second record's trailing global64
    inputs = [(r['record'], review.raw_windows(r, mean, std)) for r in prefix]
    expected = predict_raw_phase(decoder, inputs, device)
    ft_evidence = []
    for row in selected:
        actual, batch_info = review.finetuned_record(decoder, row, rows, mean, std, device)
        error = float(np.max(np.abs(actual-expected[row['record']]))); assert error == 0
        ft_evidence.append(dict(record=row['record'], maximum_difference=error, **batch_info))
    assert rows[2]['record'] in ft_evidence[1]['batch_fill_records']
    assert rows[0]['record'] in ft_evidence[1]['batch_fill_records']
    rng = np.random.default_rng(361)
    physical = [f'scalp{i}' for i in range(64)]; labels = physical+['EXG1', 'EXG2']
    shuffle = rng.permutation(66); labels = [labels[i] for i in shuffle]
    signal = rng.normal(size=(3200, 66))[:, shuffle]
    synthetic_cases = []
    for name, eeg in [('permuted66', signal), ('scale_outlier', signal*17050),
        ('all_time_constant', np.repeat(rng.normal(size=(1, 66)), 3200, axis=0))]:
        a = review.dtu_windows(eeg, labels, physical); b, starts, constants = native_windows(eeg, labels, physical)
        assert np.array_equal(a, b), name
        synthetic_cases.append({'case': name, 'adapter_max_abs': float(np.max(np.abs(a-b)))})
    # Perturb future samples in two channels: first current window unchanged.
    changed = signal.copy(); changed[192:, 0] += 7; changed[192:, 9] -= 3
    original = review.dtu_windows(signal, labels, physical)
    future = review.dtu_windows(changed, labels, physical)
    assert np.array_equal(original[0], future[0]) and np.max(np.abs(original[1:]-future[1:])) > 0
    # Gate must run before any scientific payload on premature actual review.
    calls = []
    def fail_gate(): raise AssertionError('synthetic incomplete final summary')
    def fail_payload(*a, **k): calls.append('premature'); raise RuntimeError('payload')
    with patch.object(review, 'gated', fail_gate), patch.object(review.np, 'load', fail_payload), \
         patch.object(review, 'loadmat', fail_payload), patch.object(review.torch, 'load', fail_payload):
        try: review.main(device)
        except AssertionError: pass
        else: raise AssertionError('gate bypassed')
    assert not calls
    dump(ROOT/'INDEPENDENT_RAW_INPUT_VAL_CHECK.json', {'passed': True, 'no_test_or_actual_DTU_signal_reads': True,
        'actual_completed_native_and_FT_seed': 1024, 'actual_VAL_evidence': evidence, 'actual_FT_VAL_evidence': ft_evidence,
        'synthetic_DTU_adaptation': synthetic_cases, 'future_samples_do_not_change_first_current_window': True,
        'premature_scientific_payload_loads': len(calls),
        'source_sha256': {name: sha(ROOT/'input'/name) for name in ['pub_01_s03_ss07_review_raw.py', 'pub_01_s03_ss07_review_raw_check.py']}})
    print(json.dumps({'independent_raw_VAL_and_synthetic_adapter': 'PASS', 'records': len(evidence)}))


if __name__ == '__main__': main()
