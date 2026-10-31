"""Prospectively sampled raw-input replay AFTER complete final evaluation.

Independent readers, checkpoint roles, batching, DTU adapter and fusion.
Uses the frozen neural architecture/forward implementations, so this is NOT
an independent implementation of every neural operation, nor exhaustive
raw-input replay of all records. All metrics are reviewed separately.
"""
import argparse
import json
import time
import numpy as np
import torch
from scipy.io import loadmat
from pub_01_s03_ss07_baseline import ROOT, OLD, PREPARED, read, dump, sha, native, cfg
from pub_01_s03_ss07_train import NativeHead, make
from pub_01_s03_ss07_finetune import Decoder
from pub_01_s03_ss07_review_metrics import gated
from pub_01_s03_ss07_review_math import pearson


def raw_windows(row, mean, std, indices=None):
    source = np.load(row['eeg_path'], mmap_mode='r')
    eeg = source.T if source.shape[0] == 64 else source
    assert eeg.ndim == 2 and eeg.shape[1] == 64 and len(eeg) >= row['common_samples']
    count = (row['common_samples'] - 192) // 96 + 1
    chosen = range(count) if indices is None else indices
    windows = []
    for i in chosen:
        assert 0 <= i < count
        window = np.asarray(eeg[96*i:96*i+192], dtype=np.float32)
        windows.append(((window - mean) / std).astype(np.float32))
    return np.stack(windows)


@torch.no_grad()
def encoded(encoder, windows, device):
    outputs = []
    for first in range(0, len(windows), 256):
        output = encoder(torch.from_numpy(windows[first:first+256]).to(device)).transpose(1, 2)
        assert output.shape == (min(256, len(windows)-first), 8, 192)
        outputs.append(output.cpu().numpy())
    result = np.concatenate(outputs); assert np.isfinite(result).all()
    return result


@torch.no_grad()
def residual_windows(model, latent, record, offset, mode='full'):
    model.eval(); state = None; output = []; first = 0
    while first < len(latent):
        end = min(len(latent), ((offset + first)//64 + 1)*64 - offset)
        value, state, _ = model(latent[first:end], state, first, record, mode=mode)
        assert value.shape == (end-first, 1, 192) and torch.isfinite(value).all()
        output.append(value[:, 0].cpu().numpy()); first = end
    return np.concatenate(output)


@torch.no_grad()
def finetuned_record(decoder, row, phase_rows, mean, std, device):
    """Snap to global64; fill neighboring record windows before/after target.

    Never repack a record alone, split raw64 at record boundaries, or feed the
    frozen P encoder to FT. Only selected record output is retained in memory.
    """
    starts = []; counts = []; offset = 0
    for source in phase_rows:
        assert source['phase'] == row['phase'] and source['phase_offset'] == offset
        starts.append(offset); count = (source['common_samples'] - 192)//96 + 1
        counts.append(count); offset += count
    index = next(i for i, r in enumerate(phase_rows) if r['record'] == row['record'])
    begin = starts[index]; finish = begin + counts[index]
    lower = begin//64*64; upper = min(offset, ((finish+63)//64)*64)
    chunks = []; source_records = set()
    for batch_start in range(lower, upper, 64):
        batch_end = min(batch_start+64, upper); windows = []; cursor = batch_start
        while cursor < batch_end:
            record_index = int(np.searchsorted(starts, cursor, side='right')-1)
            stop = min(batch_end, starts[record_index] + counts[record_index])
            source = phase_rows[record_index]; source_records.add(source['record'])
            windows.extend(raw_windows(source, mean, std,
                range(cursor-starts[record_index], stop-starts[record_index])))
            cursor = stop
        value = decoder(torch.from_numpy(np.stack(windows)).to(device))
        assert value.shape == (batch_end-batch_start, 1, 192) and torch.isfinite(value).all()
        a = max(begin, batch_start); b = min(finish, batch_end)
        chunks.append(value[a-batch_start:b-batch_start, 0].cpu().numpy())
    result = np.concatenate(chunks); assert result.shape == (counts[index], 192)
    return result, {'snapped_global_range': [lower, upper], 'target_global_range': [begin, finish],
                    'batch_fill_records': sorted(source_records), 'raw_global64_not_record_split': True}


def dtu_windows(eeg, labels, physical):
    assert len(labels) == len(set(labels)) and len(physical) == len(set(physical)) == 64
    index = {label: i for i, label in enumerate(labels)}
    assert set(physical).issubset(index)
    # Match the frozen floating-point layout and reduction order, while
    # independently resolving label names and implementing the adapter.
    selected = np.asarray(eeg[:, [index[name] for name in physical]], dtype=np.float64)
    assert selected.shape == (3200, 64) and np.isfinite(selected).all()
    referenced = selected - selected.mean(axis=1, keepdims=True)
    output = []
    for start in range(0, 2977, 96):
        current = referenced[start:start+192]
        constant = current.max(axis=0) == current.min(axis=0)
        scale = current.std(axis=0, ddof=0); scale[constant] = 1.
        normalized = (current-current.mean(axis=0)) / scale
        normalized[:, constant] = 0.
        output.append(normalized.astype(np.float32))
    result = np.stack(output); assert result.shape == (32, 192, 64) and np.isfinite(result).all()
    return result


def fused(windows, length, tau=0.):
    assert windows.shape == ((length-192)//96+1, 192) and np.isfinite(windows).all()
    output = np.zeros(length); weight = np.zeros(length)
    taper = .5 - .5*np.cos(np.arange(192)*2*np.pi/191)
    for i, window in enumerate(windows):
        start = 96*i; output[start:start+192] += taper*window
        weight[start:start+192] += taper
    keep = weight > 1e-8; output[keep] /= weight[keep]
    if tau:
        assert tau > 0
        alpha = 1-np.exp(-1/(64*tau)); previous = None
        for i in range(length):
            if not keep[i]: previous = None; continue
            previous = output[i] if previous is None else alpha*output[i]+(1-alpha)*previous
            output[i] = previous
    return output, keep


class Models:
    def __init__(self, manifest, seed, device):
        self.manifest = manifest; self.seed = seed; self.device = device; self.models = {}
        source = manifest['encoders'][str(seed)]
        checkpoint = torch.load(source['path'], map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == seed and sha(source['path']) == source['sha256']
        self.module = native(); original = self.module.TransformerCLIP(cfg())
        original.load_state_dict(checkpoint['model'], strict=True)
        self.encoder = original.eeg.to(device).eval()

    def load(self, condition):
        assert condition != 'P_EMA'
        if condition not in self.models:
            entry = self.manifest['entries'][f'{condition}:{self.seed}']
            checkpoint = torch.load(entry['path'], map_location='cpu', weights_only=True)
            assert checkpoint['seed'] == self.seed and checkpoint['epoch'] == entry['selected_epoch']
            assert sha(entry['path']) == entry['sha256']
            if condition == 'native_P':
                assert entry['kind'] == 'native_P'
                model = NativeHead(); model.head.load_state_dict(checkpoint['head'], strict=True)
            elif condition == 'P_FT':
                assert entry['kind'] == 'P_FT'
                model = Decoder(self.module.TransformerCLIP(cfg()).eeg)
                model.load_state_dict(checkpoint['model'], strict=True)
            else:
                assert entry['kind'] == 'incremental'
                model = make(entry['config'], self.seed); model.load_state_dict(checkpoint['model'], strict=True)
            self.models[condition] = model.to(self.device).eval()
        return self.models[condition]


def choose_sparr(rows):
    chosen = []
    for phase in ['test_seen', 'test_heldout']:
        group = sorted([r for r in rows if r['phase'] == phase], key=lambda r: r['record'])
        chosen.append(group[0]['record'])
        nonzero = next(r for r in group if r['phase_offset'] % 64)
        if nonzero['record'] not in chosen: chosen.append(nonzero['record'])
    return chosen


def main(device):
    manifest, ready = gated()  # before actual checkpoint, target, MAT, EEG reads
    protocol = read(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json')
    assert protocol['frozen'] and protocol['waveform_absolute_tolerance'] == 1e-6
    for path, digest in protocol['sources'].items(): assert sha(path) == digest
    assert read(ROOT/'INDEPENDENT_RAW_INPUT_VAL_CHECK.json')['passed']
    destination = ROOT/'independent_raw_review'; destination.mkdir(exist_ok=False)
    precision = read(ROOT/'INCREMENTAL_PROTOCOL.json')['precision']
    torch.set_num_threads(2); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = precision['TF32_matmul']
    torch.backends.cudnn.allow_tf32 = precision['TF32_cudnn']
    rows = read(ROOT/'SPARR_TEST_MASK.json')['records']; chosen = choose_sparr(rows)
    assert chosen == protocol['Sparr_records']
    scaler = np.load(PREPARED/'train_only_normalization.npz'); mean, std = scaler['mean'], scaler['std']
    dtu = read(ROOT/'DTU_PROTOCOL.json'); evidence = []; started = time.time()
    def compare(prediction, path, metadata):
        original = np.load(path, mmap_mode='r')
        assert original.dtype == np.float64 and original.shape == prediction.shape
        difference = float(np.max(np.abs(prediction-original)))
        assert np.isfinite(difference) and difference <= 1e-6, (metadata, difference)
        evidence.append(dict(metadata, maximum_absolute_waveform_difference=difference,
                             prespecified_tolerance=1e-6))
    try:
        for seed in manifest['seeds']:
            models = Models(manifest, seed, device)
            for record in chosen:
                row = next(r for r in rows if r['record'] == record)
                raw = raw_windows(row, mean, std)
                z = torch.from_numpy(encoded(models.encoder, raw, device)).to(device)
                predictions = {}
                for condition in manifest['main_model_ids']:
                    extra = {}
                    if condition == 'P_EMA':
                        windows = predictions['native_P']; tau = manifest['EMA_alias']['tau_seconds']
                    elif condition == 'P_FT':
                        windows, extra = finetuned_record(models.load(condition), row,
                            [r for r in rows if r['phase'] == row['phase']], mean, std, device); tau = 0.
                    else:
                        mode = manifest['entries'][f'{condition}:{seed}']['config'].get('mode', 'full')
                        windows = residual_windows(models.load(condition), z, record, row['phase_offset'], mode)
                        predictions[condition] = windows; tau = 0.
                    prediction, keep = fused(windows, row['common_samples'], tau)
                    compare(prediction, ROOT/f'final_results/seed{seed}/predictions/internal/{condition}/{record}.npy',
                        {'seed': seed, 'condition': condition, 'dataset': 'Sparr', 'record': record, **extra})
            for person in [1, 11, 18]:
                data = loadmat(ROOT/f'external_dtu/extracted/S{person}_data_preproc.mat', simplify_cells=True)['data']
                for trial in [1, 60]:
                    key = f'S{person}_trial{trial:02d}'
                    raw = dtu_windows(data['eeg'][trial-1], data['dim']['chan']['eeg'][trial-1].tolist(), dtu['physical_scalp_names'])
                    z = torch.from_numpy(encoded(models.encoder, raw, device)).to(device); predictions = {}
                    for condition in manifest['main_model_ids']:
                        if condition == 'P_EMA':
                            windows = predictions['native_P']; tau = manifest['EMA_alias']['tau_seconds']
                        elif condition == 'P_FT':
                            with torch.no_grad(): windows = models.load(condition)(torch.from_numpy(raw).to(device))[:, 0].cpu().numpy()
                            tau = 0.
                        else:
                            mode = manifest['entries'][f'{condition}:{seed}']['config'].get('mode', 'full')
                            windows = residual_windows(models.load(condition), z, key, 0, mode)
                            predictions[condition] = windows; tau = 0.
                        prediction, keep = fused(windows, 3200, tau)
                        assert np.flatnonzero(keep).tolist() == list(range(1, 3167))
                        compare(prediction, ROOT/f'final_results/seed{seed}/predictions/external/{condition}/{key}.npy',
                            {'seed': seed, 'condition': condition, 'dataset': 'DTU', 'subject': f'S{person}', 'trial': trial})
            del models; torch.cuda.empty_cache()
        assert len(evidence) == 5*len(manifest['main_model_ids'])*(len(chosen)+6)
        dump(destination/'RAW_INPUT_REVIEW.json', {'passed': True, 'records': evidence, 'seconds': time.time()-started,
            'prospective_metadata_only_sampling': True, 'S11_prespecified_from_input_scale_not_model_scores': True,
            'all_five_seeds_all_main_models_on_selected_records': True,
            'fresh_raw_signals_and_own_checkpoint_roles': True, 'reused_frozen_neural_forward_implementation': True,
            'not_exhaustive_raw_replay_or_independent_neural_math': True,
            'history_intervention_neural_outputs_not_raw_replayed': True,
            'separate_all_fused_prediction_metric_replay_required': True, 'scientific_goal_complete': False,
            'review_protocol_sha256': sha(ROOT/'INDEPENDENT_REVIEW_PROTOCOL.json'),
            'checkpoint_manifest_sha256': sha(ROOT/'FINAL_CHECKPOINT_MANIFEST.json')})
        print(json.dumps({'raw_input_review_passed': True, 'replayed_model_records': len(evidence)}))
    except Exception as error:
        dump(destination/'RAW_INPUT_REVIEW_FAILURE.json', {'error': repr(error), 'completed_records': evidence,
            'seconds': time.time()-started, 'original_predictions_and_results_unchanged': True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cuda:0')
    main(parser.parse_args().device)
