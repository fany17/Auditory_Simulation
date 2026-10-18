"""Extract the published HDPS M representation on server2203 only.

Paper III.B.1 controls log(eps+Mel). Other explicit STFT/mel settings and
3s+50sample chunk boundaries follow the documented public implementation.
Source NPZ fs is used directly, avoiding the public WAV header inconsistency.
No raw samples or feature tensors leave server2203.
"""
import argparse
import gzip
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np


def main(config, smoke):
    cfg = json.loads(config.read_text())
    base, root = Path(cfg['work']), Path(cfg['root'])
    sys.path.insert(0, str(base / 'runtime_torchaudio_cu128'))
    import librosa
    import torch
    import torchaudio
    if platform.system() != 'Linux' or cfg['patient_data'] != 'FORBIDDEN':
        raise RuntimeError('server2203 public-only')
    split = json.loads((base / 'hdps_split.json').read_text())
    indices = json.loads((base / 'valid_segments_metadata.json').read_text())
    stories = sorted({story for phases in split.values() for phase in phases.values() for story in phase})
    if smoke:
        stories = ['audiobook_5_1']
    output = base / ('mel_smoke01' if smoke else 'mel_paper01')
    output.mkdir(exist_ok=False)
    (output / 'config_frozen.json').write_text(json.dumps(cfg, indent=2))
    torch.set_num_threads(4)
    device = 'cuda:0'
    a = cfg['audio']
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=a['fs'], n_fft=a['n_fft'],
        win_length=a['win_length'], hop_length=a['hop_length'], f_min=a['fmin'], f_max=a['fmax'],
        n_mels=a['n_mels'], power=a['power'], norm=a['norm'], mel_scale=a['mel_scale'],
        center=a['center']).to(device)
    manifest, failures = [], []
    for story in stories:
        try:
            started = time.time()
            path = root / 'pub_01/s06/sparrkulee_v3.1/raw/stimuli/eeg' / (story + '.npz.gz')
            with gzip.open(path, 'rb') as stream, np.load(stream, allow_pickle=False) as archive:
                data = np.asarray(archive['audio'], dtype=np.float32)
                fs = int(np.asarray(archive['fs']).squeeze())
            if data.ndim != 1 or not np.isfinite(data).all():
                raise ValueError('nonfinite or nonmono audio')
            native_samples = len(data)
            audio = librosa.resample(data, orig_sr=fs, target_sr=a['fs'], res_type='soxr_hq')
            del data
            pieces = []
            chunk = a['fs'] * 3
            for start in range(0, len(audio), chunk):
                clip = audio[start:start + chunk + 50]
                if len(clip) < 1024:
                    continue
                with torch.inference_mode():
                    mel = transform(torch.from_numpy(clip.copy()).to(device))
                    feature = torch.log(1e-5 + mel)[:, :-1].T.cpu().numpy()
                pieces.append(feature)
            result = np.concatenate(pieces).astype(np.float32)
            ids = indices[story + '.wav']
            effective = [i for i in ids if (i + 1) * 300 <= len(result)]
            if not np.isfinite(result).all() or not effective:
                raise ValueError('invalid features/indices')
            np.save(output / (story + '.npy'), result, allow_pickle=False)
            row = {'story': story, 'source_path': str(path), 'source_fs': fs, 'source_samples': native_samples,
                'source_duration_seconds': native_samples / fs, 'resampled_fs': 16000,
                'resampled_samples': len(audio), 'resampled_duration_seconds': len(audio) / 16000,
                'shape': list(result.shape), 'fs': 100, 'finite': True,
                'author_valid_segments': len(ids), 'effective_valid_segments': len(effective),
                'excluded_tail_indices': sorted(set(ids) - set(effective)), 'seconds': time.time() - started,
                'time_origin_seconds': 0.0, 'path': str(output / (story + '.npy'))}
            manifest.append(row)
            print(json.dumps(row), flush=True)
        except Exception as exc:
            failures.append({'story': story, 'error': repr(exc), 'no_replacement': True})
            print(json.dumps(failures[-1]), flush=True)
            if smoke:
                raise
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (output / 'failures.json').write_text(json.dumps(failures, indent=2))
    (output / 'completed.json').write_text(json.dumps({'stories': len(manifest), 'failures': len(failures)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    main(args.config, args.smoke)
