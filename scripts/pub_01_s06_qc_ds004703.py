"""Full recording non-destructive initial QC and clean eligibility indices.

Server only. No filtering, rereferencing, imputation, resampling, model fit or hashes.
Stimulus mapping follows bundled experiment script (Block directories; controls
at stimuli root). all_stimuli is checked as duplicate, not treated as a new trial.
Catalan alternatives remain excluded even when a different file fits duration.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import platform
import numpy as np
import mne
import soundfile as sf

ROOT = Path('/home/fanyu/auditory_simulation_m6a')


def save_csv(path, rows, fields=None):
    if not rows and fields is None:
        raise ValueError('Empty table requires explicit schema')
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def compare_audio(a, b):
    with sf.SoundFile(a) as left, sf.SoundFile(b) as right:
        if (left.samplerate, left.channels, len(left)) != (right.samplerate, right.channels, len(right)):
            return dict(same_shape=False, exact_waveform=False, max_abs_difference='NOT_COMPARABLE')
        difference = 0.0
        while len(x := left.read(65536, dtype='float64', always_2d=True)):
            y = right.read(len(x), dtype='float64', always_2d=True)
            difference = max(difference, float(np.max(np.abs(x - y))))
        return dict(same_shape=True, exact_waveform=difference == 0, max_abs_difference=difference)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    qc = config['ds004703']
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise RuntimeError('Server-only QC')
    out = Path(args.output).resolve()
    if ROOT not in out.parents:
        raise ValueError('Output outside server root')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'executed_config.json').write_text(json.dumps(config, indent=2))
    base = ROOT / 'data/ds004703/v1.1.0'
    templates = defaultdict(list)
    with (base / 'stimuli/stim-times.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['ex_name'] != 'ex_name':
                templates[row['ex_name']].append(row)
    candidates = defaultdict(list)
    audio_info = {}
    audio_rows = []
    audio_bad = {}
    for path in sorted(base.rglob('*.wav')):
        key = path.stem.removesuffix('_normed')
        candidates[key].append(path)
        info = sf.info(path)
        audio_info[path] = info
        nonfinite = 0
        with sf.SoundFile(path) as f:
            for block in f.blocks(blocksize=65536, dtype='float64'):
                nonfinite += int((~np.isfinite(block)).sum())
        audio_bad[path] = nonfinite
        audio_rows.append(dict(path=str(path), story=key, rate=info.samplerate,
                               frames=info.frames, duration_s=info.duration, channels=info.channels,
                               nonfinite=nonfinite, scan='ALL_SAMPLES'))
    save_csv(out / 'audio_qc.csv', audio_rows)
    canonical = {}
    blocks = {}
    for path in sorted((base / 'stimuli/excerpts').glob('Block */*_normed.wav')):
        key = path.stem.removesuffix('_normed')
        if key in canonical:
            raise ValueError('Multiple Block definitions for ' + key)
        canonical[key] = path
        blocks[key] = path.parent.name
    for path in (base / 'stimuli').glob('*_normed.wav'):
        canonical[path.stem.removesuffix('_normed')] = path
    comparisons = []
    for key, path in canonical.items():
        for alternative in candidates[key]:
            if alternative != path:
                comparisons.append(dict(story=key, canonical=str(path), alternative=str(alternative),
                                        **compare_audio(path, alternative)))
    save_csv(out / 'audio_duplicate_comparison.csv', comparisons)
    cue = base / 'stimuli/pleasePressSpace_normed.wav'
    save_csv(out / 'catalan_cue_comparison.csv', [dict(story=k, canonical=str(v), cue=str(cue), **compare_audio(v, cue))
                                               for k, v in canonical.items() if k.startswith('catalan-')])
    records, channels, pairs, exclusions = [], [], [], []
    for path in sorted(base.glob('sub-*/*/ieeg/*.edf')):
        recording = path.name.removesuffix('_ieeg.edf')
        raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
        rate = raw.info['sfreq']
        size = len(raw.ch_names)
        lo, hi, bad = np.full(size, np.inf), np.full(size, -np.inf), np.zeros(size, dtype=np.int64)
        for start in range(0, raw.n_times, int(rate * qc['scan_chunk_seconds'])):
            data = raw.get_data(start=start, stop=min(raw.n_times, start + int(rate * qc['scan_chunk_seconds'])))
            good = np.isfinite(data)
            bad += (~good).sum(axis=1)
            lo = np.minimum(lo, np.where(good, data, np.inf).min(axis=1))
            hi = np.maximum(hi, np.where(good, data, -np.inf).max(axis=1))
        meta_path = path.with_name(recording + '_channels.tsv')
        with meta_path.open() as f:
            channel_meta = {r['name']: r for r in csv.DictReader(f, delimiter='\t')}
        eligible = 0
        for i, name in enumerate(raw.ch_names):
            meta = channel_meta.get(name, {})
            reasons = []
            if not meta or meta.get('units') in [None, '', 'n/a']:
                reasons.append('UNKNOWN_CHANNEL_METADATA_OR_UNIT')
            if name.startswith('C'):
                reasons.append('README_UNUSED_C_PREFIX')
            if meta.get('status') == 'bad':
                reasons.append('OFFICIAL_BAD:' + meta.get('status_description', 'UNKNOWN'))
            if meta.get('type') not in ['SEEG', 'ECOG', 'EEG']:
                reasons.append('NOT_NEURAL_CHANNEL')
            if bad[i]:
                reasons.append('NONFINITE')
            if lo[i] == hi[i]:
                reasons.append('FLAT_EXACT')
            eligible += not reasons
            channels.append(dict(recording=recording, channel=name, type=meta.get('type', 'UNKNOWN'),
                                 original_unit=meta.get('units', 'UNKNOWN'), mne_data_unit='SI volts for voltage channels',
                                 samples=raw.n_times, nonfinite=int(bad[i]), min_value=lo[i], max_value=hi[i],
                                 eligible=not reasons, reasons=';'.join(reasons), scan='ALL_SAMPLES'))
            if reasons:
                exclusions.append(dict(recording=recording, item=name, kind='channel', reason=';'.join(reasons)))
        duration = raw.n_times / rate
        with (path.parent.parent / (recording + '_events.tsv')).open() as f:
            events = list(csv.DictReader(f, delimiter='\t'))
        grouped = []
        for row in events:
            if not grouped or grouped[-1][0]['ex_name'] != row['ex_name']:
                grouped.append([])
            grouped[-1].append(row)
        oob = sum(float(r['onset']) < 0 or float(r['onset']) + float(r['duration']) > duration for r in events)
        for trial, group in enumerate(grouped):
            key = group[0]['ex_name']
            template = templates.get(key, [])
            labels = len(group) == len(template) and all(
                all(a[k] == b[k] for k in ['word', 'pos', 'phone']) and abs(float(a['duration']) - float(b['duration'])) < qc['template_duration_tolerance_seconds']
                for a, b in zip(group, template))
            offset, residual = None, None
            if labels:
                offsets = np.array([float(a['onset']) - float(b['t_start']) for a, b in zip(group, template)])
                offset = float(np.median(offsets))
                residual = float(np.ptp(offsets))
            audio = canonical.get(key)
            info = audio_info.get(audio)
            template_end = max([float(x['t_start']) + float(x['duration']) for x in template], default=0)
            margin = info.duration - template_end if info else None
            reasons = []
            if not labels or residual is None or residual > qc['alignment_max_spread_seconds']:
                reasons.append('TEMPLATE_ALIGNMENT')
            if key.startswith('catalan-'):
                reasons.append('CATALAN_PLAYED_AUDIO_PROVENANCE_CONFLICT')
            elif not (key.startswith('s') and '-ex' in key):
                reasons.append('CONTROL_NOT_PASSAGE')
            if info is None or margin is None or not qc['audio_template_margin_seconds'][0] <= margin <= qc['audio_template_margin_seconds'][1]:
                reasons.append('AUDIO_TEMPLATE_DURATION')
            if audio_bad.get(audio, 1):
                reasons.append('AUDIO_NONFINITE_OR_UNVERIFIED')
            if offset is None or info is None or offset < 0 or offset + info.duration > duration:
                reasons.append('RECORDING_BOUNDARY')
            if not eligible:
                reasons.append('NO_ELIGIBLE_CHANNELS')
            pairs.append(dict(dataset='ds004703', subject=path.parts[-4], session=path.parts[-3], recording=recording,
                              trial=trial, story=key, block=blocks.get(key, 'NOT_APPLICABLE'),
                              audio_path=str(audio) if audio else 'UNKNOWN', n_events=len(group),
                              labels_match=labels, offset_s=offset, offset_spread_s=residual,
                              audio_duration_s=info.duration if info else None, template_margin_s=margin,
                              eligible=not reasons, reason=';'.join(reasons),
                              physical_sync='UNKNOWN', split='UNASSIGNED_S01_REVIEW_REQUIRED'))
            if reasons:
                exclusions.append(dict(recording=recording, item=f'trial-{trial}:{key}', kind='segment', reason=';'.join(reasons)))
        records.append(dict(recording=recording, server_raw_path=str(path), sample_rate_hz=rate,
                            samples=raw.n_times, duration_s=duration, channels=size, eligible_channels=eligible,
                            nonfinite_samples=int(bad.sum()), flat_channels=int((lo == hi).sum()),
                            event_count=len(events), event_out_of_bounds=oob, scan='ALL_SAMPLES',
                            raw_modified=False, filtering=False, rereference=False, resampling=False))
        raw.close()
        save_csv(out / 'recording_qc.csv', records)
        print(recording, 'ALL_SAMPLES', eligible, 'eligible channels', flush=True)
    save_csv(out / 'channel_qc.csv', channels)
    save_csv(out / 'audio_neural_pairing_audit.csv', pairs)
    save_csv(out / 'clean_segment_index.csv', [r for r in pairs if r['eligible']], list(pairs[0]))
    save_csv(out / 'clean_channel_index.csv', [r for r in channels if r['eligible']], list(channels[0]))
    save_csv(out / 'exclusion_inventory.csv', exclusions, ['recording', 'item', 'kind', 'reason'])
    manifest = dict(task='PUB-01-S06', dataset='ds004703', version='1.1.0',
                    original_server_path=str(base), derivatives_path=str(out),
                    recordings=len(records), audio_files=len(audio_rows), event_groups=len(pairs),
                    eligible_segments=sum(r['eligible'] for r in pairs),
                    scan='Every neural/audio sample read for finite/flat checks; metadata pairing audited',
                    cleaning='Virtual derivative: channel/segment exclusions + executable pub_01_virtual_derivative.py; validation required',
                    config=qc,
                    signal_transforms='NONE: no filtering, resampling, re-reference, ICA or imputation',
                    original_preserved=True, benchmark_admission='HOLD pending S01 complete audit and split review',
                    physical_sync='UNKNOWN; metadata consistency is not physical sync evidence')
    (out / 'cleaning_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
