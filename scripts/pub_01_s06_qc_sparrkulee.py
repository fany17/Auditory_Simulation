"""Incremental full-sample QC of official NPY derivatives on server2203.

Run repeatedly as download advances. Each run is a new output snapshot.
No pickle loading or arbitrary waveform trimming; mismatch means HOLD.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import re
import numpy as np

ROOT = Path('/home/fanyu/auditory_simulation_m6a')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise RuntimeError('Server only')
    raw, out = Path(args.raw).resolve(), Path(args.output).resolve()
    if ROOT not in raw.parents or ROOT not in out.parents:
        raise ValueError('Outside server project')
    out.mkdir(parents=True, exist_ok=False)
    meta_path = raw / 'derivatives/preprocessed_eeg/.save_metadata.json'
    metadata = json.loads(meta_path.read_text())
    reverse = {item: source for source, items in metadata.items() for item in items}
    rows = []
    for path in sorted((raw / 'derivatives').rglob('*.npy')):
        # Downloader exposes final names only after whole-file byte-count check.
        row = dict(path=str(path), shape='', dtype='', samples='UNKNOWN', channels='UNKNOWN',
                   nonfinite='UNKNOWN', flat_channels='UNKNOWN', role='EEG' if 'preprocessed_eeg' in path.parts else 'STIMULUS_FEATURE',
                   source_recording='UNKNOWN', story='UNKNOWN', paired_feature_path='UNKNOWN',
                   paired_samples='UNKNOWN', sample_count_difference='UNKNOWN', sampling_rate_hz='64_PUBLISHER_PIPELINE_NOT_PER_FILE_VERIFIED',
                   channel_names='UNKNOWN_NPY_HAS_NO_NAMES', units='UNKNOWN_NPY_HAS_NO_UNITS',
                   status='HOLD', reason='', scan='ALL_SAMPLES')
        try:
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            row.update(shape=str(a.shape), dtype=str(a.dtype))
            eeg = row['role'] == 'EEG'
            if a.ndim != 2 or (eeg and a.shape[0] != 64):
                raise ValueError('Unexpected array axes/shape')
            values = a if eeg else a.T
            row.update(samples=values.shape[1], channels=values.shape[0])
            lo, hi = np.full(values.shape[0], np.inf), np.full(values.shape[0], -np.inf)
            nonfinite = 0
            for start in range(0, values.shape[1], 65536):
                x = values[:, start:start + 65536]
                good = np.isfinite(x)
                nonfinite += int((~good).sum())
                lo = np.minimum(lo, np.where(good, x, np.inf).min(axis=1))
                hi = np.maximum(hi, np.where(good, x, -np.inf).max(axis=1))
            row.update(nonfinite=nonfinite, flat_channels=int((lo == hi).sum()), status='PASS_FORMAT_QC_ONLY')
            if nonfinite or (lo == hi).any():
                row.update(status='HOLD', reason='NONFINITE_OR_FLAT')
            if eeg:
                rel = str(path.relative_to(raw / 'derivatives/preprocessed_eeg'))
                row['source_recording'] = reverse.get(rel, 'UNKNOWN')
                match = re.search(r'_desc-preproc-audio-(.+)_eeg.npy$', path.name)
                if not match:
                    raise ValueError('Unrecognized story key')
                story = match.group(1)
                row['story'] = story
                feature = raw / 'derivatives/preprocessed_stimuli' / (story + '_envelope.npy')
                row['paired_feature_path'] = str(feature)
                if feature.is_file():
                    b = np.load(feature, mmap_mode='r', allow_pickle=False)
                    row['paired_samples'] = b.shape[0]
                    row['sample_count_difference'] = values.shape[1] - b.shape[0]
                    if values.shape[1] != b.shape[0]:
                        row.update(status='HOLD', reason=(row['reason'] + ';PAIR_LENGTH_MISMATCH').strip(';'))
                else:
                    row.update(status='HOLD', reason=(row['reason'] + ';PAIR_NOT_YET_DOWNLOADED').strip(';'))
                if row['source_recording'] == 'UNKNOWN':
                    row.update(status='HOLD', reason=(row['reason'] + ';SOURCE_MAPPING_UNKNOWN').strip(';'))
            rows.append(row)
        except Exception as exc:
            row.update(status='HOLD', reason=type(exc).__name__ + ':' + str(exc))
            rows.append(row)
    if not rows:
        raise RuntimeError('No completed NPY payload available yet')
    with (out / 'recording_qc.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = dict(task='PUB-01-S06', dataset='SparrKULee', version='3.1',
                    npy_files_scanned=len(rows), eeg_files=sum(r['role'] == 'EEG' for r in rows),
                    hold_files=sum(r['status'] == 'HOLD' for r in rows),
                    scope='Completed NPY derivatives present at run start only; not full 4142-file QC',
                    transformations='NONE; preserve publisher preprocessing, no truncation/padding/refiltering',
                    benchmark_admission='HOLD: per-file timebase/channel/unit provenance and missing scope pending',
                    excluded_formats='data_dict not unpickled; raw BDF/NPZ QC pending',
                    publisher_pipeline_source='https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/preprocessing_code/sparrKULee.yaml')
    (out / 'cleaning_manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
