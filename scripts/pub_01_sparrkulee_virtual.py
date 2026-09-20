"""Publisher common-overlap virtual derivative, no adjacent split/normalization.

Contract source: exporl/auditory-eeg-dataset master technical_validation/util/
split_and_normalize.py, lines 48-86, inspected 2026-09-19. The original arrays
are unchanged. This reproduces a publisher consumption convention; it does not
independently establish physical synchronization or historical code revision.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import platform
import re
import numpy as np

ROOT = Path('/home/fanyu/auditory_simulation_m6a')
SOURCE = 'https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/technical_validation/util/split_and_normalize.py'


class CommonOverlap:
    def __init__(self, eeg_path, envelope_path):
        self.eeg = np.load(eeg_path, mmap_mode='r', allow_pickle=False)
        self.envelope = np.load(envelope_path, mmap_mode='r', allow_pickle=False)
        if self.eeg.ndim != 2 or self.eeg.shape[0] != 64 or self.envelope.ndim != 2 or self.envelope.shape[1] != 1:
            raise ValueError('Unexpected publisher array shapes')
        self.length = min(self.eeg.shape[1], self.envelope.shape[0])

    def read(self, start, stop):
        if not 0 <= start < stop <= self.length:
            raise ValueError('Outside common-overlap interval')
        return self.eeg[:, start:stop].T, self.envelope[start:stop]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    raw, out = Path(args.raw).resolve(), Path(args.output).resolve()
    if platform.system() != 'Linux' or ROOT not in raw.parents or ROOT not in out.parents:
        raise ValueError('Server project only')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'executed_config.json').write_text(json.dumps(config, indent=2))
    prefix = raw / 'derivatives/preprocessed_eeg'
    metadata = json.loads((prefix / '.save_metadata.json').read_text())
    with open(args.inventory) as f:
        official = {r['relative_path'] for r in csv.DictReader(f) if r['restricted'] == 'False'}
    rows = []
    aliases = defaultdict(list)
    for source, files in metadata.items():
        for relative in files:
            aliases[relative].append(source)
    normalized_metadata = defaultdict(list)
    for relative, sources in aliases.items():
        if len({s.removesuffix('.gz') for s in sources}) != 1:
            raise ValueError('Ambiguous source identity for ' + relative)
        # .bdf and .bdf.gz source aliases refer to one published derivative path.
        normalized_metadata[sorted(sources)[0]].append(relative)
    for source, files in sorted(normalized_metadata.items()):
        for relative in files:
            path = prefix / relative
            match = re.search(r'_desc-preproc-audio-(.+)_eeg.npy$', path.name)
            story = match.group(1) if match else 'UNKNOWN'
            envelope = raw / 'derivatives/preprocessed_stimuli' / (story + '_envelope.npy')
            row = dict(source_recording=source, eeg_path=str(path), envelope_path=str(envelope),
                       source_aliases=';'.join(sorted(aliases[relative])),
                       subject=path.parts[-3], session=path.parts[-2], story=story,
                       status='MISSING', samples=0, eeg_tail_samples='UNKNOWN', envelope_tail_samples='UNKNOWN',
                       tail_flag='UNKNOWN', sample_rate_hz=64, rate_basis='PUBLISHER_CONSUMPTION_CONTRACT',
                       offset_samples=0, time_scale=1, split='UNASSIGNED',
                       nonfinite='UNKNOWN', flat_eeg_channels='UNKNOWN', numerical_max_difference='UNKNOWN',
                       physical_sync='UNKNOWN', historical_generator_version='UNKNOWN', error='')
            if 'derivatives/preprocessed_eeg/' + relative not in official:
                row.update(status='NOT_IN_OFFICIAL_INVENTORY', error='Publisher save metadata references unpublished/absent object; do not invent download')
                rows.append(row)
                continue
            if not path.is_file() or not envelope.is_file():
                rows.append(row)
                continue
            try:
                data = CommonOverlap(path, envelope)
                row.update(samples=data.length, eeg_tail_samples=data.eeg.shape[1] - data.length,
                           envelope_tail_samples=data.envelope.shape[0] - data.length)
                # 3 seconds is an advisory flag: nominal 2s author tail plus rounding,
                # not a pass threshold, and never used to change the data window.
                row['tail_flag'] = 'LARGE_TAIL_REVIEW' if max(row['eeg_tail_samples'], row['envelope_tail_samples']) > config['large_tail_advisory_samples'] else 'WITHIN_3S_ADVISORY'
                nonfinite = 0
                lo, hi = np.full(64, np.inf), np.full(64, -np.inf)
                for start in range(0, data.length, 8192):
                    eeg, env = data.read(start, min(start + 8192, data.length))
                    nonfinite += int((~np.isfinite(eeg)).sum() + (~np.isfinite(env)).sum())
                    lo, hi = np.minimum(lo, eeg.min(axis=0)), np.maximum(hi, eeg.max(axis=0))
                a = np.load(path, mmap_mode='r', allow_pickle=False)
                b = np.load(envelope, mmap_mode='r', allow_pickle=False)
                difference = 0.0
                for start in [0, data.length // 2, max(data.length - 64, 0)]:
                    stop = min(start + 64, data.length)
                    eeg, env = data.read(start, stop)
                    difference = max(difference, float(np.max(np.abs(eeg - a[:, start:stop].T))), float(np.max(np.abs(env - b[start:stop]))))
                row.update(nonfinite=nonfinite, flat_eeg_channels=int((lo == hi).sum()), numerical_max_difference=difference,
                           status='PASS_VIRTUAL_INITIAL_QC' if nonfinite == 0 and not (lo == hi).any() and difference == 0 else 'HOLD_VALUES')
            except Exception as exc:
                row.update(status='HOLD', error=type(exc).__name__ + ': ' + str(exc))
            rows.append(row)
    fields = list(rows[0])
    for filename, subset in [('cleaning_index.csv', rows), ('usable_index.csv', [r for r in rows if r['status'] == 'PASS_VIRTUAL_INITIAL_QC'])]:
        with (out / filename).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(subset)
    summary = dict(task='PUB-01-S06', dataset='SparrKULee', version='3.1',
                    metadata_mapping_rows=sum(len(v) for v in metadata.values()),
                    duplicate_source_alias_rows=sum(len(v)-1 for v in aliases.values()),
                    expected_eeg_derivatives=len(rows), available_validated=sum(r['status'] == 'PASS_VIRTUAL_INITIAL_QC' for r in rows),
                    missing=sum(r['status'] == 'MISSING' for r in rows), hold=sum(r['status'].startswith('HOLD') for r in rows),
                    not_in_official_inventory=sum(r['status'] == 'NOT_IN_OFFICIAL_INVENTORY' for r in rows),
                    large_tail_flag=sum(r['tail_flag'] == 'LARGE_TAIL_REVIEW' for r in rows),
                    loader='pub_01_sparrkulee_virtual.CommonOverlap', source=SOURCE,
                    transformation='Time-first view of EEG, common prefix at index 0; excluded tails explicit; raw arrays unchanged',
                    numerical_validation='All indexed samples finite/flat-checked through loader; 3 positions per recording compared directly to raw arrays',
                    rate_hz=64, rate_basis='Publisher preprocessing YAML and consumption contract; NPY has no embedded sample-rate header',
                    forbidden_actions_not_done=['train', 'adjacent_time_split', 'normalize', 'refilter', 'rereference'],
                    limitations=['Physical sync not independently established', 'Historical generator revision unknown',
                                 'Large-tail flags retained for review, not silently accepted scientific pairing',
                                 'Full public download and other-format QC separate', 'Group split remains unassigned'])
    (out / 'cleaning_manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
