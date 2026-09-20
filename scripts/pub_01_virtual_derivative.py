"""Consume PUB-01 S06 virtual cleaned data; originals remain immutable.

Public API: VirtualDerivative(folder).read(recording, start, stop) yields
eligible-channel voltage samples at native rate. segments lists admissible
half-open sample intervals; no filtering/resampling or data fitting occurs.
"""
import csv
import json
import math
from pathlib import Path
import mne
import numpy as np


def rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


class VirtualDerivative:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.records = {r['recording']: r for r in rows(self.folder / 'recording_qc.csv')}
        self.channels = {}
        for row in rows(self.folder / 'clean_channel_index.csv'):
            self.channels.setdefault(row['recording'], []).append(row['channel'])
        self.segments = rows(self.folder / 'clean_segment_index.csv')
        self._recording = None
        self._raw = None

    def raw(self, recording):
        if recording != self._recording:
            self.close()
            self._raw = mne.io.read_raw_edf(self.records[recording]['server_raw_path'], preload=False, verbose='ERROR')
            self._recording = recording
        return self._raw

    def read(self, recording, start, stop):
        if recording not in self.channels or not self.channels[recording]:
            raise ValueError('No eligible channels')
        raw = self.raw(recording)
        if not 0 <= start < stop <= raw.n_times:
            raise ValueError('Sample interval outside recording')
        return raw.get_data(picks=self.channels[recording], start=start, stop=stop)

    def segment_interval(self, segment):
        rate = float(self.records[segment['recording']]['sample_rate_hz'])
        start = math.floor(float(segment['offset_s']) * rate)
        stop = math.ceil((float(segment['offset_s']) + float(segment['audio_duration_s'])) * rate)
        return start, stop

    def close(self):
        if self._raw is not None:
            self._raw.close()
        self._raw = None
        self._recording = None


def validate(folder):
    consumer = VirtualDerivative(folder)
    mapping, checks = [], []
    for recording, record in consumer.records.items():
        source = mne.io.read_raw_edf(record['server_raw_path'], preload=False, verbose='ERROR')
        names = consumer.channels.get(recording, [])
        if not names:
            checks.append(dict(recording=recording, channels=0, numeric_comparisons=0,
                               max_abs_difference=0.0, status='NO_ELIGIBLE_CHANNELS'))
            source.close()
            continue
        max_difference = 0.0
        for start in [0, source.n_times // 2, source.n_times - min(256, source.n_times)]:
            stop = min(start + 256, source.n_times)
            value = consumer.read(recording, start, stop)
            expected = source.get_data(start=start, stop=stop)[[source.ch_names.index(n) for n in names]]
            max_difference = max(max_difference, float(np.max(np.abs(value - expected))))
            if not np.array_equal(value, expected):
                raise ValueError('Virtual derivative changes numeric samples')
        for name in names:
            mapping.append(dict(recording=recording, channel=name, source_channel_index=source.ch_names.index(name),
                                sample_rate_hz=source.info['sfreq'], sample_offset=0, time_scale=1, transformation='CHANNEL_SELECTION_ONLY'))
        checks.append(dict(recording=recording, channels=len(names), numeric_comparisons=3,
                           max_abs_difference=max_difference, status='PASS'))
        source.close()
    segment_checks = []
    for segment in consumer.segments:
        recording = segment['recording']
        start, stop = consumer.segment_interval(segment)
        # Read every indexed sample through the actual consumer, in bounded chunks.
        count = 0
        for pos in range(start, stop, 8192):
            values = consumer.read(recording, pos, min(pos + 8192, stop))
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite virtual segment')
            count += values.shape[1]
        if count != stop - start:
            raise ValueError('Incomplete segment')
        segment_checks.append(dict(recording=recording, trial=segment['trial'], story=segment['story'],
                                   start_sample=start, stop_sample_exclusive=stop,
                                   samples_read=count, status='PASS'))
    consumer.close()
    from pub_01_s06_qc_ds004703 import save_csv
    save_csv(Path(folder) / 'virtual_channel_mapping.csv', mapping,
             ['recording', 'channel', 'source_channel_index', 'sample_rate_hz', 'sample_offset', 'time_scale', 'transformation'])
    save_csv(Path(folder) / 'virtual_numeric_validation.csv', checks)
    save_csv(Path(folder) / 'virtual_segment_validation.csv', segment_checks,
             ['recording', 'trial', 'story', 'start_sample', 'stop_sample_exclusive', 'samples_read', 'status'])
    result = dict(recordings=len(checks), segments=len(segment_checks), numeric_comparisons=sum(r['numeric_comparisons'] for r in checks),
                  max_abs_difference=max(r['max_abs_difference'] for r in checks),
                  status='PASS' if segment_checks else 'NO_ELIGIBLE_SEGMENTS',
                  representation='Virtual derivative with executable loader; not filtered signals',
                  sample_window='floor(onset*sfreq):ceil((onset+audio_duration)*sfreq), half-open',
                  sources='Original EDF paths bound in recording_qc.csv; native samples and rates preserved')
    (Path(folder) / 'virtual_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('folder')
    validate(p.parse_args().folder)
