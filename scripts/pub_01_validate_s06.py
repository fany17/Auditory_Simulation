"""Validate lightweight S06 structured outputs, never signal payloads."""
import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def validate(folder):
    folder = Path(folder)
    records = rows(folder / 'recording_qc.csv')
    pairs = rows(folder / 'audio_neural_pairing_audit.csv')
    clean = rows(folder / 'clean_segment_index.csv')
    channel = rows(folder / 'clean_channel_index.csv')
    numeric = rows(folder / 'virtual_numeric_validation.csv')
    virtual = rows(folder / 'virtual_segment_validation.csv')
    manifest = json.loads((folder / 'cleaning_manifest.json').read_text())
    assert len(records) == manifest['recordings'] == 11
    assert len(pairs) == manifest['event_groups'] == 438
    assert len(clean) == manifest['eligible_segments'] == len(virtual)
    assert all(r['scan'] == 'ALL_SAMPLES' for r in records)
    assert all(int(r['nonfinite_samples']) == 0 and int(r['event_out_of_bounds']) == 0 for r in records)
    assert all(r['eligible'] == 'True' and not r['reason'] and not r['story'].startswith('catalan-') for r in clean)
    assert all(r['eligible'] == 'True' and not r['reasons'] for r in channel)
    assert len(numeric) == 11 and all(float(r['max_abs_difference']) == 0 and r['status'] == 'PASS' for r in numeric)
    assert all(int(r['samples_read']) == int(r['stop_sample_exclusive']) - int(r['start_sample']) > 0 for r in virtual)
    assert all(r['physical_sync'] == 'UNKNOWN' and r['split'] == 'UNASSIGNED_S01_REVIEW_REQUIRED' for r in pairs)
    assert manifest['original_preserved'] is True
    print(json.dumps(dict(status='PASS', recordings=len(records), indexed_segments=len(clean),
                          indexed_channels=len(channel), checked_numeric_rows=len(numeric),
                          scope='Technical schema/count/eligibility validation; not scientific acceptance')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder')
    validate(p.parse_args().folder)
