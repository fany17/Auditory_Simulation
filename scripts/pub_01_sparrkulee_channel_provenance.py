"""Map public derivative channels to actual available raw BDF headers.

No assumption that a restricted/missing raw source has been inspected.
Source selection is by frozen official inventory and publisher mapping only.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/fanyu/auditory_simulation_m6a')
BAD = {
    'sub-006_ses-shortstories01_task-listeningActive_run-06_eeg.bdf.gz',
    'sub-017_ses-shortstories01_task-listeningActive_run-03_eeg.bdf.gz',
    'sub-048_ses-varyingStories05_task-listeningActive_run-04_eeg.bdf.gz',
}


def header(path):
    with gzip.open(path, 'rb') as f:
        h = f.read(256)
        if not h.startswith(b'\xffBIOSEMI'):
            raise ValueError('Not BDF')
        n, size, duration = int(h[252:256]), int(h[184:192]), float(h[244:252])
        if n < 64 or size != (n + 1) * 256 or not np.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid/short BDF channel header or duration')
        data = f.read(size - 256)
        if len(data) != size - 256:
            raise ValueError('Truncated BDF signal header')
    def strings(offset, width):
        return [data[offset*n+i*width:offset*n+(i+1)*width].decode('ascii').strip() for i in range(n)]
    fields = dict(channel_names=strings(0,16), physical_units=strings(96,8))
    if not all(fields['channel_names'][:64]) or not all(fields['physical_units'][:64]):
        raise ValueError('Unknown channel names/physical units in first 64 channels')
    for name, offset in [('physical_min',104),('physical_max',112),('digital_min',120),('digital_max',128)]:
        fields[name] = [float(v) for v in strings(offset,8)]
    fields['sample_rates_hz'] = [int(v)/duration for v in strings(216,8)]
    if not all(len(fields[k][:64]) == 64 for k in fields) or not all(r > 0 and np.isfinite(r) for r in fields['sample_rates_hz'][:64]):
        raise ValueError('Invalid 64-channel metadata/rates')
    fields['calibration_valid'] = [np.isfinite([fields[k][i] for k in ['physical_min','physical_max','digital_min','digital_max']]).all().item()
                                   and fields['physical_min'][i] != fields['physical_max'][i]
                                   and fields['digital_min'][i] < fields['digital_max'][i] for i in range(n)]
    return fields


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--virtual-index', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    raw, out = Path(args.raw).resolve(), Path(args.output).resolve()
    if ROOT not in raw.parents or ROOT not in out.parents:
        raise ValueError('Server project only')
    out.mkdir(parents=True, exist_ok=False)
    with open(args.inventory) as f:
        official = list(csv.DictReader(f))
    by_name = {Path(r['relative_path']).name:r for r in official if r['relative_path'].endswith('.bdf.gz')}
    by_path = {r['relative_path']:r for r in official}
    with open(args.virtual_index) as f:
        index = list(csv.DictReader(f))
    sidecar = json.loads((raw/'task-listeningActive_eeg.json').read_text())
    rows = []
    for entry in index:
        name = entry['source_recording'].removesuffix('.gz') + '.gz'
        source = by_name.get(name)
        row = dict(eeg_path=entry['eeg_path'], source_recording=name, source_raw_path='UNKNOWN',
                   source_access='UNKNOWN', status='HOLD', official_recording_bad=name in BAD,
                   channel_index_mapping='0..63 in original order per publisher config',
                   channel_names='UNKNOWN', channel_types='EEG per publisher dataset/config',
                   source_physical_units='UNKNOWN', source_sample_rates_hz='UNKNOWN',
                   calibration_valid='UNKNOWN', publisher_npy_units='microvolts by MNE volts x 1e6 contract; historical generator revision UNKNOWN',
                   derivative_rate_hz=64, filter_provenance='publisher highpass/artifact/MWF/CAR/resample; not repeated here',
                   dataset_sidecar_sampling_hz=sidecar.get('SamplingFrequency','UNKNOWN'), header_sidecar_rate_match='UNKNOWN',
                   source_stimulation_path='UNKNOWN', stimulation_access='UNKNOWN', stimulation_story_match='UNKNOWN',
                   stimulation_limitation='', error='')
        if source is None:
            row['error'] = 'NO_OFFICIAL_RAW_SOURCE'
        else:
            row['source_access'] = 'RESTRICTED' if source['restricted']=='True' else 'PUBLIC'
            path = raw / source['relative_path']
            row['source_raw_path'] = str(path)
            stimulation = path.with_name(name.removesuffix('_eeg.bdf.gz') + '_stimulation.tsv')
            row['source_stimulation_path'] = str(stimulation)
            stim_item=by_path.get(str(stimulation.relative_to(raw)))
            row['stimulation_access']='RESTRICTED' if stim_item and stim_item['restricted']=='True' else 'PUBLIC' if stim_item else 'NOT_IN_OFFICIAL_INVENTORY'
            if row['stimulation_access']=='RESTRICTED':
                row['stimulation_limitation']='OBJECT_LEVEL_ACCESS_RESTRICTION; not requested, not download missing'
            if stimulation.is_file():
                with stimulation.open() as f:
                    table = list(csv.DictReader(f, delimiter='\t'))
                paths = [r.get('apx_file','') for r in table]
                row['stimulation_story_match'] = any(Path(s).stem == entry['story'] for s in paths)
            if source['restricted']=='True':
                row['error'] = 'RAW_RESTRICTED_NOT_ACCESSED; public derivative may still be usable with provenance limitation'
            elif not path.is_file():
                row['error'] = 'RAW_DOWNLOAD_PENDING'
            else:
                try:
                    h = header(path)
                    row.update(channel_names=json.dumps(h['channel_names'][:64]),
                               source_physical_units=json.dumps(h['physical_units'][:64]),
                               source_sample_rates_hz=json.dumps(h['sample_rates_hz'][:64]),
                               header_sidecar_rate_match=all(r==sidecar.get('SamplingFrequency') for r in h['sample_rates_hz'][:64]),
                               calibration_valid=all(h['calibration_valid'][:64]),
                               status='PASS_HEADER_PROVENANCE' if all(h['calibration_valid'][:64]) and not row['official_recording_bad'] else 'HOLD',
                               error='')
                except Exception as exc:
                    row['error'] = type(exc).__name__ + ':' + str(exc)
        rows.append(row)
    with (out / 'channel_provenance.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bad_rows = [dict(source_recording=name, official_reason='README insufficient triggers; cannot accurately align',
                     derivative_references=sum(r['source_recording']==name for r in rows)) for name in sorted(BAD)]
    with (out / 'official_bad_recordings.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(bad_rows[0])); w.writeheader(); w.writerows(bad_rows)
    summary = dict(derivatives=len(rows), header_verified=sum(r['status']=='PASS_HEADER_PROVENANCE' for r in rows),
                    restricted_raw=sum(r['source_access']=='RESTRICTED' for r in rows),
                    missing_raw=sum(r['error']=='RAW_DOWNLOAD_PENDING' for r in rows),
                    unknown_raw=sum(r['error']=='NO_OFFICIAL_RAW_SOURCE' for r in rows),
                    scope='Actual raw header and publisher mapping; not independent reproduction of preprocessing',
                    historical_generator_version='UNKNOWN')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
