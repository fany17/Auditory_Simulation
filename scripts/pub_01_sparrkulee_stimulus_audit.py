"""Bind published envelope files to safe-decoded data_dict provenance.

One file at a time; no serialized callable execution and no signal export.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from pub_01_s06_format_qc import NumericUnpickler, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--inventory', required=True)
    args = p.parse_args()
    raw, out = Path(args.raw).resolve(), Path(args.output).resolve()
    if ROOT not in raw.parents or ROOT not in out.parents:
        raise ValueError('Server project only')
    out.mkdir(parents=True, exist_ok=False)
    base = raw / 'derivatives/preprocessed_stimuli'
    metadata = json.loads((base / '.save_metadata.json').read_text())
    aliases = defaultdict(list)
    for source, items in metadata.items():
        for item in items:
            aliases[item].append(source)
    with open(args.inventory) as f:
        official = [r for r in csv.DictReader(f) if r['restricted'] == 'False']
    official_dicts = {Path(r['relative_path']).name for r in official if r['relative_path'].endswith('.data_dict')}
    eeg_paths = [r['relative_path'] for r in official if r['relative_path'].startswith('derivatives/preprocessed_eeg/') and r['relative_path'].endswith('.npy')]
    rows = []
    for relative in sorted(official_dicts):
        sources = aliases.get(relative, [])
        path = base / relative
        envelope = base / (path.stem + '_envelope.npy')
        row = dict(story=path.stem, data_dict_path=str(path), envelope_path=str(envelope),
                   source_aliases=';'.join(sorted(sources)), status='MISSING', error='',
                   in_save_metadata=bool(sources), corresponding_envelope_exists=envelope.is_file(),
                   exact_story_eeg_references=sum(f'-audio-{path.stem}_eeg.npy' in q for q in eeg_paths),
                   published_envelope_sr='UNKNOWN', envelope_samples='UNKNOWN',
                   raw_trigger_sr='UNKNOWN', raw_stimulus_samples='UNKNOWN',
                   trigger_count='UNKNOWN', first_trigger_s='UNKNOWN', last_trigger_s='UNKNOWN',
                   envelope_matches_data_dict='UNKNOWN', publisher_steps='UNKNOWN')
        if not path.is_file():
            rows.append(row)
            continue
        try:
            with path.open('rb') as f:
                data = NumericUnpickler(f).load()
                if f.read(1):
                    raise ValueError('Trailing serialized content')
            if not envelope.is_file():
                raise ValueError('No matching envelope for official data_dict')
            values = np.load(envelope, mmap_mode='r', allow_pickle=False)
            embedded = data['envelope_data']
            if values.dtype.hasobject or embedded.dtype.hasobject:
                raise ValueError('Object dtype forbidden')
            same = values.shape == embedded.shape and np.array_equal(values, embedded)
            trigger = np.asarray(data['trigger_data']).reshape(-1)
            indices = np.flatnonzero(trigger > 0.5)
            rising = indices[np.r_[True, np.diff(indices) > 1]] if indices.size else indices
            trigger_sr = float(np.asarray(data['trigger_sr']).item())
            row.update(status='PASS_STIMULUS_PROVENANCE' if same else 'HOLD_ENVELOPE_MISMATCH',
                       published_envelope_sr=float(data['stimulus_sr']), envelope_samples=values.shape[0],
                       raw_trigger_sr=trigger_sr, raw_stimulus_samples=len(data['stimulus_data']),
                       trigger_count=len(rising), first_trigger_s=float(rising[0]/trigger_sr) if len(rising) else 'UNKNOWN',
                       last_trigger_s=float(rising[-1]/trigger_sr) if len(rising) else 'UNKNOWN',
                       envelope_matches_data_dict=same,
                       publisher_steps=';'.join(step.get('step_name','UNKNOWN') for step in data.get('previous_steps', [])))
            del data, embedded, trigger
        except Exception as exc:
            row.update(status='HOLD', error=type(exc).__name__ + ': ' + str(exc))
        rows.append(row)
        print(row['story'], row['status'], flush=True)
    with (out / 'stimulus_provenance.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = dict(unique_stimulus_dicts=len(rows), alias_rows=sum(len(v) for v in metadata.values()),
                    pass_count=sum(r['status']=='PASS_STIMULUS_PROVENANCE' for r in rows),
                    missing=sum(r['status']=='MISSING' for r in rows),
                    hold=sum(r['status'].startswith('HOLD') for r in rows),
                    provenance='Actual published data_dict metadata + direct numerical comparison to envelope NPY',
                    physical_sync='UNKNOWN; no neural recording trigger verification here')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
