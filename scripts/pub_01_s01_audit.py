"""PUB-01 S01 read-only audit; run only on server2203, not Windows.

No checksum/object identifiers or payload are exported. Originals stay untouched.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import urllib.request
from collections import defaultdict
import numpy as np

ROOT = Path('/home/fanyu/auditory_simulation_m6a')
RDR = 'https://rdr.kuleuven.be'
DOI = 'doi:10.48804/K3VSND'


def get(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def official(out):
    url = RDR + '/api/datasets/:persistentId/?persistentId=' + DOI
    data = json.loads(get(url))['data']
    version = data['latestVersion']
    # Explicit whitelist: do not save raw API or inspect integrity metadata.
    summary = {k: version.get(k, 'UNKNOWN') for k in
               ['versionNumber', 'versionMinorNumber', 'versionState',
                'releaseTime', 'license', 'termsOfUse', 'termsOfAccess',
                'restrictions', 'fileAccessRequest', 'confidentialityDeclaration']}
    summary['source_url'] = url
    summary['metadataBlocks'] = version.get('metadataBlocks', {})
    rows = []
    for item in version['files']:
        f = item['dataFile']
        rows.append(dict(dataset='SparrKULee', version=f"{version['versionNumber']}.{version['versionMinorNumber']}",
                         file_id=f['id'], relative_path='/'.join(filter(None, [item.get('directoryLabel'), item['label']])),
                         bytes=f.get('filesize', 'UNKNOWN'), restricted=item.get('restricted', 'UNKNOWN'),
                         content_type=f.get('contentType', 'UNKNOWN'),
                         source_url=RDR + '/api/access/datafile/' + str(f['id'])))
    write_json(out / 'sparrkulee_official_metadata.json', summary)
    write_csv(out / 'sparrkulee_official_inventory.csv', rows)
    print(json.dumps({'SparrKULee': {k: summary[k] for k in ['versionNumber', 'versionMinorNumber', 'license', 'termsOfUse', 'termsOfAccess', 'restrictions', 'fileAccessRequest']},
                      'files': len(rows), 'public_files': sum(r['restricted'] is False for r in rows),
                      'public_bytes': sum(r['bytes'] for r in rows if r['restricted'] is False)}, ensure_ascii=False))
    for row in rows[:15]:
        print(json.dumps(row))
    sources = {
        'ds004703_official_readme.txt': 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds004703/master/README',
        'ds004703_official_description.json': 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds004703/master/dataset_description.json',
        'sparrkulee_code_readme.txt': 'https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/README.md',
        'sparrkulee_code_config.json': 'https://raw.githubusercontent.com/exporl/auditory-eeg-dataset/master/config.json',
    }
    for name, source in sources.items():
        try:
            (out / name).write_bytes(get(source))
            print('Official document retrieved:', name, source)
        except Exception as exc:
            print('Official document failed:', name, str(exc))
    write_json(out / 'official_document_sources.json', sources)


def probe_ds(out):
    base = ROOT / 'data/ds004703/v1.1.0'
    print('Existing description:', (base / 'dataset_description.json').read_text())
    print('Existing README:', (base / 'README').read_text())
    wavs = sorted(base.rglob('*.wav'))
    print('WAV count:', len(wavs))
    print('WAV example paths:', [str(x.relative_to(base)) for x in wavs[:20]])
    print('TSV paths:', [str(x.relative_to(base)) for x in base.glob('sub-*/*/*.tsv')][:30])
    print('Stimulus table sample:', (base / 'stimuli/stim-times.tsv').read_text().splitlines()[:5])


def ds_audit(out):
    import mne
    import soundfile as sf
    base = ROOT / 'data/ds004703/v1.1.0'
    stimuli = defaultdict(list)
    with (base / 'stimuli/stim-times.tsv').open() as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['t_start'] == 't_start':
                continue  # documented duplicate header, never silently treated as an event
            stimuli[r['ex_name']].append(r)
    wavs = defaultdict(list)
    audios = []
    for path in sorted(base.rglob('*.wav')):
        info = sf.info(path)
        key = path.stem.removesuffix('_normed')
        wavs[key].append(path)
        audios.append(dict(dataset='ds004703', path=str(path), story=key,
                           sample_rate=info.samplerate, frames=info.frames,
                           channels=info.channels, duration_s=info.duration))
    recordings, pairs, exclusions = [], [], []
    for path in sorted(base.glob('sub-*/*/ieeg/*.edf')):
        stem = path.name.removesuffix('_ieeg.edf')
        raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
        meta = json.loads(path.with_suffix('.json').read_text())
        event_path = path.parent.parent / (stem + '_events.tsv')
        with event_path.open() as f:
            events = list(csv.DictReader(f, delimiter='\t'))
        duration = raw.n_times / raw.info['sfreq']
        sample = raw.get_data(start=0, stop=min(raw.n_times, int(raw.info['sfreq'])))
        bounds = sum(float(e['onset']) < 0 or float(e['onset']) + float(e['duration']) > duration for e in events)
        recordings.append(dict(dataset='ds004703', recording=stem, subject=path.parts[-4], session=path.parts[-3],
                               server_path=str(path), bytes=path.stat().st_size, sample_rate_hz=raw.info['sfreq'],
                               metadata_sample_rate_hz=meta.get('SamplingFrequency', 'UNKNOWN'),
                               channels=len(raw.ch_names), samples=raw.n_times, duration_s=duration,
                               events=len(events), event_out_of_bounds=bounds,
                               first_second_nonfinite=int((~np.isfinite(sample)).sum()),
                               reference=meta.get('iEEGReference', 'UNKNOWN'),
                               sample_scope='header + first second only; not full QC'))
        for channel in raw.ch_names:
            if channel.startswith('C'):
                exclusions.append(dict(dataset='ds004703', recording=stem, item=channel, reason='README unused C-prefix channel; exclude at S06'))
        grouped = []
        for e in events:
            if not grouped or grouped[-1][0]['ex_name'] != e['ex_name']:
                grouped.append([])
            grouped[-1].append(e)
        for trial, block in enumerate(grouped):
            name = block[0]['ex_name']
            source = stimuli.get(name, [])
            same = len(source) == len(block) and all((s['word'], s['phone']) == (e['word'], e['phone']) for s, e in zip(source, block))
            residual = 'UNKNOWN'
            offset = 'UNKNOWN'
            if same:
                offsets = np.array([float(e['onset']) - float(s['t_start']) for s, e in zip(source, block)])
                offset = float(np.median(offsets))
                residual = float(np.max(np.abs(offsets - offset)))
            candidates = wavs.get(name, [])
            status = 'PASS_METADATA_PAIRING' if same and len(candidates) == 1 and residual < 0.002 else 'HOLD'
            pairs.append(dict(dataset='ds004703', recording=stem, subject=path.parts[-4], session=path.parts[-3],
                              trial=trial, story=name, n_events=len(block), source_events=len(source),
                              audio_candidates=len(candidates), audio_path=str(candidates[0]) if len(candidates) == 1 else 'UNKNOWN',
                              labels_match=same, offset_s=offset, max_residual_s=residual, status=status,
                              physical_audio_sync='UNKNOWN: metadata alignment only'))
        raw.close()
    write_csv(out / 'ds004703_recording_inventory.csv', recordings)
    write_csv(out / 'ds004703_audio_inventory.csv', audios)
    write_csv(out / 'audio_neural_pairing_audit.csv', pairs)
    write_csv(out / 'exclusion_inventory.csv', exclusions, ['dataset', 'recording', 'item', 'reason'])
    print('ds audit:', len(recordings), 'recordings;', len(audios), 'audio files;', len(pairs), 'trials;', sum(p['status'] == 'HOLD' for p in pairs), 'HOLD trials')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=['official', 'ds'], default='official')
    args = parser.parse_args()
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise RuntimeError('Server-only audit: required Linux project root absent')
    out = Path(args.output).resolve()
    if ROOT not in out.parents:
        raise ValueError('Output must stay within server project root')
    out.mkdir(parents=True, exist_ok=False)
    if args.phase == 'official':
        official(out)
        probe_ds(out)
    else:
        ds_audit(out)


if __name__ == '__main__':
    main()
