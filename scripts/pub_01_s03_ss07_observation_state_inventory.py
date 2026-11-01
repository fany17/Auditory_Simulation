"""D11 metadata/header-only inventory, on server2203. No signal/fit/test."""
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np

BASE = Path('/home/fanyu/auditory_simulation_m6a/pub_01')
D8 = BASE / 's03_finetuned_readout_transfer_20261101'
OLD = BASE / 's03_strong_baselines_20261015/transformer_available01'
ROOT = BASE / 's03_observation_state_probe_20261101'
SEEDS = (1024, 2026, 31415)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(stream):
    version = np.lib.format.read_magic(stream)
    shape, order, dtype = np.lib.format._read_array_header(stream, version)
    assert not dtype.hasobject
    return dict(shape=list(shape), fortran_order=order, dtype=str(dtype))


def fingerprint(path):
    path = Path(path)
    st = path.stat()
    result = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    if path.suffix == '.npy':
        with path.open('rb') as stream:
            result.update(header(stream))
    if path.suffix == '.npz':
        with zipfile.ZipFile(path) as archive:
            result['arrays'] = {}
            for name in archive.namelist():
                assert name.endswith('.npy')
                with archive.open(name) as stream:
                    result['arrays'][name] = header(stream)
    return result


def main():
    ROOT.mkdir(exist_ok=True)
    output = ROOT / 'INPUT_INVENTORY.json'
    assert not output.exists()
    protocol = json.loads((D8 / 'PROTOCOL.json').read_text())
    counts, inputs, pins = {}, {}, {}
    records = protocol['records']
    for phase, rows in records.items():
        assert phase in ('train', 'val')
        for row in rows:
            assert row['windows'] > 41
            assert row['windows'] == (row['common_samples'] - 192) // 96 + 1
            for key in ('eeg_path', 'envelope_path'):
                path = Path(row[key])
                inputs[str(path)] = fingerprint(path)
            assert inputs[row['eeg_path']]['shape'][0] == 64
            assert inputs[row['eeg_path']]['shape'][1] >= row['common_samples']
        counts[phase] = dict(records=len(rows), people=len({r['subject'] for r in rows}),
                             windows=sum(r['windows'] for r in rows),
                             fit_windows=sum(r['windows'] - 41 for r in rows),
                             cold_windows=41 * len(rows))
        target = OLD / f'envelope_readout_seed1024/{phase}_envelope.npy'
        inputs[str(target)] = fingerprint(target)
    train_people = {r['subject'] for r in records['train']}
    assert all(r['subject'] in train_people for r in records['val'])
    for person in train_people:
        assert sum(r['subject'] == person for r in records['train']) >= 2
    for seed in SEEDS:
        for name in ('TRAIN_TRANSFORMS.npz', 'TRAIN_MOMENTS.npz',
                     'Ccap_lambda0.01_COEFFICIENTS.npz', 'Ccap_lambda0.01_METRICS.json',
                     'P_FT_BASELINE.json'):
            path = D8 / f'seed{seed}/{name}'
            inputs[str(path)] = fingerprint(path)
            pins[str(path)] = sha(path)
        for phase in records:
            for name in ('latent', 'P'):
                path = D8 / f'seed{seed}/{phase}_{name}.npy'
                inputs[str(path)] = fingerprint(path)
                assert inputs[str(path)]['shape'][0] == counts[phase]['windows']
    normalization = OLD / 'prepared/train_only_normalization.npz'
    inputs[str(normalization)] = fingerprint(normalization)
    for path in (normalization, D8 / 'PROTOCOL.json', Path(__file__)):
        pins[str(path)] = sha(path)
    existing = sum(p.stat().st_size for root in BASE.glob('s03_*_20261101')
                   for p in root.rglob('*.npy'))
    state_payload = sum(c['windows'] for c in counts.values()) * 5 * 8 * 4
    npy_increment = state_payload + 10 * 128
    free = shutil.disk_usage(ROOT).free
    budget = dict(existing_npy_bytes=existing, state_payload_bytes=state_payload,
                  new_npy_bytes=npy_increment, projected_npy_bytes=existing+npy_increment,
                  array_cap_bytes=25*1024**3, root_cap_bytes=256*1024**2,
                  new_numerical_array_cap_bytes=128*1024**2,
                  disk_free_bytes=free, minimum_free_bytes=20*1024**3,
                  peak_RSS_cap_bytes=8*1024**3, wall_cap_seconds=3600,
                  BLAS_threads=16, GPU=False, new_ridge_fits=15,
                  nofit_intervention_roles=6, NN_forwards=0, NN_updates=0)
    assert existing+npy_increment <= budget['array_cap_bytes']
    assert free-budget['root_cap_bytes'] >= budget['minimum_free_bytes']
    assert counts['train']['fit_windows'] == 234988 and counts['val']['fit_windows'] == 28491
    result = dict(scope='ORIGINAL_TRAIN_VAL_METADATA_HEADERS_ONLY', time=time.time(),
                  counts=counts, budget=budget, inputs=inputs, source_pins=pins,
                  no_signal_values=True, no_fit=True, no_test_DTU=True,
                  patients='FORBIDDEN', new_download=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','source_pins')},
                     indent=2))


if __name__ == '__main__':
    main()
