"""Incremental format audit for every public SparrKULee object on 2203.

Runs alongside the single downloader. Reads only final files of expected size.
Uses normal format decoders, no integrity digest audit. Preserves all inputs and
any decompressed NPZ in a dedicated server directory. No arbitrary pickle globals.
"""
import argparse
from collections import Counter, OrderedDict
import csv
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

ROOT = Path('/home/fanyu/auditory_simulation_m6a')


def inert_publisher_loader(*args, **kwargs):
    raise pickle.UnpicklingError('Serialized publisher loader execution forbidden')


class NumericUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        allowed['collections', 'OrderedDict'] = OrderedDict
        # Known publisher metadata reference, replaced by an inert sentinel.
        # It can be stored, but any attempted REDUCE invocation fails closed.
        allowed['__main__', 'temp_stimulus_load_fn'] = inert_publisher_loader
        for prefix in ['numpy.core.multiarray', 'numpy._core.multiarray']:
            allowed[prefix, '_reconstruct'] = np.core.multiarray._reconstruct
            allowed[prefix, 'scalar'] = np.core.multiarray.scalar
        for prefix in ['numpy.core.numeric', 'numpy._core.numeric']:
            allowed[prefix, '_frombuffer'] = np.core.numeric._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('Forbidden global: ' + module + '.' + name)
        return allowed[module, name]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent references forbidden')


def array_summary(a):
    if a.dtype.hasobject:
        raise ValueError('Object dtype forbidden')
    result = dict(shape=list(a.shape), dtype=str(a.dtype))
    if np.issubdtype(a.dtype, np.number):
        flat = a.reshape(-1)
        nonfinite = 0
        for start in range(0, flat.size, 1048576):
            nonfinite += int((~np.isfinite(flat[start:start + 1048576])).sum())
        result['nonfinite'] = nonfinite
    return result


def describe(value, depth=0):
    if depth > 8:
        raise ValueError('Nested object too deep')
    if value is inert_publisher_loader:
        return {'inert_callable_reference': '__main__.temp_stimulus_load_fn', 'executed': False}
    if isinstance(value, np.ndarray):
        result = array_summary(value)
        return result
    if isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 500:
            return {'sequence_length': len(value), 'elements': 'NOT_EXPORTED'}
        return [describe(v, depth + 1) for v in value]
    if isinstance(value, (str, int, float, bool, type(None), np.number)):
        return value.item() if isinstance(value, np.number) else value
    raise ValueError('Unsupported decoded type: ' + type(value).__name__)


def bdf_qc(path):
    with gzip.open(path, 'rb') as f:
        h = f.read(256)
        if len(h) != 256 or not h.startswith(b'\xffBIOSEMI'):
            raise ValueError('Not BDF header')
        n = int(h[252:256])
        header_bytes, nrecords, interval = int(h[184:192]), int(h[236:244]), float(h[244:252])
        sh = f.read(header_bytes - 256)
        if len(sh) != header_bytes - 256 or len(sh) < 256 * n:
            raise ValueError('Incomplete signal header')
        labels = [sh[i * 16:(i + 1) * 16].decode('ascii').strip() for i in range(n)]
        units = [sh[96*n + i*8:96*n + (i+1)*8].decode('ascii').strip() for i in range(n)]
        fields = {}
        for field, offset in [('physical_min', 104), ('physical_max', 112), ('digital_min', 120), ('digital_max', 128)]:
            fields[field] = [float(sh[offset*n + i*8:offset*n + (i+1)*8]) for i in range(n)]
        calibration_valid = [all(np.isfinite(fields[k][i]) for k in fields) and
                             fields['physical_max'][i] != fields['physical_min'][i] and
                             fields['digital_max'][i] > fields['digital_min'][i] for i in range(n)]
        sizes = [int(sh[216*n + i*8:216*n + (i+1)*8]) for i in range(n)]
        block_bytes = 3 * sum(sizes)
        lo, hi = np.full(n, 2**23), np.full(n, -2**23)
        read_records = 0
        while block := f.read(block_bytes):
            if len(block) != block_bytes:
                raise ValueError('Truncated BDF data record')
            b = np.frombuffer(block, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
            signal = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            signal = (signal ^ (1 << 23)) - (1 << 23)
            offset = 0
            for i, size in enumerate(sizes):
                v = signal[offset:offset + size]
                lo[i], hi[i] = min(lo[i], int(v.min())), max(hi[i], int(v.max()))
                offset += size
            read_records += 1
        if nrecords >= 0 and read_records != nrecords:
            raise ValueError('BDF record count mismatch')
        return dict(format='BDF_GZIP', channels=n, records=read_records, duration_s=read_records * interval,
                    sample_rates_hz=[s / interval for s in sizes], channel_names=labels,
                    physical_dimensions=units, **fields, calibration_valid=calibration_valid,
                    quality_status='CALIBRATION_VALID' if all(calibration_valid) else 'HOLD_CALIBRATION',
                    flat_channels=[labels[i] for i in range(n) if lo[i] == hi[i]],
                    all_samples_decoded=True, interpretation='digital int24 values, not reprocessed EEG')


def inspect(path, expanded):
    name = path.name
    if name.endswith('.bdf.gz'):
        return bdf_qc(path)
    if name.endswith('.data_dict'):
        with path.open('rb') as f:
            magic = f.read(2)
            if len(magic) != 2 or magic[0] != 128:
                raise ValueError('Expected binary pickle header')
            f.seek(0)
            value = NumericUnpickler(f).load()
            if f.read(1):
                raise ValueError('Trailing pickle data')
        if not isinstance(value, dict):
            raise ValueError('Expected data_dict')
        return dict(format='RESTRICTED_NUMPY_PICKLE', protocol=magic[1], fields=describe(value))
    if name.endswith('.npy'):
        return dict(format='NPY', array=array_summary(np.load(path, mmap_mode='r', allow_pickle=False)))
    if name.endswith('.gz'):
        dest = expanded / name.removesuffix('.gz')
        # Per-file sequential expansion; any failed expansion remains explicit.
        if dest.exists():
            raise ValueError('Expanded file exists; review rather than overwrite')
        if shutil.disk_usage(expanded).free < 50 * 1024**3:
            raise ValueError('Free space below 50GiB reserve')
        with gzip.open(path, 'rb') as src, dest.open('xb') as target:
            shutil.copyfileobj(src, target, length=1024 * 1024)
        if dest.suffix == '.npz':
            with np.load(dest, allow_pickle=False) as data:
                return dict(format='NPZ_GZIP', expanded_server_path=str(dest), arrays={k: array_summary(data[k]) for k in data.files})
        if dest.suffix == '.npy':
            return dict(format='NPY_GZIP', expanded_server_path=str(dest), array=array_summary(np.load(dest, mmap_mode='r', allow_pickle=False)))
        return dict(format='UNSUPPORTED_GZIP_PAYLOAD', expanded_server_path=str(dest), status='HOLD')
    if path.suffix == '.json':
        value = json.loads(path.read_text())
        return dict(format='JSON', top_level_type=type(value).__name__)
    if path.suffix == '.docx':
        with zipfile.ZipFile(path) as z:
            ET.fromstring(z.read('word/document.xml'))
        return dict(format='DOCX', document_xml_readable=True)
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix == '.tsv':
        table = list(csv.reader(text.splitlines(), delimiter='\t'))
        return dict(format='TSV', rows=max(len(table)-1, 0), columns=len(table[0]) if table else 0,
                    variable_width_rows=sum(len(r) != len(table[0]) for r in table) if table else 0)
    if text.lstrip().startswith('<'):
        ET.fromstring(text)
        return dict(format='XML', parse=True)
    return dict(format='TEXT', characters=len(text), semantics='NOT_VALIDATED')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inventory', required=True)
    p.add_argument('--download-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-hold', action='store_true')
    args = p.parse_args()
    if platform.system() != 'Linux' or not ROOT.is_dir():
        raise RuntimeError('Server only')
    out, download = Path(args.output).resolve(), Path(args.download_root).resolve()
    if ROOT not in out.parents or ROOT not in download.parents:
        raise ValueError('Outside server project')
    out.mkdir(parents=True, exist_ok=args.resume)
    expanded = out / 'expanded_archives'
    expanded.mkdir(exist_ok=args.resume)
    with open(args.inventory) as f:
        rows = [r for r in csv.DictReader(f) if r['restricted'] == 'False']
    done = set()
    counts = Counter()
    failures = 0
    if args.resume and (out / 'format_qc.jsonl').exists():
        latest = {}
        with (out / 'format_qc.jsonl').open() as prior:
            for line in prior:
                result = json.loads(line)
                latest[result['relative_path']] = result
        for result in latest.values():
            if args.retry_hold and result['status'] == 'HOLD':
                continue
            done.add(result['relative_path'])
            counts[result.get('format', 'FAILED_FORMAT')] += 1
            failures += result['status'] == 'HOLD'
    status = dict(task='PUB-01-S06', files_checked=len(done), expected_files=len(rows), state='RUNNING')
    with (out / 'format_qc.jsonl').open('a') as log:
        while len(done) < len(rows):
            progressed = False
            for row in rows:
                rel = row['relative_path']
                path = download / 'raw' / rel
                if rel in done or not path.is_file() or path.stat().st_size != int(row['bytes']):
                    continue
                try:
                    info = inspect(path, expanded)
                    state = info.pop('status', 'PASS_FORMAT_READABILITY')
                except Exception as exc:
                    info = dict(error=type(exc).__name__ + ': ' + str(exc))
                    state = 'HOLD'
                result = dict(relative_path=rel, expected_bytes=int(row['bytes']), status=state, **info)
                log.write(json.dumps(result, ensure_ascii=False) + '\n')
                log.flush()
                done.add(rel)
                counts[info.get('format', 'FAILED_FORMAT')] += 1
                failures += state == 'HOLD'
                progressed = True
                status = dict(task='PUB-01-S06', files_checked=len(done), expected_files=len(rows),
                              hold_files=failures, format_counts=dict(counts),
                              updated_utc=datetime.now(timezone.utc).isoformat(), state='RUNNING',
                              scope='Format readability/QC, not verified time pairing or full cleaning')
                tmp = out / 'status.json.tmp'
                tmp.write_text(json.dumps(status, indent=2))
                tmp.replace(out / 'status.json')
                print(len(done), state, rel, flush=True)
            dstate = json.loads((download / 'download_status.json').read_text())
            try:
                os.kill(int(dstate['pid']), 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if not progressed and (dstate['state'] != 'RUNNING' or not alive):
                status['download_process_alive'] = alive
                break
            if not progressed:
                time.sleep(30)
        status['state'] = 'COMPLETE_FORMAT_PASS' if len(done) == len(rows) and not failures else 'COMPLETE_WITH_HOLD_OR_MISSING'
        (out / 'status.json').write_text(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
