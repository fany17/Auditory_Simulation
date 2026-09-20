"""Independent BDF int24/calibration check against MNE, server2203 only.

Preserves the original gzip. A diagnostic two-record BDF excerpt is retained
on the server; only its record-count header is changed to describe the excerpt.
This checks numerical interpretation, not whole-dataset quality or alignment.
"""
import gzip
import json
from pathlib import Path
import numpy as np
import mne

ROOT = Path('/home/fanyu/auditory_simulation_m6a')


def main():
    root = ROOT / 'pub_01/s06'
    latest = {}
    for line in (root / 'sparrkulee_format_qc_20260919_v1/format_qc.jsonl').read_text().splitlines():
        entry = json.loads(line)
        latest[entry['relative_path']] = entry
    source = next(x for x in latest.values() if x.get('format') == 'BDF_GZIP')
    path = root / 'sparrkulee_v3.1/raw' / source['relative_path']
    out = root / 'review_bdf_calibration_20260920_v1'
    out.mkdir(exist_ok=False)
    with gzip.open(path, 'rb') as f:
        base = bytearray(f.read(256))
        channels = int(base[252:256])
        header_size = int(base[184:192])
        signal_header = f.read(header_size - 256)
        def fields(offset):
            return np.array([float(signal_header[offset * channels + i * 8:offset * channels + (i + 1) * 8]) for i in range(channels)])
        sizes = fields(216).astype(int)
        records = min(int(base[236:244]), 2)
        assert records > 0 and channels >= 64 and (sizes[:64] == sizes[0]).all()
        block_size = int(sizes.sum()) * 3
        payload = f.read(records * block_size)
        assert len(payload) == records * block_size
    base[236:244] = str(records).encode().ljust(8)
    excerpt = out / 'two_record_diagnostic.bdf'
    with excerpt.open('xb') as f:
        f.write(base + signal_header + payload)
    units = [signal_header[96 * channels + i * 8:96 * channels + (i + 1) * 8].decode().strip() for i in range(64)]
    assert set(units) == {'uV'}, units
    physical_min, physical_max = fields(104), fields(112)
    digital_min, digital_max = fields(120), fields(128)
    pieces = []
    for record in range(records):
        packed = np.frombuffer(payload[record * block_size:(record + 1) * block_size], np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] + 256 * packed[:, 1] + 65536 * packed[:, 2]
        values = np.where(values >= 8388608, values - 16777216, values)
        pieces.append(np.stack(np.split(values, np.cumsum(sizes)[:-1])[:64]))
    digital = np.concatenate(pieces, axis=1)
    gain = (physical_max[:64] - physical_min[:64]) / (digital_max[:64] - digital_min[:64])
    expected_uv = (digital - digital_min[:64, None]) * gain[:, None] + physical_min[:64, None]
    with mne.io.read_raw_bdf(str(excerpt), preload=True, verbose='ERROR') as raw:
        observed_uv = raw.get_data(picks=list(range(64))) * 1e6
    difference = float(np.max(np.abs(expected_uv - observed_uv)))
    result = dict(source_server_path=str(path), diagnostic_excerpt_server_path=str(excerpt),
                  records=records, channels_compared=64, samples_per_channel=observed_uv.shape[1],
                  compared_unit='microvolt', max_absolute_difference=difference,
                  absolute_tolerance_microvolt=1e-8,
                  negative_digital_values=int((digital < 0).sum()), positive_digital_values=int((digital > 0).sum()),
                  status='PASS' if difference <= 1e-8 else 'FAIL', original_preserved=True,
                  scope='Independent numerical decoder/calibration sample, not full raw QC or physical synchronization')
    (out / 'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    assert result['status'] == 'PASS', result


if __name__ == '__main__':
    main()
