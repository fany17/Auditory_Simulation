"""Independently verify every cached TRAIN/VAL target against record coordinates."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

BASE = Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT = BASE / 's03_full_ft_history_20261102'
started, cpu = time.time(), time.process_time()
protocol = json.loads((BASE / 's03_finetuned_readout_transfer_20261101/PROTOCOL.json').read_text())
checks = []
for phase in ('train', 'val'):
    cached_path = BASE / f's03_strong_baselines_20261015/transformer_available01/envelope_readout_seed1024/{phase}_envelope.npy'
    cached = np.load(cached_path, mmap_mode='r')
    windows, elements, unequal, maximum = 0, 0, 0, 0.
    for row in protocol['records'][phase]:
        target = np.load(row['envelope_path'], mmap_mode='r').reshape(-1)
        n, offset, count = row['common_samples'], row['offset'], row['windows']
        assert count == (n - 192) // 96 + 1
        view = np.lib.stride_tricks.sliding_window_view(target[:n], 192)[::96]
        assert len(view) == count
        for first in range(0, count, 256):
            last = min(first + 256, count)
            expected = view[first:last].astype('f4')
            actual = np.asarray(cached[offset + first:offset + last], dtype='f4').reshape(last - first, 192)
            assert np.isfinite(actual).all() and np.isfinite(expected).all()
            unequal += int(np.count_nonzero(actual != expected))
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
            elements += actual.size
        windows += count
    assert windows == len(cached)
    checks.append(dict(phase=phase, records=len(protocol['records'][phase]), windows=windows,
        target_elements=elements, unequal_float32_elements=unequal, maximum_abs_difference=maximum,
        record_order_and_window_coordinate_passed=unequal == 0))
result = dict(passed=all(c['record_order_and_window_coordinate_passed'] for c in checks), checks=checks,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), NN_forwards=0, optimizer_updates=0,
    external_access=False, wall_seconds=time.time() - started, CPU_seconds=time.process_time() - cpu)
with (ROOT / 'COORDINATOR_TARGET_MAPPING.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
assert result['passed']
