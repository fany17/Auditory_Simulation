"""Bounded D26 raw-EEG final-model audit. CPU on server2203, no fitting."""
import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

WORKER_STARTED, WORKER_CPU = time.monotonic(), time.process_time()

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_contrastive_history_20261106')
INPUT = ROOT / 'input_v05'
OUT = ROOT / 'coordinator_final_replay_v01'
GATE_SHA = '8c249eabeb9edb9a5360006f1d3dabb9552e2a22ff649b9bae72801f75311b4d'
SEEDS, ARMS = (1024, 2026, 31415), ('HH', 'CH', 'CC')
COUNTS = dict(backbone_windows=0, native_unused_projection_windows=0,
              final_projection_windows=0, head_windows=0,
              manual_memory_query_windows=0, library_memory_query_windows=0)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def prefix(backbone, x):
    captured = []
    handle = backbone.output.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        COUNTS['backbone_windows'] += len(x)
        COUNTS['native_unused_projection_windows'] += len(x)
        backbone(x)
    finally:
        handle.remove()
    assert len(captured) == 1 and captured[0].shape == (len(x), 192, 64)
    return captured[0]


def memory_delta(memory, query, past):
    assert query.shape[1:] == (192, 64) and past.shape == (len(query), 384, 64)
    n = len(query)
    qn = F.layer_norm(query, (64,), memory.query_norm.weight, memory.query_norm.bias, memory.query_norm.eps)
    vn = F.layer_norm(past, (64,), memory.memory_norm.weight, memory.memory_norm.bias, memory.memory_norm.eps)
    q, k, v = qn + memory.query_position, vn + memory.memory_position, vn
    attention = memory.attention
    assert attention.num_heads == 8 and attention.dropout == 0. and attention.embed_dim == 64
    projected = []
    for j, value in enumerate((q, k, v)):
        value = F.linear(value, attention.in_proj_weight[j*64:(j+1)*64], attention.in_proj_bias[j*64:(j+1)*64])
        projected.append(value.reshape(n, value.shape[1], 8, 8).transpose(1, 2))
    q, k, v = projected
    COUNTS['manual_memory_query_windows'] += n
    probability = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(8), dim=-1)
    message = (probability @ v).transpose(1, 2).contiguous().reshape(n, 192, 64)
    manual = F.linear(message, attention.out_proj.weight, attention.out_proj.bias)
    COUNTS['library_memory_query_windows'] += n
    reference = memory(query, past)
    error = float((manual - reference).abs().max())
    assert error <= 2e-5, ('memory bridge', error)
    return manual, error


def correct(z, prediction, fit):
    assert z.shape == (40, 8, 192) and prediction.shape == (40, 192)
    a, b = float(fit['a']), float(fit['b'])
    assert a > 0 and fit['coef'].shape == (3264, 192) and fit['bias'].shape == (192,)
    base = a * prediction.astype(np.float64) + b
    def norm(value, index):
        mean, scale, constant = (fit[f'{key}_{index}'] for key in ('mean', 'std', 'constant'))
        assert np.all(scale > 0)
        answer = (value - mean) / scale
        answer[:, constant.astype(bool)] = 0.
        return answer
    u = norm(z[4:].astype(np.float64).reshape(36, 1536), 0)
    v, w = norm(base[4:], 1), norm(u * u, 2)
    features = np.concatenate((u / math.sqrt(1536), v / math.sqrt(192), w / math.sqrt(1536)), axis=1)
    result = base.copy()
    result[4:] += features @ fit['coef'] + fit['bias']
    assert np.array_equal(result[:4], base[:4])
    return base, result


def metrics(windows, target, active=False):
    assert windows.shape == (40, 192) and target.shape == (3840,)
    wave, weight = np.zeros(3936), np.zeros(3936)
    hann = .5 - .5 * np.cos(2 * np.pi * np.arange(192) / 191)
    for i in range(40):
        s = i * 96
        wave[s:s+192] += windows[i] * hann
        weight[s:s+192] += hann
    keep = weight[:3840] > 1e-8
    if active:
        keep[:960] = False
    p, y = wave[:3840][keep] / weight[:3840][keep], target[keep]
    assert len(p) == (2880 if active else 3839) and np.ptp(p) > 0 and np.ptp(y) > 0
    pc, yc = p - p.mean(), y - y.mean()
    return dict(r=float(pc @ yc / np.sqrt((pc @ pc) * (yc @ yc))),
                MSE=float(np.mean((p - y)**2)), samples=len(p))


def run(seed, arm):
    started, cpu = time.monotonic(), time.process_time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    assert len(os.sched_getaffinity(0)) <= 4
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(seed)
    gate_path = ROOT / 'PROTOCOL_V05.json'
    assert sha(gate_path) == GATE_SHA
    gate, inv = read(gate_path), read(ROOT / 'INPUTS.json')
    assert sha(ROOT / 'INPUTS.json') == gate['inventory_sha256']
    assert all(sha(p) == v for p, v in gate['sources'].items())
    sys.path.insert(0, str(INPUT))
    import pub_01_s03_ss07_contrastive_history_model as model
    import pub_01_s03_ss07_contrastive_history_data as data
    row = inv['phases']['val'][0]
    assert row['phase'] == 'val' and row['offset'] == 0 and row['common_samples'] >= 3936
    mask = read(data.MASK)
    first_mask = next(b for b in mask['blocks'] if b['record'] == row['record'] and b['start'] == 0)
    assert first_mask['eligible'] and first_mask['samples'] == 3839
    seed_root = ROOT / 'formal' / f'seed{seed}'
    readout = seed_root / f'READOUT_{arm}'
    receipt = read(readout / 'COMPLETED.json')
    checkpoint = seed_root / f'FT_{arm}' / 'FINAL.pt'
    ft = read(checkpoint.parent / 'COMPLETED.json')
    assert receipt['passed'] and receipt['seed'] == seed and receipt['arm'] == arm
    assert receipt['epoch'] == ft['final_epoch'] == 4 and ft['updates'] == 19725 and receipt['lambda_fixed'] == .01
    assert receipt['old_test_reads'] == 0 and sha(checkpoint) == receipt['FT_checkpoint_sha256']
    norm_path = data.PREPARED / 'train_only_normalization.npz'
    for p in (row['eeg_path'], row['envelope_path'], str(norm_path)):
        s = Path(p).stat()
        assert inv['inputs'][p] == dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns)
    with np.load(norm_path, allow_pickle=False) as f:
        mean, scale = f['mean'], f['std']
    eeg = np.load(row['eeg_path'], mmap_mode='r', allow_pickle=False)
    raw = eeg.T if eeg.shape[0] == 64 else eeg
    normalized = ((np.asarray(raw[:3936], np.float32) - mean) / scale).astype(np.float32)
    target = np.asarray(np.load(row['envelope_path'], mmap_mode='r', allow_pickle=False).reshape(-1)[:3840], np.float64)
    assert normalized.shape == (3936, 64) and np.isfinite(normalized).all() and np.isfinite(target).all()
    starts = np.arange(40) * 96
    current = torch.from_numpy(np.stack([normalized[s:s+192] for s in starts]))
    valid = starts >= 864
    assert int(valid.sum()) == 31
    with np.load(readout / 'READOUT.npz', allow_pickle=False) as f:
        fit = {k: f[k] for k in f.files}
    backbone = model.native.TransformerCLIP(read(data.NATIVE_CONFIG)).eeg
    net = model.Decoder(model.Encoder(backbone, seed), model.make_head(seed), arm[1])
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    net.eval()
    parameter_count = sum(p.numel() for p in net.parameters())
    assert parameter_count == 3389513
    comparisons, bridges, score_hashes = [], {}, {}
    names = ('near', 'far', 'copy') if arm == 'HH' else ('native',)
    with torch.inference_mode():
        query = prefix(net.encoder.backbone, current)
        query_before = query.clone()
        for name in names:
            route = 'C' if name == 'copy' else arm[1]
            if route == 'C':
                slots = (current[valid], current[valid])
            else:
                offsets = (-864, -672) if name == 'far' else (-480, -288)
                slots = tuple(torch.from_numpy(np.stack([normalized[s+d:s+d+192] for s in starts[valid]])) for d in offsets)
            encoded = torch.cat([prefix(net.encoder.backbone, x) for x in slots], dim=1)
            delta, bridges[name] = memory_delta(net.encoder.memory, query[valid], encoded)
            fused = query.clone()
            fused[valid] += delta
            assert torch.equal(fused[~valid], query[~valid]) and torch.equal(query, query_before)
            COUNTS['final_projection_windows'] += 40
            z = net.encoder.backbone.output(fused).transpose(1, 2)
            COUNTS['head_windows'] += 40
            prediction = net.head(z)[:, 0].numpy()
            base, calibrated = correct(z.numpy(), prediction, fit)
            saved_path = readout / f'SCORE_{name}.json'
            saved = read(saved_path)
            score_hashes[name] = sha(saved_path)
            alternatives = {'Ccap': calibrated}
            if name in ('native', 'near'):
                alternatives.update(raw=prediction.astype(np.float64), affine=base)
            for kind, windows in alternatives.items():
                value = metrics(windows, target)
                expected = [b for b in saved[kind]['segments'] if b['record'] == row['record'] and b['start'] == 0]
                assert len(expected) == 1 and expected[0]['samples'] == value['samples']
                errors = {k: abs(value[k] - expected[0][k]) for k in ('r', 'MSE')}
                assert max(errors.values()) <= 1e-4, (seed, arm, name, kind, errors)
                comparisons.append(dict(path=name, output=kind, support='first_original_block', observed=value, errors=errors))
            if arm == 'HH':
                value = metrics(calibrated, target, active=True)
                expected = [b for b in saved['Ccap']['pure_memory']['records'] if b['record'] == row['record'] and b['start'] == 0]
                assert len(expected) == 1 and expected[0]['retained_samples'] == value['samples']
                errors = {k: abs(value[k] - expected[0][k]) for k in ('r', 'MSE')}
                assert max(errors.values()) <= 1e-4, (seed, arm, name, 'pure_memory', errors)
                comparisons.append(dict(path=name, output='Ccap', support='physical_960_to_3839', observed=value, errors=errors))
    expected_calls = dict(backbone_windows=226 if arm == 'HH' else 102,
                          native_unused_projection_windows=226 if arm == 'HH' else 102,
                          final_projection_windows=120 if arm == 'HH' else 40,
                          head_windows=120 if arm == 'HH' else 40,
                          manual_memory_query_windows=93 if arm == 'HH' else 31,
                          library_memory_query_windows=93 if arm == 'HH' else 31)
    assert COUNTS == expected_calls
    assert len(comparisons) == (8 if arm == 'HH' else 3)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert peak < 4 * 1024**3 and not torch.cuda.is_initialized()
    result = dict(passed=True, seed=seed, arm=arm, source_sha256=sha(__file__), protocol_sha256=GATE_SHA,
                  record=row['record'], EEG_samples_read=3936, target_samples_read=3840,
                  parameter_count=parameter_count, normalization_sha256=sha(norm_path),
                  normalized_EEG_prefix_sha256=hashlib.sha256(normalized.tobytes()).hexdigest(),
                  target_prefix_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                  FT_checkpoint_sha256=sha(checkpoint), readout_sha256=sha(readout / 'READOUT.npz'),
                  score_sha256=score_hashes, memory_bridge_max_abs=bridges, comparisons=comparisons,
                  actual_calls=COUNTS, CPU_seconds=time.process_time()-WORKER_CPU,
                  wall_seconds=time.monotonic()-WORKER_STARTED,
                  replay_CPU_seconds=time.process_time()-cpu, replay_wall_seconds=time.monotonic()-started,
                  peak_RSS_bytes=peak, threadpools=threadpool_info(), CPU_affinity=sorted(os.sched_getaffinity(0)),
                  new_fits=0, optimizer_updates=0, audio_forwards=0, old_test_reads=0, GPU=0,
                  scope='First fixed VAL record, first 60-second block only; raw EEG and final model replay. Not full-cohort signal replay or a new independent test.')
    publish(OUT / f'REPLAY_{seed}_{arm}.json', result)
    print(json.dumps({k: result[k] for k in ('passed', 'seed', 'arm', 'actual_calls', 'CPU_seconds', 'wall_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    assert not (OUT / f'REPLAY_{args.seed}_{args.arm}.json').exists()
    began, cpu = time.monotonic(), time.process_time()
    try:
        with threadpool_limits(limits=4):
            run(args.seed, args.arm)
    except BaseException:
        publish(OUT / f'FAILURE_{args.seed}_{args.arm}.json', dict(traceback=traceback.format_exc(),
                actual_calls=COUNTS, wall_seconds=time.monotonic()-WORKER_STARTED,
                CPU_seconds=time.process_time()-WORKER_CPU,
                new_fits=0, optimizer_updates=0))
        raise
