"""D24 independent CPU audit of actual lane statistics and twelve saved fits.

No EEG or target arrays, new model forwards, fitting, or new evaluation split.
The normalizer reference weights each four-point mean by its exact multiplicity
in the 96 past bins, independently of the production gather/chunk implementation.
"""
import argparse
import collections
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_source_family_crossfit_20261106')
D19 = ROOT.parent / 's03_physical_stream_20261106'
OUT = ROOT / 'coordinator_downstream_audit_v01'
GATE = ROOT / 'DOWNSTREAM_FORMAL_GATE_V05.json'
WORKER_SHA = 'afffe133a001cfaa3553d62cd271fd622e9c8af5be53dfb69d3d28a93ca96f41'
ASSIGN_SHA = '7ab57f1818a4aef312c43342768893ae8a49ff39407fa8bd76812ef2c81e3d24'
SEEDS = (1024, 2026, 31415)
ARMS = ('B_IS', 'S_IS', 'B_OOF', 'S_OOF')
ROLE = 'P_past_prediction'
NTRAIN, NVAL, STEPS = 23703510, 2866799, 1447


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def near(x, y, tolerance=1e-12):
    require(math.isfinite(float(x)) and math.isfinite(float(y)), 'Nonfinite number')
    error = abs(float(x)-float(y))
    require(error <= tolerance, f'Numeric mismatch: {x} versus {y}')
    return error


def weighted_moments(wave):
    n = len(wave)
    require(n > 1344 and np.isfinite(wave).all(), 'Finite physical record')
    # Direct four-term averages, rather than the production cumulative sum.
    prior = ((wave[:-3]+wave[1:-2]+wave[2:-1]+wave[3:])/4).astype(np.float32)
    delta = np.zeros(n+1, np.int64)
    starts = 1344-381+4*np.arange(96)
    ends = n-381+4*np.arange(96)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends, -1)
    weights = np.cumsum(delta)[:-1]
    require(weights.min() == 0 and weights.max() <= 96, 'History multiplicity')
    count = int(weights.sum())
    require(count == (n-1344)*96, 'All active points and past bins counted once')
    v = prior.astype(np.float64)
    w = weights[3:]
    return float(v@w), float((v*v)@w), count


def finish(sums, squares, count):
    mean = sums/count
    std = math.sqrt(max(squares/count-mean*mean, 0.))
    constant = std < 1e-8
    return dict(mean=float(np.float32(mean)), std=float(np.float32(1. if constant else std)), constant=constant)


def lane_statistics(seed, lane, manifest, gate, counters):
    sums = squares = 0.
    count = physical = 0
    ids = []
    for fold in range(3):
        folder = ROOT / f'formal/seed{seed}_fold{fold}'
        path = folder/f'{lane}_ANCHOR.npy'
        pin = gate['upstream_payloads'][str(path)]
        require(path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'Actual anchor payload changed')
        rows = [r for r in manifest if r[f'{lane}_model_fold'] == fold]
        entries = read(folder/f'{lane}_ANCHOR.json')
        require(len(rows) == len(entries), 'Lane record count')
        wave = np.load(path, mmap_mode='r', allow_pickle=False)
        offset = 0
        for row, entry in zip(rows, entries):
            require(entry == dict(record=row['record'], offset=offset, stop=row['covered_stop'], windows=row['windows']), 'Lane metadata coordinates')
            stop = entry['stop']
            a, b, c = weighted_moments(wave[offset:offset+stop])
            sums += a
            squares += b
            count += c
            physical += stop
            offset += stop
            ids.append(row['record'])
        require(offset == len(wave), 'Lane packed array support')
    require(len(ids) == len(set(ids)) == 426 and set(ids) == {r['record'] for r in manifest}, 'All and only frozen TRAIN records')
    require(count == NTRAIN*96, 'Complete lane normalization exposure')
    counters['TRAIN_anchor_points_read'] += physical
    counters['context_values_represented'] += count
    expected = finish(sums, squares, count)
    saved = read(ROOT/f'downstream_formal/seed{seed}/{lane}_TRAIN_STATS.json')
    require(saved['points'] == NTRAIN and saved['context_values'] == count and saved['constant'] == expected['constant'], 'Saved statistics boundary')
    errors = {k:near(saved[k], expected[k], 2e-7) for k in ('mean', 'std')}
    require(not expected['constant'], 'Nonconstant lane')
    for variant in ('B', 'S'):
        require(read(ROOT/f'downstream_formal/seed{seed}/{variant}_{lane}/seed{seed}_TRAIN_STATS.json') == saved, 'Shared B/S lane statistics')
    return dict(passed=True, lane=lane, reference=expected, saved=saved, absolute_errors=errors, TRAIN_records=426, physical_points=physical)


def aggregate(metric, mask):
    require(len(metric['blocks']) == len(mask) == 752 and len(metric['people']) == 49, 'Full VAL support')
    grouped = collections.defaultdict(list)
    seen = set()
    error = 0.
    for block in metric['blocks']:
        key = block['record'], block['start']
        require(key in mask and key not in seen, 'VAL block identity')
        seen.add(key)
        ref = mask[key]
        require(all(block[k] == ref[k] for k in ('subject','source_family','samples')), 'VAL block coordinates')
        require(math.isfinite(block['r']) and abs(block['r']) <= 1.0000001 and math.isfinite(block['MSE']) and block['MSE'] >= 0, 'Finite block metrics')
        grouped[block['subject']].append(block)
    people = {p['subject']:p for p in metric['people']}
    require(len(people) == 49 and set(people) == set(grouped), 'Person macro denominator')
    for subject, blocks in grouped.items():
        require(people[subject]['blocks'] == len(blocks), 'Person block count')
        for field in ('r','MSE'):
            error = max(error, near(people[subject][field], sum(b[field] for b in blocks)/len(blocks)))
    for field in ('r','MSE'):
        error = max(error, near(metric[field], sum(p[field] for p in people.values())/49))
    require(metric['active_points'] == NVAL, 'Physical VAL denominator')
    return error


def state_sha(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def audit_fit(seed, arm, mask, completed, gate_sha, core):
    folder = ROOT/f'downstream_formal/seed{seed}/{arm}/seed{seed}_{ROLE}'
    result, cfg = read(folder/'RESULT.json'), read(folder/'CONFIG.json')
    require(result['seed'] == cfg['seed'] == seed and result['role'] == cfg['role'] == ROLE, 'Fit identity')
    require(cfg['protocol_sha256'] == gate_sha and cfg['batch'] == 16384 and cfg['epochs'] == 20, 'Frozen fit budget')
    require(cfg['optimizer'] == 'Adam' and cfg['lr'] == .001 and cfg['weight_decay'] == 0 and cfg['betas'] == [.9,.999] and cfg['eps'] == 1e-8, 'Frozen optimizer')
    require(cfg['shuffle'] == 'global_without_replacement_int32' and cfg['shuffle_seed_recipe'] == [seed,20261106,'epoch'], 'Frozen order')
    epochs = result['epochs']
    require([x['epoch'] for x in epochs] == list(range(-1,20)), 'Complete trajectory')
    require([x['epoch'] for x in epochs if 'train' in x] == [-1,0,4,9,19], 'Fixed TRAIN checks')
    best_epoch, best_r = -1, epochs[0]['val']['r']
    for row in epochs:
        require(math.isfinite(row['val']['r']) and row['val']['active_points'] == NVAL, 'Epoch VAL')
        if row['epoch'] >= 0:
            require(math.isfinite(row['online_TRAIN_MSE']) and row['online_TRAIN_MSE'] >= 0, 'Actual epoch loss')
        if 'train' in row:
            require(row['train']['active_points'] == NTRAIN, 'Epoch TRAIN support')
        if row['val']['r'] > best_r:
            best_epoch, best_r = row['epoch'], row['val']['r']
    require(result['selected_epoch'] == best_epoch, 'Earliest strict best with anchor fallback')
    near(result['selected']['r'], best_r)
    maximum = aggregate(result['selected'], mask)
    pair = result['val_pair']
    require(set(pair) == {'near','older'}, 'Recorded history intervention')
    for value in pair.values():
        maximum = max(maximum, aggregate(value, mask))
    near(pair['near']['r'], best_r)
    if arm.startswith('S_'):
        require(pair['near'] == pair['older'], 'Memoryless intervention invariance')
    for epoch in range(20):
        first = read(folder/f'epoch{epoch:02}_FIRST_BATCH.json')
        require(first['epoch'] == epoch and first['active_points'] == 16384 and len(first['gradients']) == 10, 'Actual first batch')
        require(all(math.isfinite(x) and x >= 0 for x in first['gradients'].values()), 'Finite actual gradients')
        if epoch == 0:
            require(first['residual_RMS'] == 0 and first['gradients']['output.weight'] > 0, 'Zero residual initial condition')
            require(all(x == 0 for key,x in first['gradients'].items() if not key.startswith('output.')), 'Initial gated gradients')
    recovery = folder/'recovery'
    require(sorted(p.name for p in recovery.glob('epoch*.pt')) == [f'epoch{x:02}.pt' for x in range(20)], 'Twenty recovery files')
    last = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
    expected = 9441 if arm.startswith('B_') else 3361
    require(len(last) == 10 and sum(v.numel() for v in last.values()) == cfg['trainable_parameters'] == expected, 'Actual active parameters')
    require(all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in last.values()), 'Finite saved weights')
    original = core.PhysicalDecoder(seed, ROLE).state_dict()
    original['current.weight'] = original['current.weight'][:,:1]*math.sqrt(8)
    if arm.startswith('S_'):
        original['context_in.weight'] = original['context_in.weight'][:,:1]*math.sqrt(96)
    require(state_sha(original) == cfg['initialization_sha256'] != state_sha(last), 'Actual initialization and weight update')
    checkpoint = torch.load(recovery/'epoch19.pt', map_location='cpu', weights_only=True)
    require(checkpoint['epoch'] == 19 and checkpoint['next_epoch'] == 20 and checkpoint['seed'] == seed and checkpoint['role'] == ROLE, 'Terminal optimizer checkpoint')
    require(checkpoint['best_epoch'] == best_epoch and checkpoint['scores'] == epochs, 'Saved trajectory matches recovery')
    require(all(torch.equal(v,checkpoint['model'][k]) for k,v in last.items()), 'Final parameters match recovery')
    require(len(checkpoint['optimizer']['state']) == 10 and all(int(s['step']) == 20*STEPS for s in checkpoint['optimizer']['state'].values()), 'Actual Adam update count')
    require(checkpoint['numpy_bit_generator']['bit_generator'] == 'PCG64' and checkpoint['torch_rng'].numel() > 0 and checkpoint['torch_cuda_rng'].numel() > 0, 'Saved generator states')
    stage = checkpoint['stage']
    require(stage['protocol_sha256'] == gate_sha and stage['process'] == completed['process'], 'Actual process and gate identity')
    ordinal = ARMS.index(arm)+1
    require(stage['counters']['optimizer_updates'] == ordinal*20*STEPS and stage['counters']['backward_points'] == ordinal*20*NTRAIN, 'Actual cumulative physical budget')
    require(read(folder/'LAST.json')['epoch'] == 19, 'Last pointer')
    if best_epoch >= 0:
        require(read(folder/'BEST.json')['epoch'] == best_epoch, 'Selected pointer')
        selected = torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)
        chosen = torch.load(recovery/f'epoch{best_epoch:02}.pt', map_location='cpu', weights_only=True)
        require(all(torch.equal(v,chosen['model'][k]) for k,v in selected.items()), 'Selected parameters match actual trained checkpoint')
    else:
        require(not (folder/'selected.pt').exists() and not (folder/'BEST.json').exists(), 'Anchor fallback carries no invented model')
    require(completed['results'][arm]['epoch'] == best_epoch, 'Terminal result selection')
    near(completed['results'][arm]['r'], best_r)
    return dict(passed=True, arm=arm, selected_epoch=best_epoch, selected_r=best_r, last_r=epochs[-1]['val']['r'], parameters=expected, maximum_saved_aggregate_error=maximum, result_sha256=sha(folder/'RESULT.json'), last_weights_sha256=sha(folder/'last.pt'), selected_weights_sha256=sha(folder/'selected.pt') if best_epoch >= 0 else None, recovery_files_named=20, recovery_files_loaded=2 if best_epoch not in (-1,19) else 1)


def main(wait):
    wall, cpu = time.time(), time.process_time()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only inspection')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    require(sha(ROOT/'TRAIN_RECORD_ASSIGNMENT.json') == ASSIGN_SHA, 'Frozen record assignment')
    require(sha(ROOT/'input_downstream_v05/pub_01_s03_ss07_crossfit_downstream.py') == WORKER_SHA, 'Frozen downstream worker')
    manifest = read(ROOT/'TRAIN_RECORD_ASSIGNMENT.json')['records']
    metadata = read(D19/'PREPARATION_METADATA.json')
    require(len(metadata['phases']['train']['records']) == 426 and metadata['phases']['train']['active_points'] == NTRAIN, 'Frozen data scope')
    records = {r['record']:r for r in metadata['phases']['val']['records']}
    mask = {(b['record'],b['start']):dict(b, subject=records[b['record']]['subject'], source_family=records[b['record']]['source_family']) for b in metadata['phases']['val']['blocks']}
    sys.path.insert(0,str(D19/'input'))
    core = importlib.import_module('pub_01_s03_ss07_physical_stream_core')
    OUT.mkdir()
    identity = dict(pid=os.getpid(), uid=os.getuid(), startticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]), source_sha256=sha(Path(__file__)), deadline_unix=wall+24*3600)
    write(OUT/'PROCESS.json', identity)
    counters = dict(TRAIN_anchor_points_read=0, context_values_represented=0, neural_forwards=0, label_arrays_read=0, fitting_updates=0)
    done = {}
    try:
        while len(done) < 3:
            require(time.time() <= identity['deadline_unix'], 'Audit observation deadline; never restart training')
            failure = ROOT/'downstream_queue_v02/FAILURE.json'
            require(not failure.exists(), 'Downstream controller failure')
            for seed in SEEDS:
                folder = ROOT/f'downstream_formal/seed{seed}'
                require(not (folder/'FAILURE.json').exists(), f'Downstream seed {seed} failed')
                if seed in done or not (folder/'COMPLETED.json').exists():
                    continue
                gate = read(GATE)
                require(gate['passed'], 'Actual admitted downstream gate')
                for name,digest in gate['sources'].items():
                    require(sha(Path(name)) == digest, 'Frozen source or evidence drift: '+name)
                completed = read(folder/'COMPLETED.json')
                require(completed['passed'] and completed['seed'] == seed and completed['protocol_sha256'] == sha(GATE), 'Actual seed terminal')
                counts = completed['counters']
                require(counts['formal_fits_completed'] == 4 and counts['optimizer_updates'] == 115760 and counts['backward_points'] == 1896280800, 'Complete four-arm seed budget')
                statistics = [lane_statistics(seed,lane,manifest,gate,counters) for lane in ('IS','OOF')]
                fits = [audit_fit(seed,arm,mask,completed,sha(GATE),core) for arm in ARMS]
                chosen = sum(x['selected_epoch'] >= 0 for x in fits)
                common = 4*(24*NTRAIN+20*NVAL)
                require(counts['current_stem_points'] == common+chosen*NVAL and counts['context_output_points'] == common+2*chosen*NVAL, 'Actual forward cost includes selected history pair')
                result = dict(passed=True, seed=seed, statistics=statistics, fits=fits, terminal_sha256=sha(folder/'COMPLETED.json'), formal_gate_sha256=sha(GATE), actual_training_statistics_files=6)
                write(OUT/f'seed{seed}.json', result)
                done[seed] = result
                print(json.dumps(dict(audited_seeds=len(done), seed=seed, counters=counters)), flush=True)
            status = dict(process=identity, completed_seeds=list(done), counters=counters)
            temp = OUT/'STATUS.writing'
            temp.write_text(json.dumps(status,indent=2))
            temp.replace(OUT/'STATUS.json')
            if len(done) < 3:
                require(wait, 'Incomplete formal matrix; use --wait, never duplicate a live auditor')
                time.sleep(30)
        write(OUT/'COMPLETED.json',dict(passed=True, process=identity, seeds=list(done), fits=12, counters=counters, CPU_seconds=time.process_time()-cpu, wall_seconds_including_wait=time.time()-wall, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, scope='Independent complete lane TRAIN statistics, saved scoring aggregation/selection, actual last/selected weights and optimizer states. No new labels, model forward or scientific claim.'))
    except BaseException as error:
        write(OUT/'FAILURE.json',dict(process=identity, message=str(error), traceback=traceback.format_exc(), counters=counters))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
