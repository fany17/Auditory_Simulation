"""Independent D24 replay audit V03. No labels, validation scores or fitting.

Check all fold normalization statistics and replay predetermined physical points
from raw EEG through saved FT weights and a separate NumPy Ccap/Hann calculation.
The exact global-64 prediction batches are reconstructed from each output manifest.
V03 waits for all upstream training before creating any CUDA context.
It matches formal CUDA cuDNN TF32 precision; normalization and independent Ccap/Hann
remain NumPy float64. V01 precision failure and V02 scheduling collision are preserved.
"""
import argparse
import bisect
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

ROOT = Path('/sata5/fanyu/auditory_simulation_m6a/pub_01/s03_source_family_crossfit_20261106')
OLD = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_strong_baselines_20261015/transformer_available01')
OUT = ROOT / 'coordinator_upstream_audit_v03'
GATE_SHA = '0ad69424fb76f827b268b63f7d45cbd954ae514c7f687d13a870e66e52322482'
ASSIGN_SHA = '7ab57f1818a4aef312c43342768893ae8a49ff39407fa8bd76812ef2c81e3d24'
ATOL, RTOL = 2e-5, 2e-5
NN_WINDOW_CAP = 16384


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def raw(row):
    p = Path(row['eeg_path']).resolve()
    require(p.is_relative_to(Path('/home/fanyu/auditory_simulation_m6a')), 'Raw path outside authorized root')
    a = np.load(p, mmap_mode='r', allow_pickle=False)
    if a.shape[0] == 64:
        a = a.T
    require(a.ndim == 2 and a.shape[1] == 64 and len(a) >= row['common_samples'], 'Raw support')
    return a[:row['common_samples']]


def norm_reference(rows):
    # Independent chunked accumulation; all fit samples, no OOF or labels.
    count = 0
    total = np.zeros(64, np.float64)
    squared = np.zeros_like(total)
    for row in rows:
        a = raw(row)
        for start in range(0, len(a), 8192):
            block = np.asarray(a[start:start+8192], np.float64)
            total += block.sum(axis=0)
            squared += (block*block).sum(axis=0)
            count += len(block)
    mean = total/count
    std = np.sqrt(np.maximum(squared/count-mean*mean, 1e-12))
    return mean, std, count


class Decoder(nn.Module):
    def __init__(self, native, cfg):
        super().__init__()
        original = native.TransformerCLIP(cfg)
        self.encoder = original.eeg
        self.head = nn.Sequential(nn.Conv1d(8, 1, 32, padding='same'), nn.LeakyReLU(.01))


def readout(z, p, arrays, affine, local_frames):
    base = affine['a']*p.astype(np.float64)+affine['b']
    def normalize(x, i):
        result = (x-arrays[f'mean_{i}'])/arrays[f'std_{i}']
        result[:, arrays[f'constant_{i}'].astype(bool)] = 0
        return result
    u = normalize(z.astype(np.float64).reshape(len(z), 1536), 0)
    v = normalize(base, 1)
    w = normalize(u*u, 2)
    x = np.concatenate([u/np.sqrt(1536), v/np.sqrt(192), w/np.sqrt(1536)], axis=1)
    prediction = base+x@arrays['coef']+arrays['bias']
    prediction[np.asarray(local_frames) < 4] = base[np.asarray(local_frames) < 4]
    return prediction


@torch.no_grad()
def replay_lane(folder, lane, rows, model, arrays, affine, counts):
    entries = read(folder/f'{lane}_ANCHOR.json')
    saved = np.load(folder/f'{lane}_ANCHOR.npy', mmap_mode='r', allow_pickle=False)
    require(len(entries) == len(rows) and saved.dtype == np.float64, 'Prediction manifest schema')
    offsets = [0]
    physical = 0
    for row, entry in zip(rows, entries):
        require(entry == dict(record=row['record'], offset=physical, stop=row['covered_stop'], windows=row['windows']), 'Prediction geometry')
        physical += row['covered_stop']
        offsets.append(offsets[-1]+row['windows'])
    require(physical == len(saved), 'Saved physical length')
    selected = sorted({0, len(rows)-1})
    required_batches, probes = set(), {}
    for ri in selected:
        stop = rows[ri]['covered_stop']
        points = np.unique(np.r_[np.arange(1, min(33, stop)), np.arange(1344, min(1408, stop)), np.arange(max(1344, stop-64), stop)])
        probes[ri] = points
        for t in points:
            first = max(0, (int(t)-190+95)//96)
            last = min(rows[ri]['windows']-1, (int(t)-1)//96)
            require(first <= last, 'No positive Hann contributor')
            required_batches.update((offsets[ri]+fi)//64 for fi in range(first, last+1))
    frames, mapped = {}, {}
    for batch in sorted(required_batches):
        first, stop = batch*64, min((batch+1)*64, offsets[-1])
        require(counts['NN_windows']+stop-first <= NN_WINDOW_CAP, 'Preregistered replay NN cap')
        data, local_ids = [], []
        for global_frame in range(first, stop):
            ri = bisect.bisect_right(offsets, global_frame)-1
            local_frame = global_frame-offsets[ri]
            if ri not in mapped:
                mapped[ri] = raw(rows[ri])
            start = local_frame*96
            x = ((np.asarray(mapped[ri][start:start+192], np.float32)-arrays['mean'])/arrays['std']).astype(np.float32)
            data.append(x)
            local_ids.append(local_frame)
        x = torch.from_numpy(np.stack(data)).to("cuda")
        z = model.encoder(x).transpose(1, 2)
        p = model.head(z)[:, 0]
        prediction = readout(z.cpu().numpy(), p.cpu().numpy(), arrays, affine, local_ids)
        require(np.isfinite(prediction).all(), 'Finite independent predictions')
        for i, global_frame in enumerate(range(first, stop)):
            frames[global_frame] = prediction[i]
        counts['NN_windows'] += stop-first
        counts['NN_batches'] += 1
    h = np.hanning(192)
    result = []
    for ri, points in probes.items():
        reference = []
        for t in points:
            first = max(0, (int(t)-190+95)//96)
            last = min(rows[ri]['windows']-1, (int(t)-1)//96)
            numerator = denominator = 0.
            for fi in range(first, last+1):
                j = int(t)-96*fi
                numerator += h[j]*frames[offsets[ri]+fi][j]
                denominator += h[j]
            reference.append(numerator/denominator)
        reference = np.asarray(reference)
        observed = np.asarray(saved[entries[ri]['offset']+points])
        error = np.abs(reference-observed)
        require(np.isfinite(observed).all() and np.all(error <= ATOL+RTOL*np.abs(observed)), 'Raw-FT-Ccap-Hann replay mismatch')
        result.append(dict(record=rows[ri]['record'], points=len(points), max_abs_error=float(error.max()), first_point=int(points[0]), last_point=int(points[-1])))
        counts['physical_points'] += len(points)
    return result


def inspect_job(job, manifest, norm_cache, native, cfg, counts):
    seed, fold = job['seed'], job['model_fold']
    folder = ROOT/f'formal/seed{seed}_fold{fold}'
    completed = read(folder/'COMPLETED.json')
    frozen = read(folder/'UPSTREAM_FROZEN.json')
    fit = [r for r in manifest if r['OOF_model_fold'] != fold]
    fit_ids = {r['record'] for r in fit}
    require(completed['seed'] == seed and completed['fold'] == fold and completed['stage'] == 'formal', 'Job identity')
    require(set(completed['fit_records']) == set(frozen['fit_records']) == fit_ids, 'Fold fit identity')
    require(completed['original_VAL_reads'] == completed['old_test_reads'] == completed['OOF_label_reads'] == 0, 'Reported phase boundary')
    require(set(completed['contrastive']['record_presentations']) <= fit_ids, 'Contrastive record boundary')
    require(completed['contrastive']['updates'] == job['contrastive']['updates'], 'Contrastive updates')
    for stage in ('head', 'FT'):
        require(len(completed[stage]) == job[stage]['epochs'], 'Full stage epochs')
        for key in ('updates', 'target_windows'):
            require(sum(x[key] for x in completed[stage]) == job[stage][key], 'Full stage budget')
    require(frozen['Ccap']['TRAIN_fit_windows'] == job['Ccap']['fit_windows'] and frozen['Ccap']['TRAIN_physical_points'] == job['affine_physical_points'], 'Ccap training support')
    if fold not in norm_cache:
        norm_cache[fold] = norm_reference(fit)
    mean, std, n = norm_cache[fold]
    with np.load(folder/'READOUT.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    require(n == completed['normalization_samples'] == job['fit_raw_channel_normalization_samples'], 'Normalization sample count')
    require(np.allclose(mean, arrays['mean'], rtol=1e-8, atol=1e-10) and np.allclose(std, arrays['std'], rtol=1e-8, atol=1e-10), 'Independent complete fold normalization')
    require(arrays['coef'].shape == (3264, 192) and arrays['bias'].shape == (192,) and all(np.isfinite(v).all() for v in arrays.values()), 'Saved readout layout')
    model = Decoder(native, cfg).eval().to("cuda")
    model.load_state_dict(torch.load(folder/'FT.pt', map_location='cuda', weights_only=True), strict=True)
    require(sum(x.numel() for x in model.parameters()) == 3372617, 'Actual saved FT parameters')
    replay = {}
    for lane in ('IS', 'OOF'):
        rows = [r for r in manifest if r[f'{lane}_model_fold'] == fold]
        require(all((r['record'] in fit_ids) == (lane == 'IS') for r in rows), 'Independent lane routing')
        replay[lane] = replay_lane(folder, lane, rows, model, arrays, frozen['affine'], counts)
    payloads = ['FT.pt','contrastive.pt','head.pt','READOUT.npz','IS_ANCHOR.npy','IS_ANCHOR.json','OOF_ANCHOR.npy','OOF_ANCHOR.json','COMPLETED.json','UPSTREAM_FROZEN.json']
    return dict(passed=True, seed=seed, fold=fold, fit_records=len(fit), full_normalization_points=n,
                normalization_max_abs=dict(mean=float(np.abs(mean-arrays['mean']).max()), std=float(np.abs(std-arrays['std']).max())),
                replay=replay, payloads={name:dict(bytes=(folder/name).stat().st_size, sha256=sha(folder/name)) for name in payloads})


def main(wait):
    started, cpu = time.time(), time.process_time()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-be77d5f7-abf2-5e41-2a1c-9510e4a832d9', 'Pinned audit GPU')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    require(sha(ROOT/'UPSTREAM_FORMAL_GATE_V04.json') == GATE_SHA, 'Frozen upstream gate')
    gate = read(ROOT/'UPSTREAM_FORMAL_GATE_V04.json')
    require(all(sha(Path(p)) == h for p,h in gate['sources'].items()), 'Frozen sources/configs')
    require(sha(ROOT/'TRAIN_RECORD_ASSIGNMENT.json') == ASSIGN_SHA, 'Frozen assignment')
    manifest = read(ROOT/'TRAIN_RECORD_ASSIGNMENT.json')['records']
    jobs = read(ROOT/'FOLD_AND_BUDGET_CANDIDATE_V2.json')['jobs']
    require(len(jobs) == 9 and len({(j['seed'],j['model_fold']) for j in jobs}) == 9, 'Full upstream matrix')
    OUT.mkdir()
    fields = Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    identity = dict(pid=os.getpid(), uid=os.getuid(), startticks=int(fields[19]), source_sha256=sha(Path(__file__)), deadline_unix=started+24*3600, CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"], TF32_matmul=False, TF32_cudnn=True)
    write(OUT/'PROCESS.json', identity)
    try:
        # Merely importing torch must not retain a CUDA context during upstream work.
        require(not torch.cuda.is_initialized(), 'No CUDA context before the matrix finishes')
        recovery_terminal = ROOT/'UPSTREAM_RECOVERY_TERMINAL.json'
        while True:
            require(time.time() <= identity['deadline_unix'], 'Matrix observation deadline; never restart workers')
            failure = ROOT/'upstream_recovery_v01/FAILURE.json'
            require(not failure.exists(), 'Upstream recovery failed')
            ready = recovery_terminal.is_file()
            if ready:
                terminal = read(recovery_terminal)
                require(terminal['passed'] and len(terminal['jobs']) == 9
                        and terminal['frozen_upstream_gate_sha256'] == GATE_SHA,
                        'Actual completed recovered matrix')
                require(all((ROOT/'formal'/f"seed{j['seed']}_fold{j['model_fold']}"/'COMPLETED.json').is_file()
                            for j in jobs), 'All nine formal artifacts exist')
                resident = subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                            '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
                require(resident.isdecimal(), 'GPU memory observation')
                ready = int(resident) <= 256
            if ready:
                break
            require(wait, 'Upstream matrix not released; waiting requires --wait')
            state = dict(process=identity, stage='WAIT_COMPLETE_MATRIX_WITHOUT_CUDA',
                         CUDA_context_initialized=torch.cuda.is_initialized(), completed_audits=0)
            temp = OUT/'STATUS.writing'
            temp.write_text(json.dumps(state,indent=2))
            temp.replace(OUT/'STATUS.json')
            time.sleep(30)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    except BaseException as error:
        write(OUT/'FAILURE.json', dict(process=identity, message=str(error), traceback=traceback.format_exc(), stage='WAIT_COMPLETE_MATRIX', NN_windows=0, fitting_updates=0, labels_read=0))
        raise
    sys.path.insert(0, str(OLD/'input'))
    native = importlib.import_module('pub_01_s03_transformer_train_v2')
    cfg = read(OLD/'training_seed1024_attempt04/config_frozen.json')
    done, norm_cache = {}, {}
    counts = dict(NN_windows=0, NN_batches=0, physical_points=0, fitting_updates=0, labels_read=0)
    try:
        while len(done) < 9:
            require(time.time() <= identity['deadline_unix'], 'Audit observation deadline; do not restart upstream')
            for job in jobs:
                key = f"seed{job['seed']}_fold{job['model_fold']}"
                folder = ROOT/'formal'/key
                require(not (folder/'FAILURE.json').exists(), 'Upstream failure: '+key)
                if key in done or not (folder/'COMPLETED.json').exists():
                    continue
                result = inspect_job(job, manifest, norm_cache, native, cfg, counts)
                torch.cuda.synchronize()
                require(torch.cuda.max_memory_reserved() < 2*1024**3, "Bounded replay GPU memory")
                write(OUT/(key+'.json'), result)
                done[key] = result
                print(json.dumps(dict(completed_audits=len(done), job=key, **counts)), flush=True)
            state = dict(process=identity, completed_audits=len(done), pending=[f"seed{j['seed']}_fold{j['model_fold']}" for j in jobs if f"seed{j['seed']}_fold{j['model_fold']}" not in done], counters=counts)
            temp = OUT/'STATUS.writing'
            temp.write_text(json.dumps(state, indent=2))
            temp.replace(OUT/'STATUS.json')
            if len(done) < 9:
                require(wait, 'Upstream incomplete; use --wait without restarting an existing audit')
                terminal = ROOT/'UPSTREAM_RECOVERY_TERMINAL.json'
                require(not terminal.exists() or read(terminal)['passed'], 'Upstream queue failed')
                time.sleep(30)
        write(OUT/'COMPLETED.json', dict(passed=True, process=identity, jobs=list(done), counters=counts,
                normalization_folds=len(norm_cache), upstream_matrix_receipt_sha256=sha(recovery_terminal), peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(), peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(), total_CPU_seconds=time.process_time()-cpu,
                wall_seconds_including_dependency_wait=time.time()-started, peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                tolerances=dict(atol=ATOL, rtol=RTOL), scope='All fold statistics; actual raw EEG, saved FT weights and independent NumPy Ccap/Hann at predetermined points only. No label, VAL score, fit or scientific effect test.'))
    except BaseException as error:
        write(OUT/'FAILURE.json', dict(process=identity, message=str(error), traceback=traceback.format_exc(), counters=counts))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
