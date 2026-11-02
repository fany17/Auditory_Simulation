"""Fixed TRAIN-only nonzero last-checkpoint CPU/manual-vs-GPU bridge.

Run after all nine fits, using one actually idle GPU. No VAL scoring, adjustment,
optimizer updates, or encoder/head. Absolute/relative waveform bounds predeclared.
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import pub_01_s03_ss07_full_ft_history_independent_review as audit
from pub_01_s03_ss07_full_ft_history_core import Conditional


@torch.no_grad()
def run():
    started, cpu = time.time(), time.process_time()
    torch.set_num_threads(4)
    assert (audit.ROOT / 'QUEUE_COMPLETED.json').exists()
    assert audit.sha(audit.ROOT / 'PROTOCOL.json') == audit.PIN
    apps = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not [s for s in apps.splitlines() if s.strip().isdigit()], 'GPU 0 not idle; defer bridge'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    folder = audit.D8 / 'seed1024'
    cal = audit.read(folder / 'P_FT_BASELINE.json')['calibration']
    with np.load(folder / 'TRAIN_TRANSFORMS.npz') as f:
        stats = [{k: f[f'{k}_{i}'].copy() for k in ('mean', 'std', 'constant')} for i in range(3)]
    raw = np.asarray(np.load(folder / 'train_latent.npy', mmap_mode='r')[:25], dtype='f8').reshape(25, 1536)
    z64 = audit.transformed(raw, stats[0])
    z = z64.reshape(25, 8, 192).astype('f4')
    p = np.asarray(np.load(folder / 'train_P.npy', mmap_mode='r')[:25], dtype='f8')
    anchor = cal['a'] * p + cal['b']
    with np.load(folder / 'Ccap_lambda0.01_COEFFICIENTS.npz') as f:
        feat = np.concatenate((z64[4:] / np.sqrt(1536), audit.transformed(anchor[4:], stats[1]) / np.sqrt(192),
            audit.transformed(z64[4:] ** 2, stats[2]) / np.sqrt(1536)), axis=1)
        anchor[4:] += feat @ f['coef'] + f['bias']
    prior = audit.transformed(anchor, stats[1]).astype('f4')
    rows = audit.read(audit.D8 / 'PROTOCOL.json')['records']['train']
    assert rows[0]['offset'] == 0 and rows[0]['windows'] >= 25
    ix = np.arange(9, 25)
    current = z[ix].copy()
    checks = []
    for arm in audit.ARMS:
        path = audit.ROOT / f'fits/{arm}_seed1024/last.pt'
        cp = torch.load(path, map_location='cpu', weights_only=True)
        assert cp['epoch'] == 19 and cp['arm'] == arm and cp['seed'] == 1024
        state = cp['model']
        assert torch.count_nonzero(state['output.weight']).item() > 0
        if arm == 'C_match':
            history = np.repeat(current[:, None], 2, axis=1)
        else:
            hi = np.column_stack((ix - 5, ix - 3))
            history = prior[hi, None, :].copy() if arm == 'P_prior' else z[hi].copy()
        own = audit.manual_residual(state, current, history).astype('f8') * .1
        model = Conditional(1024, arm).eval().cuda(0)
        model.load_state_dict(state)
        reference = model(torch.from_numpy(current).cuda(0), torch.from_numpy(history).cuda(0)).cpu().numpy().astype('f8') * .1
        error = np.abs(own - reference)
        bound = 2e-4 + 2e-4 * np.abs(reference)
        checks.append(dict(arm=arm, checkpoint_sha256=audit.sha(path), samples=16,
            nonzero_output_weight_scalars=int(torch.count_nonzero(state['output.weight'])),
            scaled_residual_max_abs_error=float(error.max()), scaled_residual_max_abs_reference=float(np.abs(reference).max()),
            bound_absolute=2e-4, bound_relative=2e-4, passed=bool(np.all(error <= bound))))
        del model
    result = dict(passed=all(c['passed'] for c in checks), source_sha256=audit.sha(__file__),
        review_source_sha256=audit.sha(Path(audit.__file__)), protocol_sha256=audit.PIN,
        phase='train', seed=1024, record=rows[0]['record'], record_local_indices=[9, 24], checkpoints='fixed last epoch19, never selected using validation',
        checks=checks, residual_NN_window_forwards_CPU=48, residual_NN_window_forwards_GPU=48,
        new_encoder_head_forwards=0, optimizer_updates=0, targets_or_VAL_scores_accessed=False,
        CPU_seconds=time.process_time() - cpu, wall_seconds=time.time() - started,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(0))
    audit.write(audit.ROOT / 'COORDINATOR_REVIEW_TRAIN_BRIDGE.json', result)
    print(json.dumps(result))
    assert result['passed']


if __name__ == '__main__':
    run()
