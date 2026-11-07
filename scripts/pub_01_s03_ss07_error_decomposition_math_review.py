"""Independent synthetic calculus check via torch.fold/autograd; no project imports."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_full_ft_error_decomposition_20261102')


def run():
    start, cpu = time.time(), time.process_time()
    torch.set_num_threads(2)
    rng = np.random.default_rng(963121)
    windows, width, hop = 82, 192, 96
    covered = (windows - 1) * hop + width
    n = covered + 33
    target = rng.normal(size=n).astype('f4').astype('f8')
    pred = rng.normal(size=(windows, width)) + .4
    hann = np.hanning(width)
    active = np.ones_like(pred)
    active[:9] = 0
    tensor = torch.tensor(pred, dtype=torch.float64, requires_grad=True)
    pp = torch.tensor(pred) + torch.tensor(active) * (tensor - torch.tensor(pred))
    hh = torch.tensor(hann)
    folded = F.fold((pp * hh).T[None], output_size=(1, covered), kernel_size=(1, width), stride=(1, hop)).flatten()
    weights = F.fold(hh[:, None].repeat(1, windows)[None], output_size=(1, covered), kernel_size=(1, width), stride=(1, hop)).flatten()
    keep = weights > 1e-8
    safe = weights.clone()
    safe[~keep] = 1
    fused = folded / safe
    yy = torch.tensor(target[:covered])
    yw = torch.stack([torch.tensor(target[i * hop:i * hop + width]) for i in range(windows)])
    loss_m = ((pp - yw) ** 2).mean()
    loss_f = torch.zeros((), dtype=torch.float64)
    loss_r = torch.zeros((), dtype=torch.float64)
    gf = np.zeros(covered)
    gr = np.zeros(covered)
    angular_errors = []
    # Unequal artificial weights exercise accounting; not a real person scheme.
    for first, block_weight in [(0, .25), (3840, .75)]:
        k = keep[first:first + 3840]
        x, y = fused[first:first + 3840][k], yy[first:first + 3840][k]
        u, v = x - x.mean(), y - y.mean()
        rho = torch.dot(u, v) / (torch.linalg.vector_norm(u) * torch.linalg.vector_norm(v))
        loss_f = loss_f + block_weight * ((x - y) ** 2).mean()
        loss_r = loss_r - block_weight * rho
        xn, yn = x.detach().numpy(), y.numpy()
        un, vn = xn - xn.mean(), yn - yn.mean()
        umag, vmag = np.linalg.norm(un), np.linalg.norm(vn)
        r = np.dot(un, vn) / (umag * vmag)
        one_f = 2 * (xn - yn) / len(xn)
        one_r = -vn / (umag * vmag) + r * un / np.dot(un, un)
        tangent = one_f - one_f.mean()
        tangent -= np.dot(tangent, un) / np.dot(un, un) * un
        angular_errors.append(float(np.max(np.abs(tangent - (2 * umag * vmag / len(xn)) * one_r))))
        ix = np.flatnonzero(k.numpy()) + first
        gf[ix] = block_weight * one_f
        gr[ix] = block_weight * one_r
    expected_m = 2 * (pred - yw.numpy()) / pred.size * active
    def transpose(physical):
        lifted = np.zeros_like(pred)
        weight = weights.numpy()
        for i in range(windows):
            first = i * hop
            kk = weight[first:first + width] > 1e-8
            lifted[i, kk] = hann[kk] * physical[first:first + width][kk] / weight[first:first + width][kk]
        return lifted * active
    expected = dict(M=expected_m, F=transpose(gf), R=transpose(gr))
    losses = dict(M=loss_m, F=loss_f, R=loss_r)
    errors = {}
    for name, loss in losses.items():
        actual = torch.autograd.grad(loss, tensor, retain_graph=True)[0].numpy()
        error = float(np.max(np.abs(actual - expected[name])))
        assert error < 1e-12, (name, error)
        assert np.count_nonzero(actual[:9]) == 0
        errors[name] = error
    assert np.count_nonzero(expected['M'][-1]) > 0
    assert np.count_nonzero(expected['F'][-1]) == 0 and np.count_nonzero(expected['R'][-1]) == 0
    assert np.count_nonzero(expected['R'][39, :96]) > 0 and np.count_nonzero(expected['R'][39, 96:]) > 0
    assert max(angular_errors) < 1e-12
    decompositions = []
    for gain in [.2, -.4, -1.4]:
        a = rng.normal(size=101) + .7
        y = rng.normal(size=101) + .1
        d = .3 + gain * (a - a.mean()) + rng.normal(scale=.3, size=101)
        u, v = a - a.mean(), d - d.mean()
        alpha = np.dot(u, v) / np.dot(u, u)
        e = v - alpha * u
        vy = np.var(y)
        va, ve = np.var(a), np.var(e)
        cay, cey = np.mean(u * (y - y.mean())), np.mean(e * (y - y.mean()))
        ra = cay / np.sqrt(va * vy)
        rr = ((1 + alpha) * ra + cey / np.sqrt(va * vy)) / np.sqrt((1 + alpha) ** 2 + ve / va)
        actual_r = np.corrcoef(a + d, y)[0, 1]
        offset = 2 * d.mean() * (a.mean() - y.mean()) + d.mean() ** 2
        along = (alpha ** 2 + 2 * alpha) * va - 2 * alpha * cay
        orth = ve - 2 * cey
        actual_m = np.mean((a + d - y) ** 2) - np.mean((a - y) ** 2)
        rerror, merror = abs(rr - actual_r), abs(offset + along + orth - actual_m)
        assert rerror < 1e-12 and merror < 1e-12
        decompositions.append(dict(designed_gain=gain, alpha=float(alpha), r_reconstruction_error=float(rerror), MSE_sum_error=float(merror)))
    result = dict(passed=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        methods='torch.fold/autograd versus independent NumPy physical derivatives and explicit H transpose',
        synthetic_windows=windows, first_fixed_windows=9, physical_blocks=2, physical_block_weights=[.25, .75],
        gradient_max_abs=errors, angular_identity_max_abs=max(angular_errors), decomposition=decompositions,
        cross_60s_window_both_halves_checked=True, unscored_tail_MSE_only_checked=True, negative_total_gain_checked=True,
        actual_data_or_NN_or_optimizer_access=False, CPU_seconds=time.process_time() - cpu, wall_seconds=time.time() - start)
    ROOT.mkdir(exist_ok=True)
    with (ROOT / 'COORDINATOR_AUTOGRAD_CHECK.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    run()
