"""Additional independent audit of every fixed last epoch19, after zero selections.

All nine selected checkpoints were the zero residual incumbent. Their replay
does not validate trained nonzero predictions. Audit all nine saved last models
by a uniform epoch rule, retaining the already declared numerical tolerances.
This is verification of existing VAL results, not new selection or fitting.
"""
import json
import resource
import time
from pathlib import Path
import torch
import pub_01_s03_ss07_full_ft_history_independent_review as a


def run():
    started, cpu = time.time(), time.process_time()
    torch.set_num_threads(4)
    prior = a.ROOT / 'coordinator_review/REVIEW.json'
    assert a.read(prior)['passed']
    assert a.sha(a.ROOT / 'PROTOCOL.json') == a.PIN
    out = a.ROOT / 'coordinator_last_review'
    out.mkdir(exist_ok=False)
    a.write(out / 'STARTED.json', dict(source_sha256=a.sha(__file__), protocol_sha256=a.PIN,
        primary_review_sha256=a.sha(prior), reason='All nine selected checkpoints were zero heads; verify nonzero trained VAL path.',
        selection='Every arm and seed, fixed last epoch19; no choice by scores', tolerances=a.TOLERANCES,
        device='CPU', threads=4, expected_NN_windows=9 * 30059))
    protocol = a.read(a.D8 / 'PROTOCOL.json')
    rows, mask = protocol['records']['val'], protocol['mask']['blocks']
    summaries = []
    for seed in a.SEEDS:
        z, anchor, history_prior = a.inputs(seed, rows)
        for arm in a.ARMS:
            folder = a.ROOT / f'fits/{arm}_seed{seed}'
            checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
            assert checkpoint['epoch'] == 19 and checkpoint['arm'] == arm and checkpoint['seed'] == seed
            assert torch.count_nonzero(checkpoint['model']['output.weight']).item() > 0
            hist = [json.loads(line) for line in (folder / 'HISTORY.jsonl').read_text().splitlines()]
            assert len(hist) == 20 and hist[-1]['epoch'] == 19
            pred = a.predictions(checkpoint['model'], arm, rows, z, anchor, history_prior)
            metrics = a.score(pred, rows, mask)
            a.compare(f'{arm}/{seed}/last19', metrics['phases']['val'], hist[-1]['validation'])
            a.write(out / f'{arm}_seed{seed}.json', metrics)
            summaries.append(dict(seed=seed, arm=arm, epoch=19, checkpoint_sha256=a.sha(folder / 'last.pt'),
                actual_r=metrics['phases']['val']['r'], reported_r=hist[-1]['validation']['r'],
                absolute_r_difference=abs(metrics['phases']['val']['r'] - hist[-1]['validation']['r'])))
            print(json.dumps(summaries[-1]), flush=True)
        del z, anchor, history_prior
    assert a.COUNT['residual_NN_window_forwards'] == 9 * 30059
    result = dict(passed=not a.DIFFERENCES, source_sha256=a.sha(__file__), protocol_sha256=a.PIN,
        role='Verification of existing last19 results, not selected model evaluation', tolerances=a.TOLERANCES,
        checks=a.CHECKS, failures=a.DIFFERENCES, summaries=summaries,
        cost=dict(**a.COUNT, CPU_seconds=time.process_time() - cpu, wall_seconds=time.time() - started,
                  peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, GPU_bytes=0),
        unretained_intermediate_epoch_checkpoints_replayed=False, goal_complete=False)
    a.write(out / 'REVIEW.json', result)
    assert result['passed'], a.DIFFERENCES


if __name__ == '__main__':
    run()
