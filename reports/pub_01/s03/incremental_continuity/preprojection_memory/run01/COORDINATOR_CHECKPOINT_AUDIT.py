"""Inspect completed D18 checkpoint contents on 2203, without neural execution."""
import argparse
import hashlib
import json
import math
import resource
import time
from collections import Counter
from pathlib import Path

ROOT = Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_preprojection_memory_20261103')
PROTOCOL = '255ce2b95457d807e41775eaa208d40024dc5d1637c3a37b24edfec189c90632'
ARMS = ('H_history', 'C_current', 'T_continue', 'P_prediction_history')


def audit(seed):
    wall, cpu = time.time(), time.process_time()
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    assert hashlib.sha256((ROOT / 'PROTOCOL.json').read_bytes()).hexdigest() == PROTOCOL
    results = []
    for arm in ARMS:
        folder = ROOT / 'fits' / f'{arm}_seed{seed}'
        completed_bytes = (folder / 'COMPLETED.json').read_bytes()
        assert not (folder / 'FAILURE.json').exists()
        completed = json.loads(completed_bytes)
        history_bytes = (folder / 'HISTORY.jsonl').read_bytes()
        history = [json.loads(line) for line in history_bytes.splitlines()]
        assert completed['seed'] == seed and completed['arm'] == arm
        assert completed['protocol_sha256'] == PROTOCOL and len(history) == 20
        expected = {'encoder': 3372360, 'head': 257}
        if arm != 'T_continue':
            expected['memory'] = 16896
        if arm == 'P_prediction_history':
            expected['prior_projection'] = 640
        assert sum(expected.values()) == completed['registered_trainable_parameters']
        checkpoints, states = [], []
        for filename, epoch, score in (
            ('best_validation.pt', completed['selected_epoch'], completed['selected_validation_r']),
            ('last.pt', 19, history[-1]['validation']['r']),
        ):
            path = folder / filename
            assert path.resolve().is_relative_to(ROOT.resolve())
            before = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert set(checkpoint) == {'trainable_state', 'seed', 'arm', 'epoch', 'validation_r', 'protocol_sha256', 'frozen_assets'}
            assert checkpoint['seed'] == seed and checkpoint['arm'] == arm
            assert checkpoint['protocol_sha256'] == PROTOCOL and checkpoint['epoch'] == epoch
            assert math.isfinite(checkpoint['validation_r']) and checkpoint['validation_r'] == score
            assert checkpoint['frozen_assets'] == 'Source pins only; no repeated E0/Ccap/position/optimizer'
            state = checkpoint['trainable_state']
            assert isinstance(state, dict) and state
            groups, dtypes = Counter(), Counter()
            schema = []
            for name, value in sorted(state.items()):
                assert isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                assert value.is_floating_point() and bool(torch.isfinite(value).all())
                groups[name.split('.', 1)[0]] += value.numel()
                dtypes[str(value.dtype)] += value.numel()
                schema.append((name, list(value.shape), str(value.dtype)))
            assert dict(groups) == expected
            checkpoints.append(dict(file=filename, sha256=digest, bytes=before.st_size,
                                    epoch=epoch, validation_r=score,
                                    parameter_tensors=len(state), parameters_by_module=dict(groups),
                                    parameters_by_dtype=dict(dtypes), all_tensors_finite=True,
                                    tensor_schema_sha256=hashlib.sha256(json.dumps(schema).encode()).hexdigest(),
                                    header_matches_completed_and_history=True))
            states.append(state)
        assert states[0].keys() == states[1].keys()
        assert checkpoints[0]['tensor_schema_sha256'] == checkpoints[1]['tensor_schema_sha256']
        differing = sum(not torch.equal(states[0][name], states[1][name]) for name in states[0])
        results.append(dict(arm=arm, seed=seed,
                            completed_sha256=hashlib.sha256(completed_bytes).hexdigest(),
                            history_sha256=hashlib.sha256(history_bytes).hexdigest(),
                            checkpoints=checkpoints,
                            best_vs_last_differing_parameter_tensors=differing))
        del states, checkpoint, state
    assert not torch.cuda.is_initialized()
    return dict(seed=seed, passed=True, protocol_sha256=PROTOCOL, fits=results,
                audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                execution_host='server2203', map_location='cpu', weights_only=True,
                checkpoint_files_inspected=8, dataset_arrays_read=0, neural_forwards=0,
                backwards=0, optimizer_updates=0, cuda_initialized=False,
                audit_CPU_seconds=time.process_time()-cpu, audit_wall_seconds=time.time()-wall,
                audit_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                timer_scope='Includes torch import, reads/hashes, header/parameter/schema/finite/equality checks; no dataset or model execution',
                scope='One completed seed: actual serialized checkpoint integrity and metadata only; no independent prediction or metric replay',
                goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=(1024, 2026, 31415), required=True)
    print(json.dumps(audit(parser.parse_args().seed), allow_nan=False))
