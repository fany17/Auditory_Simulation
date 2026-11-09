"""Scalar comparison/control validation only; no model, signal, or weight load."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'coordinator_final_replay_v01'
SOURCE = HERE / 'COORDINATOR_FINAL_REPLAY.py'
spec = importlib.util.spec_from_file_location('replay_v02_scalar_check', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not module.torch.cuda.is_initialized()

def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)}

old, new = functions(PRIOR / SOURCE.name), functions(SOURCE)
for name in ('prefix', 'memory_delta', 'correct', 'metrics'):
    assert old[name] == new[name], name
check = module.compare_metrics
records = []
records.append(check(dict(r=.2, MSE=10001.), dict(r=.2, MSE=10000.)))
records.append(check(dict(r=.2, MSE=.10002), dict(r=.2, MSE=.1)))
records.append(check(dict(r=.2, MSE=.1002), dict(r=.2, MSE=.1)))
assert len(records) == 3 and [r['strict_passed'] for r in records] == [False, True, False]
assert records[0]['original_checks'] == dict(r=True, MSE=False)
edge = check(dict(r=1e-4, MSE=1e-4), dict(r=0., MSE=0.))
assert edge['strict_passed'] and edge['relative_MSE_error'] is None
for invalid in (float('nan'), float('inf'), -1.):
    try:
        check(dict(r=.2, MSE=invalid), dict(r=.2, MSE=.1))
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid metric must remain a hard failure')
assert not any(module.COUNTS.values())
result = dict(passed=True, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              plan_sha256=hashlib.sha256((HERE / 'COORDINATOR_FINAL_REPLAY_PLAN.md').read_bytes()).hexdigest(),
              unchanged_forward_and_score_functions=['prefix', 'memory_delta', 'correct', 'metrics'],
              original_threshold_unchanged=True, raw_failure_preserved_with_following_outputs=True,
              calibrated_failure_preserved=True, equality_and_zero_denominator_checked=True,
              nonfinite_and_negative_MSE_rejected=True, raw_signal_reads=0, weight_reads=0,
              new_fits=0, model_forwards=0, optimizer_updates=0)
with (HERE / 'PREFLIGHT.json').open('x') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(result))
