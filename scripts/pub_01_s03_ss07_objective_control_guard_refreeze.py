"""D15 v2 IO/log amendment; reuse unchanged mathematical engineering, zero NN."""
import ast
from pub_01_s03_ss07_objective_control_core import *
def main():
    start=time.time();cpu=time.process_time();prior=ROOT/'engineering_v1'
    p=read(prior/'PROTOCOL.json');engineering=read(prior/'ENGINEERING_CHECK.json')
    assert sha(prior/'PROTOCOL.json')=='7b78b356a5565ae72f5ae3486f832c8f428ba88d50672202c001174e7257016b'
    assert engineering['passed'] and engineering['NN_windows']==288
    old_ast=ast.parse((prior/'input/pub_01_s03_ss07_objective_control_core.py').read_text())
    new_ast=ast.parse((ROOT/'input/pub_01_s03_ss07_objective_control_core.py').read_text())
    for name in ['window_losses','objective','target_counts','train_score']:
        left=next(x for x in old_ast.body if isinstance(x,ast.FunctionDef) and x.name==name)
        right=next(x for x in new_ast.body if isinstance(x,ast.FunctionDef) and x.name==name)
        assert ast.dump(left)==ast.dump(right),name
    mock=guard_checks();guard();assert read(ROOT/'PATH_PROBE.json')['passed']
    allowed={str(ROOT/'PLAN.md'),str(ROOT/'input/pub_01_s03_ss07_objective_control_core.py'),str(ROOT/'input/pub_01_s03_ss07_objective_control_train.py'),str(ROOT/'input/pub_01_s03_ss07_objective_control_prepare.py')}
    for path,digest in p['source_pins'].items():
        if path not in allowed:assert sha(path)==digest,path
    proof=dict(passed=True,source_sha256=sha(__file__),mathematical_function_AST_unchanged=True,checks=mock,new_NN=0,extra_optimizer_updates=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start)
    dump(ROOT/'GUARD_ENGINEERING_CHECK.json',proof)
    for path in list(p['source_pins']):
        if path in allowed:p['source_pins'][path]=sha(path)
    for path in [ROOT/'GUARD_AMENDMENT.md',ROOT/'GUARD_ENGINEERING_CHECK.json',ROOT/'PATH_PROBE.json',prior/'PROTOCOL.json',prior/'ENGINEERING_CHECK.json',prior/'input/pub_01_s03_ss07_objective_control_core.py']+sorted((ROOT/'input').glob('pub_01_s03_ss07_objective_control_*.py')):
        p['source_pins'][str(path)]=sha(path)
    p.update(time=time.time(),engineering_reuse='v1 numerical engineering288NN; identical objective/statistic AST; guard/log amendment0NN; authorized fresh/saturation probe192NN',v1_protocol_sha256=sha(prior/'PROTOCOL.json'),first_batch_gradients='Existing first shuffled batch after backward/before step, all5module nonzero and L2;0newforward/update, never selection')
    p['budget'].update(inflight_reserve_bytes=INFLIGHT_RESERVE_BYTES,measured_plus_reserve_below_root_cap=True)
    assert read(ROOT/'STORAGE_ESTIMATE.json')['conservative_total_bytes']+INFLIGHT_RESERVE_BYTES<160*1024**2
    dump(ROOT/'PROTOCOL.json',p)
    engineering.update(guard_mock=mock,guard_amendment_proof_sha256=sha(ROOT/'GUARD_ENGINEERING_CHECK.json'),path_probe=read(ROOT/'PATH_PROBE.json'),NN_windows=288+192,original_v1_NN_windows=288,path_probe_extra_NN_windows=192,guard_amendment_new_NN=0,formal_models_reconstructed_clean=True)
    dump(ROOT/'ENGINEERING_CHECK.json',engineering)
    print(json.dumps(dict(passed=True,engineering_NN=480,protocol_sha256=sha(ROOT/'PROTOCOL.json'),new_NN_this_refreeze=0)),flush=True)
if __name__=='__main__':main()
