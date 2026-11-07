"""Bind the authorized independent output bridge; no new neural computation."""
from pub_01_s03_ss07_token_norm_core import *
def main():
    start=time.time();cpu=time.process_time();v1=ROOT/'engineering_v1'
    assert sha(ROOT/'PROTOCOL.json')=='24204cbde539ca165e76b312b53817dc4dc07e4bbdc2a365b17f73d5557bb647'
    p=read(ROOT/'PROTOCOL.json');first=read(ROOT/'ENGINEERING_CHECK.json');bridge=read(ROOT/'REFERENCE_BRIDGE.json')
    assert first['passed'] and first['NN_windows']==528 and bridge['passed'] and bridge['NN_windows']==288
    for path,digest in p['source_pins'].items():
        if path!=str(ROOT/'PLAN.md'):assert sha(path)==digest,path
    p['source_pins'][str(ROOT/'PLAN.md')]=sha(ROOT/'PLAN.md')
    for path in [ROOT/'BRIDGE_AMENDMENT.md',ROOT/'REFERENCE_BRIDGE.json',ROOT/'input/pub_01_s03_ss07_token_norm_reference_bridge.py',Path(__file__),ROOT/'coordinator_input/pub_01_s03_ss07_token_norm_independent_review.py',ROOT/'coordinator_input/pub_01_s03_ss07_full_ft_history_independent_review.py',v1/'PROTOCOL.json',v1/'ENGINEERING_CHECK.json']:
        p['source_pins'][str(path)]=sha(path)
    p.update(time=time.time(),v1_protocol_sha256=sha(v1/'PROTOCOL.json'),primary_four_source_files_unchanged=True,engineering_actual_NN_windows=816,engineering_actual_GPU_NN_windows=672,engineering_actual_manual_CPU_NN_windows=144,independent_review_max_NN_windows=631239,bridge='All3 fixed48 valid windows, nonzero fresh probe; pointwise GPU vs independent manually normalized network')
    (ROOT/'PROTOCOL.json').rename(v1/'CANONICAL_PROTOCOL_AT_REFREEZE.json');(ROOT/'ENGINEERING_CHECK.json').rename(v1/'CANONICAL_ENGINEERING_AT_REFREEZE.json')
    dump(ROOT/'PROTOCOL.json',p)
    result=dict(first);result.update(NN_windows=816,original_v1_NN_windows=528,additional_GPU_NN_windows=144,additional_manual_CPU_NN_windows=144,reference_bridge=bridge,primary_sources_unchanged=True,timing_note='Top-level times/peaks remain original528 pass; reference_bridge contains separate actual288 costs; not cumulative')
    dump(ROOT/'ENGINEERING_CHECK.json',result)
    dump(ROOT/'REFREEZE_RECORD.json',dict(passed=True,source_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL.json'),new_NN=0,extra_optimizer_updates=0,CPU_seconds=time.process_time()-cpu,wall_seconds=time.time()-start,original_evidence_preserved=True))
    print(json.dumps(dict(passed=True,engineering_NN_windows=816,protocol_sha256=sha(ROOT/'PROTOCOL.json'))),flush=True)
if __name__=='__main__':main()
