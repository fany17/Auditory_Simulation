"""Validate independent reconstruction code on validation only, before holdout."""
import json
import numpy as np
import torch
from pub_01_s03_ss05_data import ROOT,ContinuousPool,dump
from pub_01_s03_ss05_finite_history import FiniteHistoryDecoder
from pub_01_s03_ss05_evaluate import evaluate
from pub_01_s03_ss05_review import independent_cpu

torch.set_num_threads(4);pool=ContinuousPool('validation','cuda:1');pool.segments=pool.segments[:1]
scale=np.load(ROOT/'newtrain_normalization.npz');answers=[]
for variant in ('local','state'):
    torch.manual_seed(1024);gpu=FiniteHistoryDecoder(variant).to('cuda:1').eval()
    _,_,_,rows=evaluate(gpu,pool,8,('true','reset','wrong_record'),batch=1)
    cpu=FiniteHistoryDecoder(variant).eval();cpu.load_state_dict({k:v.cpu() for k,v in gpu.state_dict().items()})
    rid=pool.segments[0]['record_id'];expected={r['condition']:r for r in rows}
    errors=independent_cpu(cpu,pool.records[rid],pool.records,scale,8,expected)
    answers.append({'variant':variant,'max_raw_CPU_CUDA_r_error':max(errors)})
assert max(r['max_raw_CPU_CUDA_r_error'] for r in answers)<1e-4
dump(ROOT/'review_method_validation_smoke.json',{'status':'PASS','results':answers,'holdout_payload_used':False,'optimizer_steps':0})
print(json.dumps(answers),flush=True)
