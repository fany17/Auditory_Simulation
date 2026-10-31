"""Same-device real-input inference costs; no fitting or target access."""
import json
import time
import numpy as np
import torch
from m6a_public.continuity.execution import build_variant,VARIANTS
from pub_01_s03_ss06_data import ROOT,LENGTH,Records,dump
from pub_01_s03_ss06_train import tensor


def main():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Records('train');x,_=data.arrays(data.rows[0]);rows=[]
    local=64*32*9+32*9
    for device in ('cpu','cuda:0'):
        for variant in VARIANTS:
            torch.manual_seed(1024);model=build_variant(variant).to(device).eval()
            prefix=tensor(x[:LENGTH],device);current=tensor(x[LENGTH:2*LENGTH],device)
            with torch.no_grad():
                state=model(prefix).state
                for _ in range(2):model(current,state)
                if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                before=time.perf_counter()
                for _ in range(5):model(current,state)
                if device.startswith('cuda'):torch.cuda.synchronize()
                ms=(time.perf_counter()-before)*1000/5
            if variant=='local' or variant=='m5_eeg':macc=local+32
            elif variant.startswith('m1_'):macc=local+4*32*32*5+32*32+32
            elif variant=='m2':macc=local+6*32*32+32*32+3*32*32+32
            elif variant in ('gru','m3_frozen','m3_finetune'):macc=local+6*32*32+32
            elif variant.startswith('m4_'):macc=local+4*32+32+(4*32 if variant=='m4_adaptive' else 0)
            elif variant=='m5_prior':macc=32*32+32
            else:macc=local+32*32*4+32
            rows.append({'variant':variant,'device':device,'registered_parameters':sum(p.numel() for p in model.parameters()),
                '60s_inference_ms_after_warmup':ms,'approximate_MACCs_per60s':macc*LENGTH,
                'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0})
    dump(ROOT/'resource_profile.json',{'rows':rows,'input':'same real newtrain60s after60s prefix','weights':'initialization, no optimization',
        'MACC_convention':'multiply-accumulate in affine/conv/readout/lag mixture; excludes bias/nonlinear/elementwise scan/data movement; not equal wallclock',
        'interpretation':'same-device timings available; actual training M5 CPU and other models GPU must still be disclosed'})
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
