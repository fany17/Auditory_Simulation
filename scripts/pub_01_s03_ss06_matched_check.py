"""Required run02 engineering checks; run only on2203, actual EEG train/val."""
import json
import time
import numpy as np
import torch
from pub_01_s03_ss06_strong import CausalStrong,NativeStrong,detach,objective
from pub_01_s03_ss06_matched import ROOT,Inputs,setup,batch,dump,sync


def main():
    setup(1024,'cuda:0');device='cuda:0';data=Inputs('train');x,y=data.arrays(0)
    xx=batch(x[3840:3840+3840],device);yy=batch(y[3840:3840+3840],device);report={};started=time.perf_counter()
    m=CausalStrong().to(device).eval()
    with torch.no_grad():
        a,_=m(xx[:,:600]);changed=xx[:,:600].clone();changed[:,317:]+=torch.randn_like(changed[:,317:])*4
        b,_=m(changed);error=float((a[:,:317]-b[:,:317]).abs().max());assert error<1e-5
        full,_=m(xx[:,:600],chunk=600);tiny,_=m(xx[:,:600],chunk=37)
        chunk_error=float((full-tiny).abs().max());assert torch.allclose(full,tiny,atol=2e-4,rtol=2e-4)
        combined,_=m(xx[:,:600]);_,s=m(xx[:,:300]);second,s=m(xx[:,300:600],s)
        carry_error=float((combined[:,300:]-second).abs().max());assert torch.allclose(combined[:,300:],second,atol=2e-4,rtol=2e-4)
        reset,_=m(xx[:,300:600],None);assert float((reset-second).abs().max())>1e-6
        assert s['position']==600 and all(v.shape[1]==63 for v in s['conv']) and all(v[0].shape[2]==191 for v in s['kv']) and s['head'].shape[1]==31
    report['C']={'future_perturbation_max':error,'full_chunk_max':chunk_error,'carry_max':carry_error,
        'cache_conv':63,'cache_KV':191,'cache_head':31,'attention_window_including_current':192,'position_after_carry':600,
        'parameters':sum(p.numel() for p in m.parameters())}
    # Gradient from a LATER microchunk reaches earlier input: no hidden detach.
    z=xx[:,:384].clone().requires_grad_();p,_=m(z);p[:,192:].square().mean().backward()
    cross_grad=float(z.grad[:,:192].abs().sum());assert cross_grad>0;report['C']['cross_microchunk_gradient']=cross_grad
    del m
    for name,model in [('C',CausalStrong()),('O',NativeStrong())]:
        model=model.to(device);model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        t=time.perf_counter();p,state=model(xx);loss=objective(p,yy);loss.backward()
        assert p.shape==yy.shape and torch.isfinite(p).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();sync(device)
        report[name]=report.get(name,{})|{'real_60s_loss':float(loss.detach()),'all_grad_finite':True,
            'parameters':sum(p.numel() for p in model.parameters()),'real_step_seconds':time.perf_counter()-t}
        if name=='O':
            # Check overlap coordinates with identity window predictor. Every
            # sample, including the two Hann endpoints, must be reconstructed.
            model.windows=lambda windows:windows[:,:,:1]
            with torch.no_grad():identity,_=model(xx)
            e=float((identity-xx[:,:,:1]).abs().max());assert e<1e-5
            report[name]['Hann_identity_max']=e;report[name]['native_windows_per_60s']=39
        del model
    dump(ROOT/'ENGINEERING_CHECK.json',{'passed':True,'checks':report,'seconds':time.perf_counter()-started,
        'audit_read':False,'actual_input':'newtrain normalized cache x[s]=rawEEG[s+8]','target':'same envelope coordinate s'})
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
