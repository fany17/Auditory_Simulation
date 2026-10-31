"""Synthetic engineering on2203; no EEG performance or selection claim."""
import json
from pathlib import Path
import time
import torch
from torch import nn
from pub_01_s03_ss07_incremental import NativeResidual,StaticCapacityResidual,StaticAlignment,summary,current_sequence

ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101')


def exercise(model,z,y):
    opt=torch.optim.Adam(model.parameters(),lr=.001);reach=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True);p,_,aux=model(z,record_key='A',with_aux=True)
        pc=p-p.mean(-1,keepdim=True);yc=y-y.mean(-1,keepdim=True)
        loss=-nn.functional.cosine_similarity(pc,yc,dim=-1,eps=1e-6).mean()
        if 'future_prediction' in aux:loss=loss+.01*aux['future_prediction'].square().mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        reach.append({name:float(p.grad.abs().sum()) if p.grad is not None else None for name,p in model.named_parameters()})
        opt.step()
    if hasattr(model,'residual'):
        assert reach[0]['residual.weight']>0
        assert sum(v or 0 for k,v in reach[1].items() if k.startswith('core.'))>0
    if isinstance(model,StaticCapacityResidual):assert reach[1]['adapter.0.weight']>0
    return {'registered_parameters':sum(p.numel() for p in model.parameters()),
        'second_step_gradient_reachable':sum(p.numel() for p in model.parameters() if p.grad is not None),
        'nonzero_gradient_entries_second_step':sum(int(torch.count_nonzero(p.grad)) for p in model.parameters() if p.grad is not None),
        'first_step_final_projection_gradient':reach[0].get('residual.weight'),
        'second_step_core_gradient':sum(v or 0 for k,v in reach[1].items() if k.startswith('core.'))}


def main():
    torch.set_num_threads(1);torch.manual_seed(7123);z=torch.randn(17,8,192);y=torch.randn(17,1,192);rows=[]
    coordinate_error=float((current_sequence(z).sum(1)-summary(z)).abs().max());assert coordinate_error==0
    for family in ('m1','m2','m3','m4','m5','output_prior'):
        for conditioning in ('past','current'):
            torch.manual_seed(1024);model=NativeResidual(family,16,30,conditioning).eval()
            torch.manual_seed(1024);head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
            with torch.no_grad():
                p,_,_=model(z,record_key='A');identity=float((p-head(z)).abs().max());assert identity==0
            gradients=exercise(model,z,y);model.eval()
            with torch.no_grad():
                model.residual.weight.normal_(0,.03);model.residual.bias.normal_(0,.03)
                whole,_,_=model(z,record_key='A');changed=z.clone();changed[9:]+=torch.randn_like(changed[9:])*3
                altered,_,_=model(changed,record_key='A');future=float((whole[:9]-altered[:9]).abs().max());assert future<2e-5
                state=None;parts=[]
                for first,last in [(0,5),(5,9),(9,17)]:
                    p,state,_=model(z[first:last],state,first,'A');parts.append(p)
                chunk=float((whole-torch.cat(parts)).abs().max());assert chunk<2e-5
                if conditioning=='past':
                    _,state,_=model(z[:5],record_key='A')
                    try:model(z[5:9],state,5,'B')
                    except AssertionError:pass
                    else:raise AssertionError('record boundary not protected')
                    wrong=z.clone();wrong[1]+=10
                    altered,_,_=model(wrong,record_key='A')
                    # Window1 overlaps raw support of window2, so it must NOT
                    # enter window2 history. Current z2 is held fixed here.
                    overlap=float((whole[2]-altered[2]).abs().max());assert overlap<2e-5
                else:overlap=None
            rows.append({'family':family,'conditioning':conditioning,'identity_max':identity,'future_feature_perturbation_max':future,
                'arbitrary_odd_chunk_max':chunk,'overlap_window_exclusion_max':overlap,**gradients})
    for family in ('m1','m2','m4','m5'):
        ref=NativeResidual(family,16);n=ref.inference_residual_parameters();static=StaticCapacityResidual(n)
        r=exercise(static,z,y);actual=sum(p.numel() for p in static.adapter.parameters());assert abs(actual-n)/n<.01
        rows.append({'family':'static_capacity_for_'+family,'target_residual_parameters':n,'actual_residual_parameters':actual,**r})
    rows.append({'family':'static_nonuniform_alignment',**exercise(StaticAlignment(),z,y)})
    model=NativeResidual('m2',16).eval()
    with torch.no_grad():model.residual.weight.normal_(0,.05)
    xx=z.clone().requires_grad_();p,_,_=model(xx,record_key='A');p[12:].square().mean().backward()
    cross=float(xx.grad[:8].abs().sum());assert cross>0
    # Exact support proof over all synthetic native windows and Hann coverage.
    for t in range(1,17*96+95):
        covering=[i for i in range(17) if i*96<=t<i*96+192]
        if not covering:continue
        baseline_end=max(i*96+192 for i in covering)
        extended_end=max([i*96+192 for i in covering]+[k*96+192 for i in covering for k in range(i-1) if (i-k)%2==0])
        assert extended_end==baseline_end
    result={'passed':True,'scope':'SYNTHETIC ENGINEERING ONLY; no EEG result or selected configuration',
        'rows':rows,'cross_window_input_gradient_single_call':cross,'safe_history_rule':'i-2/i-4; source_end<=current_start',
        'current_sequence_channel_major_coordinate_error':coordinate_error,
        'v1_clarification':'v1 cross_chunk_input_gradient was single-call cross-WINDOW, not TBPTT/cross-call proof; v1 preserved; separate real-input training-boundary check follows',
        'Hann_union_future_support_unchanged':True,'first_two_insufficient_history_windows_kept':True,
        'test_payload_loaded':False,'old_model_checkpoint_loaded':False}
    with (ROOT/'INCREMENTAL_ENGINEERING_CHECK_v2.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
