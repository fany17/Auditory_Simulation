"""Equation/gradient parity of exact performance adapters, server execution only."""
import torch
import pytest
from m6a_public.continuity.execution import FastPersistent,FastPredictive,FastAlignment,FastInnovation,MatchedDense,affine_scan
from m6a_public.continuity.dense_state import PersistentStateDecoder
from m6a_public.continuity.predictive import PredictiveContinuityDecoder
from m6a_public.continuity.alignment_innovation import AdaptiveAlignmentDecoder,InnovationStateDecoder


def test_exact_adapter_outputs_states_and_gradients():
    torch.set_num_threads(1)
    for base,fast in ((PersistentStateDecoder,FastPersistent),(PredictiveContinuityDecoder,FastPredictive),
                      (AdaptiveAlignmentDecoder,FastAlignment),(InnovationStateDecoder,FastInnovation)):
        torch.manual_seed(19);a=base(hidden=8);b=fast(hidden=8);b.load_state_dict(a.state_dict())
        x=torch.randn(1,257,64);oa=a(x);ob=b(x)
        torch.testing.assert_close(oa.prediction,ob.prediction,atol=2e-6,rtol=2e-5)
        oa.prediction.square().mean().backward();ob.prediction.square().mean().backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            assert na==nb
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,atol=2e-6,rtol=2e-4)
        split=b(x[:,:71]);rest=b(x[:,71:],split.state)
        torch.testing.assert_close(torch.cat((split.prediction,rest.prediction),1),ob.prediction,atol=2e-6,rtol=2e-5)


def test_affine_scan_causal_and_long():
    x=torch.randn(1,3840,4);a=torch.tensor([.01,.9,.99,.9999]);initial=torch.randn(1,4);states=[];h=initial
    for value in x.unbind(1):h=a*h+value;states.append(h)
    torch.testing.assert_close(affine_scan(x,a,initial),torch.stack(states,1),atol=3e-4,rtol=3e-4)
    changed=x.clone();changed[:,2000:]+=3
    assert torch.equal(affine_scan(x,a,initial)[:,:2000],affine_scan(changed,a,initial)[:,:2000])


def test_matched_m1_all_branches_active():
    counts=[]
    for mode in ('short','long','multi'):
        model=MatchedDense(mode);model(torch.randn(1,1400,64)).prediction.square().mean().backward()
        counts.append(sum(p.numel() for p in model.parameters() if p.grad is not None))
    assert counts[0]==counts[1]==counts[2]


def test_innovation_mode_isolation_and_prior_purity():
    full=FastInnovation(hidden=8);prior=FastInnovation(hidden=8,mode='prior_only');prior.load_state_dict(full.state_dict())
    x=torch.randn(1,257,64)
    with pytest.raises(ValueError):prior(x,full(x).state)
    a=prior(x);b=prior(x+100)
    assert torch.equal(a.prediction,b.prediction)
    assert torch.equal(prior(x,a.state).prediction,prior(x*0,b.state).prediction)


def test_innovation_manual_BPTT_includes_observation_loss_and_initial_state():
    torch.manual_seed(31);a=InnovationStateDecoder(hidden=8);b=FastInnovation(hidden=8);b.load_state_dict(a.state_dict())
    x1=torch.randn(2,73,64,requires_grad=True);x2=x1.detach().clone().requires_grad_(True)
    state1=a(x1[:,:19]).state;state2=b(x2[:,:19]).state
    oa=a(x1[:,19:],state1);ob=b(x2[:,19:],state2)
    for out in (oa,ob):(out.prediction.square().mean()+.1*out.aux['predicted_observation'].square().mean()).backward()
    torch.testing.assert_close(x1.grad,x2.grad,atol=1e-7,rtol=1e-4)
    for aa,bb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(aa.grad,bb.grad,atol=1e-6,rtol=1e-4)


def test_alignment_zero_lag_cache_empty():
    model=FastAlignment(hidden=8,lags=(0,));x=torch.randn(1,257,64)
    out=model(x);assert out.state['lag'].shape==(1,0,8)
    a=model(x[:,:71]);b=model(x[:,71:],a.state)
    torch.testing.assert_close(torch.cat((a.prediction,b.prediction),1),out.prediction,atol=1e-6,rtol=1e-5)


def test_cuda_actual_dimensions_output_gradient_and_chunk_parity():
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for base,fast in ((PersistentStateDecoder,FastPersistent),(PredictiveContinuityDecoder,FastPredictive),
                      (AdaptiveAlignmentDecoder,FastAlignment)):
        torch.manual_seed(1024);a=base().cuda();b=fast().cuda();b.load_state_dict(a.state_dict())
        x=torch.randn(1,3840,64,device='cuda');pa=a(x).prediction;pb=b(x).prediction
        torch.testing.assert_close(pa,pb,atol=2e-6,rtol=3e-5)
        pa.square().mean().backward();pb.square().mean().backward()
        for (na,aa),(nb,bb) in zip(a.named_parameters(),b.named_parameters()):
            assert na==nb
            if aa.grad is not None:torch.testing.assert_close(aa.grad,bb.grad,atol=3e-6,rtol=5e-4)
        with torch.no_grad():
            first=b(x[:,:257]);rest=b(x[:,257:],first.state)
            torch.testing.assert_close(torch.cat((first.prediction,rest.prediction),1),pb,atol=2e-6,rtol=3e-5)
