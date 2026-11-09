"""SS06 exact-equation execution adapters and declared within-framework controls.

Prototype modules remain available unchanged. The fused GRU kernel and affine
parallel scan are verified against original recurrences before real training.
"""
from typing import List
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .common import LocalEncoder, CausalConv, DecoderOutput
from .dense_state import DenseMultiscaleDecoder, PersistentStateDecoder
from .predictive import PredictiveContinuityDecoder
from .alignment_innovation import AdaptiveAlignmentDecoder, InnovationStateDecoder


def affine_scan(inputs, decay, initial):
    """h[t]=decay*h[t-1]+inputs[t], no resets or future dependence."""
    result=inputs
    shift=1
    while shift<inputs.shape[1]:
        result=torch.cat((result[:,:shift],result[:,shift:]+decay.pow(shift)*result[:,:-shift]),dim=1)
        shift*=2
    powers=decay**torch.arange(1,inputs.shape[1]+1,device=inputs.device,dtype=inputs.dtype)[:,None]
    return result+powers[None]*initial[:,None]


def gru_sequence(cell, features, initial):
    # Same gate equations/order as GRUCell; use installed PyTorch's sequence
    # kernel with the SAME four parameter tensors, without copied weights.
    return torch._VF.gru(features,initial.unsqueeze(0),
        [cell.weight_ih,cell.weight_hh,cell.bias_ih,cell.bias_hh],
        True,1,0.0,cell.training,False,True)


class LocalDecoder(nn.Module):
    def __init__(self,hidden=32):
        super().__init__();self.encoder=LocalEncoder(64,hidden);self.head=nn.Linear(hidden,1)
    def forward(self,x,state=None):
        features,cache=self.encoder(x,None if state is None else state['encoder'])
        return DecoderOutput(self.head(features),{'encoder':cache},{'features':features})


class MatchedDense(DenseMultiscaleDecoder):
    """Four active independent branches in each mode; equal active capacity.

    short: four dilation1; long: four dilation256; multi:1/8/64/256.
    Compared with original single-branch toggles this removes idle capacity.
    """
    def __init__(self,mode):
        super().__init__();self.actual_dilations=self.DILATIONS if mode=='multi' else ((1,)*4 if mode=='short' else (256,)*4)
        for key,d in zip(self.branches,self.actual_dilations):self.branches[key]=CausalConv(32,32,5,dilation=d)
    def forward(self,x,state=None):
        out=super().forward(x,state);out.aux['actual_dilations']=self.actual_dilations
        out.aux['receptive_field_samples']=17+4*max(self.actual_dilations)
        return out


class FastPersistent(PersistentStateDecoder):
    def forward(self,x,state=None):
        state={} if state is None else state
        local,enc=self.encoder(x,state.get('encoder'))
        fast,fh=gru_sequence(self.fast_cell,local,state.get('fast',x.new_zeros(x.shape[0],self.hidden)))
        alpha=self.slow_coefficient
        slow=affine_scan(alpha*torch.tanh(self.slow_input(local)),1-alpha,state.get('slow',x.new_zeros(x.shape[0],self.hidden)))
        fused=F.gelu(self.fusion(torch.cat((local,fast,slow),-1)))
        return DecoderOutput(self.head(fused),{'encoder':enc,'fast':fh[0],'slow':slow[:,-1]},
             {'fast':fast,'slow':slow,'slow_coefficient':alpha})


class FastPredictive(PredictiveContinuityDecoder):
    def forward(self,x,state=None):
        local,enc=self.encoder(x,None if state is None else state['encoder'])
        initial=x.new_zeros(x.shape[0],self.hidden) if state is None else state['context']
        trajectory,last=gru_sequence(self.context,local,initial)
        return DecoderOutput(self.envelope_head(trajectory),{'encoder':enc,'context':last[0]},
            {'features':local,'context':trajectory,'latent_predictions':{h:self.predictors[str(h)](trajectory) for h in self.horizons} if self.stage=='pretrain' else {},
             'input_shape':tuple(x.shape),'starts_at_boundary':state is None})


class FastAlignment(AdaptiveAlignmentDecoder):
    def forward(self,x,state=None):
        state={} if state is None else state;batch,steps,_=x.shape
        features,enc=self.encoder(x,state.get('encoder'))
        previous=state.get('lag',x.new_zeros(batch,self.max_lag,self.hidden))
        extended=torch.cat((previous,features),1)
        bank=torch.stack([extended[:,self.max_lag-lag:self.max_lag-lag+steps] for lag in self.lags],2)
        gate=state.get('gate',x.new_zeros(batch,len(self.lags)))
        if self.mode=='adaptive':
            gates=affine_scan((1-self.slow_decay)*self.lag_logits(features),self.slow_decay,gate)
            weights=gates.softmax(-1);gate=gates[:,-1]
        else:weights=self.fixed_weights.view(1,1,-1).expand(batch,steps,-1)
        aligned=(bank*weights.unsqueeze(-1)).sum(2)
        return DecoderOutput(self.readout(aligned),{'encoder':enc,'lag':extended[:,-self.max_lag:],'gate':gate},
            {'weights':weights,'aligned_features':aligned,'slow_decay':self.slow_decay})


@torch.jit.script
def innovation_scan(features:Tensor,latent:Tensor,aw:Tensor,ab:Tensor,hw:Tensor,hb:Tensor,gw:Tensor,gb:Tensor,full:bool):
    zs=torch.jit.annotate(List[Tensor],[]);obs=torch.jit.annotate(List[Tensor],[])
    for t in range(features.size(1)):
        prior=torch.tanh(F.linear(latent,aw,ab))
        if full:
            observation=F.linear(prior,hw,hb);innovation=features[:,t]-observation
            gain=torch.sigmoid(F.linear(torch.cat((prior,innovation),-1),gw,gb))
            latent=torch.tanh(prior+gain*innovation)
        else:observation=torch.zeros_like(prior);latent=prior
        zs.append(latent);obs.append(observation)
    return torch.stack(zs,1),torch.stack(obs,1)


class FastInnovation(InnovationStateDecoder):
    def forward(self,x,state=None):
        if self.mode=='eeg_only':return super().forward(x,state)
        state={} if state is None else state
        if self.mode=='full':features,enc=self.encoder(x,state.get('encoder'))
        else:features=x.new_zeros(x.shape[0],x.shape[1],self.hidden);enc=None
        latent=state.get('latent',x.new_zeros(x.shape[0],self.hidden))
        trajectory,observations=innovation_scan(features,latent,self.prior_dynamics.weight,self.prior_dynamics.bias,
             self.observation_predictor.weight,self.observation_predictor.bias,self.gain_net.weight,self.gain_net.bias,self.mode=='full')
        return DecoderOutput(self.readout(trajectory),{'encoder':enc,'latent':trajectory[:,-1],'mode':self.mode},
            {'features':features,'predicted_observation':observations,'latent':trajectory})


VARIANTS=('local','m1_short','m1_long','m1_multi','m2','gru','m3_frozen','m3_finetune','m4_fixed','m4_adaptive','m5_eeg','m5_prior','m5_full')


def build_variant(name):
    if name=='local':return LocalDecoder()
    if name.startswith('m1_'):return MatchedDense(name[3:])
    if name=='m2':return FastPersistent()
    if name in ('gru','m3_frozen','m3_finetune'):
        return FastPredictive(mode={'gru':'no_pretrain','m3_frozen':'frozen','m3_finetune':'finetune'}[name])
    if name.startswith('m4_'):return FastAlignment(mode=name[3:])
    if name.startswith('m5_'):return FastInnovation(mode={'m5_eeg':'eeg_only','m5_prior':'prior_only','m5_full':'full'}[name])
    raise ValueError(name)
