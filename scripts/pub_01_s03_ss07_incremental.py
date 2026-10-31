"""Residuals on the FULL frozen native eight-channel EEG representation.

Current z[i] has original EEG support [96i,96i+192). Only i-2/i-4...
enter its main history: two parity streams, no overlap-only observation.
No target envelope enters a history update. Compute on2203 only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def detach(v):
    if isinstance(v,torch.Tensor):return v.detach()
    if isinstance(v,dict):return {k:detach(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return type(v)(detach(x) for x in v)
    return v


def summary(z):return F.adaptive_avg_pool1d(z,6).flatten(1)


def current_sequence(z):
    # Match summary's CHANNEL-major coordinates: feature index=channel*6+bin.
    bins=F.adaptive_avg_pool1d(z,6).transpose(1,2)
    basis=torch.eye(6,device=z.device,dtype=z.dtype)
    return (bins[:,:,:,None]*basis[None,:,None,:]).reshape(len(z),6,-1)


def affine_ema(sequence,old,alpha):
    """Closed-form causal EMA, exact gradients, no hidden state detach.

    sequence [B,L,H], old [B,H], independent decay per channel.
    L<=64 for the fixed native batch. No unstable division by alpha**L.
    """
    n=sequence.shape[1];ids=torch.arange(n,device=sequence.device)
    distance=ids[:,None]-ids[None,:]
    weights=alpha[None,None,:].pow(distance.clamp_min(0)[:,:,None])
    weights=weights*(distance>=0)[:,:,None]*(1-alpha)[None,None,:]
    after=torch.einsum('ijh,bjh->bih',weights,sequence)
    after=after+old[:,None,:]*alpha[None,None,:].pow((ids+1)[None,:,None])
    before=torch.cat((old[:,None,:],after[:,:-1]),1)
    return before,after


class ContextCore(nn.Module):
    def __init__(self,family,width,tau=30.,input_dim=48):
        super().__init__();self.family=family;self.width=width;self.input_dim=input_dim
        self.project=nn.Linear(input_dim,width)
        if family=='m1':
            self.register_buffer('alpha',torch.exp(-3/torch.tensor([3.,12.,48.,192.])).repeat_interleave(width))
            self.context=nn.Linear(4*width,width)
        else:
            self.gru=nn.GRU(input_dim,width,batch_first=True)
            self.log_tau=nn.Parameter(torch.full((width,),math.log(tau)))
            self.context=nn.Linear(2*width,width)

    def evolve(self,seq,state=None,dt=3.):
        b,n,_=seq.shape;state=state or {}
        projected=torch.tanh(self.project(seq))
        if self.family=='m1':
            alpha=self.alpha if dt==3 else self.alpha.pow(dt/3)
            projected=projected.repeat(1,1,4)
            old=state.get('slow',seq.new_zeros(b,4*self.width))
            before,after=affine_ema(projected,old,alpha)
            return torch.tanh(self.context(before)),torch.tanh(self.context(after)),{'slow':after[:,-1]}
        old_fast=state.get('fast',seq.new_zeros(1,b,self.width))
        fast,new_fast=self.gru(seq,old_fast)
        before_fast=torch.cat((old_fast.transpose(0,1),fast[:,:-1]),1)
        tau=self.log_tau.exp().clamp(3,300);alpha=torch.exp(-dt/tau)
        old_slow=state.get('slow',seq.new_zeros(b,self.width))
        before_slow,after_slow=affine_ema(projected,old_slow,alpha)
        before=torch.tanh(self.context(torch.cat((before_fast,before_slow),-1)))
        after=torch.tanh(self.context(torch.cat((fast,after_slow),-1)))
        return before,after,{'fast':new_fast,'slow':after_slow[:,-1]}

    def safe_history(self,values,state=None,local_start=0,record_key=None):
        if state is not None:
            assert state['next_window']==local_start,'window order/record-reset drift'
            assert state['record_key']==record_key,'state crossed recording'
        prior=[None,None] if state is None else state['parity']
        out=values.new_zeros(len(values),self.width);after=out.clone();states=[]
        for parity in range(2):
            first=(parity-local_start)%2
            ids=torch.arange(first,len(values),2,device=values.device)
            if len(ids)==0:states.append(prior[parity]);continue
            before,following,new=self.evolve(values.index_select(0,ids)[None],prior[parity])
            out=out.index_copy(0,ids,before[0]);after=after.index_copy(0,ids,following[0]);states.append(new)
        return out,after,{'parity':states,'next_window':local_start+len(values),'record_key':record_key}

    def current_only(self,z):
        # Exact same learned core/parameter count, recurrent weights ACTIVE:
        # six ordered within-window bins embedded in disjoint slots, not h=0
        # at every recurrent step (which would leave weight_hh unused).
        bins=F.adaptive_avg_pool1d(z,6).transpose(1,2)
        if self.input_dim==48:
            seq=current_sequence(z)
        else:
            # Output-prior current-only uses its six-bin signal once per step.
            seq=bins.flatten(1)[:,None,:].expand(-1,6,-1)
        _,after,_=self.evolve(seq,None,dt=.5)
        return after[:,-1]


class NativeResidual(nn.Module):
    def __init__(self,family='m2',width=16,tau=30.,conditioning='past'):
        super().__init__();assert family in ('m1','m2','m3','m4','m5','output_prior')
        assert conditioning in ('past','current')
        self.family=family;self.conditioning=conditioning;self.width=width
        # Construct FIRST so manual_seed(seed) yields native head initialization.
        self.head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
        self.core=ContextCore(family,width,tau,input_dim=6 if family=='output_prior' else 48)
        output=24 if family=='m4' else 6 if family=='output_prior' else 64
        self.residual=nn.Linear(width,output)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        if family=='m3':self.predictor=nn.Linear(width,48)
        if family=='m5':self.prior=nn.Linear(width,48)

    def forward(self,z,state=None,local_start=0,record_key=None,enabled=True,with_aux=False,mode='full'):
        assert z.ndim==3 and z.shape[1:]==(8,192)
        values=summary(self.head(z)) if self.family=='output_prior' else summary(z)
        if self.conditioning=='current':
            current_signal=self.head(z) if self.family=='output_prior' else z
            context=self.core.current_only(current_signal);after=context;new=None
        else:context,after,new=self.core.safe_history(values,state,local_start,record_key)
        aux={};delta=self.residual(context)
        if self.family=='output_prior':
            base=self.head(z);residual=F.interpolate(delta[:,None,:],size=192,mode='linear',align_corners=False)
            prediction=base+.1*residual if enabled else base
        elif self.family=='m4':
            coeff=delta.reshape(-1,8,3);transformed=z
            for j,lag in enumerate((4,8,16)):
                shifted=F.pad(z[:,:,:-lag],(lag,0))
                transformed=transformed+.1*coeff[:,:,j,None]*(shifted-z)
            prediction=self.head(transformed if enabled else z)
            aux['alignment_coefficients']=coeff
        else:
            innovation=z
            if self.family=='m5':
                prior=F.interpolate(self.prior(context).reshape(-1,8,6),size=192,mode='linear',align_corners=False)
                innovation=z-prior;aux['latent_prior']=prior
            matrix=delta.reshape(-1,8,8)
            transformed=z+.1*torch.bmm(matrix,innovation)
            prediction=self.head((aux['latent_prior'] if mode=='prior_only' else transformed) if enabled else z)
        if self.family=='m3' and with_aux:aux['future_prediction']=self.predictor(after)
        return prediction,new,aux

    def inference_residual_parameters(self):
        return sum(p.numel() for name,p in self.named_parameters() if not name.startswith(('head.','predictor.')))


class StaticCapacityResidual(nn.Module):
    """All static capacity acts on CURRENT z; no padding/unreachable params."""
    def __init__(self,target_parameters):
        super().__init__();self.head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
        width=max(1,round((target_parameters-8)/17));self.width=width
        self.adapter=nn.Sequential(nn.Conv1d(8,width,1),nn.GELU(),nn.Conv1d(width,8,1))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
    def forward(self,z,state=None,local_start=0,record_key=None,enabled=True,**kwargs):
        return self.head(z+.1*self.adapter(z) if enabled else z),None,{}


class StaticAlignment(nn.Module):
    """Best learned static nonuniform feature-shift residual, not uniform lag."""
    def __init__(self):
        super().__init__();self.head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
        self.coefficients=nn.Parameter(torch.zeros(8,3))
    def forward(self,z,state=None,local_start=0,record_key=None,enabled=True,**kwargs):
        transformed=z
        for j,lag in enumerate((4,8,16)):
            transformed=transformed+.1*self.coefficients[None,:,j,None]*(F.pad(z[:,:,:-lag],(lag,0))-z)
        return self.head(transformed if enabled else z),None,{}
