"""D18 model proposal: pretrained64-token memory + differentiable fixed Ccap.

No construction or execution at import. Public TRAIN/VAL on2203 only.
"""
import math,sys
from pathlib import Path
import torch
from torch import nn
BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
OLD=BASE/'s03_strong_baselines_20261015/transformer_available01'
sys.path.insert(0,str(OLD/'input'))
from pub_01_s03_method_models import AuditoryTransformer
ARMS=('H_history','C_current','T_continue','P_prediction_history')

def prefix(encoder,x):
    """Exact original loop through192x64, before original Linear64->8."""
    initial=encoder.spatial(x.transpose(1,2));state=initial
    for index,(conv,attention) in enumerate(zip(encoder.temporal,encoder.attend)):
        state=conv(state+initial).transpose(1,2)
        if index<len(encoder.temporal)-1:state=state+initial.transpose(1,2)
        state=attention(state).transpose(1,2)
    return state.transpose(1,2)

def fixed_position(positions):
    # Fixed relative sample coordinates, not a learned position/length grid.
    p=torch.as_tensor(positions,dtype=torch.float64)[:,None]
    rate=torch.exp(-math.log(10000)*torch.arange(0,64,2,dtype=torch.float64)/64)
    result=torch.empty(len(p),64,dtype=torch.float64)
    result[:,0::2]=torch.sin(p*rate);result[:,1::2]=torch.cos(p*rate)
    return result.float()[None]

class CcapReadout(nn.Module):
    def __init__(self,stats,coefficient,bias,calibration):
        super().__init__()
        for i,item in enumerate(stats):
            for key in ('mean','std'):self.register_buffer(f'{key}{i}',torch.as_tensor(item[key],dtype=torch.float64).clone())
            self.register_buffer(f'constant{i}',torch.as_tensor(item['constant'],dtype=torch.bool).clone())
        self.register_buffer('coefficient',torch.as_tensor(coefficient,dtype=torch.float64).clone())
        self.register_buffer('bias',torch.as_tensor(bias,dtype=torch.float64).clone())
        self.register_buffer('cal_a',torch.tensor(calibration['a'],dtype=torch.float64))
        self.register_buffer('cal_b',torch.tensor(calibration['b'],dtype=torch.float64))
        assert self.coefficient.shape==(3264,192) and self.bias.shape==(192,)
        assert self.mean0.numel()==1536 and self.mean1.numel()==192 and self.mean2.numel()==1536
        assert self.cal_a>0
        assert all(torch.isfinite(v).all() for v in self.buffers())
        assert all((getattr(self,f'std{i}')>0).all() for i in range(3))
    def normalize(self,x,index):
        v=(x-getattr(self,f'mean{index}'))/getattr(self,f'std{index}')
        return torch.where(getattr(self,f'constant{index}'),0.,v)
    def forward(self,z,native_prediction):
        assert z.shape[1:]==(8,192) and native_prediction.shape==(len(z),1,192)
        u=self.normalize(z.double().reshape(len(z),1536),0)
        base=self.cal_a*native_prediction[:,0].double()+self.cal_b
        v=self.normalize(base,1);w=self.normalize(u.square(),2)
        features=torch.cat((u/math.sqrt(1536),v/math.sqrt(192),w/math.sqrt(1536)),1)
        # Preserve original base+(features@coef+bias) grouping. Buffers fixed,
        # but all operations retain the graph to raw encoder/head parameters.
        return base+(features@self.coefficient+self.bias)

class TokenMemory(nn.Module):
    def __init__(self):
        super().__init__();self.query_norm=nn.LayerNorm(64);self.memory_norm=nn.LayerNorm(64)
        self.attention=nn.MultiheadAttention(64,8,dropout=0.,batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
        self.register_buffer('query_position',fixed_position(range(192)),persistent=False)
        # Chronological i-5 then i-3. C/P use the SAME slots. Older replacement
        # changes data only; slot codes are kept fixed, so it is not pure delay.
        positions=list(range(-480,-288))+list(range(-288,-96))
        self.register_buffer('memory_position',fixed_position(positions),persistent=False)
    def inputs(self,query,memory):
        assert query.shape[1:]==(192,64) and memory.shape==(len(query),384,64)
        q=self.query_norm(query);v=self.memory_norm(memory)
        return q+self.query_position,v+self.memory_position,v
    def forward(self,query,memory):
        q,k,v=self.inputs(query,memory)
        result,_=self.attention(q,k,v,need_weights=False)
        return result

class AdaptedDecoder(nn.Module):
    def __init__(self,seed,arm,config,trained_state,readout):
        super().__init__();assert arm in ARMS;self.arm=arm;self.seed=seed
        # Common constructor sequence across all arms. All modules start onCPU.
        self.encoder=AuditoryTransformer(config)
        self.head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
        self.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in trained_state.items() if k.startswith('encoder.')},strict=True)
        self.head.load_state_dict({k.removeprefix('head.'):v for k,v in trained_state.items() if k.startswith('head.')},strict=True)
        # Frozen independent copy, not an alias that train()/optimizer can mutate.
        import copy
        self.reference_encoder=copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.readout=readout
        if arm!='T_continue':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+18000);self.memory=TokenMemory()
        if arm=='P_prediction_history':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+18001);self.prior_projection=nn.Conv1d(1,64,9,padding=4)
        self.count={k:0 for k in ('current_encoder_prefix_windows','current_projection_windows','current_head_windows','frozen_encoder_prefix_windows','attention_query_windows','prior_projection_windows')}
        self.capture=False;self.last_diagnostics=None
    def train(self,mode=True):
        super().train(mode);self.reference_encoder.eval();return self
    def encode_current(self,x):
        self.count['current_encoder_prefix_windows']+=len(x);return prefix(self.encoder,x)
    def memory_values(self,x,valid,history_raw=None,prior=None):
        n=int(valid.sum());assert n>0
        if self.arm=='H_history':
            assert history_raw is not None and history_raw.shape==(n,2,192,64)
            with torch.no_grad():
                self.count['frozen_encoder_prefix_windows']+=2*n
                values=prefix(self.reference_encoder,history_raw.reshape(n*2,192,64))
            return values.reshape(n,384,64)
        if self.arm=='C_current':
            with torch.no_grad():
                self.count['frozen_encoder_prefix_windows']+=n
                values=prefix(self.reference_encoder,x[valid])
            # Actual token reuse, no ignored past forward or fake FLOP parity.
            return torch.cat((values,values),1)
        assert self.arm=='P_prediction_history' and prior is not None and prior.shape==(n,2,192)
        self.count['prior_projection_windows']+=2*n
        values=self.prior_projection(self.readout.normalize(prior.double(),1).float().reshape(n*2,1,192)).transpose(1,2)
        return values.reshape(n,384,64)
    def predict_from_query(self,query,x,anchor,valid,history_raw=None,prior=None):
        assert query.shape==(len(x),192,64) and anchor.shape==(len(x),192)
        if self.arm=='T_continue':fused=query
        else:
            memory=self.memory_values(x,valid,history_raw,prior)
            self.count['attention_query_windows']+=int(valid.sum())
            delta=self.memory(query[valid],memory)
            ids=torch.nonzero(valid,as_tuple=False)[:,0]
            full=torch.zeros_like(query).index_copy(0,ids,delta);fused=query+full
            if self.capture:
                with torch.no_grad():
                    self.last_diagnostics=dict(query_RMS=float(query.double().square().mean().sqrt()),memory_RMS=float(memory.double().square().mean().sqrt()),delta_RMS=float(delta.double().square().mean().sqrt()),delta_max_abs=float(delta.abs().max()))
        self.count['current_projection_windows']+=len(x)
        z=self.encoder.output(fused).transpose(1,2)
        self.count['current_head_windows']+=len(x)
        native=self.head(z)
        prediction=self.readout(z,native)
        return torch.where(valid[:,None],prediction,anchor)
    def forward(self,x,anchor,valid,history_raw=None,prior=None,enabled=True):
        if not enabled or not bool(valid.any()):return anchor
        query=self.encode_current(x)
        return self.predict_from_query(query,x,anchor,valid,history_raw,prior)
    def trainable_state(self):
        # No frozen E0, statistics/coefs, fixed positions or optimizer duplication.
        return {k:v.detach().cpu() for k,v in self.named_parameters() if v.requires_grad}
    def load_trainable(self,state):
        expected={k:v for k,v in self.named_parameters() if v.requires_grad}
        assert expected.keys()==state.keys()
        with torch.no_grad():
            for name,value in expected.items():value.copy_(state[name].to(device=value.device,dtype=value.dtype))
    def trainable_groups(self):
        groups=[dict(params=list(self.encoder.parameters()),lr=1e-5),dict(params=list(self.head.parameters()),lr=1e-3)]
        if self.arm!='T_continue':groups.append(dict(params=list(self.memory.parameters()),lr=1e-3))
        if self.arm=='P_prediction_history':groups.append(dict(params=list(self.prior_projection.parameters()),lr=1e-3))
        return groups

def objective(prediction,target):
    assert prediction.dtype==target.dtype==torch.float64
    p=prediction-prediction.mean(-1,keepdim=True);y=target-target.mean(-1,keepdim=True)
    return -nn.functional.cosine_similarity(p,y,dim=-1,eps=1e-6).mean()
