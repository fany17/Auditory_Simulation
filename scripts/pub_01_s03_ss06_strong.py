"""SS06 matched strong controls; no historical checkpoint is loaded here."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from pub_01_s03_method_models import AuditoryTransformer, AttentionBlock, MethodPair
from m6a_public.continuity.objectives import dense_objective

CFG = dict(depth=10,width=64,heads=8,ff_multiplier=4,latent_dim=8,
    conv_kernel=64,conv_dropout=.2,attention_dropout=.5,speech_conv_kernel=32,
    speech_dropout=.4,speech_lstm_hidden=128,initial_logit_parameter=.075)
WINDOW=192


def detach(value):
    if isinstance(value,torch.Tensor):return value.detach()
    if isinstance(value,dict):return {k:detach(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(detach(v) for v in value)
    return value


def causal_conv(layer,x,cache):
    # x is B,T,C. Exactly kernel-1 past inputs, never future padding.
    keep=layer.kernel_size[0]-1
    if cache is None:cache=x.new_zeros(x.shape[0],keep,x.shape[2])
    joined=torch.cat((cache,x),1)
    return layer(joined.transpose(1,2)).transpose(1,2),joined[:,-keep:]


class CausalStrong(nn.Module):
    """Native SS02 residual layout; sample LN, left conv, 192-sample attention.

    No positional encoding exists in native SS02. Relative sample differences
    define the sliding mask; cache lengths retain the correct coordinate at
    arbitrary chunk/60s boundaries. Returned caches remain attached until the
    caller detaches AFTER the entire 3840-sample optimizer step.
    """
    def __init__(self):
        super().__init__()
        self.spatial=nn.Conv1d(64,64,1)
        self.temporal=nn.ModuleList([nn.Conv1d(64,64,64) for _ in range(10)])
        self.norm=nn.ModuleList([nn.LayerNorm(64) for _ in range(10)])
        self.drop=nn.Dropout(.2)
        self.attend=nn.ModuleList([AttentionBlock(64,8,.5) for _ in range(10)])
        self.output=nn.Linear(64,8)
        self.head=nn.Conv1d(8,1,32)

    def piece(self,x,state=None):
        state=state or {'conv':[None]*10,'kv':[None]*10,'head':None,'position':0}
        initial=self.spatial(x.transpose(1,2)).transpose(1,2);z=initial;convs=[];kvs=[]
        for i,(conv,norm,a) in enumerate(zip(self.temporal,self.norm,self.attend)):
            z,cache=causal_conv(conv,z+initial,state['conv'][i]);convs.append(cache)
            z=F.leaky_relu(norm(self.drop(z)))
            if i<9:z=z+initial
            b,t,w=z.shape
            q,k,v=a.qkv(a.before_attention(z)).reshape(b,t,3,8,8).permute(2,0,3,1,4).unbind(0)
            old=state['kv'][i]
            if old is not None:k=torch.cat((old[0],k),2);v=torch.cat((old[1],v),2)
            past=k.shape[2]-t
            relative=torch.arange(k.shape[2],device=x.device)[None,:]-(past+torch.arange(t,device=x.device)[:,None])
            mask=(relative<=0)&(relative>=-191)
            message=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                dropout_p=.5 if self.training else 0.,scale=1/math.sqrt(64))
            message=message.transpose(1,2).reshape(b,t,w)
            z=z+a.residual_drop(a.mix(message))
            z=z+a.residual_drop(a.ff(a.before_ff(z)))
            kvs.append((k[:,:,-191:],v[:,:,-191:]))
        p,h=causal_conv(self.head,self.output(z),state['head'])
        return F.leaky_relu(p,.01),{'conv':convs,'kv':kvs,'head':h,'position':state['position']+x.shape[1]}

    def forward(self,x,state=None,chunk=192):
        pieces=[]
        for start in range(0,x.shape[1],chunk):
            p,state=self.piece(x[:,start:start+chunk],state);pieces.append(p)
        return torch.cat(pieces,1),state


class NativeStrong(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=AuditoryTransformer(CFG)
        self.head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))

    def windows(self,x):return self.head(self.encoder(x).transpose(1,2)).transpose(1,2)

    def forward(self,x,state=None):
        # Independent 60s blocks, starts 0,96,...,3648: 39 native 3s
        # windows. Both endpoint predictions retain positive fallback weight;
        # no scoring sample is omitted. All predictions stay differentiable.
        assert x.shape[1]%96==0 and x.shape[1]>=192
        t=x.shape[1];starts=list(range(0,t-191,96))
        windows=torch.cat([x[:,s:s+192] for s in starts],0)
        pred=self.windows(windows).reshape(len(starts),x.shape[0],192,1)
        weight=torch.hann_window(192,periodic=False,device=x.device,dtype=x.dtype)
        # Two exactly-zero Hann endpoints: sole available window prediction.
        numerator=x.new_zeros(x.shape[0],t,1);denominator=x.new_zeros(1,t,1)
        for i,s in enumerate(starts):
            w=weight.clone()
            if i==0:w[0]=1
            if i==len(starts)-1:w[-1]=1
            numerator=numerator+F.pad(pred[i]*w[None,:,None],(0,0,s,t-s-192))
            denominator=denominator+F.pad(w[None,:,None],(0,0,s,t-s-192))
        assert bool((denominator>0).all())
        return numerator/denominator,None


def pair():return MethodPair('transformer',{'transformer':CFG},0)


def objective(pred,y):
    return dense_objective(pred,y,mse_weight=.1)['loss']


def pearson_loss(pred,y):
    pc=pred-pred.mean(1,keepdim=True);yc=y-y.mean(1,keepdim=True)
    return -((pc*yc).sum(1)/(pc.square().sum(1)*yc.square().sum(1)).sqrt().clamp_min(1e-8)).mean()
