"""Audited finite-history execution wrapper for the SS05 window prototypes.

Each 60s score block contains20 current3s EEG windows. Each current window
queries exactly its preceding20 contiguous windows, with a fresh state at the
left context boundary. No state is ever reused between unrelated examples.
The historical prototype's 'local' used two past tokens; this wrapper fixes
local to reset on EACH current window. Original prototype files stay intact.
"""
import torch
from torch import nn
from torch.nn import functional as F
from pub_01_s03_ss05_window_decoder import WindowPredictiveDecoder,WindowGRUDecoder

VARIANTS=('local','state','assoc_no_pred','full','gru')


class FiniteHistoryDecoder(nn.Module):
    def __init__(self,variant='full',width=32,gru_hidden=104):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.width=width;self.window_samples=192
        self.base=WindowGRUDecoder(width=width,gru_hidden=gru_hidden) if variant=='gru' else WindowPredictiveDecoder(width=width,memory_variant=variant)

    def encode(self,eeg):
        if eeg.ndim!=4 or eeg.shape[-2:]!=(192,64):raise ValueError('raw EEG[B,W,192,64] only')
        b,w,t,c=eeg.shape
        hidden=self.base.extract(eeg.reshape(b*w,t,c).transpose(1,2))
        token=F.gelu(self.base.summary(torch.cat((hidden.mean(-1),hidden.abs().amax(-1)),dim=-1)))
        return hidden.transpose(1,2).reshape(b,w,t,self.width),token.reshape(b,w,self.width)

    def queries(self,local,current,history,reset=False):
        # local[B,Q,192,width], current[B,Q,width], history[B,Q,20,width]
        if history.shape[-2:]!=(20,self.width) or current.shape!=history.shape[:2]+(self.width,):raise ValueError('history/query shape')
        b,q,t,h=local.shape
        tokens=current.reshape(b*q,1,h) if self.variant=='local' or reset else torch.cat((history,current[:,:,None,:]),dim=2).reshape(b*q,21,h)
        readout=self.base.memory(tokens)[:,-1].reshape(b,q,1,h).expand(b,q,t,h)
        gate=torch.sigmoid(self.base.fusion_gate(torch.cat((local,readout),dim=-1)))
        fused=local+gate*self.base.fusion_memory(readout)
        return self.base.output(fused.reshape(b*q,t,h).transpose(1,2)).transpose(1,2).reshape(b,q,t,1)

    def forward(self,eeg):
        if eeg.shape[1]!=40:raise ValueError('20 contiguous past+20 current windows required')
        local,tokens=self.encode(eeg)
        context=tokens.unfold(1,21,1).permute(0,1,3,2)
        return self.queries(local[:,20:],context[:,:,-1],context[:,:,:-1])


def pcc_loss(pred,target):
    # Per3s window, averaged;60s metrics are reserved for validation selection.
    a=pred-pred.mean(2,keepdim=True);b=target-target.mean(2,keepdim=True)
    return -F.cosine_similarity(a,b,dim=2,eps=1e-6).mean()


def parameter_profile(model,x,y):
    model.zero_grad(set_to_none=True);loss=pcc_loss(model(x),y);loss.backward()
    total=sum(p.numel() for p in model.parameters())
    reachable=sum(p.numel() for p in model.parameters() if p.grad is not None)
    nonzero=sum(p.numel() for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad)>0)
    disconnected=[name for name,p in model.named_parameters() if p.grad is None]
    macs=[0];handles=[]
    def linear(module,args,result):macs[0]+=args[0].numel()//module.in_features*module.in_features*module.out_features
    def conv(module,args,result):macs[0]+=result.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
    def gru(module,args,result):
        seq=args[0];macs[0]+=seq.shape[0]*seq.shape[1]*3*(module.input_size*module.hidden_size+module.hidden_size**2)
    for module in model.modules():
        hook=linear if isinstance(module,nn.Linear) else conv if isinstance(module,nn.Conv1d) else gru if isinstance(module,nn.GRU) else None
        if hook:handles.append(module.register_forward_hook(hook))
    with torch.no_grad():model(x)
    for h in handles:h.remove()
    assoc_macs=0 if model.variant not in ('full','assoc_no_pred') else x.shape[0]*20*21*3*(model.width//2)*model.width
    return {'instantiated_parameters':total,'gradient_reachable_parameters':reachable,'nonzero_gradient_parameter_tensors_elements':nonzero,
        'gradient_disconnected_parameter_names':disconnected,'executed_conv_linear_GRU_MACCs_per_batch':macs[0],
        'associative_read_write_MACCs_approx_per_batch':assoc_macs,
        'MACC_limit':'bias, normalization, gates/activations/pointwise arithmetic excluded; inactive prediction Linear still executes in historical prototype'}
