"""D16: one affine-free per-token normalization; unchanged frozen D15 inputs."""
import sys,os,time,json,hashlib,resource,shutil,stat
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01');D15=BASE/'s03_full_ft_objective_control_20261102'
ROOT=BASE/'s03_full_ft_token_normalization_20261102'
sys.path.insert(0,str(D15/'input'))
import pub_01_s03_ss07_objective_control_core as previous
Data=previous.Data;predict=previous.predict;permutation=previous.permutation;train_score=previous.train_score
objective=previous.objective;window_losses=previous.window_losses;target_counts=previous.target_counts
module_gradient_summary=previous.module_gradient_summary;SEEDS=previous.SEEDS;ARMS=previous.ARMS;read=previous.read;sha=previous.sha
INFLIGHT_RESERVE_BYTES=28*1024**2
def dump(path,value):
    p=Path(path);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def directory_bytes():
    total=0
    for p in ROOT.rglob('*'):
        try:info=p.stat()
        except FileNotFoundError:continue
        if stat.S_ISREG(info.st_mode):total+=info.st_size
    return total
def guard():
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<8*1024**2
    if torch.cuda.is_initialized():assert torch.cuda.max_memory_allocated()<2*1024**3
    assert min(shutil.disk_usage(ROOT).free,shutil.disk_usage('/sata5/fanyu/auditory_simulation_m6a').free)>=20*1024**3
    assert directory_bytes()+INFLIGHT_RESERVE_BYTES<160*1024**2
def normalize_tokens(tokens):return F.layer_norm(tokens,(32,),weight=None,bias=None,eps=1e-5)
@torch.no_grad()
def token_state_summary(before,after,states):
    p=before.detach().double();q=after.detach().double();h=states.detach().double()
    values=torch.stack([p.square().mean().sqrt(),p.abs().max(),q.square().mean().sqrt(),q.abs().max(),q.mean(-1).abs().max(),q.var(-1,correction=0).min(),q.var(-1,correction=0).max(),(h.abs()>=1-1e-6).double().mean(),h.abs().max()]).cpu().tolist()
    keys=['token_pre_RMS','token_pre_max_abs','token_post_RMS','token_post_max_abs','max_abs_per_token_mean','min_per_token_population_variance','max_per_token_population_variance','GRU_fraction_abs_ge_one_minus_1e_minus6','GRU_state_max_abs']
    return dict(zip(keys,values))
class Conditional(previous.Conditional):
    def __init__(self,seed,arm):
        super().__init__(seed,arm);self.capture_diagnostics=False;self.last_scalar_diagnostics=None
    def forward(self,current,history,return_state=False):
        b=len(current);before=self.token(history.reshape(b*2,history.shape[2],192)).reshape(b,2,32)
        tokens=normalize_tokens(before)
        seq,_=self.gru(tokens);h=seq[:,-1];q=self.current(current);g,beta=self.film(h).chunk(2,1)
        out=self.output(q*(1+.1*torch.tanh(g))+.1*beta)
        if self.capture_diagnostics:self.last_scalar_diagnostics=token_state_summary(before,tokens,seq)
        return (out,h,q) if return_state else out
    def take_token_diagnostics(self):
        assert self.last_scalar_diagnostics is not None
        value=self.last_scalar_diagnostics;self.last_scalar_diagnostics=None;return value
