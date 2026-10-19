"""Window-recurrent EEG→envelope model with across-window history.

Input [batch, windows, 192 samples, 64 channels] from ONE record.
Output [batch, windows, 192 samples, 1] without candidate/audio input.
Past windows can influence current predictions via PredictiveEventMemory.
The encoder is *offline within each 3-s window*, but its cross-window state is
causal. Do not advertise this model as fully sample-causal/zero-latency.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .pub_01_s03_ss05_predictive_memory import PredictiveEventMemory, PredictiveMemoryState
except ImportError:  # server2203 flat-script execution
    from pub_01_s03_ss05_predictive_memory import PredictiveEventMemory, PredictiveMemoryState


class WindowPredictiveDecoder(nn.Module):
    def __init__(self, *, eeg_channels=64, width=32, memory_variant='full', window_samples=192):
        super().__init__()
        if min(eeg_channels, width, window_samples) < 1:
            raise ValueError('dimensions must be positive')
        self.eeg_channels=eeg_channels
        self.window_samples=window_samples
        self.width=width
        self.extract=nn.Sequential(
            nn.Conv1d(eeg_channels, width, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=9, padding=4, groups=width),
            nn.GELU(),
        )
        self.summary=nn.Linear(width*2, width)
        self.memory=PredictiveEventMemory(
            input_channels=width,hidden=width,key_dim=max(4,width//2),
            output_channels=width,variant=memory_variant)
        self.fusion_gate=nn.Linear(width*2,width)
        self.fusion_memory=nn.Linear(width,width)
        self.output=nn.Conv1d(width,1,kernel_size=1)

    def forward(self,eeg,state:PredictiveMemoryState|None=None,*,return_state=False):
        if eeg.ndim!=4 or eeg.shape[-2:]!=(self.window_samples,self.eeg_channels):
            raise ValueError(f'expected [batch,windows,{self.window_samples},{self.eeg_channels}]')
        if eeg.shape[1]<1: raise ValueError('one or more windows required')
        b,w,t,c=eeg.shape
        hidden=self.extract(eeg.reshape(b*w,t,c).transpose(1,2)) # [b*w,width,t]
        avg=hidden.mean(dim=-1)
        amplitude=hidden.abs().amax(dim=-1)
        tokens=F.gelu(self.summary(torch.cat([avg,amplitude],dim=-1))).reshape(b,w,self.width)
        readout,state=self.memory(tokens,state,return_state=True)
        local=hidden.transpose(1,2).reshape(b,w,t,self.width)
        readout=readout[:,:,None,:].expand(b,w,t,self.width)
        gate=torch.sigmoid(self.fusion_gate(torch.cat([local,readout],dim=-1)))
        combined=local+gate*self.fusion_memory(readout)
        prediction=self.output(combined.reshape(b*w,t,self.width).transpose(1,2))
        prediction=prediction.transpose(1,2).reshape(b,w,t,1)
        return (prediction,state) if return_state else prediction


class _GRUTokenMemory(nn.Module):
    """Established recurrent control, with active parameters close to PI-MRAM."""
    def __init__(self, width:int, gru_hidden:int):
        super().__init__()
        self.gru=nn.GRU(width,gru_hidden,batch_first=True)
        self.project=nn.Linear(gru_hidden,width)

    def forward(self,tokens,state=None,*,return_state=False):
        hidden,state=self.gru(tokens,state)
        output=self.project(hidden)
        return (output,state) if return_state else output


class WindowGRUDecoder(WindowPredictiveDecoder):
    """EEG frontend, fusion, output held fixed; memory replaced by standard GRU.

    The matching target is trainable parameter count (tolerance reported).
    Equivalent throughput, mathematical receptive fields or train dynamics are
    NOT implied. State is a GRU tensor and must also reset at record boundaries.
    """
    def __init__(self, *, eeg_channels=64, width=32, window_samples=192, gru_hidden=104):
        super().__init__(eeg_channels=eeg_channels,width=width,
                         memory_variant='local',window_samples=window_samples)
        self.memory=_GRUTokenMemory(width,gru_hidden)
