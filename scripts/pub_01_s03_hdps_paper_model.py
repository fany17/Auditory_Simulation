"""Functional implementation of HDPS paper equations1–8 and public details.

Wang et al., IEEE TNSRE34:2349–2360, DOI10.1109/TNSRE.2026.3688564,
CC BY4.0. Public architecture was inspected; not a clean-room claim.
No author model module is imported or redistributed.
"""
import torch
from torch import nn
from pub_01_s03_method_models import AuditoryConvConcat


def similarity(predicted, targets):
    """Per-channel temporal Pearson, averaged over channels; float32."""
    centered_p = (predicted - predicted.mean(1, keepdim=True)).transpose(0, 2)
    centered_t = (targets - targets.mean(1, keepdim=True)).transpose(0, 2)
    # transpose(0,2) gives channels,time,batch; put batch before time.
    centered_p = centered_p.transpose(1, 2)
    centered_t = centered_t.transpose(1, 2)
    divisor = predicted.shape[1] - 1
    covariance = (centered_p @ centered_t.transpose(1, 2)) / divisor
    sd_p = torch.sqrt((centered_p.square().sum(2, keepdim=True)) / divisor)
    sd_t = torch.sqrt((centered_t.square().sum(2, keepdim=True)) / divisor)
    denominator = sd_p @ sd_t.transpose(1, 2)
    return (covariance / (denominator + 1e-8)).mean(0)


class PaperHDPS(nn.Module):
    def __init__(self, cfg, subjects):
        super().__init__()
        self.encoder = AuditoryConvConcat(cfg['model'], subjects)
        # Paper equation1: adapter then final linear output; no extra GELU.
        width = cfg['model']['target_dim']
        self.feature = nn.Sequential(nn.Linear(width, width), nn.Linear(width, width))

    def forward(self, eeg, target, subject):
        return self.encoder(eeg, subject), self.feature(target)


def normalize(eeg):
    mean = eeg.mean(1, keepdims=True)
    sd = eeg.std(1, keepdims=True)
    return ((eeg - mean) / (sd + 1e-8)).clip(-20, 20)
