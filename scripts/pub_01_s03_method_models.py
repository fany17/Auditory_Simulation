"""Functional reimplementations of two specified published baseline methods.

Architecture/function inspection: HDPS commit42a5e1e and Transformer-CLIP
commit8be963ab. Author source is not imported, copied or redistributed here.
This is not a clean-room claim. Training/data deviations live in frozen config.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeNormConv(nn.Module):
    def __init__(self, channels, kernel, time, dropout):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels, kernel, padding="same")
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm((channels, time))

    def forward(self, sequence):
        return F.leaky_relu(self.norm(self.drop(self.filter(sequence))))


class AttentionBlock(nn.Module):
    def __init__(self, width, heads, dropout=0.5):
        super().__init__()
        self.heads, self.width = heads, width
        self.before_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.mix = nn.Linear(width, width)
        self.attention_drop = nn.Dropout(dropout)
        self.residual_drop = nn.Dropout(dropout)
        self.before_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(width * 4, width))

    def forward(self, sequence):
        batch, time, width = sequence.shape
        projected = self.qkv(self.before_attention(sequence)).reshape(batch, time, 3, self.heads, width // self.heads)
        q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
        # Published source scales by sqrt(full embedding size), not head width.
        attention = self.attention_drop(torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.width), dim=-1))
        message = (attention @ v).transpose(1, 2).reshape(batch, time, width)
        sequence = sequence + self.residual_drop(self.mix(message))
        return sequence + self.residual_drop(self.ff(self.before_ff(sequence)))


class AuditoryTransformer(nn.Module):
    def __init__(self, cfg, time=192):
        super().__init__()
        width = cfg["width"]
        self.spatial = nn.Conv1d(64, width, 1)
        self.temporal = nn.ModuleList([TimeNormConv(width, cfg["conv_kernel"], time, cfg["conv_dropout"]) for _ in range(cfg["depth"])])
        self.attend = nn.ModuleList([AttentionBlock(width, cfg["heads"], cfg["attention_dropout"]) for _ in range(cfg["depth"])])
        self.output = nn.Linear(width, cfg["latent_dim"])

    def forward(self, eeg, subject=None):
        initial = self.spatial(eeg.transpose(1, 2))
        state = initial
        for index, (conv, attention) in enumerate(zip(self.temporal, self.attend)):
            state = conv(state + initial).transpose(1, 2)
            if index < len(self.temporal) - 1:
                state = state + initial.transpose(1, 2)
            state = attention(state).transpose(1, 2)
        return self.output(state.transpose(1, 2))


class SpeechLSTM(nn.Module):
    def __init__(self, cfg, time=192):
        super().__init__()
        self.map = nn.Conv1d(1024, 64, 1)
        self.conv = TimeNormConv(64, cfg["speech_conv_kernel"], time, cfg["speech_dropout"])
        self.first = nn.LSTM(64, cfg["speech_lstm_hidden"], batch_first=True, bidirectional=True)
        self.second = nn.LSTM(cfg["speech_lstm_hidden"] * 2, cfg["latent_dim"] // 2, batch_first=True, bidirectional=True)

    def forward(self, audio):
        sequence = self.conv(self.map(audio.transpose(1, 2))).transpose(1, 2)
        sequence, _ = self.first(sequence)
        sequence, _ = self.second(sequence)
        return sequence


class ConvConcatStage(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg["spatial_hidden"]
        base, merged = width * 3, width * 4
        self.spatial_steps = nn.ModuleList([nn.Conv1d(base if i == 0 else merged, width, 1) for i in range(4)])
        self.temporal_steps = nn.ModuleList([nn.Conv1d(merged, merged, cfg["temporal_kernel"], groups=merged, padding="same") for _ in range(4)])
        self.spatial_norm = nn.LayerNorm(width)
        self.merged_norm = nn.LayerNorm(merged)
        self.read = nn.Conv1d(merged, cfg["stack_output"], cfg["temporal_kernel"], padding="same")
        self.read_norm = nn.LayerNorm(cfg["stack_output"])

    def forward(self, initial):
        state = initial
        for spatial, temporal in zip(self.spatial_steps, self.temporal_steps):
            spatial_value = F.leaky_relu(self.spatial_norm(spatial(state.transpose(1, 2)).transpose(1, 2)))
            state = torch.cat((initial, spatial_value), dim=-1)
            state = F.leaky_relu(self.merged_norm(temporal(state.transpose(1, 2)).transpose(1, 2)))
        return F.leaky_relu(self.read_norm(self.read(state.transpose(1, 2)).transpose(1, 2)))


class AuditoryConvConcat(nn.Module):
    def __init__(self, cfg, subjects):
        super().__init__()
        width = cfg["spatial_hidden"]
        self.subject_input = nn.Linear(64, cfg["subject_dim"])
        self.subject_operator = nn.Parameter(torch.randn(subjects, cfg["subject_dim"], cfg["subject_dim"]))
        self.subject_output = nn.Linear(cfg["subject_dim"], width)
        self.stage = ConvConcatStage(cfg)
        self.reduce = nn.Linear(cfg["stack_output"], width)
        self.smooth = nn.Conv1d(width, width, cfg["output_kernel"], padding="same")
        self.norm = nn.LayerNorm(width)
        self.gate = nn.Sequential(nn.Linear(width, width), nn.LeakyReLU(), nn.Linear(width, width))
        self.output = nn.Linear(width, cfg["target_dim"])
        self.repetitions = cfg["blocks"]

    def forward(self, eeg, subject):
        base = self.subject_output(torch.bmm(self.subject_input(eeg), self.subject_operator[subject]))
        state, gated = torch.zeros_like(base), torch.zeros_like(base)
        for _ in range(self.repetitions):
            state = self.reduce(self.stage(torch.cat((base, state, gated), dim=-1)))
            state = F.leaky_relu(self.norm(self.smooth(state.transpose(1, 2)).transpose(1, 2)))
            gated = self.gate(state) * state
        return self.output(state)


def temporal_pearson_matrix(predicted, candidates):
    # Per output dimension: Pearson across time, then dimension average.
    predicted = F.normalize((predicted - predicted.mean(dim=1, keepdim=True)).transpose(1, 2), dim=-1)
    candidates = F.normalize((candidates - candidates.mean(dim=1, keepdim=True)).transpose(1, 2), dim=-1)
    return torch.einsum("bct,nct->bn", predicted, candidates) / predicted.shape[1]


class MethodPair(nn.Module):
    def __init__(self, method, cfg, subjects):
        super().__init__()
        self.method = method
        if method == "hdps":
            self.eeg = AuditoryConvConcat(cfg["hdps"], subjects)
            dim = cfg["hdps"]["target_dim"]
            self.audio = nn.Sequential(nn.Linear(dim, dim))
            self.fusion = nn.Conv1d(dim, dim, 1)
            self.temperature = cfg["hdps"]["temperature"]
        else:
            self.eeg = AuditoryTransformer(cfg["transformer"])
            self.audio = SpeechLSTM(cfg["transformer"])
            self.logit_parameter = nn.Parameter(torch.tensor(cfg["transformer"]["initial_logit_parameter"]))

    def encode_audio(self, audio):
        result = self.audio(audio)
        return F.gelu(self.fusion(result.transpose(1, 2))).transpose(1, 2) if self.method == "hdps" else result

    def similarity(self, predicted, candidates):
        if self.method == "hdps":
            return temporal_pearson_matrix(predicted, candidates)
        return F.normalize(predicted.flatten(1), dim=-1) @ F.normalize(candidates.flatten(1), dim=-1).T

    def loss(self, eeg, audio, subject):
        scores = self.similarity(self.eeg(eeg, subject), self.encode_audio(audio))
        labels = torch.arange(len(eeg), device=eeg.device)
        if self.method == "hdps":
            return F.cross_entropy(scores / self.temperature, labels)
        scores = scores * self.logit_parameter.exp()
        return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
