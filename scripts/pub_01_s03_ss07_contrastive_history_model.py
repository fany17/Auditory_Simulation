"""D26 shared trainable preprojection encoder and original supervised head."""
import importlib
import sys

import torch
from torch import nn
from torch.nn import functional as F

from pub_01_s03_ss07_contrastive_history_data import BASE, D18, D24, HOME, OLD, read


def dependencies():
    paths = (OLD/"input",D18/"input",D24/"input_v04",BASE/"input",
             HOME/"s03_conditional_history_probe_20261101/input",
             HOME/"s03_current_readout_internal_audit_20261101/input")
    for path in reversed(paths):
        sys.path.insert(0,str(path))
    return tuple(importlib.import_module(name) for name in (
        "pub_01_s03_transformer_train_v2","pub_01_s03_ss07_preprojection_model",
        "pub_01_s03_ss07_finetune","pub_01_s03_ss07_crossfit_upstream",
        "pub_01_s03_ss07_conditional_probe","pub_01_s03_ss07_generalization_core",
    ))


native, original_memory, native_ft, ridge, ridge_helpers, metrics = dependencies()


class Encoder(nn.Module):
    def __init__(self, backbone, seed):
        super().__init__()
        self.backbone = backbone
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+18000)
            self.memory = original_memory.TokenMemory()

    def query(self, x):
        return original_memory.prefix(self.backbone,x)

    def project(self, query, x, mask, past, route, inspect=None):
        if past is None and route == "H":
            if mask.any():
                raise ValueError("Valid historical rows require raw memory")
            return self.backbone.output(query)
        if not mask.any():
            return self.backbone.output(query)
        indices = torch.nonzero(mask,as_tuple=False).flatten()
        if route == "C":
            x_valid = x.index_select(0,indices)
            slots = (x_valid,x_valid)
        elif route == "H":
            slots = past
        else:
            raise ValueError("Unknown memory route")
        first, second = (original_memory.prefix(self.backbone,t) for t in slots)
        if inspect is not None:
            for name,value in (("query",query),("slot1",first),("slot2",second)):
                value.retain_grad()
                inspect[name] = value
        correction = self.memory(query.index_select(0,indices),torch.cat((first,second),1))
        result = query + torch.zeros_like(query).index_copy(0,indices,correction)
        return self.backbone.output(result)

    def forward(self, packed, route, inspect=None):
        x, mask, past = packed
        return self.project(self.query(x),x,mask,past,route,inspect)


class CL(nn.Module):
    def __init__(self, config, seed, route):
        super().__init__()
        upstream = native.TransformerCLIP(config)
        self.eeg = Encoder(upstream.eeg,seed)
        self.audio = upstream.audio
        self.log_scale = upstream.log_scale
        self.route = route
        self.inspect = None

    def forward(self, packed, audio):
        eeg = self.eeg(packed,self.route,self.inspect)
        speech = self.audio(audio)
        if self.inspect is not None:
            speech.retain_grad()
            self.inspect["audio"] = speech
        return F.normalize(eeg.flatten(1),dim=-1),F.normalize(speech.flatten(1),dim=-1)


def make_head(seed):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Sequential(nn.Conv1d(8,1,32,padding="same"),nn.LeakyReLU(.01))


class Decoder(nn.Module):
    def __init__(self, encoder, head, route):
        super().__init__()
        self.encoder,self.head,self.route = encoder,head,route

    def forward(self, packed):
        return self.head(self.encoder(packed,self.route).transpose(1,2))
