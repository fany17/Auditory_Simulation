"""Isolated SS07 warm round02: original train/val only, server2203 only.

Whole global64 encoder forward precedes record-specific readout/state splits.
No round01 mutation; no test loader, downloads, new latent cache or retries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from pub_01_s03_ss07_baseline import native, cfg, seed_all, OLD
from pub_01_s03_ss07_finetune import RawWindows, Decoder
from pub_01_s03_ss07_incremental import NativeResidual, detach, summary
from pub_01_s03_ss07_train import NativeHead, objective

ROUND01=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_continuity_20261101')
ROOT=Path('/home/fanyu/auditory_simulation_m6a/pub_01/s03_incremental_warm_20261101')
SEEDS=(1024,2026,31415)
CONDITIONS=tuple(f'{mode}_{kind}' for mode in ('frozen','joint') for kind in ('p','current','past'))

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    with p.open('x') as f:json.dump(v,f,indent=2)
def update(p,v):
    p=Path(p);assert p.is_relative_to(ROOT)
    tmp=Path(str(p)+'.writing');tmp.write_text(json.dumps(v,indent=2));os.replace(tmp,p)
def disk_guard():
    free=shutil.disk_usage(ROOT).free
    assert free>=20*1024**3,'preserve all evidence: below20GiB reserve'
def precision():
    p=read(ROUND01/'INCREMENTAL_PROTOCOL.json')['precision']
    torch.backends.cuda.matmul.allow_tf32=p['TF32_matmul']
    torch.backends.cudnn.allow_tf32=p['TF32_cudnn']
    return p
def rng_snapshot():
    return {'cpu':torch.get_rng_state().clone(),'cuda':[v.clone() for v in torch.cuda.get_rng_state_all()]}
def rng_restore(s):
    torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state_all(s['cuda'])
def rng_equal(a,b):
    return torch.equal(a['cpu'],b['cpu']) and len(a['cuda'])==len(b['cuda']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))

def original_p(seed,device):
    """Replay original FT constructor ordering, including its temporary audio net."""
    assert seed in SEEDS
    seed_all(seed,device);torch.set_num_threads(2);precision()
    original=native().TransformerCLIP(cfg()).to(device)
    pre=torch.load(ROUND01/f'baseline/pretrain_seed{seed}/best_validation.pt',map_location=device,weights_only=True)
    assert pre['seed']==seed;original.load_state_dict(pre['model'])
    ref=Decoder(original.eeg).to(device)
    cp=torch.load(ROUND01/f'baseline/head_seed{seed}/best_validation.pt',map_location=device,weights_only=True)
    assert cp['seed']==seed;ref.head.load_state_dict(cp['head'])
    return ref

class WarmDecoder(nn.Module):
    def __init__(self,reference,condition):
        super().__init__();assert condition in CONDITIONS
        self.condition=condition;self.joint=condition.startswith('joint_');self.kind=condition.split('_')[1]
        self.encoder=reference.encoder
        # Constructor isolation preserves reference CPU and ALL-device CUDA streams.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            self.readout=NativeHead() if self.kind=='p' else NativeResidual('m4',16,30.,self.kind)
            self.readout.to(next(self.encoder.parameters()).device)
            self.readout.head.load_state_dict(reference.head.state_dict())
        self.encoder.requires_grad_(self.joint)

    def train(self,mode=True):
        super().train(mode)
        if not self.joint:self.encoder.eval()
        return self

    def latent_forward(self,z,spans,state=None,active=None):
        if self.kind=='p':return self.readout(z)[0],None,None
        parts=[]
        for a,b,row,local in spans:
            if active!=row['record']:state=None
            value=z[a:b]
            if self.kind=='current':context=self.readout.core.current_only(value);state=None
            else:context,_,state=self.readout.core.safe_history(summary(value),state,local,row['record'])
            coeff=self.readout.residual(context).reshape(-1,8,3)
            transformed=value
            for j,lag in enumerate((4,8,16)):
                shifted=nn.functional.pad(value[:,:,:-lag],(lag,0))
                transformed=transformed+.1*coeff[:,:,j,None]*(shifted-value)
            parts.append(transformed);active=row['record']
        # Same M4 equations and parameters as NativeResidual; concatenate before
        # the original head so zero residual uses the EXACT original B64 call.
        return self.readout.head(torch.cat(parts,0)),state,active

def make(seed,condition,device):return WarmDecoder(original_p(seed,device),condition)

class Data:
    def __init__(self,seed,phase,device,joint):
        assert seed in SEEDS and phase in ('train','val')
        self.raw=RawWindows(phase);self.phase=phase;self.rows=self.raw.rows;self.ids=self.raw.ids
        self.device=device;self.y=self.raw.y;self.cache=None
        if not joint:
            folder=ROUND01/f'baseline/head_seed{seed}';m=read(folder/'cache_manifest.json')
            assert m['only_train_val'] and m['encoder_sha256']==sha(ROUND01/f'baseline/pretrain_seed{seed}/best_validation.pt')
            self.cache=np.load(folder/f'{phase}_latent.npy',mmap_mode='r')
            assert self.cache.shape==(len(self),8,192)
            for r in self.rows:
                cached=m['records'][r['record']]
                assert cached['phase']==phase and cached['offset']==r['offset'] and cached['windows']==r['windows']
    def __len__(self):return len(self.raw)
    def spans(self,first,last):
        cursor=first
        while cursor<last:
            row=self.rows[self.ids[cursor]];end=min(last,row['offset']+row['windows'])
            yield cursor-first,end-first,row,cursor-row['offset'];cursor=end
    def batch(self,model,first,last,state=None,active=None):
        assert 0<=first<last<=len(self) and last-first<=64
        if model.joint:
            assert self.cache is None
            x,y=self.raw.batch(first,last,self.device)
            # Crucial: NEVER split the encoder at a record boundary.
            z=model.encoder(x).transpose(1,2)
        else:
            assert self.cache is not None and not model.encoder.training
            z=torch.from_numpy(np.array(self.cache[first:last],copy=True)).to(self.device)
            y=torch.from_numpy(np.array(self.y[first:last],copy=True)).to(self.device)
        p,state,active=model.latent_forward(z,list(self.spans(first,last)),state,active)
        return p,y,state,active

def macro_from_predictions(pred,data):
    assert data.phase=='val' and pred.shape==(30500,192)
    mask=read(ROUND01/'block_eligibility/val.json')
    assert mask['records']==[r['record'] for r in data.rows]
    eligible={(r['record'],r['start']):r for r in mask['blocks']};segments=[];window=np.hanning(192)
    for row in data.rows:
        n=row['common_samples'];start=row['offset'];total=np.zeros(n);weight=np.zeros(n)
        for i,value in enumerate(pred[start:start+row['windows']]):
            s=i*96;total[s:s+192]+=value*window;weight[s:s+192]+=window
        keep=weight>1e-8;total[keep]/=weight[keep]
        target=np.load(row['envelope_path'],mmap_mode='r').reshape(-1);assert len(target)>=n
        for s in range(0,n-3840+1,3840):
            k=keep[s:s+3840];e=eligible[(row['record'],s)];assert int(k.sum())==e['samples']
            if not e['eligible']:continue
            p=total[s:s+3840][k];t=np.asarray(target[s:s+3840])[k]
            assert np.isfinite(p).all() and np.isfinite(t).all() and np.std(p)>0 and np.std(t)>0
            segments.append({'record':row['record'],'subject':row['subject'],'story':row['story'],'start':s,'samples':len(p),'r':float(np.corrcoef(p,t)[0,1])})
    assert len(segments)==mask['eligible_blocks']
    people=[{'subject':s,'r':float(np.mean([r['r'] for r in segments if r['subject']==s]))} for s in sorted({r['subject'] for r in segments})]
    return {'macro_r':float(np.mean([r['r'] for r in people])),'subjects':len(people),'blocks':len(segments),'people':people,'segments':segments}

@torch.no_grad()
def validate(model,data,continuous=False,keep_predictions=False):
    assert data.phase=='val';model.eval();state=None;active=None;total=0.;pieces=[];resets=0
    for first in range(0,len(data),64):
        last=min(first+64,len(data));resets+=sum(local==0 for _,_,_,local in data.spans(first,last))
        p,y,state,active=data.batch(model,first,last,state,active)
        total+=float(objective(p,y))*(last-first)
        if continuous or keep_predictions:pieces.append(p[:,0].cpu().numpy())
    assert resets==49
    result={'negative_Pearson':total/len(data),'windows':len(data),'record_starts':resets}
    assert np.isfinite(result['negative_Pearson'])
    if continuous or keep_predictions:
        pred=np.concatenate(pieces)
        if continuous:result['continuous']=macro_from_predictions(pred,data)
        if keep_predictions:return result,pred
    return result

def optimizer(model):
    groups=[]
    if model.joint:groups.append({'params':list(model.encoder.parameters()),'lr':1e-5})
    groups.append({'params':list(model.readout.parameters()),'lr':.001})
    return torch.optim.Adam(groups,weight_decay=0.)

def finite_gradients(model):
    flags=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    return bool(torch.stack(flags).all())

def save_best(folder,model,seed,epoch,score):
    temporary=folder/'best_validation.pt.writing'
    torch.save({'model':model.state_dict(),'seed':seed,'condition':model.condition,'epoch':epoch,'validation_loss':score},temporary)
    os.replace(temporary,folder/'best_validation.pt')

def train(condition,seed,device):
    protocol=read(ROOT/'WARM_PROTOCOL.json');assert protocol['frozen'] and protocol['test_operations'] is False
    assert condition in protocol['new_fit_conditions'] and seed in protocol['seeds']
    for p,d in protocol['sources'].items():assert sha(p)==d,p
    assert read(ROOT/'ENGINEERING_COMPLETED.json')['passed']
    disk_guard();model=make(seed,condition,device)
    folder=ROOT/f'fits/{condition}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'resolved_config.json',{'seed':seed,'condition':condition,'protocol_sha256':sha(ROOT/'WARM_PROTOCOL.json'),
        'incumbent_epoch_minus1_eligible':True,'fixed_epochs':5,'test_read':False,'initial_encoder_sha256':sha(ROUND01/f'baseline/pretrain_seed{seed}/best_validation.pt'),
        'initial_head_sha256':sha(ROUND01/f'baseline/head_seed{seed}/best_validation.pt'),'precision':precision(),
        'optimizer':'freshAdam,EEG1e-5/readout1e-3,no wd/no clip/no shuffle','global64_encoder_before_record_splits':True})
    train_data=Data(seed,'train',device,model.joint);val=Data(seed,'val',device,model.joint);started=time.time()
    incumbent=validate(model,val,continuous=True);dump(folder/'INCUMBENT_VALIDATION.json',incumbent)
    best=incumbent['negative_Pearson'];selected=-1;updates=0;history=[]
    save_best(folder,model,seed,-1,best);opt=optimizer(model)
    for epoch in range(5):
        model.train();state=None;active=None;total=0.;count=0;record_starts=0
        for first in range(0,len(train_data),64):
            last=min(first+64,len(train_data));record_starts+=sum(local==0 for _,_,_,local in train_data.spans(first,last))
            opt.zero_grad(set_to_none=True)
            p,y,state,active=train_data.batch(model,first,last,state,active)
            loss=objective(p,y);assert torch.isfinite(loss);loss.backward();assert finite_gradients(model)
            opt.step();state=detach(state);updates+=1;total+=float(loss.detach())*(last-first);count+=last-first
            if updates%200==0:update(folder/'status.json',{'epoch':epoch,'updates':updates,'windows_epoch':count,'seconds':time.time()-started})
        assert count==252454 and record_starts==426 and updates==(epoch+1)*3945
        v=validate(model,val);score=v['negative_Pearson']
        if score<best:best=score;selected=epoch;save_best(folder,model,seed,epoch,score)
        history.append({'epoch':epoch,'train_negative_Pearson':total/count,'validation_negative_Pearson':score,
            'selected_epoch_so_far':selected,'updates':updates,'record_starts':record_starts,'seconds':time.time()-started})
        update(folder/'history.json',history);print(json.dumps({'seed':seed,'condition':condition,**history[-1]}),flush=True);disk_guard()
    cp=torch.load(folder/'best_validation.pt',map_location=device,weights_only=True)
    assert cp['epoch']==selected and cp['seed']==seed and cp['condition']==condition
    model.load_state_dict(cp['model']);selected_val=validate(model,val,continuous=True)
    assert abs(selected_val['negative_Pearson']-best)<=1e-7
    dump(folder/'SELECTED_VALIDATION.json',selected_val)
    dump(folder/'completed.json',{'condition':condition,'seed':seed,'epochs':5,'updates':updates,'selected_epoch':selected,
        'best_validation_negative_Pearson':best,'incumbent_negative_Pearson':incumbent['negative_Pearson'],
        'selected_val60s_macro_r':selected_val['continuous']['macro_r'],'incumbent_val60s_macro_r':incumbent['continuous']['macro_r'],
        'supervised_window_presentations':1262270,'target_time_sample_presentations':1262270*192,
        'validation_full_passes':7,'validation_continuous_summaries':2,'encoder_train_mode':model.joint,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'seconds':time.time()-started,'GPU_peak_bytes':torch.cuda.max_memory_allocated(device),
        'checkpoint_sha256':sha(folder/'best_validation.pt'),'protocol_sha256':sha(ROOT/'WARM_PROTOCOL.json'),'test_read':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=CONDITIONS);p.add_argument('--seed',required=True,type=int,choices=SEEDS);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    train(a.condition,a.seed,a.device)
