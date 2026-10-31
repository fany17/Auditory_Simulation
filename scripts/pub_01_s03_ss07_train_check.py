"""REAL TRAIN EEG engineering; random native encoder, discarded weights.

No historical EEG checkpoint/latent or test signal/score is loaded. This is
not a pilot ranking or an SS07 performance result. Includes actual two-call
TBPTT, mixed-record global64, prior-only path and future/current masks.
"""
import json
from pathlib import Path
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT,OLD,PREPARED,native,cfg,read,dump
from pub_01_s03_ss07_incremental import NativeResidual,detach,summary
from pub_01_s03_ss07_train import NativeHead,LatentData,objective,forward_global_batch,validation,fixed_block_mask


def metadata(rows):
    result=[];offset=0;ids=[]
    for i,r in enumerate(rows):
        count=(r['common_samples']-192)//96+1
        result.append(dict(r,offset=offset,windows=count));ids.extend([i]*count);offset+=count
    return result,ids


class Probe:
    spans=LatentData.spans
    def __init__(self,rows,x=None,y=None):
        self.rows,self.ids=metadata(rows);self.x=x if x is not None else torch.empty(len(self.ids),0)
        self.y=y;self.reset_count=0
        if x is not None:self.teacher=summary(x)


def main():
    torch.set_num_threads(2);torch.manual_seed(7711);manifest=read(OLD/'envelope_readout_seed1024/manifest_frozen.json')
    trainrows=[r for r in manifest['records'] if r['phase']=='train'];allmeta=Probe(trainrows)
    steps=0;visits=0;crossing=0
    for first in range(0,len(allmeta.ids),64):
        pieces=list(allmeta.spans(first,min(first+64,len(allmeta.ids))))
        assert sum(b-a for a,b,_,_ in pieces)==min(64,len(allmeta.ids)-first)
        crossing+=len(pieces)>1;visits+=sum(b-a for a,b,_,_ in pieces);steps+=1
    assert steps==3945 and visits==252454 and len(allmeta.rows)==426 and crossing>0
    masks={phase:fixed_block_mask(phase,[r for r in manifest['records'] if r['phase']==phase]) for phase in ('train','val')}
    # Fresh random FULL native encoder, actual normalized TRAIN EEG, original
    # overlapping zero-lag grid. Engineering weights never enter formal runs.
    mod=native();encoder=mod.TransformerCLIP(cfg()).eeg.cuda().eval();stats=np.load(PREPARED/'train_only_normalization.npz')
    selected=trainrows[:2];zs=[];ys=[]
    with torch.no_grad():
        for row in selected:
            raw=np.load(row['eeg_path'],mmap_mode='r');x=raw.T if raw.shape[0]==64 else raw
            target=np.load(row['envelope_path'],mmap_mode='r').reshape(-1);count=(row['common_samples']-192)//96+1
            for first in range(0,count,128):
                starts=np.arange(first,min(count,first+128))*96
                xx=np.stack([(np.asarray(x[s:s+192],np.float32)-stats['mean'])/stats['std'] for s in starts]).astype(np.float32)
                zs.append(encoder(torch.from_numpy(xx).cuda()).transpose(1,2).cpu())
                ys.append(torch.from_numpy(np.stack([target[s:s+192] for s in starts])[:,None].astype(np.float32)))
    del encoder;z=torch.cat(zs);y=torch.cat(ys);data=Probe(selected,z,y)
    # Two ACTUAL forward calls, later loss must reach first-call latent inputs.
    model=NativeResidual('m2',16).eval()
    with torch.no_grad():model.residual.weight.normal_(0,.04)
    a=z[:16].clone().requires_grad_();b=z[16:32].clone().requires_grad_()
    _,state,_=model(a,record_key='A');p2,_,_=model(b,state,16,'A')
    detached=detach(state);q2,_,_=model(b,detached,16,'A')
    no_detach_grad=torch.autograd.grad(p2.square().mean(),a,retain_graph=True)[0].abs().sum().item()
    detached_grad=torch.autograd.grad(q2.square().mean(),a,allow_unused=True)[0]
    assert no_detach_grad>0 and detached_grad is None and torch.equal(p2,q2)
    try:model(b,state,16,'B')
    except AssertionError:pass
    else:raise AssertionError('cross-record state not rejected')
    # Prior-only validation path must match direct manual prior readout, notfull.
    prior=NativeResidual('m5',16).eval();part=z[:64]
    with torch.no_grad():
        p,_,aux=prior(part,record_key='A',mode='prior_only')
        manual=prior.head(aux['latent_prior']);assert torch.equal(p,manual)
        expected=0.;state=None;active=None
        for first in range(0,len(z),64):
            last=min(first+64,len(z));pred,state,active,_,_=forward_global_batch(prior,data,first,last,state,active,mode='prior_only')
            expected+=float(objective(pred,y[first:last]))*(last-first)
        expected/=len(z)
        prior_val=validation(prior,data,mode='prior_only');assert prior_val==expected
    # Native head segmented forward/one update agrees with direct nativebatch,
    # including the real record boundary straddling a global64 batch.
    boundary=data.rows[0]['windows'];first=(boundary//64)*64;last=first+64
    assert first<boundary<last
    torch.manual_seed(1024);m1=NativeHead();torch.manual_seed(1024);m2=NativeHead()
    opt1=torch.optim.Adam(m1.parameters(),lr=.001);opt2=torch.optim.Adam(m2.parameters(),lr=.001)
    out,_,_,_,_=forward_global_batch(m1,data,first,last)
    direct=m2.head(z[first:last]);forward_error=float((out-direct).abs().max())
    objective(out,y[first:last]).backward();objective(direct,y[first:last]).backward();opt1.step();opt2.step()
    weight_error=max(float((a-b).abs().max()) for a,b in zip(m1.parameters(),m2.parameters()))
    assert forward_error<1e-5 and weight_error<1e-5
    # Same current/future teacher mask, even at the actual recordingtail.
    m3=NativeResidual('m3',16);scaling=(data.teacher.mean(0),data.teacher.std(0,unbiased=False).clamp_min(1e-5))
    counts={}
    for mode in ('future','current'):
        counts[mode]=0;state=None;active=None
        for first in range(0,len(z),64):
            _,state,active,aux,n=forward_global_batch(m3,data,first,min(first+64,len(z)),state,active,scaling,mode)
            counts[mode]+=n;state=detach(state)
        assert counts[mode]==sum(r['windows']-2 for r in data.rows)
    assert counts['future']==counts['current']
    # Repeated validation starts both recording/phase passes from freshstate.
    repeat=validation(prior,data,mode='prior_only');assert repeat==prior_val
    result={'passed':True,'kind':'REAL_TRAIN_ENGINEERING_RANDOM_NATIVE_ENCODER; no performance/config selection',
        'global_train_windows':visits,'updates_per_full_epoch':steps,'mixed_record_batches':crossing,
        'mixed_record_once_loss_and_once_optimizer_forward_max':forward_error,'mixed_record_optimizer_weight_max':weight_error,
        'two_forward_no_detach_first_input_gradient':no_detach_grad,'explicit_detach_first_input_gradient':None,
        'detach_preserves_numeric_state_prediction_exact':True,'cross_record_state_rejected':True,
        'record_phase_epoch_state_restart_repeated_val_identical':True,'prior_only_validation_equals_manual':True,
        'same_teacher_valid_counts':counts,'teacher_scaling':'one common TRAIN-only mean/std for current/future',
        'actual_records':[r['record'] for r in selected],'old_checkpoint_or_latent_loaded':False,
        'test_payload_or_scores':False,'all_probe_weights_discarded':True,
        'fixed_prediction_independent_masks':{p:{'eligible':d['eligible_blocks'],'excluded':len(d['excluded_blocks'])} for p,d in masks.items()}}
    dump(ROOT/'TRAINER_ENGINEERING_CHECK.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
