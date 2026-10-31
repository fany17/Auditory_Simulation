"""Stronger native-P control: full encoder/head supervised fine tuning.

Only original train/val. No new latent cache and no auxiliary objective.
Five seeds/budget/rates fixed by INCREMENTAL_PROTOCOL; separate phase guard.
"""
import argparse
import json
import time
import numpy as np
import torch
from torch import nn
from pub_01_s03_ss07_baseline import ROOT,OLD,PREPARED,read,dump,update,sha,disk_guard,native,cfg,seed_all
from pub_01_s03_ss07_train import objective


class RawWindows:
    def __init__(self,phase):
        assert phase in ('train','val');self.phase=phase
        self.rows=[];self.ids=[];offset=0;self.cached_record=None;self.raw=None
        for r in read(OLD/'envelope_readout_seed1024/manifest_frozen.json')['records']:
            if r['phase']!=phase:continue
            n=(r['common_samples']-192)//96+1
            assert r['phase_offset']==offset and r['readout_windows']==n
            self.rows.append(dict(r,offset=offset,windows=n));self.ids.extend([len(self.rows)-1]*n);offset+=n
        self.y=np.load(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy',mmap_mode='r')
        assert len(self.y)==len(self.ids)==({'train':252454,'val':30500}[phase])
        scaling=np.load(PREPARED/'train_only_normalization.npz');self.mean,self.std=scaling['mean'],scaling['std']

    def __len__(self):return len(self.ids)

    def batch(self,first,last,device):
        assert 0<=first<last<=len(self);pieces=[];cursor=first
        while cursor<last:
            r=self.rows[self.ids[cursor]];end=min(last,r['offset']+r['windows'])
            if self.cached_record!=r['record']:
                raw=np.load(r['eeg_path'],mmap_mode='r');self.raw=raw.T if raw.shape[0]==64 else raw
                self.cached_record=r['record'];assert self.raw.shape[1]==64
            starts=np.arange(cursor-r['offset'],end-r['offset'])*96
            pieces.extend([(np.asarray(self.raw[s:s+192],np.float32)-self.mean)/self.std for s in starts])
            cursor=end
        x=torch.from_numpy(np.stack(pieces).astype(np.float32)).to(device)
        y=torch.from_numpy(np.array(self.y[first:last],copy=True)).to(device)
        assert x.shape==(last-first,192,64) and y.shape==(last-first,1,192)
        return x,y


class Decoder(nn.Module):
    def __init__(self,encoder):
        super().__init__();self.encoder=encoder
        self.head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
    def forward(self,x):return self.head(self.encoder(x).transpose(1,2))


@torch.no_grad()
def validate(model,data,device):
    model.eval();total=0.
    for first in range(0,len(data),64):
        last=min(len(data),first+64);x,y=data.batch(first,last,device)
        total+=float(objective(model(x),y))*(last-first)
    result=total/len(data);assert np.isfinite(result);return result


def main(seed,device):
    protocol=read(ROOT/'FINETUNE_PROTOCOL.json');assert protocol['frozen'] and seed in protocol['seeds']
    for path,digest in protocol['sources'].items():assert sha(path)==digest
    p=read(ROOT/'INCREMENTAL_PROTOCOL.json');assert protocol['incremental_protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
    budget=p['strong_control'];assert budget['max_epochs']==5 and budget['patience']==3
    pre=ROOT/f'baseline/pretrain_seed{seed}';head=ROOT/f'baseline/head_seed{seed}'
    assert read(pre/'completed.json')['seed']==read(head/'completed.json')['seed']==seed
    disk_guard();seed_all(seed,device);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=p['precision']['TF32_matmul'];torch.backends.cudnn.allow_tf32=p['precision']['TF32_cudnn']
    mod=native();original=mod.TransformerCLIP(cfg()).to(device)
    checkpoint=torch.load(pre/'best_validation.pt',map_location=device,weights_only=True);assert checkpoint['seed']==seed
    original.load_state_dict(checkpoint['model']);model=Decoder(original.eeg).to(device)
    head_checkpoint=torch.load(head/'best_validation.pt',map_location=device,weights_only=True)
    assert head_checkpoint['seed']==seed;model.head.load_state_dict(head_checkpoint['head'])
    del original,checkpoint
    folder=ROOT/f'finetune/native_P_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    dump(folder/'resolved_config.json',{'seed':seed,'budget':budget,'initial_encoder_sha256':sha(pre/'best_validation.pt'),
        'initial_head_sha256':sha(head/'best_validation.pt'),'incumbent_epoch_minus1_eligible':True,
        'optimizer':'Adam, EEG1e-5/head1e-3, no wd/no clip/no shuffle','encoder_train_mode_during_training':True,
        'objective':'original global64 3s negativePearson eps1e-6','test_read':False})
    train=RawWindows('train');val=RawWindows('val');started=time.time()
    best=validate(model,val,device);initial=best;selected=-1;stale=0;updates=0;history=[]
    reference=read(head/'completed.json')['best_validation'];difference=abs(best-reference)
    tolerance=protocol['incumbent_loss_absolute_tolerance'];assert tolerance==1e-4
    dump(folder/'INCUMBENT_EQUIVALENCE.json',{'raw_reencoded_B64_validation_loss':best,
        'original_cached_B512_validation_loss':reference,'absolute_difference':difference,
        'prespecified_absolute_tolerance':tolerance,'passed':difference<=tolerance,
        'full_validation_windows':30500,'optimizer_steps_before_check':0,'test_read':False})
    assert difference<=tolerance,'native P incumbent mismatch; investigate paths/norm/grid/source before any training'
    torch.save({'model':model.state_dict(),'seed':seed,'epoch':-1,'validation_loss':best},folder/'best_validation.pt')
    opt=torch.optim.Adam([{'params':model.encoder.parameters(),'lr':budget['EEG_lr']},
        {'params':model.head.parameters(),'lr':budget['head_lr']}],weight_decay=0.)
    for epoch in range(budget['max_epochs']):
        model.train();total=0.;count=0
        for first in range(0,len(train),64):
            last=min(first+64,len(train));x,y=train.batch(first,last,device);opt.zero_grad(set_to_none=True)
            loss=objective(model(x),y);assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            opt.step();updates+=1;total+=float(loss.detach())*(last-first);count+=last-first
            if updates%100==0:update(folder/'status.json',{'epoch':epoch,'updates':updates,'samples_epoch':count,'seconds':time.time()-started})
        assert count==252454 and updates==(epoch+1)*3945
        score=validate(model,val,device);stale=0 if score<best else stale+1
        if score<best:
            best=score;selected=epoch
            torch.save({'model':model.state_dict(),'seed':seed,'epoch':epoch,'validation_loss':best},folder/'best_validation.pt')
        history.append({'epoch':epoch,'train_negative_Pearson':total/count,'validation_negative_Pearson':score,
            'updates':updates,'stale':stale,'seconds':time.time()-started})
        update(folder/'history.json',history);print(json.dumps({'seed':seed,**history[-1]}),flush=True);disk_guard()
        if stale>=budget['patience']:break
    dump(folder/'completed.json',{'seed':seed,'epochs':len(history),'updates':updates,'selected_epoch':selected,
        'initial_validation_negative_Pearson':initial,'best_validation_negative_Pearson':best,
        'target_window_presentations':252454*len(history),'validation_passes_including_incumbent':len(history)+1,
        'seconds':time.time()-started,'peak_GPU_bytes':torch.cuda.max_memory_allocated(device),
        'registered_trainable_parameters':sum(v.numel() for v in model.parameters()),
        'all_extra_costs_separate':True,'test_read':False,'protocol_sha256':sha(ROOT/'FINETUNE_PROTOCOL.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',required=True,type=int);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    main(a.seed,a.device)
