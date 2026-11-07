"""Incremental head trainer: full native latent caches, same global64 batches.

Train/val only. A record boundary splits the forward, NEVER creates an extra
optimizer step. No pilot/final scientific training starts without frozen matrix.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from pub_01_s03_ss07_baseline import ROOT,OLD,read,dump,update,sha,disk_guard
from pub_01_s03_ss07_incremental import NativeResidual,StaticCapacityResidual,StaticAlignment,summary,detach


class NativeHead(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
    def forward(self,z,*args,**kwargs):return self.head(z),None,{}


class LatentData:
    def __init__(self,seed,phase,device):
        assert phase in ('train','val')
        folder=ROOT/f'baseline/head_seed{seed}';m=read(folder/'cache_manifest.json')
        assert m['only_train_val'] and m['encoder_sha256']==sha(ROOT/f'baseline/pretrain_seed{seed}/best_validation.pt')
        x=np.load(folder/f'{phase}_latent.npy',mmap_mode='r')
        y=np.load(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy',mmap_mode='r')
        self.x=torch.from_numpy(np.array(x,copy=True)).to(device);self.y=torch.from_numpy(np.array(y,copy=True)).to(device)
        source={r['record']:r for r in read(OLD/'envelope_readout_seed1024/manifest_frozen.json')['records'] if r['phase']==phase}
        self.rows=sorted([dict(source[k],**v) for k,v in m['records'].items() if v['phase']==phase],key=lambda r:r['offset'])
        self.ids=[]
        for i,r in enumerate(self.rows):
            assert len(self.ids)==r['offset'];self.ids.extend([i]*r['windows'])
        assert len(self.x)==len(self.y)==len(self.ids)
        self.teacher=summary(self.x)
        self.eligible_teacher_samples=sum(max(0,r['windows']-2) for r in self.rows)
        self.fixed_mask=fixed_block_mask(phase,self.rows)

    def spans(self,first,last):
        while first<last:
            r=self.rows[self.ids[first]];end=min(last,r['offset']+r['windows'])
            yield first,end,r,first-r['offset'];first=end


def fixed_block_mask(phase,rows):
    """Frozen target/geometry-only eligibility; never conditioned on model."""
    assert phase in ('train','val')
    folder=ROOT/'block_eligibility';folder.mkdir(exist_ok=True);path=folder/f'{phase}.json'
    if path.exists():
        old=read(path);assert old['records']==[r['record'] for r in rows];return old
    blocks=[];window=np.hanning(192)
    for row in rows:
        n=row['common_samples'];weight=np.zeros(n)
        count=(n-192)//96+1
        for i in range(count):weight[i*96:i*96+192]+=window
        covered=weight>1e-8;target=np.load(row['envelope_path'],mmap_mode='r').reshape(-1);assert len(target)>=n
        for s in range(0,n-3840+1,3840):
            keep=covered[s:s+3840];truth=np.asarray(target[s:s+3840])[keep]
            reason='insufficient_overlap_support' if len(truth)<3800 else 'nonfinite_target' if not np.isfinite(truth).all() else 'constant_target' if np.std(truth)==0 else None
            blocks.append({'record':row['record'],'start':s,'samples':len(truth),'eligible':reason is None,'reason':reason})
    result={'phase':phase,'records':[r['record'] for r in rows],'blocks':blocks,
        'eligible_blocks':sum(r['eligible'] for r in blocks),'excluded_blocks':[r for r in blocks if not r['eligible']],
        'prediction_independent':True,'test_read':False}
    temporary=folder/f'{phase}.{os.getpid()}.writing';temporary.write_text(json.dumps(result,indent=2))
    try:os.link(temporary,path)
    except FileExistsError:assert read(path)==result
    # No permanent deletion: temporary tiny same-content file retained.
    return read(path)


def make(config,seed):
    torch.manual_seed(seed);family=config['family'];width=config.get('width',16)
    if family=='native':return NativeHead()
    if family=='static':
        # Obtain target active inference count without consuming native head RNG.
        ref=NativeResidual(config.get('reference_family','m2'),width);n=ref.inference_residual_parameters()
        torch.manual_seed(seed);return StaticCapacityResidual(n)
    if family=='static_alignment':return StaticAlignment()
    return NativeResidual(family,width,config.get('tau',30.),config.get('conditioning','past'))


def objective(pred,y):
    pc=pred-pred.mean(-1,keepdim=True);yc=y-y.mean(-1,keepdim=True)
    return -nn.functional.cosine_similarity(pc,yc,dim=-1,eps=1e-6).mean()


def forward_global_batch(model,data,first,last,state=None,active=None,teacher_scaling=None,teacher_mode='future',mode='full'):
    parts=[];aux_sum=data.x.new_zeros(());aux_n=0
    for a,b,row,local in data.spans(first,last):
        if active!=row['record']:
            state=None
            if hasattr(data,'reset_count'):data.reset_count+=1
        z=data.x[a:b];prediction,state,aux=model(z,state,local,row['record'],with_aux=teacher_scaling is not None,mode=mode)
        parts.append(prediction);active=row['record']
        if teacher_scaling is not None and 'future_prediction' in aux:
            shift=2 if teacher_mode=='future' else 0
            # BOTH future and current teacher use the SAME valid tail mask.
            end=min(b,row['offset']+row['windows']-2);count=max(0,end-a)
            if count:
                teacher=data.teacher[a+shift:a+shift+count].detach()
                mean,std=teacher_scaling;target=(teacher-mean)/std
                aux_sum=aux_sum+(aux['future_prediction'][:count]-target).square().mean()*count;aux_n+=count
        # NEVER detach between record parts inside this optimizer batch.
    return torch.cat(parts,0),state,active,aux_sum/max(1,aux_n),aux_n


@torch.no_grad()
def validation(model,data,mode='full'):
    model.eval();state=None;active=None;total=0.;data.reset_count=0
    for first in range(0,len(data.x),512):
        last=min(len(data.x),first+512)
        # ContextCore EMA matrix deliberately bounded: split into native64
        # forwards while keeping states. This does not alter score aggregation.
        for a in range(first,last,64):
            b=min(last,a+64);p,state,active,_,_=forward_global_batch(model,data,a,b,state,active,mode=mode)
            total+=float(objective(p,data.y[a:b]))*(b-a)
    assert data.reset_count==len(data.rows)
    result=total/len(data.x);assert np.isfinite(result)
    return result


@torch.no_grad()
def macro_validation(model,data,mode='full'):
    """Native full-record Hann/60s→person metric for CONFIG choice only."""
    model.eval();state=None;active=None;predictions=[];data.reset_count=0
    for first in range(0,len(data.x),64):
        p,state,active,_,_=forward_global_batch(model,data,first,min(first+64,len(data.x)),state,active,mode=mode)
        predictions.append(p[:,0].cpu().numpy())
    assert data.reset_count==len(data.rows)
    predictions=np.concatenate(predictions);window=np.hanning(192);segments=[]
    eligible={(r['record'],r['start']):r for r in data.fixed_mask['blocks']}
    for row in data.rows:
        n=row['common_samples'];start=row['offset'];pred=predictions[start:start+row['windows']]
        total=np.zeros(n);weight=np.zeros(n)
        for i,value in enumerate(pred):
            s=i*96;total[s:s+192]+=value*window;weight[s:s+192]+=window
        covered=weight>1e-8;total[covered]/=weight[covered]
        target=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        for s in range(0,n-3840+1,3840):
            keep=covered[s:s+3840]
            mask=eligible[(row['record'],s)];assert int(keep.sum())==mask['samples']
            if not mask['eligible']:continue
            a=total[s:s+3840][keep];b=np.asarray(target[s:s+3840])[keep]
            assert np.isfinite(b).all() and np.std(b)>0,'target changed after fixed eligibility'
            if not np.isfinite(a).all() or np.std(a)==0:raise RuntimeError('undefined prediction; no selective block exclusion')
            segments.append({'record':row['record'],'subject':row['subject'],'story':row['story'],
                'start':s,'samples':int(keep.sum()),'r':float(np.corrcoef(a,b)[0,1])})
    people=[{'subject':p,'r':float(np.mean([r['r'] for r in segments if r['subject']==p]))} for p in sorted({r['subject'] for r in segments})]
    macro=float(np.mean([r['r'] for r in people]));assert np.isfinite(macro)
    assert len(segments)==data.fixed_mask['eligible_blocks']
    return {'macro_r':macro,'subjects':len(people),'blocks':len(segments)},segments,people


def train_config(config,seed,device,stage='screen',epochs=12,patience=4):
    assert stage in ('screen','full')
    protocol=read(ROOT/'INCREMENTAL_PROTOCOL.json');assert protocol['frozen']
    budget=protocol['screen_budget' if stage=='screen' else 'full_budget']
    assert epochs==budget['epochs'] and patience==budget['patience']
    assert seed in protocol['screening_seeds'] if stage=='screen' else seed in protocol['final_seeds']
    assert config in protocol['conditions'] if stage=='screen' else config in protocol['full_conditions']
    if stage=='full':assert config['id'] in read(ROOT/'FULL_SELECTION.json')['approved_condition_ids']
    for path,value in protocol['sources'].items():assert sha(path)==value
    disk_guard();torch.set_num_threads(2);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=protocol['precision']['TF32_matmul']
    torch.backends.cudnn.allow_tf32=protocol['precision']['TF32_cudnn']
    model=make(config,seed).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    folder=ROOT/f'{stage}/{config["id"]}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    train=LatentData(seed,'train',device);val=LatentData(seed,'val',device)
    scaling=None
    if config['family']=='m3':
        # Native encoder fixed; statistics fit solely on this seed's TRAIN.
        scaling=(train.teacher.mean(0),train.teacher.std(0,unbiased=False).clamp_min(1e-5))
    dump(folder/'resolved_config.json',{'condition':config,'seed':seed,'epochs_cap':epochs,'patience':patience,
        'encoder_sha256':sha(ROOT/f'baseline/pretrain_seed{seed}/best_validation.pt'),
        'global_batch64':True,'optimizer':'Adam.001/no wd/no clip','shuffle':False,'test_read':False})
    history=[];best=float('inf');selected=None;stale=0;updates=0;started=time.time();active_grad=None
    for epoch in range(epochs):
        model.train();state=None;active=None;total=0.;samples=0;teacher_count=0;train.reset_count=0
        for first in range(0,len(train.x),64):
            last=min(first+64,len(train.x));opt.zero_grad(set_to_none=True)
            p,state,active,aux,naux=forward_global_batch(model,train,first,last,state,active,scaling,config.get('teacher','future'),config.get('mode','full'))
            pearson=objective(p,train.y[first:last]);loss=pearson+config.get('aux_weight',.01)*aux
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            opt.step();state=detach(state);updates+=1;total+=float(pearson.detach())*(last-first);samples+=last-first;teacher_count+=naux
            if updates==2:
                active_grad={'registered':sum(v.numel() for v in model.parameters()),
                    'gradient_reachable':sum(v.numel() for v in model.parameters() if v.grad is not None),
                    'nonzero_gradient_scalars':sum(int(torch.count_nonzero(v.grad)) for v in model.parameters() if v.grad is not None)}
            if updates%500==0:update(folder/'status.json',{'epoch':epoch,'updates':updates,'samples_epoch':samples,'elapsed_seconds':time.time()-started})
        assert samples==252454 and updates==(epoch+1)*3945 and train.reset_count==426
        score=validation(model,val,config.get('mode','full'));stale=0 if score<best else stale+1
        if score<best:
            best=score;selected=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'seed':seed,'condition':config,'val_loss':score},folder/'best_validation.pt')
        item={'epoch':epoch,'train_negative_Pearson':total/samples,'validation_negative_Pearson':score,
            'stale':stale,'updates':updates,'teacher_presentations_epoch':teacher_count,'seconds':time.time()-started}
        history.append(item);update(folder/'history.json',history);print(json.dumps({'condition':config['id'],'seed':seed,**item}),flush=True)
        disk_guard()
        if stale>=patience:break
    selected_checkpoint=torch.load(folder/'best_validation.pt',map_location=device,weights_only=True)
    model.load_state_dict(selected_checkpoint['model'])
    macro,segments,people=macro_validation(model,val,config.get('mode','full'))
    dump(folder/'validation_macro.json',{'summary':macro,'people':people,'segments':segments,
        'checkpoint_selection':'original fullval3s negativePearson, NOT this macro score','config_selection_only':True})
    dump(folder/'completed.json',{'condition':config,'seed':seed,'stage':stage,'epochs':len(history),'updates':updates,
        'selected_epoch':selected,'best_validation_negative_Pearson':best,'supervised_window_presentations':252454*len(history),
        'supervised_target_samples':252454*192*len(history),'teacher_presentations':sum(r['teacher_presentations_epoch'] for r in history),
        'teacher_target48_scalar_cost_separate':True,'gradient_parameters':active_grad,'test_read':False,
        'seconds':time.time()-started,'validation_macro':macro,'record_resets_per_train_epoch':426,
        'validation_record_resets':49,'protocol_sha256':sha(ROOT/'INCREMENTAL_PROTOCOL.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--stage',default='screen',choices=['screen','full']);a=p.parse_args()
    protocol=read(ROOT/'INCREMENTAL_PROTOCOL.json');rows=protocol['conditions'] if a.stage=='screen' else protocol['full_conditions']
    config=next(c for c in rows if c['id']==a.condition)
    train_config(config,a.seed,a.device,a.stage,12 if a.stage=='screen' else 250,4 if a.stage=='screen' else 10)
