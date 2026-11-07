"""Server2203-only run02. One pass; never load old EEG checkpoints."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from pub_01_s03_ss06_data import ROOT as OLD, Records, LENGTH, WARMUP
from pub_01_s03_ss06_train import metrics,aggregate,write_csv
from pub_01_s03_ss06_strong import CausalStrong,NativeStrong,pair,detach,objective,pearson_loss

ROOT=OLD.parent/'eeg_run02_matched_baseline'
FEATURES=OLD.parents[1]/'s03_strong_baselines_20261015/transformer_available01/features/manifest.json'
SEEDS=(1024,2026,31415)


def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(a,device):return torch.from_numpy(np.array(a,copy=True)).to(device)
def batch(a,device):return tensor(a,device).unsqueeze(0)
def setup(seed,device):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
def sync(device):torch.cuda.synchronize(device)


def schedule(rows,seed):
    """Each eligible nonoverlap scored 3s span exactly once, except marked anchors.

    Each batch contains one story, unique sound timestamps. Pick one person at
    each available timestamp per round. Redistribute one item when a last chunk
    would be singleton; if only one timestamp remains, add a declared repeat
    anchor at another time from the SAME train story. No singleton optimizer.
    """
    rng=np.random.default_rng(seed);result=[]
    for story in sorted({r['story'] for r in rows}):
        groups={}
        for i,r in enumerate(rows):
            if r['story']!=story:continue
            for s in range(WARMUP,WARMUP+r['score_blocks']*LENGTH,192):groups.setdefault(s,[]).append(i)
        original={s:list(v) for s,v in groups.items()}
        for v in groups.values():rng.shuffle(v)
        while groups:
            times=rng.permutation(sorted(groups)).tolist()
            if len(times)==1:
                s=times[0];i=groups[s].pop();other=next(t for t in original if t!=s)
                result.append([(i,s,False),(original[other][0],other,True)])
            else:
                parts=[times[j:j+128] for j in range(0,len(times),128)]
                if len(parts[-1])==1:parts[-1].append(parts[-2].pop())
                for part in parts:result.append([(groups[s].pop(),s,False) for s in part])
            groups={s:v for s,v in groups.items() if v}
    return result


class Inputs:
    def __init__(self,phase):self.rows=Records(phase).rows;self._arrays={};self._features={}
    def arrays(self,i):
        if i not in self._arrays:self._arrays[i]=Records.arrays(None,self.rows[i])
        return self._arrays[i]
    def contrast(self,entries,device):
        index={r['story']:r for r in read(FEATURES)};xx=[];aa=[]
        for i,s,_ in entries:
            r=self.rows[i];x,_=self.arrays(i);story=r['story']
            if story not in self._features:self._features[story]=np.load(index[story]['path'],mmap_mode='r')
            xx.append(x[s:s+192]);aa.append(self._features[story][s:s+192])
        assert len({self.rows[i]['story'] for i,_,_ in entries})==1
        assert len({s for _,s,_ in entries})==len(entries)>=2
        return tensor(np.stack(xx),device),tensor(np.stack(aa),device)


def intake():
    ROOT.mkdir(exist_ok=True);(ROOT/'input').mkdir(exist_ok=True);train=Inputs('train');val=Inputs('validation')
    features={r['story']:r for r in read(FEATURES)};norm=read(OLD/'normalization_contract.json')
    assert set(norm['fit_records'])=={r['record'] for r in train.rows}
    verified=[]
    for data in (train,val):
        for i,r in enumerate(data.rows):
            f=features[r['story']];end=WARMUP+r['score_blocks']*LENGTH
            assert f['tuple_index']==19 and f['revision']=='46f221381d200f7bef268309b3f02023ccf11fcc' and f['fs']==64
            a=np.load(f['path'],mmap_mode='r');x,y=data.arrays(i)
            assert a.shape[1]==1024 and len(a)>=end and x.shape[1]==64 and len(x)>=end and len(y)>=end
            assert np.isfinite(a[WARMUP:WARMUP+192]).all() and np.isfinite(x[WARMUP:WARMUP+192]).all()
        verified.append({'phase':data.rows[0]['phase'],'records':len(data.rows),'blocks':sum(r['score_blocks'] for r in data.rows)})
    ledgers={}
    for seed in SEEDS:
        plan=schedule(train.rows,seed);seen=[(i,s) for b in plan for i,s,repeat in b if not repeat]
        assert len(set(seen))==len(seen)==3262*20
        assert all(2<=len(b)<=128 for b in plan)
        ledgers[str(seed)]={'contrastive_steps':len(plan),'unique_3s_windows':len(seen),
            'anchor_repeat_windows':sum(repeat for b in plan for _,_,repeat in b),
            'actual_EEG_and_audio_sample_presentations':sum(len(b)*192 for b in plan),
            'batch_sizes':{str(n):sum(len(b)==n for b in plan) for n in sorted({len(b) for b in plan})}}
    head_windows=sum(40*r['score_blocks']-1 for r in train.rows)
    dump(ROOT/'INTAKE.json',{'phases':verified,'normalizer':str(OLD/'newtrain_EEG_target_normalization.npz'),
        'normalization_contract_sha256':sha(OLD/'normalization_contract.json'),
        'split_sha256':sha(OLD/'SPLIT_PROVENANCE.json'),'feature_manifest_sha256':sha(FEATURES),
        'matched_targets_per_seed':3262*3840,'O_EEG_window_presentations':3262*39*192,
        'P_pretrain':ledgers,'P_head':{'windows':head_windows,'target_presentations':head_windows*192,
        'steps':math.ceil(head_windows/64),'unique_scored_target_support':3262*3840},
        'audit_payload_read':False,'old_checkpoints_loaded':False,'kind':'POST-HOC INTERNAL EXPLORATORY'})
    print(json.dumps(read(ROOT/'INTAKE.json')),flush=True)


@torch.no_grad()
def evaluate(model,data,device,pilot=False):
    model.eval();rows=[]
    for i,r in enumerate(data.rows[:3] if pilot else data.rows):
        x,y=data.arrays(i);_,state=model(batch(x[:WARMUP],device))
        for block in range(min(1,r['score_blocks']) if pilot else r['score_blocks']):
            s=WARMUP+block*LENGTH;p,state=model(batch(x[s:s+LENGTH],device),state)
            rows.append({k:r[k] for k in ('record','subject','source_family')}|{'block':block,**metrics(p.cpu().numpy(),y[s:s+LENGTH])})
    return aggregate(rows),rows


def guard_formal():
    protocol=read(ROOT/'FORMAL_PROTOCOL.json')
    assert protocol['frozen'] and protocol['seeds']==list(SEEDS)
    for name,value in protocol['source_sha256'].items():assert sha(ROOT/'input'/name)==value
    for path,value in protocol['upstream_source_sha256'].items():assert sha(path)==value
    assert sha(OLD/'SPLIT_PROVENANCE.json')==protocol['split_sha256']
    assert sha(OLD/'normalization_contract.json')==protocol['normalization_contract_sha256']
    return protocol


def joint(name,seed,device,pilot=False):
    setup(seed,device)
    if not pilot:guard_formal()
    folder=ROOT/('pilot01' if pilot else 'formal01')/f'{name}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    model=(CausalStrong() if name=='C' else NativeStrong()).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    train=Inputs('train');val=Inputs('validation');order=np.random.default_rng(seed).permutation(len(train.rows))
    maxsteps=20 if pilot else 3262;history=[];seen=[];best=-float('inf');selected=None;step=0;losses=[];started=time.perf_counter();active=None
    for i in order:
        r=train.rows[i];x,y=train.arrays(int(i));model.train()
        with torch.no_grad():_,state=model(batch(x[:WARMUP],device))
        for block in range(r['score_blocks']):
            s=WARMUP+block*LENGTH;opt.zero_grad(set_to_none=True)
            p,state=model(batch(x[s:s+LENGTH],device),detach(state));loss=objective(p,batch(y[s:s+LENGTH],device))
            assert torch.isfinite(loss);loss.backward()
            if active is None:
                assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                active={'registered':sum(p.numel() for p in model.parameters()),
                    'gradient_reachable':sum(p.numel() for p in model.parameters() if p.grad is not None),
                    'all_parameter_gradients_finite':True}
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();state=detach(state)
            step+=1;losses.append(float(loss.detach()));seen.append({'record':r['record'],'block':block,'start':s,'samples':LENGTH})
            if step%10==0:print(json.dumps({'condition':name,'seed':seed,'stage':'pilot' if pilot else 'formal','step':step,'loss':float(loss.detach()),'elapsed':time.perf_counter()-started}),flush=True)
            if step in ((10,20) if pilot else (1000,2000,3000,3262)):
                (summary,_,_),_=evaluate(model,val,device,pilot);score=summary['r']
                if score>best:
                    selected=step;best=score;torch.save({'model':model.state_dict(),'step':step,'validation':summary},folder/'best_validation.pt')
                item={'step':step,'mean_loss':float(np.mean(losses)),'validation':summary,'elapsed_seconds':time.perf_counter()-started}
                history.append(item);losses=[];(folder/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(item),flush=True);model.train()
            if step==maxsteps:break
        if step==maxsteps:break
    sync(device);dump(folder/'completed.json',{'condition':name,'seed':seed,'steps':step,'targets':step*3840,
        'selected_step':selected,'selected_validation':best,'seconds':time.perf_counter()-started,'parameters':active,
        'max_GPU_bytes':torch.cuda.max_memory_allocated(device),'actual_sequence':seen,
        'EEG_window_presentations_excluding_warmup':step*(3840 if name=='C' else 39*192),
        'warmup_records':len({r['record'] for r in seen}),'old_checkpoints_loaded':False,'audit_read':False})


@torch.no_grad()
def contrast_val(model,data,plan,device):
    model.eval();total=0.
    for entries in plan:
        x,a=data.contrast(entries,device);total+=float(model.loss(x,a,None))
    return total/len(plan)


def pretrain(seed,device,pilot=False):
    setup(seed,device)
    if not pilot:guard_formal()
    folder=ROOT/('pilot01' if pilot else 'formal01')/f'P_pretrain_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    model=pair().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    train=Inputs('train');val=Inputs('validation');plan=schedule(train.rows,seed);vp=schedule(val.rows,20261026)
    if pilot:plan=plan[:20];vp=vp[:2]
    best=float('inf');selected=None;history=[];started=time.perf_counter();seen=[]
    checks=sorted(set([len(plan)] if pilot else [max(1,len(plan)*i//4) for i in (1,2,3,4)]))
    for step,entries in enumerate(plan,1):
        model.train();opt.zero_grad(set_to_none=True);x,a=train.contrast(entries,device);loss=model.loss(x,a,None)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        seen.extend({'record':train.rows[i]['record'],'start':s,'repeat_anchor':repeat,'samples':192} for i,s,repeat in entries)
        if step%10==0:print(json.dumps({'condition':'P_pretrain','seed':seed,'step':step,'loss':float(loss.detach()),'elapsed':time.perf_counter()-started}),flush=True)
        if step in checks:
            score=contrast_val(model,val,vp,device)
            if score<best:best=score;selected=step;torch.save({'model':model.state_dict(),'step':step,'validation_loss':score},folder/'best_validation.pt')
            history.append({'step':step,'loss':float(loss.detach()),'validation_InfoNCE':score});(folder/'history.json').write_text(json.dumps(history,indent=2))
    sync(device);dump(folder/'completed.json',{'seed':seed,'steps':len(plan),'selected_step':selected,'validation_InfoNCE':best,
        'EEG_and_audio_sample_presentations':len(seen)*192,'parameters':sum(p.numel() for p in model.parameters()),
        'seconds':time.perf_counter()-started,'actual_sequence':seen,'old_checkpoints_loaded':False,'audit_read':False,
        'max_GPU_bytes':torch.cuda.max_memory_allocated(device)})


def head(seed,device,pilot=False):
    setup(seed,device)
    if not pilot:guard_formal()
    stage='pilot01' if pilot else 'formal01';folder=ROOT/stage/f'P_head_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    pre=ROOT/stage/f'P_pretrain_seed{seed}/best_validation.pt';assert pre.is_relative_to(ROOT/stage)
    learned=pair().to(device);learned.load_state_dict(torch.load(pre,map_location=device,weights_only=True)['model'])
    model=NativeStrong().to(device);model.encoder.load_state_dict(learned.eeg.state_dict());del learned
    for p in model.encoder.parameters():p.requires_grad_(False)
    model.encoder.eval();opt=torch.optim.AdamW(model.head.parameters(),lr=.001,weight_decay=.01)
    datasets={p:Inputs(p) for p in ('train','validation')};latents={};targets={}
    # Cache only newtrain/val selected-encoder latents; stays on2203.
    for phase,data in datasets.items():
        zs=[];ys=[]
        for i,r in enumerate(data.rows[:3] if pilot else data.rows):
            x,y=data.arrays(i);end=WARMUP+r['score_blocks']*LENGTH
            starts=list(range(WARMUP,end-191,96))
            if pilot:starts=starts[:64]
            for j in range(0,len(starts),64):
                times=starts[j:j+64];xx=tensor(np.stack([x[s:s+192] for s in times]),device)
                with torch.no_grad():zs.append(model.encoder(xx).cpu().numpy())
                ys.append(np.stack([y[s:s+192] for s in times]))
        latents[phase]=np.concatenate(zs);targets[phase]=np.concatenate(ys)
        np.save(folder/f'{phase}_latent.npy',latents[phase]);np.save(folder/f'{phase}_targets.npy',targets[phase])
    order=np.random.default_rng(seed).permutation(len(latents['train']));batches=[order[j:j+64] for j in range(0,len(order),64)]
    if pilot:batches=batches[:min(20,len(batches))]
    checks=sorted(set([len(batches)] if pilot else [len(batches)*i//4 for i in (1,2,3,4)]))
    best=float('inf');selected=None;history=[];started=time.perf_counter();seen=[]
    for step,indices in enumerate(batches,1):
        model.head.train();opt.zero_grad(set_to_none=True)
        z=tensor(latents['train'][indices],device);y=tensor(targets['train'][indices],device)
        loss=pearson_loss(model.head(z.transpose(1,2)).transpose(1,2),y);assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.,error_if_nonfinite=True);opt.step();seen.extend(indices.tolist())
        if step in checks:
            model.head.eval();total=0.
            with torch.no_grad():
                for j in range(0,len(latents['validation']),128):
                    z=tensor(latents['validation'][j:j+128],device);y=tensor(targets['validation'][j:j+128],device)
                    total+=float(pearson_loss(model.head(z.transpose(1,2)).transpose(1,2),y))*len(z)
            score=total/len(latents['validation'])
            if score<best:best=score;selected=step;torch.save({'model':model.state_dict(),'step':step,'validation_3s_negative_r':score},folder/'best_validation.pt')
            history.append({'step':step,'validation_3s_negative_r':score});(folder/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps({'condition':'P_head','seed':seed,**history[-1]}),flush=True)
    dump(folder/'completed.json',{'seed':seed,'steps':len(batches),'selected_step':selected,'validation_3s_negative_r':best,
        'head_parameters':sum(p.numel() for p in model.head.parameters()),'target_sample_presentations':len(seen)*192,
        'unique_window_indices':len(set(seen)),'actual_window_index_sequence':seen,
        'cache_windows':{p:len(v) for p,v in latents.items()},'seconds_head_optimization':time.perf_counter()-started,
        'pretrain_checkpoint':str(pre),'pretrain_checkpoint_sha256':sha(pre),'audit_read':False})


def audit(device):
    guard_formal()
    assert (ROOT/'ALL_TRAINING_REVIEW.json').exists()
    folder=ROOT/'audit01';folder.mkdir(exist_ok=False);data=Inputs('audit_holdout');results=[];allrecords=[];allpersons=[]
    for seed in SEEDS:
        for name in ('C','O','P'):
            model=(CausalStrong() if name=='C' else NativeStrong()).to(device)
            source=ROOT/'formal01'/f'{name if name!="P" else "P_head"}_seed{seed}/best_validation.pt'
            model.load_state_dict(torch.load(source,map_location=device,weights_only=True)['model'])
            (summary,records,persons),blocks=evaluate(model,data,device)
            results.append({'condition':name,'seed':seed,**summary})
            allrecords.extend({'condition':name,'seed':seed,**r} for r in records)
            allpersons.extend({'condition':name,'seed':seed,**r} for r in persons)
            write_csv(folder/f'{name}_seed{seed}_blocks.csv',blocks);print(json.dumps(results[-1]),flush=True)
    write_csv(folder/'RESULTS.csv',results);write_csv(folder/'per_record.csv',allrecords);write_csv(folder/'per_subject.csv',allpersons)
    dump(folder/'completed.json',{'conditions':['C','O','P'],'seeds':list(SEEDS),'blocks_per_condition_seed':538,'kind':'POST-HOC INTERNAL EXPLORATORY'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['intake','joint','pretrain','head','audit']);p.add_argument('--condition',choices=['C','O']);p.add_argument('--seed',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    if a.action=='intake':intake()
    elif a.action=='joint':joint(a.condition,a.seed,a.device,a.pilot)
    elif a.action=='pretrain':pretrain(a.seed,a.device,a.pilot)
    elif a.action=='head':head(a.seed,a.device,a.pilot)
    else:setup(1024,a.device);audit(a.device)
