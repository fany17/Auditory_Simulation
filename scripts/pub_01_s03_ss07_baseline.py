"""SS07 full native SS02 training, explicitly train/val-only. Server2203.

Imports immutable native classes/batching/loss, NEVER either historical main
or evaluate function. Full128 negatives on ONE GPU per seed. No old EEG cp.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch import nn

PUBLIC=Path('/home/fanyu/auditory_simulation_m6a')
OLD=PUBLIC/'pub_01/s03_strong_baselines_20261015/transformer_available01'
ROOT=PUBLIC/'pub_01/s03_incremental_continuity_20261101'
PREPARED=OLD/'prepared'
SEEDS=(1024,2026,31415)


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2)
def update(p,d):
    # Only our active current status/latest checkpoint, never historical assets.
    assert Path(p).is_relative_to(ROOT)
    tmp=Path(str(p)+'.writing');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,p)
def native():
    path=OLD/'input/pub_01_s03_transformer_train_v2.py'
    spec=importlib.util.spec_from_file_location('ss07_native_reference',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def cfg():return read(OLD/'training_seed1024_attempt04/config_frozen.json')
def phase_features(phase):
    stories={r['story'] for r in read(PREPARED/'manifest_frozen.json')['records'] if r['phase']==phase}
    return [f for f in read(OLD/'features/manifest.json') if f['story'] in stories]
def disk_guard():
    free=shutil.disk_usage(ROOT).free
    if free<20*1024**3:raise RuntimeError('storage below20GiB reserve; preserve data and stop this stage')
def seed_all(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)


def intake():
    ROOT.mkdir(exist_ok=True);(ROOT/'input').mkdir(exist_ok=True)
    c=cfg();manifest=read(PREPARED/'manifest_frozen.json');features=read(OLD/'features/manifest.json')
    assert c==manifest['config'] and manifest['normalization_fit_phase']=='train_only'
    counts=dict(Counter(r['phase'] for r in manifest['records']))
    assert counts=={'train':426,'val':49,'test_seen':69,'test_heldout':75}
    mod=native();train=mod.StoryBatches(PREPARED,phase_features('train'),'train',128);valid=mod.StoryBatches(PREPARED,phase_features('val'),'val',128)
    assert len(train.plans)==820
    train_records={r['record'] for r in manifest['records'] if r['phase']=='train'}
    train_ids={i for i,r in enumerate(manifest['records']) if r['phase']=='train'}
    assert all(r['record_id'] in train_ids for r in train.rows)
    scaling=np.load(PREPARED/'train_only_normalization.npz')
    feature_index={f['story']:f for f in features}
    coverage=[]
    for phase,loader in [('train',train),('val',valid)]:
        eligible=len(loader.rows);visited=[];audio_positions=[]
        # Same RNG decisions as native batches without moving or reading test.
        rng=np.random.default_rng(1024 if phase=='train' else 20261015);order=np.arange(len(loader.plans))
        if phase=='train':rng.shuffle(order)
        for pi in order:
            story,times,lookup=loader.plans[pi];starts=rng.choice(times,size=128,replace=False)
            assert len(set(starts))==128
            indices=[int(rng.choice(lookup[int(s)])) for s in starts]
            visited.extend(indices);audio_positions.extend((story,int(s)) for s in starts)
        coverage.append({'phase':phase,'eligible_record_time_windows':eligible,'batches_per_epoch':len(loader.plans),
            'actual_window_presentations_epoch0':len(visited),'unique_visited_epoch0':len(set(visited)),
            'repeat_presentations_epoch0':len(visited)-len(set(visited)),'unvisited_eligible_epoch0':eligible-len(set(visited)),
            'unique_story_time_windows_epoch0':len(set(audio_positions))})
    head=read(OLD/'envelope_readout_seed1024/manifest_frozen.json')
    expected={p:sum((r['common_samples']-192)//96+1 for r in manifest['records'] if r['phase']==p) for p in ('train','val')}
    assert all(head['phase_windows'][p]==n for p,n in expected.items())
    for phase in ('train','val'):
        original=[r for r in manifest['records'] if r['phase']==phase];target_rows=[r for r in head['records'] if r['phase']==phase]
        assert [r['record'] for r in original]==[r['record'] for r in target_rows]
        offset=0
        for r in target_rows:
            assert r['phase_offset']==offset
            offset+=r['readout_windows']
        a=np.load(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy',mmap_mode='r')
        assert a.shape==(expected[phase],1,192)
    dump(ROOT/'INTAKE.json',{'cohort':counts,'normalization_fit':'original426train only; exact SS02 stats reused',
        'normalization_sha256':sha(PREPARED/'train_only_normalization.npz'),
        'prepared_manifest_sha256':sha(PREPARED/'manifest_frozen.json'),
        'train_windows_sha256':sha(PREPARED/'train_windows.json'),'val_windows_sha256':sha(PREPARED/'val_windows.json'),
        'native_training_source_sha256':sha(OLD/'input/pub_01_s03_transformer_train_v2.py'),
        'native_model_source_sha256':sha(OLD/'input/pub_01_s03_method_models.py'),
        'native_config_sha256':sha(OLD/'training_seed1024_attempt04/config_frozen.json'),
        'feature_manifest_sha256':sha(OLD/'features/manifest.json'),'XLSR19_revision':c['xlsr_revision'],
        'stochastic_coverage_example':coverage,'head_windows':expected,
        'cache_estimate_three_seeds_bytes':3*sum(expected.values())*8*192*4,
        'disk_free_bytes':shutil.disk_usage(ROOT).free,'new_cache_cap_bytes':25*1024**3,
        'reserve_bytes':20*1024**3,'test_payload_loaded':False,'old_EEG_latents_or_checkpoints_reused':False,
        'original_independent_audio_targets_reused':True,'kind':'all historical tests exposed; no independent confirmation'})
    print(json.dumps(read(ROOT/'INTAKE.json')),flush=True)


def check():
    seed_all(1024,'cuda:0');mod=native();c=cfg();features=read(OLD/'features/manifest.json')
    loader=mod.StoryBatches(PREPARED,phase_features('train'),'train',128);x,a=next(loader.batches(1024,True));x=x.cuda();a=a.cuda()
    model=mod.TransformerCLIP(c).cuda().eval();assert sum(p.numel() for p in model.parameters())==3800713
    with torch.no_grad():
        native_loss=mod.contrastive(model,x,a);z,f=model(x,a)
        logits=(z@f.T)*model.log_scale.exp();labels=torch.arange(128,device=x.device)
        explicit=(nn.functional.cross_entropy(logits,labels)+nn.functional.cross_entropy(logits.T,labels))/2
    assert logits.shape==(128,128) and torch.equal(native_loss,explicit)
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    t=time.perf_counter();loss=mod.contrastive(model,x,a);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step();torch.cuda.synchronize()
    dump(ROOT/'ENGINEERING_CHECK.json',{'passed':True,'native_vs_explicit_loss_difference':float(abs(native_loss-explicit)),
        'global_negative_matrix':[128,128],'single_GPU_seed_preserves_global_batch128':True,
        'native_registered_parameters':3800713,'all_train_grad_finite':True,'real_train_step_seconds':time.perf_counter()-t,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'pilot_weights_discarded':True,
        'TF32_matmul':torch.backends.cuda.matmul.allow_tf32,'TF32_cudnn':torch.backends.cudnn.allow_tf32,
        'test_payload_loaded':False,'original_mains_called':False})
    print(json.dumps(read(ROOT/'ENGINEERING_CHECK.json')),flush=True)


def guard():
    p=read(ROOT/'BASELINE_PROTOCOL.json');assert p['frozen']
    for name,digest in p['sources'].items():assert sha(name)==digest
    disk_guard();return p


def pretrain(seed,device):
    protocol=guard();seed_all(seed,device);mod=native();c=cfg();traincfg=c['training']
    folder=ROOT/f'baseline/pretrain_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    train=mod.StoryBatches(PREPARED,phase_features('train'),'train',128);val=mod.StoryBatches(PREPARED,phase_features('val'),'val',128)
    model=mod.TransformerCLIP(c).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma=.1)
    best=float('inf');best_epoch=None;stale=0;history=[];started=time.time();updates=0
    for epoch in range(100):
        model.train();total=0.;count=0;t=time.time()
        for step,(x,a) in enumerate(train.batches(seed+epoch,True)):
            x=x.to(device);a=a.to(device);opt.zero_grad(set_to_none=True);loss=mod.contrastive(model,x,a)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite pretraining loss')
            loss.backward();opt.step();total+=float(loss.detach());count+=1;updates+=1
            if step%100==0:
                state={'seed':seed,'epoch':epoch,'step_in_epoch':step,'updates':updates,'loss':float(loss.detach()),'seconds_epoch':time.time()-t,'stage':'FULL_NATIVE_PRETRAIN'}
                update(folder/'status.json',state);print(json.dumps(state),flush=True)
        assert count==820
        model.eval();vl=[]
        with torch.no_grad():
            for x,a in val.batches(20261015,False):vl.append(float(mod.contrastive(model,x.to(device),a.to(device))))
        score=float(np.mean(vl));assert np.isfinite(score)
        improved=score<best;stale=0 if improved else stale+1
        if improved:
            best=score;best_epoch=epoch
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation_loss':score,'seed':seed,'config':c},folder/'best_validation.pt')
        item={'epoch':epoch,'train_loss':total/count,'validation_loss':score,'stale':stale,'seconds':time.time()-t,
            'train_batches':count,'validation_batches':len(vl),'updates':updates,'lr':opt.param_groups[0]['lr']}
        history.append(item);update(folder/'history.json',history);print(json.dumps({'seed':seed,**item}),flush=True);sched.step();disk_guard()
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,
            'best_epoch':best_epoch,'stale':stale,'updates':updates},folder/'latest.pt')
        if stale>=15:stop='VALIDATION_PATIENCE15';break
    else:stop='RESOURCE_CEILING100'
    dump(folder/'completed.json',{'seed':seed,'epochs':len(history),'updates':updates,'selected_epoch':best_epoch,
        'best_validation':best,'stopping':stop,'seconds':time.time()-started,
        'peak_GPU_bytes':torch.cuda.max_memory_allocated(device),'test_payload_loaded':False,
        'stochastic_actual_presentations':updates*128,'sample_presentations':updates*128*192,
        'source_protocol_sha256':sha(ROOT/'BASELINE_PROTOCOL.json'),'fresh_initialization':True})


def head(seed,device):
    guard();seed_all(seed,device);mod=native();c=cfg();pre=ROOT/f'baseline/pretrain_seed{seed}'
    assert (pre/'completed.json').exists()
    folder=ROOT/f'baseline/head_seed{seed}';folder.mkdir(exist_ok=False)
    model=mod.TransformerCLIP(c).to(device);checkpoint=torch.load(pre/'best_validation.pt',map_location=device,weights_only=True)
    assert checkpoint['seed']==seed;model.load_state_dict(checkpoint['model']);model.eval();encoder=model.eeg
    stats=np.load(PREPARED/'train_only_normalization.npz');mean,std=stats['mean'],stats['std'];manifest=read(PREPARED/'manifest_frozen.json')
    arrays={};targets={};started=time.time();offsets={};cache_start=time.time()
    for phase in ('train','val'):
        rows=[r for r in manifest['records'] if r['phase']==phase]
        n=sum((r['common_samples']-192)//96+1 for r in rows)
        path=folder/f'{phase}_latent.npy';out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(n,8,192));offset=0
        for ri,r in enumerate(rows):
            raw=np.load(r['eeg_path'],mmap_mode='r');eeg=raw.T if raw.shape[0]==64 else raw;count=(r['common_samples']-192)//96+1
            for first in range(0,count,256):
                starts=np.arange(first,min(first+256,count))*96
                xx=np.stack([(np.asarray(eeg[s:s+192],np.float32)-mean)/std for s in starts]).astype(np.float32)
                with torch.no_grad():z=encoder(torch.from_numpy(xx).to(device)).transpose(1,2).cpu().numpy()
                out[offset+first:offset+first+len(z)]=z
            offsets[r['record']]={'phase':phase,'offset':offset,'windows':count};offset+=count
            if ri%30==0:print(json.dumps({'seed':seed,'stage':'TRAIN_VAL_ONLY_ENCODER_CACHE','phase':phase,'records':ri+1}),flush=True)
        out.flush();arrays[phase]=np.load(path,mmap_mode='r')
        targets[phase]=np.load(OLD/f'envelope_readout_seed1024/{phase}_envelope.npy',mmap_mode='r')
        assert arrays[phase].shape[0]==targets[phase].shape[0]
    dump(folder/'cache_manifest.json',{'records':offsets,'encoder_source':str(pre/'best_validation.pt'),
        'encoder_sha256':sha(pre/'best_validation.pt'),'seconds':time.time()-cache_start,
        'only_train_val':True,'shared_targets':'old encoder-independent envelope arrays, same grid/order'})
    del model,encoder;torch.cuda.empty_cache();torch.manual_seed(seed)
    head=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01)).to(device)
    opt=torch.optim.Adam(head.parameters(),lr=.001);best=float('inf');stale=0;selected=None;history=[];updates=0
    # As native SS02, whole train/val latent+targets on GPU, sequential64.
    x=torch.from_numpy(np.array(arrays['train'],copy=True)).to(device);y=torch.from_numpy(np.array(targets['train'],copy=True)).to(device)
    vx=torch.from_numpy(np.array(arrays['val'],copy=True)).to(device);vy=torch.from_numpy(np.array(targets['val'],copy=True)).to(device)
    def loss_fn(p,t):
        pc=p-p.mean(-1,keepdim=True);tc=t-t.mean(-1,keepdim=True)
        return -nn.functional.cosine_similarity(pc,tc,dim=-1,eps=1e-6).mean()
    for epoch in range(250):
        head.train();total=0.;n=0;t=time.time()
        for first in range(0,len(x),64):
            pred=head(x[first:first+64]);loss=loss_fn(pred,y[first:first+64]);assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();total+=float(loss.detach())*len(pred);n+=len(pred);updates+=1
        head.eval();total_val=0.
        with torch.no_grad():
            for first in range(0,len(vx),512):total_val+=float(loss_fn(head(vx[first:first+512]),vy[first:first+512]))*len(vx[first:first+512])
        score=total_val/len(vx);assert np.isfinite(score)
        improved=score<best;stale=0 if improved else stale+1
        if improved:best=score;selected=epoch;torch.save({'head':head.state_dict(),'epoch':epoch,'seed':seed,'validation_loss':score},folder/'best_validation.pt')
        item={'epoch':epoch,'train_negative_Pearson':total/n,'validation_negative_Pearson':score,'stale':stale,'seconds':time.time()-t,'updates':updates}
        history.append(item);update(folder/'history.json',history);update(folder/'status.json',item);print(json.dumps({'seed':seed,'stage':'NATIVE_HEAD',**item}),flush=True)
        if stale>=10:stop='VALIDATION_PATIENCE10';break
    else:stop='RESOURCE_CEILING250'
    dump(folder/'completed.json',{'seed':seed,'epochs':len(history),'updates':updates,'selected_epoch':selected,'stopping':stop,
        'best_validation':best,'seconds_including_cache':time.time()-started,'test_payload_loaded':False,
        'encoder_sha256':sha(pre/'best_validation.pt'),'target_window_presentations':len(x)*len(history),
        'target_sample_presentations':len(x)*len(history)*192,'head_parameters':257})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['intake','check','pretrain','head']);p.add_argument('--seed',type=int,default=1024);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.action=='intake':intake()
    elif a.action=='check':check()
    else:(pretrain if a.action=='pretrain' else head)(a.seed,a.device)
