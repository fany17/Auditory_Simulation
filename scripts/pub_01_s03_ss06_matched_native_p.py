"""Native P optimizer/order restoration; frozen C/O and base module unchanged.

Pre-execution amendment: SS02 envelope.py:123 Adam and sequential offsets,
train.py:131 AdamW, neither source clips gradients. Scientific budget, loss,
windows, selection, and three fresh seeds remain frozen; no P formal updates
had occurred. Old P pilot is engineering history only.
"""
from pub_01_s03_ss06_matched import *
from pub_01_s03_ss06_matched import guard_formal as common_guard


def guard_formal():
    protocol=common_guard()
    amendment=read(ROOT/'P_NATIVE_AMENDMENT_02.json')
    assert amendment['frozen_before_P_formal_updates']
    assert sha(Path(__file__))==amendment['source_sha256']
    return protocol

def pearson_loss(pred,y):
    # Same centered cosine and separate norm epsilon as SS02 pcc().
    x=pred-pred.mean(1,keepdim=True);target=y-y.mean(1,keepdim=True)
    return -torch.nn.functional.cosine_similarity(x,target,dim=1,eps=1e-6).mean()


def pretrain(seed,device,pilot=False):
    setup(seed,device)
    if not pilot:guard_formal()
    folder=ROOT/('pilot_native03' if pilot else 'formal01')/f'P_pretrain_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    model=pair().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    train=Inputs('train');val=Inputs('validation');plan=schedule(train.rows,seed);vp=schedule(val.rows,20261026)
    if pilot:plan=plan[:20];vp=vp[:2]
    best=float('inf');selected=None;history=[];started=time.perf_counter();seen=[]
    checks=sorted(set([len(plan)] if pilot else [max(1,len(plan)*i//4) for i in (1,2,3,4)]))
    for step,entries in enumerate(plan,1):
        model.train();opt.zero_grad(set_to_none=True);x,a=train.contrast(entries,device);loss=model.loss(x,a,None)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
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
    stage='pilot_native03' if pilot else 'formal01';folder=ROOT/stage/f'P_head_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    pre=ROOT/stage/f'P_pretrain_seed{seed}/best_validation.pt';assert pre.is_relative_to(ROOT/stage)
    learned=pair().to(device);learned.load_state_dict(torch.load(pre,map_location=device,weights_only=True)['model'])
    model=NativeStrong().to(device);model.encoder.load_state_dict(learned.eeg.state_dict());del learned
    torch.manual_seed(seed);model.head[0].reset_parameters()
    for p in model.encoder.parameters():p.requires_grad_(False)
    model.encoder.eval();opt=torch.optim.Adam(model.head.parameters(),lr=.001)
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
    order=np.arange(len(latents['train']));batches=[order[j:j+64] for j in range(0,len(order),64)]
    if pilot:batches=batches[:min(20,len(batches))]
    checks=sorted(set([len(batches)] if pilot else [len(batches)*i//4 for i in (1,2,3,4)]))
    best=float('inf');selected=None;history=[];started=time.perf_counter();seen=[]
    for step,indices in enumerate(batches,1):
        model.head.train();opt.zero_grad(set_to_none=True)
        z=tensor(latents['train'][indices],device);y=tensor(targets['train'][indices],device)
        loss=pearson_loss(model.head(z.transpose(1,2)).transpose(1,2),y);assert torch.isfinite(loss)
        loss.backward();opt.step();seen.extend(indices.tolist())
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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pretrain','head']);p.add_argument('--seed',type=int,default=1024);p.add_argument('--device',default='cuda:0');p.add_argument('--pilot',action='store_true');a=p.parse_args()
    (pretrain if a.action=='pretrain' else head)(a.seed,a.device,a.pilot)
