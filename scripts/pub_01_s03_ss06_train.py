"""Continuous 60-second SS06 train/validation pilot and formal execution."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from m6a_public.continuity.common import detach_state
from m6a_public.continuity.execution import build_variant,VARIANTS
from m6a_public.continuity.objectives import dense_objective
from pub_01_s03_ss06_data import ROOT,LENGTH,WARMUP,Records,dump,read
from pub_01_s03_ss06_contract import verify_contract,verify_pretrained,identities,digest


def tensor(a,device):return torch.from_numpy(np.array(a,copy=True)).unsqueeze(0).to(device)


def metrics(p,y):
    p=np.asarray(p,dtype=np.float64).reshape(-1);y=np.asarray(y,dtype=np.float64).reshape(-1)
    pc=p-p.mean();yc=y-y.mean();den=np.linalg.norm(pc)*np.linalg.norm(yc)
    prediction_std=float(np.sqrt(np.mean(pc**2)));target_std=float(np.sqrt(np.mean(yc**2)))
    return {'r':float(np.dot(pc,yc)/den) if min(prediction_std,target_std)>1e-6 else None,
            'mse':float(np.mean((p-y)**2)),'prediction_std':prediction_std}


def aggregate(rows):
    records=[];persons=[]
    for rid in sorted({r['record'] for r in rows}):
        group=[r for r in rows if r['record']==rid]
        valid=[r['r'] for r in group if r['r'] is not None]
        records.append({k:group[0][k] for k in ('record','subject','source_family')}|
            {'r':float(np.mean(valid)) if valid else None,'mse':float(np.mean([r['mse'] for r in group])),
             'blocks':len(group),'valid_r_blocks':len(valid)})
    for person in sorted({r['subject'] for r in records}):
        group=[r for r in records if r['subject']==person];valid=[r['r'] for r in group if r['r'] is not None]
        persons.append({'subject':person,'r':float(np.mean(valid)) if valid else None,
            'mse':float(np.mean([r['mse'] for r in group])),'records':len(group)})
    valid=[r['r'] for r in persons if r['r'] is not None]
    return {'r':float(np.mean(valid)) if valid else None,'mse':float(np.mean([r['mse'] for r in persons])),
            'persons':len(persons),'records':len(records),'blocks':len(rows),'undefined_r_blocks':sum(r['r'] is None for r in rows)},records,persons


def write_csv(path,rows):
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def reset_post_encoder(state):
    def zeros(v):
        if isinstance(v,torch.Tensor):return torch.zeros_like(v)
        if isinstance(v,dict):return {k:zeros(x) for k,x in v.items()}
        return v
    return {k:v if k in ('encoder','mode') else zeros(v) for k,v in state.items()}


def stream_forward(model,x,state=None,periodic_reset=False):
    if not periodic_reset:return model(x,state)
    pieces=[]
    for start in range(0,x.shape[1],192):
        if state is not None:state=reset_post_encoder(state)
        out=model(x[:,start:start+192],state);state=out.state;pieces.append(out.prediction)
    out.prediction=torch.cat(pieces,1);return out


@torch.no_grad()
def evaluate(model,data,device,pilot=False,periodic_reset=False):
    model.eval();rows=[]
    for r in data.rows:
        x,y=data.arrays(r);out=stream_forward(model,tensor(x[:WARMUP],device),None,periodic_reset);state=out.state
        for block in range(min(1,r['score_blocks']) if pilot else r['score_blocks']):
            start=WARMUP+block*LENGTH
            out=stream_forward(model,tensor(x[start:start+LENGTH],device),state,periodic_reset);state=out.state
            rows.append({'record':r['record'],'subject':r['subject'],'source_family':r['source_family'],
                'block':block,**metrics(out.prediction.cpu().numpy(),y[start:start+LENGTH])})
    return aggregate(rows),rows


def loss_for(name,out,y):
    if name=='m5_prior':return (out.prediction-y).square().mean()
    loss=dense_objective(out.prediction,y,mse_weight=.1)['loss']
    if name=='m4_adaptive':loss=loss+.001*(out.aux['weights'][:,1:]-out.aux['weights'][:,:-1]).square().mean()
    if name=='m5_full':loss=loss+.01*(out.aux['predicted_observation']-out.aux['features'].detach()).square().mean()
    return loss


def records_order(data,seed,steps,pilot=False):
    rng=np.random.default_rng(seed);done=0
    while done<steps:
        for index in rng.permutation(len(data.rows)):
            row=data.rows[index]
            yield row
            done+=min(2,row['score_blocks']) if pilot else row['score_blocks']
            if done>=steps:break


def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


def pretrain(seed,device,steps,outdir,pilot=False):
    if not pilot:verify_contract('pretrain',seed,steps)
    torch.manual_seed(seed);model=build_variant('m3_finetune').to(device);model.set_stage('pretrain')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01)
    train=Records('train');rng=np.random.default_rng(seed);order=rng.permutation(len(train.rows));history=[]
    spans=[(int(i),start) for i in order for start in (range(0,(train.rows[i]['samples']//LENGTH)*LENGTH,LENGTH) if not pilot else (0,))]
    actual=[]
    started=time.perf_counter()
    for step in range(1,steps+1):
        index,start=spans[(step-1)%len(spans)];row=train.rows[index];x,_=train.arrays(row)
        # Fresh-state complete60s trajectory. No carried EMA cache or cross-chunk
        # future targets: teacher is fixed throughout this loss, updated after step.
        xx=tensor(x[start:start+LENGTH],device);model.train();opt.zero_grad(set_to_none=True)
        actual.append({'record':row['record'],'start_sample':start,'samples':LENGTH})
        out=model(xx,None);terms=model.predictive_loss(out,xx);terms['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();model.update_teacher(.99)
        if step%25==0 or step==steps:
            rowlog={'step':step,'loss':float(terms['loss'].detach()),'prediction_loss':float(terms['prediction_loss'].detach()),
                'variance_loss':float(terms['variance_loss'].detach())};history.append(rowlog);print(json.dumps({'pretrain_seed':seed,**rowlog}),flush=True)
    synchronize(device);checkpoint=outdir/'pretrained.pt';torch.save(model.state_dict(),checkpoint)
    dump(outdir/'pretrain_completed.json',{'seed':seed,'steps':steps,'samples':steps*LENGTH,'seconds':time.perf_counter()-started,
         'eligible_records':[r['record'] for r in train.rows],'actual_records':sorted({r['record'] for r in actual}),
         'actual_span_sequence':actual,'actual_spans':len(actual),'unique_actual_records':len({r['record'] for r in actual}),
         'teacher_EMA_after_optimizer_only':True,'fresh_state_complete_segments':True,'history':history,
         'stage':'pilot' if pilot else 'formal','checkpoint_sha256':digest(checkpoint),**({} if pilot else identities())})
    return checkpoint


def run(name,seed,device,steps,pilot=False,pretrained=None):
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    if str(device).startswith('cuda'):torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
    stage='pilot01' if pilot else 'formal01';outdir=ROOT/stage/f'{name}_seed{seed}';outdir.mkdir(parents=True,exist_ok=False)
    if not pilot:
        protocol=verify_contract('downstream',seed,steps,name)
    model=build_variant(name).to(device)
    pretrained_identity=None
    if name in ('m3_frozen','m3_finetune'):
        assert pretrained is not None
        if not pilot:pretrained_identity=verify_pretrained(seed,pretrained)
        model.load_state_dict(torch.load(pretrained,map_location=device,weights_only=True));model.set_stage('downstream')
    parameters=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=.01)
    train=Records('train');valid=Records('validation');selected=None;best=-float('inf');history=[];step=0;loss_sum=0.;active=None
    interval=50 if pilot else protocol['validation_every_steps'];started=time.perf_counter();train_seconds=0.;seen=[]
    for row in records_order(train,seed,steps,pilot=pilot):
        x,y=train.arrays(row);model.train()
        with torch.no_grad():state=model(tensor(x[:WARMUP],device),None).state
        for block in range(min(2,row['score_blocks']) if pilot else row['score_blocks']):
            if step>=steps:break
            start=WARMUP+block*LENGTH;xx=tensor(x[start:start+LENGTH],device);yy=tensor(y[start:start+LENGTH],device)
            synchronize(device);t=time.perf_counter();opt.zero_grad(set_to_none=True);out=model(xx,detach_state(state));loss=loss_for(name,out,yy)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss.backward()
            if active is None:
                active={'registered':sum(p.numel() for p in model.parameters()),'requires_grad':sum(p.numel() for p in parameters),
                    'gradient_tensor_reachable':sum(p.numel() for p in parameters if p.grad is not None),
                    'nonzero_gradient_scalars_first_real_batch':sum(int(torch.count_nonzero(p.grad)) for p in parameters if p.grad is not None)}
            torch.nn.utils.clip_grad_norm_(parameters,1.);state=detach_state(out.state);opt.step();synchronize(device)
            train_seconds+=time.perf_counter()-t;step+=1;loss_sum+=float(loss.detach());seen.append(row['record'])
            if step%interval==0 or step==steps:
                (summary,_,_),_=evaluate(model,valid,device,pilot=pilot)
                score=summary['r'] if name!='m5_prior' and summary['r'] is not None else -summary['mse']
                if score>best:
                    best=score;selected=step;torch.save({'model':model.state_dict(),'variant':name,'seed':seed,'step':step,'validation':summary},outdir/'best_validation.pt')
                item={'step':step,'train_loss':loss_sum/(interval if step%interval==0 else step%interval),'validation':summary,
                    'train_seconds':train_seconds,'elapsed_seconds':time.perf_counter()-started}
                history.append(item);loss_sum=0.;(outdir/'history.json').write_text(json.dumps(history,indent=2))
                print(json.dumps({'variant':name,'seed':seed,**item}),flush=True);model.train()
        if step>=steps:break
    synchronize(device)
    dump(outdir/'train_completed.json',{'variant':name,'seed':seed,'steps':step,'samples':step*LENGTH,'selected_step':selected,
        'selected_score':best,'seconds':time.perf_counter()-started,'train_step_seconds':train_seconds,'device':device,
        'unique_training_records':len(set(seen)),'training_record_sequence':seen,'parameter_counts':active,
        'max_GPU_allocated_bytes':torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
        'audit_signal_read':False,'pretraining_checkpoint':str(pretrained) if pretrained else None,
        'pretrained_identity':pretrained_identity,'checkpoint_sha256':digest(outdir/'best_validation.pt'),
        **({} if pilot else identities()),
        'continuous_60s_loss':True,'state_detach_not_reset_between_adjacent_updates':True,
        'selection_metric':'validation_macro_r' if name!='m5_prior' else 'validation_negative_MSE; autonomous prior has possibly undefined Pearson'})


def pilot():
    dump(ROOT/'pilot_started.json',{'pid':os.getpid(),'epoch':time.time(),'phase':'REAL_TRAIN_VAL_PILOT','audit_read':False})
    # Start with the authorized shared local / M1 real baseline.
    for name in ('local','m1_multi','m2','gru','m1_short','m1_long','m4_fixed','m4_adaptive','m5_eeg','m5_prior','m5_full'):
        run(name,1024,'cpu' if name in ('m5_full','m5_prior') else 'cuda:0',100,pilot=True)
    directory=ROOT/'pilot01/pretrain_seed1024';directory.mkdir()
    checkpoint=pretrain(1024,'cuda:0',100,directory,pilot=True)
    for name in ('m3_frozen','m3_finetune'):run(name,1024,'cuda:0',100,pilot=True,pretrained=checkpoint)
    dump(ROOT/'pilot_completed.json',{'all_variants':list(VARIANTS),'steps_each':100,'selected_by_validation_only':True,'audit_read':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--variant');p.add_argument('--seed',type=int,default=1024)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int);p.add_argument('--pretrained');p.add_argument('--pretrain',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.pilot:pilot()
    elif a.pretrain:
        folder=ROOT/'formal01'/f'pretrain_seed{a.seed}';folder.mkdir(parents=True,exist_ok=False);pretrain(a.seed,a.device,a.steps,folder)
    else:run(a.variant,a.seed,a.device,a.steps,pretrained=a.pretrained)
