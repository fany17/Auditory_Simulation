"""Single published Transformer-CLIP architecture on available public folds.

Original model code is not imported. Architecture/functions inspected; not a
clean-room claim. Derived2AFC is explicitly distinct from hidden challenge IDs.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pub_01_s03_method_models import AuditoryTransformer, SpeechLSTM


class TransformerCLIP(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.eeg=AuditoryTransformer(cfg['transformer']);self.audio=SpeechLSTM(cfg['transformer'])
        self.log_scale=nn.Parameter(torch.tensor(cfg['transformer']['initial_logit_parameter']))

    def forward(self,eeg,audio):
        return F.normalize(self.eeg(eeg).flatten(1),dim=-1),F.normalize(self.audio(audio).flatten(1),dim=-1)


class StoryBatches:
    def __init__(self,prepared,features,phase,batch_size):
        self.x=np.load(prepared/(phase+'_eeg.npy'),mmap_mode='r')
        self.rows=json.loads((prepared/(phase+'_windows.json')).read_text())
        self.features={r['story']:np.load(r['path'],mmap_mode='r') for r in features}
        self.batch_size=batch_size;by_story=defaultdict(lambda:defaultdict(list));subjects=defaultdict(set)
        for i,r in enumerate(self.rows):
            by_story[r['story']][r['start']].append(i);subjects[r['story']].add(r['subject'])
        self.plans=[]
        for story,times in sorted(by_story.items()):
            starts=sorted(times);nblocks=len(starts)//batch_size
            if not nblocks:
                continue
            block_len=len(starts)//nblocks
            for b in range(nblocks):
                block=starts[b*block_len:(b+1)*block_len]
                self.plans.extend([(story,block,times)]*len(subjects[story]))
        if not self.plans:
            raise RuntimeError('no same-story batches available')

    def batches(self,seed,shuffle):
        rng=np.random.default_rng(seed);order=np.arange(len(self.plans))
        if shuffle:rng.shuffle(order)
        for i in order:
            story,times,lookup=self.plans[i]
            starts=rng.choice(times,size=self.batch_size,replace=False)
            indices=[int(rng.choice(lookup[int(start)])) for start in starts]
            eeg=torch.from_numpy(np.asarray(self.x[indices]).copy())
            audio=torch.from_numpy(np.stack([self.features[story][start:start+192] for start in starts]))
            yield eeg,audio


def contrastive(model,eeg,audio):
    z,f=model(eeg,audio);core=model.module if hasattr(model,'module') else model
    score=(z@f.T)*core.log_scale.exp();true=torch.arange(len(z),device=z.device)
    return (F.cross_entropy(score,true)+F.cross_entropy(score.T,true))/2


def evaluate_2afc(model,cfg,prepared,features):
    work=Path(cfg['work']);core=model.module if hasattr(model,'module') else model
    model.eval();records=json.loads((prepared/'manifest_frozen.json').read_text())['records']
    with (prepared/'test_queries_frozen.csv').open(newline='') as stream:queries=list(csv.DictReader(stream))
    speech={r['story']:np.load(r['path'],mmap_mode='r') for r in features}
    # One frozen inference pass; the cyclic EEG control reuses embeddings.
    phase_vectors={}
    with torch.no_grad():
        for phase in ('test_seen','test_heldout'):
            x=np.load(prepared/(phase+'_eeg.npy'),mmap_mode='r');vectors=[]
            for start in range(0,len(x),128):
                tensor=torch.from_numpy(np.asarray(x[start:start+128]).copy()).to('cuda:0')
                vectors.append(F.normalize(core.eeg(tensor).flatten(1),dim=1).cpu().numpy())
            phase_vectors[phase]=np.concatenate(vectors)
        keys=sorted({(records[int(q['record_id'])]['story'],int(q[field])) for q in queries for field in ('match_start','mismatch_start')})
        encoded_audio={}
        for first in range(0,len(keys),128):
            group=keys[first:first+128]
            tensor=torch.from_numpy(np.stack([speech[story][start:start+192] for story,start in group])).to('cuda:0')
            vectors=F.normalize(core.audio(tensor).flatten(1),dim=1).cpu().numpy()
            encoded_audio.update(zip(group,vectors))
    record_phase_indices=defaultdict(list)
    for phase in phase_vectors:
        rows=json.loads((prepared/(phase+'_windows.json')).read_text())
        for i,r in enumerate(rows):record_phase_indices[(phase,r['record_id'])].append(i)
    shuffled_index={}
    for key,indices in record_phase_indices.items():
        rolled=np.roll(indices,max(1,len(indices)//2))
        shuffled_index.update({(key[0],i):int(j) for i,j in zip(indices,rolled)})
    answers=[];subject_scores=defaultdict(list);subject_null=defaultdict(list)
    for q in queries:
        phase=q['phase'];i=int(q['eeg_window_index']);record_id=int(q['record_id']);story=records[record_id]['story']
        z=phase_vectors[phase][i];wrong=phase_vectors[phase][shuffled_index[(phase,i)]]
        a=encoded_audio[(story,int(q['match_start']))];b=encoded_audio[(story,int(q['mismatch_start']))]
        correct_candidate=int(q['matched_candidate']);scores=[float(z@a),float(z@b)];null=[float(wrong@a),float(wrong@b)]
        if correct_candidate==1:scores.reverse();null.reverse()
        pred=int(np.argmax(scores));null_pred=int(np.argmax(null))
        key=(phase,q['subject']);subject_scores[key].append(pred==correct_candidate);subject_null[key].append(null_pred==correct_candidate)
        answers.append(dict(q,prediction=pred,correct=pred==correct_candidate,score0=scores[0],score1=scores[1],EEG_shuffle_correct=null_pred==correct_candidate))
    rows=[{'group':phase,'subject':subject,'queries':len(values),'accuracy_2afc':float(np.mean(values)),
        'EEG_shuffle_accuracy':float(np.mean(subject_null[(phase,subject)]))} for (phase,subject),values in sorted(subject_scores.items())]
    aggregates={}
    for phase in ('test_seen','test_heldout'):
        selected=[r for r in rows if r['group']==phase]
        values=np.array([r['accuracy_2afc'] for r in selected]);rng=np.random.default_rng(20261015)
        ci=np.percentile(values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1),[2.5,97.5]).tolist()
        aggregates[phase]={'subjects':len(selected),'queries':sum(r['queries'] for r in selected),
            'accuracy_2afc':float(values.mean()),'subject_bootstrap95':ci,
            'EEG_shuffle_accuracy':float(np.mean([r['EEG_shuffle_accuracy'] for r in selected]))}
    aggregates['weighted_2afc']=2/3*aggregates['test_seen']['accuracy_2afc']+1/3*aggregates['test_heldout']['accuracy_2afc']
    return aggregates,rows,answers


def main(config,run_name='training_seed1024'):
    cfg=json.loads(config.read_text());work=Path(cfg['work']);prepared=work/'prepared';out=work/run_name
    if cfg['patient_data']!='FORBIDDEN' or not str(work).startswith('/home/fanyu/auditory_simulation_m6a/'):
        raise RuntimeError('2203 public-only')
    if not (prepared/'completed.json').is_file():raise RuntimeError('prepared freeze incomplete')
    if cfg!=json.loads((work/'scope/config_frozen.json').read_text()):raise RuntimeError('config differs from freeze')
    out.mkdir(exist_ok=False);(out/'config_frozen.json').write_text(json.dumps(cfg,indent=2))
    train=cfg['training'];random.seed(train['seed']);np.random.seed(train['seed']);torch.manual_seed(train['seed']);torch.cuda.manual_seed_all(train['seed']);torch.set_num_threads(4)
    features=json.loads((work/'features/manifest.json').read_text());training=StoryBatches(prepared,features,'train',train['batch_size']);validation=StoryBatches(prepared,features,'val',train['batch_size'])
    model=TransformerCLIP(cfg).to('cuda:0')
    if torch.cuda.device_count()>1:model=nn.DataParallel(model,device_ids=[0,1])
    opt=torch.optim.AdamW(model.parameters(),lr=train['lr'],weight_decay=train['weight_decay'])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=train['scheduler_step'],gamma=train['scheduler_gamma'])
    best=float('inf');best_epoch=None;stale=0;history=[];started=time.time()
    for epoch in range(train['epochs']):
        model.train();loss_sum=0;count=0;epoch_start=time.time()
        for step,(eeg,audio) in enumerate(training.batches(train['seed']+epoch,True)):
            eeg,audio=eeg.to('cuda:0'),audio.to('cuda:0');opt.zero_grad(set_to_none=True);loss=contrastive(model,eeg,audio)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            loss.backward();opt.step();loss_sum+=loss.item();count+=1
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'loss':loss.item(),'seconds':time.time()-epoch_start}),flush=True)
        model.eval();val_losses=[]
        # DataParallel propagates no_grad but not thread-local InferenceMode;
        # cudnn LSTM flattening must receive normal tensors in replica threads.
        with torch.no_grad():
            for eeg,audio in validation.batches(20261015,False):
                val_losses.append(contrastive(model,eeg.to('cuda:0'),audio.to('cuda:0')).item())
        val=float(np.mean(val_losses));improved=val<best;stale=0 if improved else stale+1
        if improved:
            best=val;best_epoch=epoch;core=model.module if hasattr(model,'module') else model
            torch.save({'model':core.state_dict(),'epoch':epoch,'validation_loss':val,'config':cfg},out/'best_validation.pt')
        row={'epoch':epoch,'train_loss':loss_sum/count,'validation_loss':val,'stale':stale,'seconds':time.time()-epoch_start,'train_batches':count,'validation_batches':len(val_losses)}
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True);scheduler.step()
        if stale>=train['patience']:stopping='VALIDATION_LOSS_PATIENCE15';break
    else:stopping='RESOURCE_CEILING100'
    checkpoint=torch.load(out/'best_validation.pt',map_location='cpu',weights_only=True);core=model.module if hasattr(model,'module') else model;core.load_state_dict(checkpoint['model'])
    aggregates,rows,queries=evaluate_2afc(model,cfg,prepared,features)
    for filename,data in (('test_per_subject.csv',rows),('test_queries_scored.csv',queries)):
        with (out/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    result={'protocol':cfg['protocol'],'test':aggregates,'seed':train['seed'],'best_epoch':best_epoch,'epochs':len(history),
        'stopping':stopping,'seconds':time.time()-started,'precommitted_differences':cfg['precommitted_differences'],
        'official_test_equivalence':False,'paper87percent_comparison':'NOT_SAME_COHORT_TEST_PACKET_OR_ENSEMBLE','test_evaluations':1}
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'completed.json').write_text(json.dumps({'status':'ACTUAL_TEST_FINISHED','acceptance':'REVIEW_REQUIRED'}));print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--run-name',default='training_seed1024');args=parser.parse_args();main(args.config,args.run_name)
