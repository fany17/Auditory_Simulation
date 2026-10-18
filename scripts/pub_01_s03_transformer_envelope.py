"""Original envelope-readout endpoint on the frozen public Transformer encoder.

Prediction is EEG→encoder→Conv32/LeakyReLU→envelope, no candidate input.
Encoder/decoder selection uses validation only. All signals stay on2203.
"""
import argparse
from collections import defaultdict
import csv
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pcc(a,b):
    x=a-a.mean(-1,keepdim=True);y=b-b.mean(-1,keepdim=True)
    return F.cosine_similarity(x,y,dim=-1,eps=1e-6)


def readout(cfg):
    return nn.Sequential(nn.Conv1d(8,1,cfg['readout']['kernel_size'],padding='same'),nn.LeakyReLU(.01))


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def freeze_and_encode(cfg,base_cfg,work,out):
    module=importlib.import_module(cfg['encoder_module']);model=module.TransformerCLIP(base_cfg).cuda()
    checkpoint=torch.load(work/cfg['encoder_run']/'best_validation.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model']);model.eval();encoder=model.eeg
    records=json.loads((work/'prepared/manifest_frozen.json').read_text())['records']
    index={r['source_recording']:r for r in rows(Path(cfg['root'])/cfg['index'])}
    scaling=np.load(work/'prepared/train_only_normalization.npz');mean,std=scaling['mean'],scaling['std']
    checked=[];offsets=defaultdict(int);counts=defaultdict(int)
    for r in records:
        contract=index[r['record']];env_path=contract.get('envelope_path')
        if not env_path:raise RuntimeError('no explicit published envelope path in input contract')
        env=np.load(env_path,mmap_mode='r').reshape(-1);length=r['common_samples']
        if len(env)<length:raise RuntimeError('envelope shorter than frozen encoder support')
        count=max(0,(length-cfg['window_samples'])//cfg['hop_samples']+1)
        if not count or not np.isfinite(env[:length]).all():raise RuntimeError('invalid envelope/support')
        item=dict(r,envelope_path=env_path,common_samples=length,readout_windows=count,
            phase_offset=offsets[r['phase']]);checked.append(item);offsets[r['phase']]+=count;counts[r['phase']]+=count
    features={p:np.lib.format.open_memmap(out/(p+'_latent.npy'),mode='w+',dtype='float32',shape=(n,8,192)) for p,n in counts.items()}
    targets={p:np.lib.format.open_memmap(out/(p+'_envelope.npy'),mode='w+',dtype='float32',shape=(n,1,192)) for p,n in counts.items()}
    torch.set_num_threads(4)
    with torch.no_grad():
        for r in checked:
            raw=np.load(r['eeg_path'],mmap_mode='r');eeg=raw.T if raw.shape[0]==64 else raw;env=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)
            for first in range(0,r['readout_windows'],256):
                ids=np.arange(first,min(first+256,r['readout_windows']));starts=ids*96
                x=np.stack([(np.asarray(eeg[s:s+192],dtype=np.float32)-mean)/std for s in starts]).astype(np.float32)
                z=encoder(torch.from_numpy(x).cuda()).transpose(1,2).cpu().numpy()
                a=r['phase_offset']+first;b=a+len(ids);features[r['phase']][a:b]=z
                targets[r['phase']][a:b,0]=np.stack([env[s:s+192] for s in starts])
            print(json.dumps({'encoded_record':r['record'],'phase':r['phase'],'windows':r['readout_windows']}),flush=True)
    for array in list(features.values())+list(targets.values()):array.flush()
    manifest={'config':cfg,'records':checked,'phase_windows':dict(counts),'encoder_best_epoch':checkpoint['epoch'],
        'encoder_frozen':True,'readout_settings_precommitted':True,'candidate_input_at_prediction':False}
    (out/'manifest_frozen.json').write_text(json.dumps(manifest,indent=2));return manifest


def evaluate(model,cfg,out,manifest):
    records=manifest['records'];window=np.hanning(192).astype(np.float64);segment_rows=[];excluded=[];model.eval()
    cached={p:np.load(out/(p+'_latent.npy'),mmap_mode='r') for p in ('test_seen','test_heldout')}
    with torch.no_grad():
        for r in records:
            if not r['phase'].startswith('test'):continue
            first=r['phase_offset'];z=np.asarray(cached[r['phase']][first:first+r['readout_windows']])
            predictions=[]
            for i in range(0,len(z),512):predictions.append(model(torch.from_numpy(z[i:i+512].copy()).cuda()).cpu().numpy()[:,0])
            prediction=np.concatenate(predictions);wrong=np.roll(prediction,max(1,len(prediction)//2),axis=0)
            total=np.zeros(r['common_samples']);null=np.zeros_like(total);weight=np.zeros_like(total)
            for i,(p,q) in enumerate(zip(prediction,wrong)):
                start=i*96;total[start:start+192]+=p*window;null[start:start+192]+=q*window;weight[start:start+192]+=window
            covered=weight>1e-8;total[covered]/=weight[covered];null[covered]/=weight[covered]
            target=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)[:r['common_samples']]
            for start in range(0,r['common_samples']-3840+1,3840):
                keep=covered[start:start+3840];truth=np.asarray(target[start:start+3840])[keep];pred=total[start:start+3840][keep];bad=null[start:start+3840][keep]
                reason=('insufficient_overlap_support' if len(truth)<3800 else
                    'constant_target' if np.std(truth)==0 else
                    'constant_prediction' if np.std(pred)==0 else
                    'constant_shuffle_prediction' if np.std(bad)==0 else None)
                if reason:
                    excluded.append({'group':r['phase'],'subject':r['subject'],'record':r['record'],
                        'start_sample':start,'samples':len(truth),'reason':reason});continue
                if not all(np.isfinite(a).all() for a in (truth,pred,bad)):raise RuntimeError('nonfinite test segment')
                segment_rows.append({'group':r['phase'],'subject':r['subject'],'record':r['record'],'start_sample':start,
                    'samples':len(truth),'pearson_r':float(np.corrcoef(truth,pred)[0,1]),'EEG_shuffle_r':float(np.corrcoef(truth,bad)[0,1])})
    subjects=[]
    for key in sorted({(r['group'],r['subject']) for r in segment_rows}):
        selected=[r for r in segment_rows if (r['group'],r['subject'])==key]
        subjects.append({'group':key[0],'subject':key[1],'segments':len(selected),
            'pearson_r':float(np.mean([r['pearson_r'] for r in selected])),'EEG_shuffle_r':float(np.mean([r['EEG_shuffle_r'] for r in selected]))})
    groups={}
    for phase in ('test_seen','test_heldout'):
        selected=[r for r in subjects if r['group']==phase];values=np.array([r['pearson_r'] for r in selected]);rng=np.random.default_rng(20261015)
        groups[phase]={'subjects':len(selected),'pearson_r':float(values.mean()),
            'subject_bootstrap95':np.percentile(values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1),[2.5,97.5]).tolist(),
            'EEG_shuffle_r':float(np.mean([r['EEG_shuffle_r'] for r in selected]))}
    groups['weighted_pearson_r']=2/3*groups['test_seen']['pearson_r']+1/3*groups['test_heldout']['pearson_r']
    groups['excluded_segments']=len(excluded)
    (out/'excluded_segments.json').write_text(json.dumps(excluded,indent=2))
    for filename,data in (('test_per_segment.csv',segment_rows),('test_per_subject.csv',subjects)):
        with (out/filename).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    return groups


def main(config):
    cfg=json.loads(config.read_text());work=Path(cfg['work']);base_cfg=json.loads((work/cfg['baseline_config']).read_text())
    if cfg['patient_data']!='FORBIDDEN' or not str(work).startswith('/home/fanyu/auditory_simulation_m6a/'):raise RuntimeError('2203 public-only')
    if not (work/cfg['encoder_run']/'completed.json').exists():raise RuntimeError('contrastive model not complete')
    out=work/cfg['output'];out.mkdir(exist_ok=False);(out/'config_frozen.json').write_text(json.dumps(cfg,indent=2))
    started=time.time();manifest=freeze_and_encode(cfg,base_cfg,work,out);torch.manual_seed(cfg['seed']);model=readout(cfg).cuda()
    x=torch.from_numpy(np.load(out/'train_latent.npy').copy()).cuda();y=torch.from_numpy(np.load(out/'train_envelope.npy').copy()).cuda()
    vx=torch.from_numpy(np.load(out/'val_latent.npy').copy()).cuda();vy=torch.from_numpy(np.load(out/'val_envelope.npy').copy()).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['readout']['lr']);best=float('inf');best_epoch=None;stale=0;history=[]
    for epoch in range(cfg['readout']['max_epochs']):
        model.train();loss_sum=0.;samples=0
        for first in range(0,len(x),64):
            pred=model(x[first:first+64]);loss=-pcc(pred,y[first:first+64]).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite readout training loss')
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();loss_sum+=loss.item()*len(pred);samples+=len(pred)
        model.eval();val_sum=0.
        with torch.no_grad():
            for first in range(0,len(vx),512):val_sum+=(-pcc(model(vx[first:first+512]),vy[first:first+512]).sum()).item()
        val=val_sum/len(vx)
        if not np.isfinite(val):raise RuntimeError('nonfinite readout validation loss')
        improved=val<best;stale=0 if improved else stale+1
        if improved:best=val;best_epoch=epoch;torch.save(model.state_dict(),out/'best_validation_readout.pt')
        row={'epoch':epoch,'train_negative_Pearson':loss_sum/samples,'validation_negative_Pearson':val,'stale':stale};history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
        if stale>=cfg['readout']['patience']:stopping='VALIDATION_PATIENCE10';break
    else:stopping='RESOURCE_CEILING250'
    model.load_state_dict(torch.load(out/'best_validation_readout.pt',weights_only=True));test=evaluate(model,cfg,out,manifest)
    result={'protocol':'Transformer_frozen_encoder_Conv32_envelope_available_public','test':test,'epochs':len(history),'best_epoch':best_epoch,
        'stopping':stopping,'seconds':time.time()-started,'test_evaluations':1,'candidate_input_at_prediction':False,'differences':cfg['differences']}
    (out/'result.json').write_text(json.dumps(result,indent=2));(out/'completed.json').write_text(json.dumps({'status':'ACTUAL_TEST_FINISHED','acceptance':'REVIEW_REQUIRED'}));print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);main(parser.parse_args().config)
