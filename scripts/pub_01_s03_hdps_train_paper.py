"""Train one precommitted HDPS100Hz/M configuration and evaluate test once.

All tensor/model work stays on2203. Validation macroTop10 selects checkpoint;
test is opened by evaluation only after stopping. Available public raw cohort
is reported explicitly and never substituted by64Hz or resplit.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pub_01_s03_hdps_paper_model import PaperHDPS, similarity
from pub_01_s03_cohort_policy import validate_training_cohort


class Windows(Dataset):
    def __init__(self, prepared, phase):
        self.x = np.load(prepared / (phase+'_eeg.npy'), mmap_mode='r')
        self.pair = np.load(prepared / (phase+'_pairs.npy'), mmap_mode='r')
        self.f = np.load(prepared / 'feature_windows.npy', mmap_mode='r')

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, i):
        subject, feature = self.pair[i]
        return torch.from_numpy(self.x[i].copy()), torch.from_numpy(self.f[feature].copy()), int(subject)


def evaluate(model, prepared, phase, groups, collect=False):
    dataset = Windows(prepared, phase)
    rows, details = [], []
    model.eval()
    with torch.inference_mode():
        for subject, indices in groups[phase].items():
            if not indices:
                rows.append({'subject':subject,'candidate_n':0,'queries':0,'status':'UNAVAILABLE','Top1':None,'Top5':None,'Top10':None})
                continue
            predictions, candidates = [], []
            for start in range(0,len(indices),256):
                selected = np.asarray(indices[start:start+256])
                pair = dataset.pair[selected]
                x = torch.from_numpy(np.asarray(dataset.x[selected]).copy()).to('cuda:0')
                f = torch.from_numpy(np.asarray(dataset.f[pair[:,1]]).copy()).to('cuda:0')
                s = torch.from_numpy(np.asarray(pair[:,0]).copy()).to('cuda:0')
                z, audio = model(x,f,s)
                predictions.append(z); candidates.append(audio)
            z, audio = torch.cat(predictions), torch.cat(candidates)
            scores = similarity(z, audio)
            if not torch.isfinite(scores).all():
                raise RuntimeError('nonfinite evaluation scores')
            order = torch.argsort(scores, dim=1, descending=True)
            true = torch.arange(len(indices),device='cuda:0')[:,None]
            row = {'subject':subject,'candidate_n':len(indices),'queries':len(indices),'status':'EVALUATED'}
            for k in (1,5,10):
                row['Top'+str(k)] = (order[:,:min(k,len(indices))]==true).any(1).float().mean().item()
            if collect:
                # A model-independent cyclic wrong-EEG control from the same
                # single forward pass; no test-based checkpoint selection.
                shifted=torch.roll(order,shifts=max(1,len(indices)//2),dims=0)
                for k in (1,5,10):
                    row['EEG_shuffle_Top'+str(k)]=(shifted[:,:min(k,len(indices))]==true).any(1).float().mean().item()
                ranks=(order==true).nonzero(as_tuple=False)[:,1]+1
                for i in range(len(indices)):
                    details.append({'subject':subject,'phase_window_index':indices[i],
                        'target_feature_id':int(dataset.pair[indices[i],1]),
                        'predicted_feature_id':int(dataset.pair[indices[int(order[i,0])],1]),
                        'candidate_n':len(indices),'true_rank':int(ranks[i])})
            rows.append(row)
    valid = [r for r in rows if r['status']=='EVALUATED']
    if not valid:
        raise RuntimeError('no evaluable subjects')
    aggregate = {metric:float(np.mean([r[metric] for r in valid])) for metric in ('Top1','Top5','Top10')}
    aggregate['subjects'] = len(valid)
    aggregate['queries'] = sum(r['queries'] for r in valid)
    aggregate['candidate_min'] = min(r['candidate_n'] for r in valid)
    aggregate['candidate_median'] = float(np.median([r['candidate_n'] for r in valid]))
    aggregate['candidate_max'] = max(r['candidate_n'] for r in valid)
    if collect:
        for metric in ('Top1','Top5','Top10'):
            aggregate['EEG_shuffle_'+metric]=float(np.mean([r['EEG_shuffle_'+metric] for r in valid]))
            values=np.asarray([r[metric] for r in valid])
            rng=np.random.default_rng(20261015)
            means=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
            aggregate[metric+'_subject_bootstrap95']=np.percentile(means,[2.5,97.5]).tolist()
        aggregate['chance_Top1']=float(np.mean([1/r['candidate_n'] for r in valid]))
        aggregate['chance_Top5']=float(np.mean([min(5,r['candidate_n'])/r['candidate_n'] for r in valid]))
        aggregate['chance_Top10']=float(np.mean([min(10,r['candidate_n'])/r['candidate_n'] for r in valid]))
        return aggregate,rows,details
    return aggregate, rows


def main(config):
    cfg = json.loads(config.read_text())
    base = Path(cfg['work'])
    if not str(base).startswith('/home/fanyu/auditory_simulation_m6a/') or cfg['patient_data']!='FORBIDDEN':
        raise RuntimeError('server2203 public-only')
    prepared = Path(cfg.get('prepared_dir', str(base / 'prepared_paper_m01')))
    if not (prepared / 'completed.json').is_file():
        raise RuntimeError('frozen inputs not ready')
    frozen = json.loads((prepared / 'config_frozen.json').read_text())
    if cfg != frozen:
        raise RuntimeError('configuration differs from pretraining freeze')
    manifest = json.loads((prepared / 'manifest_frozen.json').read_text())
    validate_training_cohort(manifest,cfg)
    groups = json.loads((prepared / 'candidate_groups.json').read_text())
    run = Path(cfg.get('run_dir', str(base / 'train_paper_m01_seed1024')))
    run.mkdir(exist_ok=False)
    (run / 'config_frozen.json').write_text(json.dumps(cfg,indent=2))
    training = cfg['training']
    random.seed(training['seed']); np.random.seed(training['seed']); torch.manual_seed(training['seed'])
    torch.cuda.manual_seed_all(training['seed'])
    torch.set_num_threads(4)
    model = PaperHDPS(cfg,len(manifest['subjects'])).to('cuda:0')
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model,device_ids=[0,1])
    optimizer = torch.optim.Adam(model.parameters(),lr=training['lr'])
    dataset = Windows(prepared,'train')
    loader = DataLoader(dataset,batch_size=training['batch_size'],shuffle=True,num_workers=2,pin_memory=True,
        drop_last=training['drop_last'],generator=torch.Generator().manual_seed(training['seed']))
    history, best, stale, best_epoch = [], -1., 0, None
    started = time.time()
    for epoch in range(training['max_epochs']):
        model.train()
        loss_sum, samples = 0.,0
        epoch_start = time.time()
        for step,(eeg,target,subject) in enumerate(loader):
            eeg,target,subject = eeg.to('cuda:0',non_blocking=True), target.to('cuda:0',non_blocking=True), subject.to('cuda:0',non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z,audio = model(eeg,target,subject)
            scores = similarity(z,audio)
            loss = torch.nn.functional.cross_entropy(scores / training['temperature'],torch.arange(len(eeg),device='cuda:0'))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(eeg); samples += len(eeg)
            if step%100==0:
                print(json.dumps({'epoch':epoch,'step':step,'batch_loss':loss.item(),'elapsed_seconds':time.time()-epoch_start}),flush=True)
        aggregate,_ = evaluate(model,prepared,'val',groups)
        improved = aggregate['Top10']>best
        stale = 0 if improved else stale+1
        if improved:
            best,best_epoch = aggregate['Top10'],epoch
            core = model.module if hasattr(model,'module') else model
            torch.save({'model':core.state_dict(),'epoch':epoch,'validation':aggregate,'config':cfg},run / 'best_validation.pt')
        row = {'epoch':epoch,'training_loss':loss_sum/samples,'samples':samples,'validation':aggregate,
            'improved':improved,'stale_epochs':stale,'seconds':time.time()-epoch_start}
        history.append(row)
        (run / 'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(row),flush=True)
        if stale>=training['patience']:
            stopping = 'VALIDATION_TOP10_PATIENCE5'
            break
    else:
        stopping = 'RESOURCE_CEILING300_BEFORE_PATIENCE'
    checkpoint = torch.load(run / 'best_validation.pt',map_location='cpu',weights_only=True)
    core = model.module if hasattr(model,'module') else model
    core.load_state_dict(checkpoint['model'])
    # The single model-independent frozen test evaluation begins here.
    aggregate,rows,details = evaluate(model,prepared,'test',groups,collect=True)
    with (run / 'test_per_subject.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (run / 'test_query_ranks.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    result = {'protocol':cfg['protocol'],'feature':'M_only','seed':training['seed'],'test':aggregate,
        'best_epoch':best_epoch,'best_validation_top10':best,'epochs':len(history),'stopping':stopping,
        'seconds':time.time()-started,'strict_published_cohort_match':manifest['strict_published_cohort_match'],
        'input_failure_count':len(manifest['failures']),'incomplete_subjects':manifest['incomplete_subjects'],
        'actual_subjects':manifest['subjects'],'excluded_subjects':manifest.get('excluded_subjects',[]),
        'cohort_policy':cfg.get('cohort_policy',{'mode':'strict_complete'}),
        'precommitted_differences':cfg['precommitted_differences'],'paper_M_reference':'NOT_YET_VISUALLY_VERIFIED',
        'MWG20percent_comparison':'FORBIDDEN','test_evaluations':1,'trained_parameters':sum(p.numel() for p in model.parameters())}
    (run / 'result.json').write_text(json.dumps(result,indent=2))
    (run / 'completed.json').write_text(json.dumps({'status':'ACTUAL_TEST_FINISHED','acceptance':'REVIEW_REQUIRED'}))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    main(parser.parse_args().config)
