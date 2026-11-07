"""Predeclared validation-only post-Hann native-P smoothing control."""
import json
import math
import numpy as np
import torch
from pub_01_s03_ss07_baseline import ROOT,read,dump,sha
from pub_01_s03_ss07_train import NativeHead,LatentData
from pub_01_s03_ss07_history import causal_ema


def score_windows(predictions,data,tau):
    assert data.fixed_mask['phase']=='val' and predictions.shape==(len(data.x),192)
    h=np.hanning(192);segments=[]
    eligibility={(r['record'],r['start']):r for r in data.fixed_mask['blocks']}
    for row in data.rows:
        n=row['common_samples'];offset=row['offset'];total=np.zeros(n);weight=np.zeros(n)
        for i in range(row['windows']):
            s=i*96;total[s:s+192]+=predictions[offset+i]*h;weight[s:s+192]+=h
        keep=weight>1e-8;total[keep]/=weight[keep]
        total=causal_ema(total,keep,tau)
        truth=np.load(row['envelope_path'],mmap_mode='r').reshape(-1)
        for s in range(0,n-3840+1,3840):
            m=eligibility[(row['record'],s)];covered=keep[s:s+3840]
            assert int(covered.sum())==m['samples']
            if not m['eligible']:continue
            a=total[s:s+3840][covered];b=np.asarray(truth[s:s+3840])[covered]
            assert np.isfinite(a).all() and np.std(a)>0 and np.isfinite(b).all() and np.std(b)>0
            segments.append({'subject':row['subject'],'record':row['record'],'story':row['story'],
                'start':s,'r':float(np.corrcoef(a,b)[0,1]),'samples':int(covered.sum())})
    people=[{'subject':s,'r':float(np.mean([r['r'] for r in segments if r['subject']==s]))}
        for s in sorted({r['subject'] for r in segments})]
    assert len(people)==49 and len(segments)==752
    return {'macro_r':float(np.mean([r['r'] for r in people])),'subjects':49,'blocks':752},people,segments


def main(device='cuda:0'):
    phase=read(ROOT/'EMA_PROTOCOL.json');assert phase['frozen']
    assert phase['incremental_protocol_sha256']==sha(ROOT/'INCREMENTAL_PROTOCOL.json')
    for path,digest in phase['sources'].items():assert sha(path)==digest
    p=read(ROOT/'INCREMENTAL_PROTOCOL.json');seeds=p['screening_seeds'];taus=p['EMA_control']['taus_seconds']
    folder=ROOT/'ema_validation';folder.mkdir(exist_ok=False);all_scores={tau:[] for tau in taus}
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=p['precision']['TF32_matmul']
    torch.backends.cudnn.allow_tf32=p['precision']['TF32_cudnn']
    for seed in seeds:
        source=ROOT/f'baseline/head_seed{seed}';assert read(source/'completed.json')['seed']==seed
        checkpoint=torch.load(source/'best_validation.pt',map_location=device,weights_only=True)
        assert checkpoint['seed']==seed
        model=NativeHead().to(device).eval();model.head.load_state_dict(checkpoint['head'])
        data=LatentData(seed,'val',device);predictions=[]
        with torch.no_grad():
            for first in range(0,len(data.x),64):predictions.append(model(data.x[first:first+64])[0][:,0].cpu().numpy())
        predictions=np.concatenate(predictions)
        for tau in taus:
            summary,people,segments=score_windows(predictions,data,tau);assert math.isfinite(summary['macro_r'])
            all_scores[tau].append(summary['macro_r'])
            dump(folder/f'seed{seed}_tau{tau}.json',{'seed':seed,'tau':tau,'summary':summary,'people':people,
                'segments':segments,'baseline_head_sha256':sha(source/'best_validation.pt'),'test_read':False})
        del model,data,predictions
        torch.cuda.empty_cache()
    means={tau:math.fsum(values)/len(seeds) for tau,values in all_scores.items()}
    best=sorted(taus,key=lambda tau:(-means[tau],tau))[0]
    dump(ROOT/'EMA_SELECTION.json',{'frozen':True,'tau_seconds':best,'validation_means':means,
        'screening_seeds':seeds,'paired_scores':all_scores,'apply_same_tau_to_all_final5seeds':p['final_seeds'],
        'selection':'mean3seed original fullrecord Hann60s->person validation macro; exacttie smaller tau',
        'parent_protocol_sha256':sha(ROOT/'INCREMENTAL_PROTOCOL.json'),'EMA_protocol_sha256':sha(ROOT/'EMA_PROTOCOL.json'),
        'extra_NN_training':False,'test_read':False})
    print(json.dumps(read(ROOT/'EMA_SELECTION.json')),flush=True)


if __name__=='__main__':main()
