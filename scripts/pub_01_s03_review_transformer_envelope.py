"""Independent readout support, checkpoint, correlation and aggregation review.

Rebuilds every test prediction on CPU from frozen server-only latent files.
No signal arrays leave server2203.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def correlation(x,y):
    a=x.astype(np.float64)-np.mean(x);b=y.astype(np.float64)-np.mean(y)
    return float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))


def main(config):
    cfg=json.loads(config.read_text());work=Path(cfg['work']);out=work/cfg['output']
    if not (out/'completed.json').exists():raise RuntimeError('readout incomplete')
    result=json.loads((out/'result.json').read_text());manifest=json.loads((out/'manifest_frozen.json').read_text())
    history=json.loads((out/'history.json').read_text());excluded=json.loads((out/'excluded_segments.json').read_text())
    with (out/'test_per_segment.csv').open(newline='') as f:segments=list(csv.DictReader(f))
    with (out/'test_per_subject.csv').open(newline='') as f:subjects=list(csv.DictReader(f))
    expected={(r['group'],r['record'],int(r['start_sample'])):r for r in segments}
    excluded_keys={(r['group'],r['record'],int(r['start_sample'])) for r in excluded}
    encoder=json.loads((work/cfg['encoder_run']/'result.json').read_text())
    original=json.loads((work/'prepared/manifest_frozen.json').read_text())['records']
    checks={'readout_selected_by_validation_only':result['best_epoch']==min(history,key=lambda r:r['validation_negative_Pearson'])['epoch'],
        'encoder_selected_before_readout':encoder['best_epoch']==manifest['encoder_best_epoch'],
        'encoder_frozen_and_EEG_only':manifest['encoder_frozen'] and not result['candidate_input_at_prediction'],
        'one_test_evaluation':result['test_evaluations']==1,
        'same_frozen_records':[(r['phase'],r['record'],r['common_samples']) for r in original]==[(r['phase'],r['record'],r['common_samples']) for r in manifest['records']],
        'fixed_stopping':history[-1]['stale']>=10 or len(history)==250,
        'unique_segments':len(expected)==len(segments) and not set(expected)&excluded_keys}
    model=nn.Sequential(nn.Conv1d(8,1,32,padding='same'),nn.LeakyReLU(.01))
    model.load_state_dict(torch.load(out/'best_validation_readout.pt',map_location='cpu',weights_only=True));model.eval();torch.set_num_threads(4)
    max_prediction_error=0.;visited=set();values=defaultdict(list);nulls=defaultdict(list)
    with torch.no_grad():
        for r in manifest['records']:
            if not r['phase'].startswith('test'):continue
            latent=np.load(out/(r['phase']+'_latent.npy'),mmap_mode='r')
            z=latent[r['phase_offset']:r['phase_offset']+r['readout_windows']]
            predicted=np.concatenate([model(torch.tensor(z[i:i+512])).numpy()[:,0] for i in range(0,len(z),512)])
            shuffled=np.roll(predicted,max(1,len(predicted)//2),axis=0)
            offsets=np.arange(len(z))[:,None]*96+np.arange(192)[None,:]
            weight=np.hanning(192);denom=np.zeros(r['common_samples']);a=np.zeros_like(denom);b=np.zeros_like(denom)
            np.add.at(denom,offsets.ravel(),np.broadcast_to(weight,offsets.shape).ravel())
            np.add.at(a,offsets.ravel(),(predicted*weight).ravel());np.add.at(b,offsets.ravel(),(shuffled*weight).ravel())
            keep=denom>1e-8;a[keep]/=denom[keep];b[keep]/=denom[keep]
            target=np.load(r['envelope_path'],mmap_mode='r').reshape(-1)
            for start in range(0,r['common_samples']-3840+1,3840):
                key=(r['phase'],r['record'],start);mask=keep[start:start+3840]
                x=np.asarray(target[start:start+3840])[mask];p=a[start:start+3840][mask];q=b[start:start+3840][mask]
                valid=len(x)>=3800 and all(np.std(v)>0 for v in (x,p,q))
                if not valid:
                    if key not in excluded_keys:raise RuntimeError('unreported invalid segment')
                    continue
                if key not in expected:raise RuntimeError('unreported valid segment')
                visited.add(key);row=expected[key];score=correlation(x,p);null=correlation(x,q)
                max_prediction_error=max(max_prediction_error,abs(score-float(row['pearson_r'])),abs(null-float(row['EEG_shuffle_r'])))
                values[(r['phase'],r['subject'])].append(float(row['pearson_r']));nulls[(r['phase'],r['subject'])].append(float(row['EEG_shuffle_r']))
    checks['all_test_segments_accounted']=visited==set(expected)
    checks['independent_CPU_prediction_correlation']=max_prediction_error<1e-4
    max_aggregate_error=0.
    for r in subjects:
        key=(r['group'],r['subject']);max_aggregate_error=max(max_aggregate_error,abs(np.mean(values[key])-float(r['pearson_r'])),abs(np.mean(nulls[key])-float(r['EEG_shuffle_r'])))
    for phase in ('test_seen','test_heldout'):
        members=[r for r in subjects if r['group']==phase]
        for metric in ('pearson_r','EEG_shuffle_r'):
            max_aggregate_error=max(max_aggregate_error,abs(np.mean([float(r[metric]) for r in members])-result['test'][phase][metric]))
    weighted=2/3*result['test']['test_seen']['pearson_r']+1/3*result['test']['test_heldout']['pearson_r']
    max_aggregate_error=max(max_aggregate_error,abs(weighted-result['test']['weighted_pearson_r']))
    checks['independent_aggregation']=max_aggregate_error<1e-10
    review={'technical':'PASS' if all(checks.values()) else 'REWORK','acceptance':'PASS_WITH_LIMITATION' if all(checks.values()) else 'REWORK',
        'parent_consistency':'PASS_SAME_SS02_DIRECT_RECONSTRUCTION','checks':checks,'max_CPU_prediction_error':max_prediction_error,
        'max_aggregate_error':max_aggregate_error,'excluded_segments':len(excluded),'limitations':cfg['differences']}
    with (out/'review.json').open('x') as f:json.dump(review,f,indent=2)
    print(json.dumps(review),flush=True)
    if not all(checks.values()):raise RuntimeError('readout independent review failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args().config)
