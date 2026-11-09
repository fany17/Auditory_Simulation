"""Read-only input dimensions/scales; no target correlations or model fit."""
import json
from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT=BASE/'s03_incremental_continuity_20261101/external_dtu'


def search(value,path='cfg'):
    hits=[]
    if isinstance(value,dict):
        for k,v in value.items():
            p=path+'/'+str(k)
            if re.search('unit|volt|scale|calib|gain',str(k),re.I):hits.append({'path':p,'field_type':type(v).__name__,'scalar_text':str(v)[:250] if np.isscalar(v) else None})
            hits+=search(v,p)
    elif isinstance(value,(list,tuple)) or isinstance(value,np.ndarray) and value.dtype.kind=='O':
        for i,v in enumerate(value):hits+=search(v,path+f'/{i}')
    elif isinstance(value,(str,np.str_)) and re.search('unit|volt|scale|calib|gain',str(value),re.I):
        hits.append({'path':path,'text':str(value)[:250]})
    return hits


def main():
    stats=np.load(BASE/'s03_strong_baselines_20261015/transformer_available01/prepared/train_only_normalization.npz')
    mean,std=stats['mean'],stats['std'];assert mean.shape==std.shape==(64,) and np.isfinite(std).all() and np.all(std>0)
    audit=json.loads((ROOT/'READABILITY_CHANNEL_EVENT_AUDIT.json').read_text());expected=audit['subjects'][0]['labels'][:64]
    rows=[]
    for person in range(1,19):
        d=loadmat(ROOT/f'extracted/S{person}_data_preproc.mat',simplify_cells=True)['data']
        unit_hits=search(d['cfg']);unit_hits+=search(d['dim'],'dim');means=[];stds=[];near_constant=[]
        for i,eeg in enumerate(d['eeg']):
            labels=d['dim']['chan']['eeg'][i].tolist();indices=[labels.index(n) for n in expected]
            x=np.asarray(eeg[:,indices],dtype=np.float64);x=x-x.mean(1,keepdims=True)
            m=x.mean(0);s=x.std(0,ddof=0)
            means.append(m);stds.append(s);near_constant.append(int(np.count_nonzero(s==0)))
        means=np.stack(means);stds=np.stack(stds)
        rows.append({'subject':f'S{person}','scalp_CAR_computed_for_readonly_audit':True,'trials':len(means),
            'mean_over_trial_means_by_channel':means.mean(0).tolist(),
            'median_within_trial_std_by_channel':np.median(stds,0).tolist(),
            'channel_trial_std_min_median_max':[float(stds.min()),float(np.median(stds)),float(stds.max())],
            'absolute_trial_mean_min_median_max':[float(np.abs(means).min()),float(np.median(np.abs(means))),float(np.abs(means).max())],
            'exact_constant_channel_trial_count':sum(near_constant),'unit_scale_field_or_text_hits':unit_hits,
            'std_ratio_to_source_train_channel_std_min_median_max':[float((stds/std).min()),float(np.median(stds/std)),float((stds/std).max())]})
        print(json.dumps({'input_audit_subject':person,'exact_constant_channel_trials':sum(near_constant)}),flush=True)
    result={'no_models_or_target_correlations':True,'normalization_parameters_fit_for_models':False,
        'source_train426_stats_only':{'mean':mean.tolist(),'std':std.tolist(),'physical_source_contract':'publisher/MNE V*1e6 microvolts; actual historical generator revision unknown'},
        'DTU_subjects':rows,'unit_interpretation':'scale comparisons are descriptive ONLY; overlapping magnitude cannot prove DTU microvolt units',
        'temporal_interpretation':'whole-trial descriptive moments here are NOT applied as inference transforms; using whole50s moments would add future input permission',
        'preprocessing_difference':'source0.5Hz/MWF/scalp64CAR vs DTU0.1Hz/EOGdenoise/66CAR; physical64 selection+scalpCAR is fixed coordinate adaptation',
        'EEG_physical_unit_verified':False}
    with (ROOT/'INPUT_DIMENSIONAL_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'PASS':'read-only audit complete','unit_hits':sum(len(r['unit_scale_field_or_text_hits']) for r in rows),
        'constant_channel_trials':sum(r['exact_constant_channel_trial_count'] for r in rows)}))


if __name__=='__main__':main()
