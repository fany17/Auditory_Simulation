"""All-subject DTU readability/coordinate/channel audit, no fitting/scores."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

BASE=Path('/home/fanyu/auditory_simulation_m6a/pub_01')
ROOT=BASE/'s03_incremental_continuity_20261101/external_dtu'
PROVENANCE=BASE/'s06/sparrkulee_channel_provenance_20260920_v3/channel_provenance.csv'


def main():
    assert (ROOT/'DOWNLOAD_EXTRACT_COMPLETED.json').exists()
    expected=['Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7','P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz','Fp2','AF8','AF4','AFz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz','C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2']
    assert PROVENANCE.exists()
    with PROVENANCE.open() as f:
        source=next(row for row in csv.DictReader(f) if row['status']=='PASS_HEADER_PROVENANCE')
    assert json.loads(source['channel_names'])==expected
    rows=[];subjects=[]
    for person in range(1,19):
        path=ROOT/f'extracted/S{person}_data_preproc.mat';d=loadmat(path,simplify_cells=True)['data']
        assert d['fsample']=={'eeg':64,'wavA':64,'wavB':64}
        arrays=d['eeg'];a=d['wavA'];b=d['wavB'];labels=d['dim']['chan']['eeg'];events=d['event']['eeg']
        assert len(arrays)==len(a)==len(b)==len(labels)==len(events)
        shapes=[]
        for i,(eeg,attended,unattended,names,event) in enumerate(zip(arrays,a,b,labels,events)):
            names=[str(n) for n in names.tolist()];assert len(names)==len(set(names))==66
            indices=[names.index(n) for n in expected]
            assert set(names)-set(expected)=={'EXG1','EXG2'}
            n=len(eeg);assert eeg.shape==(n,66) and attended.shape==unattended.shape==(n,)
            assert np.isfinite(eeg).all() and np.isfinite(attended).all() and np.isfinite(unattended).all()
            assert np.std(attended)>0 and np.std(unattended)>0
            assert event['sample']==1 and event['value'] in [1,2]
            rows.append({'subject':f'S{person}','trial':i+1,'samples':n,'seconds':n/64,
                'channels_total':66,'scalp_channels':64,'all_finite':True,'target_nonconstant':True,
                'event_sample_1based':1,'attended_gender_code':event['value'],
                'channel_order_matches_Sparr64':indices==list(range(64)),
                'mean66_max_abs_relative_to_global_RMS':float(np.max(np.abs(eeg.mean(1)))/np.sqrt(np.mean(eeg**2)))})
            shapes.append(n)
        subjects.append({'subject':f'S{person}','trials':len(arrays),'samples_min':min(shapes),'samples_max':max(shapes),
            'labels':names,'source_file':path.name})
        print(json.dumps({'readable_subject':person,'trials':len(arrays)}),flush=True)
    result={'readability_pass':True,'subjects':subjects,'trials':rows,'total_trials':len(rows),
        'sample_rate_hz':64,'physical_scalp_labels_match_Sparr64':True,
        'scalp_selection_rule':'lookup physical names in frozen expected Biosemi64 order, remove EXG1/2; indices happen to be0..63 after verification',
        'reference':'published preproc final average over66 channels including mastoids; any new scalp-only CAR must be separately frozen',
        'wavA':'attended envelope per official preproc script; NOT raw audio','wavB':'unattended envelope',
        'event_sample':'trial-local 1-based sample1; not independent hardware-sync proof',
        'unknowns':['EEG physical unit not explicitly stored in preprocessed MAT','full per-trial audio filenames/story/room expinfo not in retained top-level schema; cfg may retain only first trial provenance'],
        'model_performance_computed':False,'normalization_or_adapter_fit':False,
        'provenance_source':str(PROVENANCE),'provenance_sha256':hashlib.sha256(PROVENANCE.read_bytes()).hexdigest(),
        'source_script_sha256':hashlib.sha256((ROOT/'preproc_data.m').read_bytes()).hexdigest()}
    with (ROOT/'READABILITY_CHANNEL_EVENT_AUDIT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'PASS':True,'subjects':len(subjects),'trials':len(rows),'lengths':sorted({r['samples'] for r in rows})}))


if __name__=='__main__':main()
